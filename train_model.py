import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from model import HALOModel
from config import HALOConfig
from halo_utils import parse_args, setup, get_batch, paths


# Usage: python train_model.py --target {labels,mortality,readmission}
args = parse_args()
config = HALOConfig(target=args.target)
output_paths = paths(args.target)

# Set random seeds and training device (GPU if available, otherwise CPU)
device = setup(seed=4)

train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('./data/valDataset.pkl', 'rb'))

def shuffle_training_data(train_ehr_dataset):
  np.random.shuffle(train_ehr_dataset)

# Initialize HALO model and Adam optimizer
model = HALOModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists(output_paths['model']):
  print("Loading previous model")
  checkpoint = torch.load(output_paths['model'], map_location=torch.device(device))
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])

# Train HALO and periodically evaluate on the validation dataset
global_loss = 1e10
train_losses = []
val_losses = []

for e in tqdm(range(config.epoch)):
  shuffle_training_data(train_ehr_dataset)
  epoch_train_loss = 0
  num_batches = 0
  epoch_val_loss = None

  for i in range(0, len(train_ehr_dataset), config.batch_size):
    model.train()
    
    batch_ehr, batch_mask = get_batch(train_ehr_dataset, i, config.batch_size, config)
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
    
    optimizer.zero_grad()
    loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
    epoch_train_loss += loss.item()
    num_batches += 1

    loss.backward()
    optimizer.step()
    
    if i % (500*config.batch_size) == 0: #500
      print("Epoch %d, Iter %d: Training Loss:%.6f"%(e, i, loss * 8))
    if i % (500*config.batch_size) == 0:
      if i == 0:
        continue
    
      model.eval()
      with torch.no_grad():
        val_l = []
        for v_i in range(0, len(val_ehr_dataset), config.batch_size):
          batch_ehr, batch_mask = get_batch(val_ehr_dataset, v_i, config.batch_size, config)
          batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
          batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
  
          val_loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
          val_l.append((val_loss).cpu().detach().numpy())
          
        cur_val_loss = np.mean(val_l)
        epoch_val_loss = cur_val_loss

        print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))

        # Save the model when validation loss improves
        if cur_val_loss < global_loss:
          global_loss = cur_val_loss
          state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': i
            }
          torch.save(state, output_paths['model'])
          print('\n------------ Save best model ------------\n')
          
         
  avg_train_loss = epoch_train_loss / num_batches
  train_losses.append(avg_train_loss)

  if epoch_val_loss is not None:
    val_losses.append(epoch_val_loss)

with open(output_paths['losses'], 'wb') as f:
  pickle.dump({
    'train_losses': train_losses,
    'val_losses': val_losses
  }, f)