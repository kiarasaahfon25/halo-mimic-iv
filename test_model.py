import torch
import pickle
import numpy as np
from tqdm import tqdm
from model import HALOModel
from config import HALOConfig
from halo_utils import parse_args, setup, get_batch, paths


# Usage: python test_model.py --target {labels,mortality,readmission}
args = parse_args()
config = HALOConfig(target=args.target)
output_paths = paths(args.target)
device = setup(seed=4)

# Load training and test EHR datasets and supporting code/label mappings
train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
test_ehr_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))
index_to_code = pickle.load(open("./data/indexToCode.pkl", "rb"))
id_to_label = pickle.load(open("./data/idToLabel.pkl", "rb"))
train_c = set([c for p in train_ehr_dataset for v in p['visits'] for c in v])
test_ehr_dataset = [{config.target: p[config.target], 'visits': [[c for c in v if c in train_c] for v in p['visits']]} for p in test_ehr_dataset]

def conf_mat(x, y):
  totaltrue = np.sum(x)
  totalfalse = len(x) - totaltrue
  truepos, totalpos = np.sum(x & y), np.sum(y)
  falsepos = totalpos - truepos
  return np.array([[totalfalse - falsepos, falsepos], #true negatives, false positives
                   [totaltrue - truepos, truepos]]) #false negatives, true positives

# Load the trained HALO model checkpoint for evaluation on held-out test data
model = HALOModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

checkpoint = torch.load(output_paths['model'], map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])


# Evaluate the trained HALO model on the test dataset
confusion_matrix = None
probability_list = []
loss_list = []
n_visits = 0
n_pos_codes = 0
n_total_codes = 0
model.eval()
with torch.no_grad():
 for v_i in tqdm(range(0, len(test_ehr_dataset), 2*config.batch_size)):
   # Get batch inputs
   batch_ehr, batch_mask = get_batch(test_ehr_dataset, v_i, 2*config.batch_size, config)
   batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
   batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
    
   # Get batch outputs
   test_loss, predictions, labels = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
   batch_mask_array = batch_mask.squeeze(-1).cpu().detach().numpy()
   rounded_preds = np.around(predictions.cpu().detach().numpy()).transpose((2,0,1)) 
   rounded_preds = rounded_preds + batch_mask_array - 1 # Setting the masked visits to be -1 to be ignored by the confusion matrix
   rounded_preds = rounded_preds.flatten()
   true_values = labels.cpu().detach().numpy().transpose((2,0,1))
   true_values = true_values + batch_mask_array - 1 # Setting the masked visits to be -1 to be ignored by the confusion matrix
   true_values = true_values.flatten()

   # Append test lost
   loss_list.append(test_loss.cpu().detach().numpy())
    
   # Add number of visits and codes
   n_visits += torch.sum(batch_mask).cpu().item()
   n_pos_codes += torch.sum(labels).cpu().item()
   n_total_codes += (torch.sum(batch_mask) * config.total_vocab_size).cpu().item()

   # Add confusion matrix
   batch_cmatrix = conf_mat(true_values == 1, rounded_preds == 1)
   batch_cmatrix[0][0] = torch.sum(batch_mask) * config.total_vocab_size - batch_cmatrix[0][1] - batch_cmatrix[1][0] - batch_cmatrix[1][1] # Remove the masked values
   confusion_matrix = batch_cmatrix if confusion_matrix is None else confusion_matrix + batch_cmatrix

   # Calculate and add probabilities 
   # Note that the masked codes will have probability 1 and be ignored
   label_probs = torch.abs(labels - 1.0 + predictions)
   log_prob = torch.sum(torch.log(label_probs)).cpu().item()
   probability_list.append(log_prob)

# Save intermediate values in case of error
intermediate = {}
intermediate["Losses"] = loss_list
intermediate["Confusion Matrix"] = confusion_matrix
intermediate["Probabilities"] = probability_list
intermediate["Num Visits"] = n_visits
intermediate["Num Positive Codes"] = n_pos_codes
intermediate["Num Total Codes"] = n_total_codes
pickle.dump(intermediate, open(output_paths['test_intermediate'], "wb"))

#Extract, save, and display test metrics
avg_loss = np.nanmean(loss_list)
tn, fp, fn, tp = confusion_matrix.ravel()
acc = (tn + tp)/(tn+fp+fn+tp)
prc = tp/(tp+fp)
rec = tp/(tp+fn)
f1 = (2 * prc * rec)/(prc + rec)
log_probability = np.sum(probability_list)
pp_visit = np.exp(-log_probability/n_visits)
pp_positive = np.exp(-log_probability/n_pos_codes)
pp_possible = np.exp(-log_probability/n_total_codes)

metrics_dict = {}
metrics_dict['Test Loss'] = avg_loss
metrics_dict['Confusion Matrix'] = confusion_matrix
metrics_dict['Accuracy'] = acc
metrics_dict['Precision'] = prc
metrics_dict['Recall'] = rec
metrics_dict['F1 Score'] = f1
metrics_dict['Test Log Probability'] = log_probability
metrics_dict['Perplexity Per Visit'] = pp_visit
metrics_dict['Perplexity Per Positive Code'] = pp_positive
metrics_dict['Perplexity Per Possible Code'] = pp_possible
pickle.dump(metrics_dict, open(output_paths['test_metrics'], "wb"))

print("Average Test Loss: ", avg_loss)
print("Confusion Matrix: ", confusion_matrix)
print('Accuracy: ', acc)
print('Precision: ', prc)
print('Recall: ', rec)
print('F1 Score: ', f1)
print('Test Log Probability: ', log_probability)
print('Perplexity Per Visit: ', pp_visit)
print('Perplexity Per Positive Code: ', pp_positive)
print('Perplexity Per Possible Code: ', pp_possible)

def sample_sequence(model, length, context, batch_size, device='cuda', sample=True):
  empty = torch.zeros((1,1,config.total_vocab_size), device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
  context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
  prev = context.unsqueeze(1)
  context = None
  with torch.no_grad():
    for _ in range(length-1):
      prev = model.sample(torch.cat((prev,empty), dim=1), sample)
      if torch.sum(torch.sum(prev[:,:,config.code_vocab_size+config.label_vocab_size+1], dim=1).bool().int(), dim=0).item() == batch_size:
        break
  ehr = prev.cpu().detach().numpy()
  prev = None
  empty = None
  return ehr

def convert_ehr(ehrs, index_to_code=None):
  ehr_outputs = []
  for i in range(len(ehrs)):
    ehr = ehrs[i]
    ehr_output = []
    labels_output = ehr[1][config.code_vocab_size:config.code_vocab_size+config.label_vocab_size]
    if index_to_code is not None:
      labels_output = [index_to_code[idx + config.code_vocab_size] for idx in np.nonzero(labels_output)[0]]
    for j in range(2, len(ehr)):
      visit = ehr[j]
      visit_output = []
      indices = np.nonzero(visit)[0]
      end = False
      for idx in indices:
        if idx < config.code_vocab_size: 
          visit_output.append(index_to_code[idx] if index_to_code is not None else idx)
        elif idx == config.code_vocab_size+config.label_vocab_size+1:
          end = True
      if visit_output != []:
        ehr_output.append(visit_output)
      if end:
        break
    # Store the generated target the same way as the real data:
    # a label vector for 'labels', a single 0/1 value for mortality and readmission
    target_output = labels_output if config.label_vocab_size > 1 else int(labels_output[0])
    ehr_outputs.append({'visits': ehr_output, config.target: target_output})
  ehr = None
  ehr_output = None
  labels_output = None
  visit = None
  visit_output = None
  indices = None
  return ehr_outputs

# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
stoken = np.zeros(config.total_vocab_size)
stoken[config.code_vocab_size+config.label_vocab_size] = 1
for i in tqdm(range(0, len(train_ehr_dataset), config.sample_batch_size)):
  bs = min([len(train_ehr_dataset)-i, config.sample_batch_size])
  batch_synthetic_ehrs = sample_sequence(model, config.n_ctx, stoken, batch_size=bs, device=device, sample=True)
  batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
  synthetic_ehr_dataset += batch_synthetic_ehrs

pickle.dump(synthetic_ehr_dataset, open(output_paths['synthetic'], 'wb'))