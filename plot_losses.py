import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg') # No display on the cluster
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from halo_utils import parse_args, paths


# Usage: python plot_losses.py --target {labels,mortality,readmission}
args = parse_args()
output_paths = paths(args.target)

# Load the per-epoch losses saved by train_model.py
losses = pickle.load(open(output_paths['losses'], 'rb'))
train_losses = losses['train_losses']
val_losses = losses['val_losses']
epochs = np.arange(1, len(train_losses) + 1)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(epochs, train_losses, marker='o', markersize=3, label='Train')

# train_model.py only records a validation loss for epochs that ran a validation check
if len(val_losses) == 0:
  print("Warning: no validation losses recorded, plotting training loss only")
else:
  if len(val_losses) != len(train_losses):
    print("Warning: %d validation losses for %d epochs, plotting them against the first %d epochs" % (len(val_losses), len(train_losses), len(val_losses)))
  val_epochs = np.arange(1, len(val_losses) + 1)
  ax.plot(val_epochs, val_losses, marker='o', markersize=3, label='Validation')

  # Note: the saved checkpoint is the best over every validation check, while val_losses
  # keeps only the last check of each epoch, so this marker can differ slightly from it
  best_epoch = int(np.argmin(val_losses)) + 1
  ax.axvline(best_epoch, color='gray', linestyle='--', linewidth=1, label='Best val (epoch %d)' % best_epoch)
  print("Best validation epoch: %d, Validation Loss: %.7f" % (best_epoch, val_losses[best_epoch - 1]))

ax.set_xlabel('Epoch')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylabel('Loss (BCE)')
ax.set_title('HALO training and validation loss (%s)' % args.target)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

for ext in ['pdf', 'png']:
  fig.savefig('%s.%s' % (output_paths['loss_plot'], ext), dpi=300)
  print("Saved %s.%s" % (output_paths['loss_plot'], ext))
