import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from matplotlib.gridspec import GridSpec
import numpy as np

def plot_results(history, title='Model Results', filename=None):
  
  # Convert metrics to easy dict for easy usage.
  metrics = edict(history.history)

  # Create the grid
  fig = plt.figure(tight_layout=True, figsize=(10, 8))
  gs = GridSpec(3, 2, figure=fig)
  ax1 = fig.add_subplot(gs[:2, 0])
  ax2 = fig.add_subplot(gs[:2, 1])
  ax3 = fig.add_subplot(gs[2, :])
  
  # Plot the Loss
  ax1.plot(history.epoch, metrics.loss)
  ax1.plot(history.epoch, metrics.val_loss)
  ax1.set_xlabel('Epochs')
  ax1.set_title('Loss', fontweight='bold')
  ax1.legend(['Train', 'Test'])
  
  # Plot the F1
  ax2.plot(history.epoch, np.array(metrics.f1).mean(axis=1))
  ax2.plot(history.epoch, np.array(metrics.val_f1).mean(axis=1))
  ax2.set_xlabel('Epochs')
  ax2.set_title('F1', fontweight='bold')
  ax2.legend(['Train', 'Test'])
  
  # Plot the Table
  fmt = lambda v: f'{np.mean(v):.03f}'
  ax3.axis('off')
  ax3.grid(False)
  table = ax3.table(
    colLabels=[ 'Metric', 'Train', 'Test' ],
    cellText=[
      [ 'F1 (Avg)',    fmt(metrics.f1[-1]),    fmt(metrics.val_f1[-1])    ],
      [ 'F1 (Neg)',    fmt(metrics.f1[-1][0]), fmt(metrics.val_f1[-1][0]) ],
      [ 'F1 (Pos)',    fmt(metrics.f1[-1][1]), fmt(metrics.val_f1[-1][1]) ],
      [ 'Precision',   fmt(metrics.prec[-1]),  fmt(metrics.val_prec[-1])  ],
      [ 'Recall',      fmt(metrics.rec[-1]),   fmt(metrics.val_rec[-1])   ],
      [ 'AUC',         fmt(metrics.auc[-1]),   fmt(metrics.val_auc[-1])   ],
    ],
    colColours=plt.cm.BuPu(np.full(3, 0.2)),
    loc='center',
    cellLoc='center',
  )
  table.scale(1, 1.75)
  table.set_fontsize(12)
  
  # Add the title
  fig.suptitle(title, fontsize=16, fontweight='bold')
  if filename: plt.savefig(filename)

  plt.show()
  plt.close()