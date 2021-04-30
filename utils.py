import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as layers

from easydict import EasyDict as edict
from matplotlib.gridspec import GridSpec

def baseline_model(input_shape=(28, 28, 1), output=10):
    """Create the baseline model.

    Args:
        input_shape (tuple, optional): The input shape, without batch. Defaults to (28, 28, 1).
        output (int, optional): Number of output classes. Defaults to 10.

    Returns:
        tf.keras.Sequential: The baseline model.
    """
    return tf.keras.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(64, 2, activation='relu', padding='same'),
        layers.MaxPool2D(2),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPool2D(2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(output, activation='softmax'),
    ])

def compile(model, n_classes):
    model.compile(
      optimizer=tf.optimizers.Adam(),
      loss=tf.losses.CategoricalCrossentropy(),
      metrics=[
          tf.metrics.Precision(name='precision'),
          tf.metrics.Recall(name='recall'),
          tf.metrics.AUC(name='auc'),
          tfa.metrics.F1Score(n_classes, name='f1'),
          tf.metrics.TruePositives(name='tp'),
          tf.metrics.TrueNegatives(name='tn'),
          tf.metrics.FalsePositives(name='fp'),
          tf.metrics.FalseNegatives(name='fn'),
      ],
  )

def evaluate(model, load_dataset, epochs=10, batch_size=64):
  (x_train, y_train), (x_test, y_test) = load_dataset()
  
  print('----- Data -----')
  print(f'Train : x={x_train.shape}, y={y_train.shape}')
  print(f'Test  : x={x_test.shape}, y={y_test.shape}')
  print(f'Splits: train={y_train.sum(axis=0).astype(int)}, test={y_test.sum(axis=0).astype(int)}')
  print('----------------') 

  return model.fit(
      x_train, y_train,
      callbacks=[ tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',
          patience=5,
          restore_best_weights=True
      ) ],
      validation_data=(x_test, y_test),
      epochs=epochs,
      batch_size=batch_size,
  )

def gpu_fix():
    """Fixes GPUs for windows.

    TensorFlow GPUs support on Windows is currently broken. Alowing memory
    growth fixes the problem and shouldn't cause any issues on other platforms.

    https://github.com/tensorflow/tensorflow/issues/45779#issuecomment-747403789
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            log_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f'{len(gpus)} Physical GPUs, {len(log_gpus)} Logical GPUs')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            raise e

def load_fashion():
  return preprocess(tf.keras.datasets.fashion_mnist.load_data)

def load_mnist():
  return preprocess(tf.keras.datasets.mnist.load_data)

def load_cifar10():
  return preprocess(tf.keras.datasets.cifar10.load_data)

def preprocess(load_dataset):
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = load_dataset()

    # Add dimension if necessary
    if len(x_train.shape) == 3:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
    
    # Convet inputs from 0..255 to 0..1
    x_train = x_train / 255
    x_test = x_test / 255

    # Convert outputs to one hot
    y_train = tf.one_hot(y_train, y_train.max() + 1).numpy().squeeze()
    y_test = tf.one_hot(y_test, y_test.max() + 1).numpy().squeeze()

    # Return
    return (x_train, y_train), (x_test, y_test)

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
            [ 'F1 (Avg)',  fmt(metrics.f1[-1]),    fmt(metrics.val_f1[-1])    ],
            [ 'F1 (Neg)',  fmt(metrics.f1[-1][0]), fmt(metrics.val_f1[-1][0]) ],
            [ 'F1 (Pos)',  fmt(metrics.f1[-1][1]), fmt(metrics.val_f1[-1][1]) ],
            [ 'Precision', fmt(metrics.prec[-1]),  fmt(metrics.val_prec[-1])  ],
            [ 'Recall',    fmt(metrics.rec[-1]),   fmt(metrics.val_rec[-1])   ],
            [ 'AUC',       fmt(metrics.auc[-1]),   fmt(metrics.val_auc[-1])   ],
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

def plot_history(history, title='History', keys=['loss', 'f1']):
    fig, axes = plt.subplots(nrows=1, ncols=len(keys), figsize=(len(keys) * 4, 4))

    for i, (ax, key) in enumerate(zip(axes, keys)):
        axes[i].plot(history.epoch, np.vstack(history.history[key]).mean(axis=1))
        axes[i].plot(history.epoch, np.vstack(history.history[f'val_{key}']).mean(axis=1))
        axes[i].legend(['Train', 'Test'])
        axes[i].set_title(key.title())
        axes[i].set_xlabel('Epochs')

    fig.suptitle(title, fontweight='bold')
    plt.show()
    plt.close()
