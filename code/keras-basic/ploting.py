'''
pip3 install matplotlib seaborn numpy pandas
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from header import labels

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_metrics(history):
    metrics = ['loss', 'precision_m', 'recall_m', 'f1_m']
    num = 1
    for label in labels:
        for metric in metrics:
            key = label + '_' + metric if len(labels)>1 else metric
            name = key.replace("_m", "").replace("_", " ").capitalize()
            plt.subplot(len(labels), len(metrics), num)
            num += 1
            plt.plot(history.epoch, history.history[key], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_'+key], color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8,1])
            else:
                plt.ylim([0,1])
            plt.legend()
    plt.show()

