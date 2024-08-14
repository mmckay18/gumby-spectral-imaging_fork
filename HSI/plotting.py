import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pathlib
import torch

def plot_tm_confusion_matrix(
    confusion_matrix, class_list=None, normalized=True, save_figure=True,
    rot_xlabs=True, top_x=True,
    title=None, output_dir=None, output_path=None, figsize=(8,6), 
    fontsize=12, annotate_accuracy=True, annotation_fontsize=20):
    '''Creates and saves confusion matrix figure

    INPUTS
        confusion_matrix (arr): NxN array from torchmetrics.ConfusionMatrix
        class_list (list): ordered list of strings corresponding to integers in labels
        normalized (bool): whether or not to row-normalize the confusion matrix
        save_figure (bool): whether or not to save the confusion matrix to file
        title (str): title for figure
        output_dir (pathlib.Path object): will save fig to output_dir / confusion_matrix.png
        output_path (pathlib.Path or str): absolute path to save figure to. Superscedes input directory.
        figsize (tuple): dimensions of figure
    OUTPUTS
        
    '''
    accuracy=confusion_matrix.diagonal().sum()/confusion_matrix.sum()
    num_classes = len(class_list)

    # create row-normalized CM
    normed_confusion_matrix = confusion_matrix / confusion_matrix.sum(
        axis=1,
        keepdims=True
    )

    if normalized:
        to_plot = normed_confusion_matrix
        vmin, vmax = 0.0, 1.0
        fmt="0.2f"
        cbar_label = 'Accuracy'
    else:
        to_plot = confusion_matrix.astype(int)
        vmin, vmax = 0, np.max(confusion_matrix)
        fmt = "d"
        cbar_label = 'Number'

    labs = [x.replace('_', ' ') for x in class_list]
    figure, axes = plt.subplots(figsize=figsize)
    if top_x:
        axes.xaxis.tick_top()
    seaborn.heatmap(
        to_plot,
        annot=True,
        fmt=fmt,
        ax=axes,
        xticklabels=labs,
        yticklabels=labs,
        annot_kws={'size': fontsize+4},
        cbar_kws={'label': cbar_label},
        vmin=vmin,
        vmax=vmax,
    )
    # make ticks pretty
    axes.set_ylim([len(labs), 0])
    if rot_xlabs:
        if top_x:
            plt.xticks(rotation=45, ha='left', size=fontsize)
        else:
            plt.xticks(rotation=45, ha='right', size=fontsize)
    else:
        plt.xticks(rotation=0, ha='center', size=fontsize)
    plt.yticks(rotation=0, ha='right', size=fontsize)

    if not annotate_accuracy and title is not None:
        title += f'(accuracy: {accuracy:.3f})'
    if title is not None:
        plt.title(title, size=fontsize+4)
    
    if annotate_accuracy:
        # add accuracy to figure
        axes.annotate(
            f'Accuracy:\n{accuracy:.3f}',
            xy=(.025, .975), 
            xycoords='figure fraction',
            horizontalalignment='left', 
            verticalalignment='top',
            fontsize=annotation_fontsize
        )
    figure.tight_layout()
    
    # save to file, if needed
    if output_dir is None:
        output_dir = pathlib.Path('.')
    if output_path is None:
        output_path = output_dir / 'confusion_matrix.png'
    if save_figure:
        figure.savefig(output_path, dpi=300)
    return figure

def plot_training_curve(model_weights_dir, log=False, results_dir=None):
    if results_dir is None:
#         results_dir = pathlib.Path('/qfs/projects/thidwick/weights/manga/')
        results_dir = pathlib.Path('/gscratch/astro/mmckay18/DATA/weights/')

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, figsize=(12,3))
    [ax.set_xlabel('Epoch', size=12) for ax in [ax1,ax2,ax3]]
    ax1.set_ylabel('Loss', size=14)
    ax2.set_ylabel('Performance Metric', size=14)
    ax3.set_ylabel('Learning Rate', size=14)

    train_loss = torch.load( model_weights_dir / 'train_curve_loss.pt').numpy()
    val_loss = torch.load( model_weights_dir / 'val_curve_loss.pt').numpy()
    if (model_weights_dir / 'val_curve_metric.pt').exists():
        train_metric = torch.load( model_weights_dir / 'train_curve_metric.pt').numpy()
        val_metric = torch.load( model_weights_dir / 'val_curve_metric.pt').numpy()
    else:
        train_metric = torch.load( model_weights_dir / 'train_curve_acc.pt').numpy()
        val_metric = torch.load( model_weights_dir / 'val_curve_acc.pt').numpy()
    lr = torch.load( model_weights_dir / 'lr_curve.pt').numpy()

    train_loss = train_loss[train_loss > 0.0]
    train_metric = train_metric[train_metric > 0.0]
    val_loss = val_loss[val_loss > 0.0]
    val_metric = val_metric[val_metric > 0.0]
    lr = lr[lr > 0.0]
    print(len(train_loss))

    epochs = np.arange(len(train_loss))

    ax1.plot(np.arange(len(train_loss)), train_loss, lw=2, label='Train')
    ax1.plot(np.arange(len(val_loss)), val_loss, lw=2, label='Val')
    ax1.legend()

    ax2.plot(np.arange(len(train_metric)), train_metric, lw=2, label='Train')
    ax2.plot(np.arange(len(val_metric)), val_metric, lw=2, label='Val')

    ax3.plot(np.arange(len(lr)), lr, lw=2)

    if log:
        [ax.set_yscale('log') for ax in fig.axes]

    plt.suptitle(model_weights_dir.relative_to(results_dir).as_posix(), size=14)
    plt.tight_layout()
    return