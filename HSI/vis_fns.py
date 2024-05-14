import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from utils import get_OH_bins_and_labels, get_index_to_name

def bin_OH_labels(input, OH_key='default'):
    OH_bins, _ = get_OH_bins_and_labels(OH_key)
    output = np.digitize(input, bins=OH_bins)
    return output

def create_OH_map(label_path, OH_key='default', use_int_labels=True, 
                  task='classification',regression_norm='scaledmax'):
    OH_bins, _ = get_OH_bins_and_labels(OH_key)
    input = np.load(label_path)
    input = np.ma.masked_array(input, mask=(input == 0), fill_value=0)
    if use_int_labels:
        output = np.digitize(input, bins=OH_bins)
        output = np.ma.array(output, mask=input.mask)
    else:
        output = input
    return output

def visualize_output(output, bg_index=-1, figsize=(8, 4), cb_label=None,
                     cname='viridis', save_fig=False, output_path=None, title=None, 
                     OH_key='default', OH_log=False, index_to_name=None):
    """Makes GT vs predicted figure for datacube (output)
    output is generated via datacube_evaluator from eval_utils.py
    """
    if index_to_name is None:
        index_to_name = get_index_to_name(OH_key, log=OH_log)
    num_classes=len(index_to_name)
    
    preds_mask = np.ma.masked_where(
        output['pred_map'] == bg_index,
        output['pred_map']
    )
    labs_mask = np.ma.masked_where(
        output['label_map'] == bg_index,
        output['label_map']
    )

    cmap = mpl.cm.__getattribute__(cname)
    bounds = np.arange(num_classes+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    sM = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True, 
                             gridspec_kw={'width_ratios': [1, 1]})

    ax = fig.axes[0]
    ax.imshow(labs_mask, cmap=cname, norm=norm, origin='lower')
    ax.set_title('Ground Truth', size=18)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    ax = fig.axes[1]
    ax.imshow(preds_mask, cmap=cname, norm=norm, origin='lower')
    ax.set_title('Predictions', size=18)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.annotate(f"Acc={output['accuracy']:.2f}", 
                xy=(0.02, 0.98), xycoords='axes fraction', 
                ha='left', va='top', fontsize=14)

    if title is not None:
        fig.suptitle(title, size=18)

    cbar = fig.colorbar(
        sM,
        ticks=bounds,
        boundaries=bounds,
        ax=axes[1]
    )
    cbar.set_ticks([i+0.5 for i in range(num_classes)])
    cbar.set_ticklabels([index_to_name[i] for i in range(num_classes)])
    if cb_label is not None:
        cbar.set_label(cb_label)
    if save_fig:
        if output_path is None:
            output_path = pathlib.Path(f'./figs/tmp.png')
        fig.savefig(output_path, dpi=300)
    return fig

def simple_map_viz(label_path, index_to_name=None, cname='viridis', figsize=(4,4), 
                   OH_key='default', OH_log=False, title=None):
    label_map = create_OH_map(label_path, OH_key=OH_key)
    if index_to_name is None:
        index_to_name = get_index_to_name(OH_key, log=OH_log)
    num_classes = len(index_to_name)

    cmap = mpl.cm.__getattribute__(cname)
    bounds = np.arange(num_classes+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    sM = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(label_map, cmap=cname, norm=norm, origin='lower')
    if title is not None:
        ax.set_title(title, size=18)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    cbar = fig.colorbar(
        sM,
        ticks=bounds,
        boundaries=bounds,
        ax=ax
    )
    cbar.set_ticks([i+0.5 for i in range(num_classes)])
    cbar.set_ticklabels([index_to_name[i] for i in range(num_classes)])
    
    return fig