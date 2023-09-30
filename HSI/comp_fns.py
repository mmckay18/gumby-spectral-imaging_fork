import pandas as pd
import numpy as np
from typing import Callable
from data_fns import BinLogOHLabels
from utils import(
    plateid_to_fits_file,
    get_label_path
)
from utils import get_index_to_name
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from vis_fns import create_OH_map

data_dir = pathlib.Path('/qfs/projects/gumby/data/manga')

def eval_alt_split(split_dir, label_task='N2', split='test', easy_splits=False, 
    patch_norm='global', OH_key: str='default'):
    """Evaluate entire data split for alternative method
    """
    split_str = '_easy' if easy_splits else ''
    csv_dir = data_dir / 'splits' / split_dir
    csv_path = csv_dir / f'{split}-{patch_norm}_{label_task}{split_str}.csv'
    source_csv_path =  csv_dir / f'{split}-{patch_norm}{split_str}.csv'

    # binning
    label_fn=BinLogOHLabels(OH_key=OH_key)
    # get labels from logOH file as ground truth
    sdf = pd.read_csv(source_csv_path)
    sdf.label = sdf.label.map(float)
    sdf['encoded_label'] = label_fn(sdf.label)
    # N2 diagnostic prediction
    df = pd.read_csv(csv_path)
    df.label = df.label.map(float)
    df['encoded_label'] = label_fn(df.label)
    df['ground_truth'] = sdf.encoded_label
    accuracy = sum(df.encoded_label == df.ground_truth) / len(df)
    MSE = np.sum((sdf.label - df.label)**2.0)/len(sdf)
    MAE = np.sum(np.abs(sdf.label - df.label))/len(sdf)
    MAPE = np.sum(np.abs(pow(10,sdf.label) - pow(10,df.label))/pow(10,sdf.label))/len(sdf)
    
    out = {'accuracy':accuracy,'MSE':MSE,'MAE':MAE, 'MAPE':MAPE}
    return out

def compare_gal_labels(plate_id, label_task='N2', OH_key='default'):
    """return metrics for input galaxy
    """
    fits_file = plateid_to_fits_file(plate_id)
    GT_label_path = get_label_path(fits_file=fits_file, label_task='logOH')
    GT_map = create_OH_map(GT_label_path, OH_key=OH_key)
    label_path = get_label_path(fits_file=fits_file, label_task=label_task)
    label_map = create_OH_map(label_path, OH_key=OH_key)
    out = {
        'label_map': label_map.copy(),
        'GT_map': GT_map.copy()
    }
    # mask out pixels where they aren't equal
    label_map[label_map != GT_map] = np.ma.masked
    out['accuracy'] = label_map.count()/GT_map.count()
    out['acc_map'] = label_map.copy()

    GT_map = create_OH_map(GT_label_path,OH_key=OH_key,use_int_labels=False)
    label_map = create_OH_map(label_path,OH_key=OH_key,use_int_labels=False)
    out['MSE'] = np.sum((GT_map - label_map)**2.0)/GT_map.count()
    out['MAE'] = np.sum(np.abs(GT_map - label_map))/GT_map.count()
    out['MAPE'] = np.sum(np.abs(pow(10,GT_map) - pow(10,label_map))/pow(10,GT_map))/GT_map.count()
    return out

def plot_gal_labels(plate_id: str=None, label_task:str='N2', OH_key='default', OH_log=False,
                    cname:str='viridis', figsize:tuple=(9,3), title:str=None, 
                    plot_titles:list=['O3N2 Ground Truth', 'N2 Comparison', 'Accuracy'],
                    cb_label:str=None, save_fig:bool=False, output_path:str=None):
    
    index_to_name = get_index_to_name(OH_key, log=OH_log)

    output = compare_gal_labels(
        plate_id, 
        label_task=label_task, 
        OH_key=OH_key
    )

    # colorbar
    num_classes = len(index_to_name)
    cmap = mpl.cm.__getattribute__(cname)
    bounds = np.arange(num_classes+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    sM = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, axes = plt.subplots(1,3,
        figsize=figsize,
        constrained_layout=True, 
        gridspec_kw={'width_ratios': [1, 1, 1]},
    )
    to_plot = ['GT_map', 'label_map', 'acc_map']
    for i,this_map in enumerate(to_plot):
        ax = fig.axes[i]
        ax.imshow(output[this_map], cmap=cname, norm=norm, origin='lower')
        ax.set_title(plot_titles[i], size=18)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    fig.axes[2].annotate(f"{output['accuracy']*100.:.0f}%", 
            xy=(0.02, 0.98), xycoords='axes fraction', 
            ha='left', va='top', fontsize=14)
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

def visualize_gal_labels(plate_id:str, output:dict, label_task:str='N2', 
    OH_key='default', OH_log=False, bg_index=-1, metric='accuracy',
    plot_titles = ['Ground Truth (O3N2)', 'Predictions', 'N2 Comparison'],
    cname:str='viridis', figsize:tuple=(9,3), title:str=None, 
    cb_label:str=None, save_fig:bool=False, output_path:str=None):
    """
    """
    index_to_name=get_index_to_name(OH_key, log=OH_log)

    pred_map = np.ma.masked_where(output['pred_map'] == bg_index, output['pred_map'])
    alt_output = compare_gal_labels(
        plate_id, 
        label_task=label_task, 
        OH_key=OH_key
    )

    # colorbar
    num_classes = len(index_to_name)
    cmap = mpl.cm.__getattribute__(cname)
    bounds = np.arange(num_classes+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    sM = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, axes = plt.subplots(1,3,
        figsize=figsize,
        constrained_layout=True, 
        gridspec_kw={'width_ratios': [1, 1, 1]},
    )
    to_plot = [alt_output['GT_map'], pred_map, alt_output['label_map']]
    for i,this_map in enumerate(to_plot):
        ax = fig.axes[i]
        ax.imshow(this_map, cmap=cname, norm=norm, origin='lower')
        ax.set_title(plot_titles[i], size=18)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    if metric in ['accuracy', 'MAPE']:
        metric_str1 = f"{output[metric]*100.:.0f}%"
        metric_str2 = f"{alt_output[metric]*100.:.0f}%"
    else:
        metric_str1 = f"{metric}={output[metric]:.4f}"
        metric_str2 = f"{metric}={alt_output[metric]:.4f}"
    fig.axes[1].annotate(metric_str1, 
            xy=(0.02, 0.98), xycoords='axes fraction', 
            ha='left', va='top', fontsize=14)
    fig.axes[2].annotate(metric_str2, 
            xy=(0.02, 0.98), xycoords='axes fraction', 
            ha='left', va='top', fontsize=14)
    if title is not None:
        fig.suptitle(title, size=18)
    cbar = fig.colorbar(
        sM,
        ticks=bounds,
        boundaries=bounds,
        ax=axes[2]
    )
    cbar.set_ticks([i+0.5 for i in range(num_classes)])
    cbar.set_ticklabels([index_to_name[i] for i in range(num_classes)])
    if cb_label is not None:
        cbar.set_label(cb_label)
    if save_fig:
        if output_path is None:
            output_path = pathlib.Path(f'./figs/tmp.png')
        fig.savefig(output_path, dpi=300)
    return