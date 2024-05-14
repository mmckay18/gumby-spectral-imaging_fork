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
from regression import normalizeOxygenLabels
from eval_utils import calc_regression_metrics

data_dir = pathlib.Path('/qfs/projects/thidwick/manga')

def eval_alt_split(split_dir, label_task='N2', split='test', 
                   easy_splits=False, patch_norm='global', 
                   OH_key: str='default', task: str='classification', 
                   regression_norm: str='scaledmax'):
    """Evaluate entire data split for alternative method
    """
    split_str = '_easy' if easy_splits else ''
    csv_dir = data_dir / 'splits' / split_dir
    csv_path = csv_dir / f'{split}-{patch_norm}_{label_task}{split_str}.csv'
    source_csv_path =  csv_dir / f'{split}-{patch_norm}{split_str}.csv'

    # binning
    if task == 'regression':
        label_fn = normalizeOxygenLabels(normalization=regression_norm)
    else:
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
    if task == 'regression':
        labs = label_fn.unnormalize(df.ground_truth)
        preds = label_fn.unnormalize(df.encoded_label)
        out = {'labels':labs,'predictions':preds}
        # raw values
        labs = pow(10, labs-12.0)
        preds = pow(10, preds-12.0)
        out.update(calc_regression_metrics(labs, preds))
    else:
        labs = df.ground_truth
        preds = df.encoded_label
        out = {'labels':labs,'predictions':preds}
        # calc output metrics
        accuracy = sum(preds == labs) / len(labs)
        out.update({'accuracy':accuracy})

    MSE = np.sum((labs - preds)**2.0)/len(labs)
    MAE = np.sum(np.abs(labs - preds))/len(labs)
    MAPE = np.sum(np.abs(labs - preds)/labs)/len(labs)
    out.update({'MSE':MSE, 'MAE':MAE, 'MAPE':MAPE})
    return out

def compare_gal_labels(plate_id, label_task='N2', OH_key='default', 
                       task='classification', regression_norm='scaledmax'):
    """return metrics for input galaxy
    """
    use_int_labels = False if task == 'regression' else True
    fits_file = plateid_to_fits_file(plate_id)
    GT_label_path = get_label_path(fits_file=fits_file, label_task='logOH')
    GT_map = create_OH_map(GT_label_path, OH_key=OH_key, use_int_labels=use_int_labels)
    
    label_path = get_label_path(fits_file=fits_file, label_task=label_task)
    label_map = create_OH_map(label_path, OH_key=OH_key, use_int_labels=use_int_labels)
    out = {
        'label_map': label_map.copy(),
        'GT_map': GT_map.copy()
    }
    if task == 'classification':
        # mask out pixels where they aren't equal
        label_map[label_map != GT_map] = np.ma.masked
        out['accuracy'] = label_map.count()/GT_map.count()
        out['acc_map'] = label_map.copy()
        GT_map = create_OH_map(GT_label_path,OH_key=OH_key,use_int_labels=False)
        label_map = create_OH_map(label_path,OH_key=OH_key,use_int_labels=False)
    
    out['MSE'] = np.sum((GT_map - label_map)**2.0)/GT_map.count()
    out['MAE'] = np.sum(np.abs(GT_map - label_map))/GT_map.count()
    out['logMAPE'] = np.sum(np.abs(GT_map - label_map)/GT_map)/GT_map.count()
    out['MAPE'] = np.sum(np.abs(pow(10,GT_map) - pow(10,label_map))/pow(10,GT_map))/GT_map.count()
    return out

def plot_gal_labels(plate_id: str=None, label_task:str='N2', OH_key='default', OH_log=False,
                    cname:str='viridis', figsize:tuple=(9,3), title:str=None, 
                    plot_titles:list=['Ground Truth', 'FWHM ratio', 'Accuracy'],
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

def visualize_gal_labels(plate_id:str, output:dict, label_task="N2",
    OH_key='default', OH_log=False, bg_index=-1, metric='MAPE',
    task='classification',regression_norm='scaledmax', regression_scale='logOH12',
    plot_titles = ['Labels', 'Predictions'],
    cname:str='viridis', figsize:tuple=(6,3), title:str=None, 
    cb_label:str=None, save_fig:bool=False, output_path:str=None):
    """
    """
    index_to_name=get_index_to_name(OH_key, log=OH_log)

    pred_map = np.ma.masked_where(output['pred_map'] == bg_index, output['pred_map'])
    label_map = np.ma.masked_where(output['label_map'] == bg_index, output['label_map'])

    # colorbar
    cmap = mpl.cm.__getattribute__(cname)
    if task == 'regression':
        if regression_scale == 'logOH12':
            vmin,vmax=8.34,8.64
        if regression_scale == 'ppm':
            vmin,vmax=220,430
        if regression_scale == 'logOH':
            vmin,vmax=-3.66,-3.36
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        sM = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        bounds = np.linspace(vmin,vmax,7)
    else:
        num_classes = len(index_to_name)
        bounds = np.arange(num_classes+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        sM = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, axes = plt.subplots(1,2,
        figsize=figsize,
        constrained_layout=True, 
        gridspec_kw={'width_ratios': [1, 1]},
    )
    to_plot = [label_map, pred_map]
    for i,this_map in enumerate(to_plot):
        ax = fig.axes[i]

        if task == 'regression':
            if regression_scale == 'logOH':
                this_map = this_map - 12.0
            if regression_scale == 'ppm':
                this_map = pow(10, this_map - 6.0)

        ax.imshow(this_map, cmap=cname, norm=norm, origin='lower')
        ax.set_title(plot_titles[i], size=16)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    if metric in ['accuracy', 'MAPE']:
        metric_str1 = f"{output[metric]*100.:.0f}%"
    if metric == 'logMAPE':
        metric_str1 = f"{output[metric]*100.:.2f}%"
    if metric == 'ppmMAPE':
        metric_str1 = f"{output[metric]*100.:.0f}%"
    if 'MAPE' in metric:
        metric_str1 = metric_str1 + ' err'
    else:
        metric_str1 = f"{metric}={output[metric]:.4f}"
    fig.axes[1].annotate(metric_str1, 
            xy=(0.02, 0.98), xycoords='axes fraction', 
            ha='left', va='top', fontsize=14)
    
    if title is not None:
        fig.suptitle(title, size=18)
    if task == 'regression':
        cbar = fig.colorbar(
            sM,
            ticks=bounds,
            ax=axes[1]
        )
        if regression_scale == 'ppm':
            cbar.set_ticklabels([f'{tick:.0f}' for tick in bounds])
        if (regression_scale == 'logOH12') | (regression_scale == 'logOH'):
            cbar.set_ticklabels([f'{tick:.2f}' for tick in bounds])
    else:
        cbar = fig.colorbar(
            sM,
            ticks=bounds,
            boundaries=bounds,
            ax=axes[1]
        )
        cbar.set_ticks([i+0.5 for i in range(num_classes)])
        if regression_scale == 'ppm':
                cbar.set_ticklabels([f'{int(pow(10,float(index_to_name[i])-6.0))}' for i in range(num_classes)])
        else:
            cbar.set_ticklabels([index_to_name[i] for i in range(num_classes)])
    if cb_label is not None:
        cbar.set_label(cb_label)
    if save_fig:
        if output_path is None:
            output_path = pathlib.Path(f'./figs/tmp.png')
        else: 
            output_path = pathlib.Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return