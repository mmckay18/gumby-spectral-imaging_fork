from tqdm import tqdm
import logging

from torchmetrics import MetricCollection
from torchmetrics.classification import(
    JaccardIndex, 
    Accuracy,
    ConfusionMatrix
)
from torchmetrics.regression import(
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanAbsoluteError,
)
import numpy as np
import torch
from HSI_SSFTT import get_SSFTT_model
from HSI_SIMPLENET import SimpleNet, SpectraNet
import pathlib
from utils import get_OH_bins_and_labels, get_index_to_name,get_num_classes
from data_fns import(
    HSICubeDataset,
    HSIPatchDataset,
    BinLogOHLabels,
    preprocData,
    repatchData,
    map_collate_fn,
    default_collate_fn
)
from regression import(
    normalizeOxygenLabels, 
    reg_collate_fn,
    reg_map_collate_fn
)

data_dir = pathlib.Path('/gscratch/scrubbed/mmckay18/DATA/')

def bin_OH_map(input, OH_key='default'):
    OH_bins, _ = get_OH_bins_and_labels(OH_key)
    output = np.digitize(input, bins=OH_bins)
    if type(input) is np.ma.core.MaskedArray:
        output = np.ma.array(output, mask=input.mask)
    return output

def get_test_dataloader(arch_type='ssftt', split_dir='OH_1', patch_norm='global', default_patch_size=9, OH_key='default',
                        spatial_patch_size=7, normalize=False, min_wave_ind=0, max_wave_ind=2040, task='classification', regression_norm='minmax',
                        batch_size=512, num_workers=0, easy_splits=False, use_local_data=False, local_dir='/raid/byle431'):
    """Returns dataloader for test split of input split_dir
    """
    if arch_type == 'spectranet': spatial_patch_size=1
    if (spatial_patch_size != default_patch_size) | ((min_wave_ind,max_wave_ind) != (0,2040)):
        preprocess_fn = repatchData(
            current_patch_size=default_patch_size,
            new_patch_size=spatial_patch_size,
            wave_inds=(min_wave_ind, max_wave_ind),
            normalize=normalize,
            add_extra_dim=arch_type == 'ssftt'
        )
    else:
        preprocess_fn = preprocData(
            normalize=normalize,
            add_extra_dim=arch_type == 'ssftt'
        )
    if task == 'regression':
        logging.info(f'Using regression normalization: {regression_norm}')
        label_fn = normalizeOxygenLabels(normalization=regression_norm)
        collate_fn = reg_collate_fn
    else:
        logging.info(f'Using OH_key: {OH_key}')
        label_fn = BinLogOHLabels(OH_key=OH_key)
        collate_fn = default_collate_fn
    
    # instantiate datasets
    split_str = '_easy' if easy_splits else ''
    csv_dir = data_dir / 'splits' / split_dir
    test_data = HSIPatchDataset(
        csv_dir / f'test-{patch_norm}{split_str}.csv',
        preprocess_fn=preprocess_fn,
        label_fn=label_fn,
        use_local_data=use_local_data,
        local_dir=local_dir
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    return test_dataloader

def load_trained_model(arch_type='ssftt', model_weights_dir=None, device=torch.device('cuda:0'), num_classes=6, 
                       spatial_patch_size=7, min_wave_ind=0, max_wave_ind=2040, **kwargs):
    """Returns SSFTT model with input specs and loads weights
    """
    if model_weights_dir is None:
        model_weights_dir = './'
    if type(model_weights_dir) is str:
        model_weights_dir = pathlib.Path(model_weights_dir)
    
    if arch_type == 'ssftt':
        logging.info('Loading SSFTT')
        model = get_SSFTT_model(
            num_classes=num_classes,
            min_wave_ind=min_wave_ind,
            max_wave_ind=max_wave_ind,
            spatial_patch_size=spatial_patch_size,
            **kwargs    
        )
    elif arch_type == 'simplenet':
        logging.info('Loading SimpleNet')
        model = SimpleNet(
            num_classes=num_classes,
            in_chans=max_wave_ind - min_wave_ind,
        )
    elif arch_type == 'spectranet':
        logging.info('Loading SpectraNet')
        model = SpectraNet(
            num_classes=num_classes
        )
    
    model.load_state_dict(torch.load(model_weights_dir / 'best_wts.pt', map_location=device))
    return model

def AllPredictions(dataloader, model, device=torch.cuda.current_device(), 
                   progress=True, OH_key='extra', log=True, 
                   confidence_flag=False, confidence_file='./tmp.txt',
                   cube_flag=False,
                   ):
    """
    Aggregate predictions of a model on a dataloader.
    """
    index_to_name = get_index_to_name(OH_key, log=log)
    num_classes = len(index_to_name)

    model.to(device)
    model.eval()

    kwargs_to_pass = {
        'task':'multiclass',
        'num_classes': num_classes,
        'multidim_average':'global', 
        'compute_on_cpu':True,
    }
    metric_collection = MetricCollection({
        'accuracy': Accuracy(average='micro', **kwargs_to_pass),
        'Top2Accuracy': Accuracy(top_k=2, average='micro', **kwargs_to_pass),
        'IOU': JaccardIndex(average='micro', num_classes=num_classes, task='multiclass', compute_on_cpu=True),
        'ClasswiseAccuracy': Accuracy(average=None, **kwargs_to_pass),
        'ClasswiseIOU':JaccardIndex(average=None, num_classes=num_classes, task='multiclass', compute_on_cpu=True),
        'CM':ConfusionMatrix(task='multiclass',num_classes=num_classes)
    }).to(device)
    regression_collection = MetricCollection({
        'MAE': MeanAbsoluteError(),
        'MSE':MeanSquaredError(),
        'MAPE':MeanAbsolutePercentageError()
    }).to(device)

    if progress:
        dl = tqdm(
            enumerate(dataloader), 
            total=len(dataloader), 
            desc="collecting predictions"
        )
    else:
        dl = enumerate(dataloader)
    if cube_flag:
        logging.info('Running in DATACUBE mode')
    all_map_inds = []
    with torch.no_grad():
        for i, lump in dl:
            if cube_flag:
                batch, target, map_index = lump
            else:
                batch, target = lump
            batch = batch.to(device)
            pred = model(batch)
            pred = torch.nn.functional.softmax(pred, dim=1)
            # append to file
            if confidence_flag:
                ind_letter = "w" if i == 0 else "a"
                with open(confidence_file, ind_letter) as f:
                    for p,t in zip(pred, target):
                        line = ','.join([f'{val.item():.4e}' for val in p])+f',{t.item()}'+'\n'
                        f.write(line)
            
            # add batch to metric collection
            metric_collection.update(pred.to(device), target.to(device))

            # the rest of the loop only requires the max class
            pred = torch.argmax(pred, dim=1)
            # compute for regression
            num_pred = pow(10, torch.tensor([float(index_to_name[x.item()]) for x in pred]))
            num_targ = pow(10, torch.tensor([float(index_to_name[x.item()]) for x in target]))
            regression_collection.update(num_pred.to(device), num_targ.to(device))

            if i == 0:
                all_pred = pred
                all_targ = target
            else:
                all_pred = torch.cat((all_pred, pred))
                all_targ = torch.cat((all_targ, target))
            
            if cube_flag:
                all_map_inds += map_index
    
    # compute final numbers
    all_pred = all_pred.cpu().numpy()
    all_targ = all_targ.cpu().numpy()

    output_dict = {
        'predictions':all_pred,
        'labels':all_targ,
        'map_index':all_map_inds,
    }

    # compute final numbers
    class_dict = metric_collection.compute()
    reg_dict = regression_collection.compute()
    # grab metrics off of GPU
    for mdict in [class_dict, reg_dict]:
        for metric_name, metric_value in mdict.items():
            # pop off single value unless it's an array
            if metric_value.numel() <= 1:
                output_dict[metric_name] = metric_value.cpu().item()
            else:
                output_dict[metric_name] = metric_value.cpu().numpy()
    
    metric_collection.reset()
    regression_collection.reset()
    return output_dict

def RegPredictions(dataloader, model, device=torch.cuda.current_device(), 
                   cube_flag=False, progress=True):
    """
    Aggregate predictions of a model on a dataloader.
    """
    model.to(device)
    model.eval()

    metric_collection = MetricCollection({
        'MAE': MeanAbsoluteError(),
        'MSE':MeanSquaredError(),
        'MAPE':MeanAbsolutePercentageError()
    }).to(device)

    if progress:
        dl = tqdm(
            enumerate(dataloader), 
            total=len(dataloader), 
            desc="collecting predictions"
        )
    else:
        dl = enumerate(dataloader)
    
    map_inds = []
    with torch.no_grad():
        for i, lump in dl:
            if cube_flag:
                batch, target, map_index = lump
            else:
                batch, target = lump
            batch = batch.to(device)
            pred = model(batch).squeeze(1)

            # unnormalize values
            pred = dataloader.dataset.label_fn.unnormalize(pred)
            pred = pow(10, pred-12.0)
            target = dataloader.dataset.label_fn.unnormalize(target)
            target = pow(10, target-12.0)

            if i == 0:
                all_pred = pred
                all_targ = target
            else:
                all_pred = torch.concat([all_pred, pred])
                all_targ = torch.concat([all_targ, target])
            if cube_flag:
                map_inds += map_index
            
            # add batch to metric collection
            metric_collection.update(pred.to(device), target.to(device))
    # compute final numbers
    mdict = metric_collection.compute()

    # add nice versions to the output
    output_dict = {
        'predictions':np.log10(all_pred.cpu().numpy())+12.0,
        'labels':np.log10(all_targ.cpu().numpy())+12.0
    }
    if cube_flag:
        output_dict['map_index'] = map_inds
    
    # grab metrics off of GPU
    for metric_name, metric_value in mdict.items():
        # pop off single value unless it's an array
        if metric_value.numel() <= 1:
            output_dict[metric_name] = metric_value.cpu().item()
        else:
            output_dict[metric_name] = metric_value.cpu().numpy()
    metric_collection.reset()

    # outputs are raw, this will calculate the log versions
    output_dict.update(calc_regression_metrics(all_targ.cpu(), all_pred.cpu()))
    
    return output_dict

def datacube_evaluator(fits_file, arch_type='ssftt', model_weights_dir=None, normalize=False, 
                       patch_norm='global', label_task='logOH', OH_key='extra',
                       default_patch_size=9, spatial_patch_size=7, 
                       task='classification', regression_norm='scaledmax',
                       min_wave_ind=0, max_wave_ind=2040, num_workers=0,
                       use_local_data=True, local_dir:str='/raid/byle431',
                       device=torch.cuda.current_device(), batch_size=512, **kwargs):
    '''Evaluates input fits file and creates spatial maps for labels + preds
    '''
    if arch_type == 'spectranet': spatial_patch_size=1
    
    if (spatial_patch_size != default_patch_size) | ((min_wave_ind,max_wave_ind) != (0,2040)):
        preprocess_fn = repatchData(
            current_patch_size=default_patch_size,
            new_patch_size=spatial_patch_size,
            wave_inds=(min_wave_ind, max_wave_ind),
            normalize=normalize,
            add_extra_dim=arch_type=='ssftt'
        )
    else:
        preprocess_fn = preprocData(
            normalize=normalize,
            add_extra_dim=arch_type=='ssftt'
        )
    if task == 'regression':
        logging.info('Using REGRESSION presents')
        label_fn =  normalizeOxygenLabels(normalization=regression_norm)
        collate_fn = reg_map_collate_fn
        num_classes=1
    else:
        logging.info('Using CLASSIFICATION presents')
#         if OH_key == 'BPT':
#             label_fn = BinBPTLabels(OH_key=OH_key)
#         else:
        label_fn = BinLogOHLabels(OH_key=OH_key)
        collate_fn = map_collate_fn
        num_classes = get_num_classes(OH_key)
    
    eval_data = HSICubeDataset(
        fits_file,
        preprocess_fn=preprocess_fn,
        label_fn=label_fn,
        patch_size=spatial_patch_size,
        patch_norm=patch_norm,
        label_task=label_task,
        use_local_data=use_local_data,
        local_dir=local_dir,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    model = load_trained_model(
        arch_type=arch_type,
        model_weights_dir=model_weights_dir, 
        spatial_patch_size=spatial_patch_size, 
        device=device,
        num_classes=num_classes
    )

    if task == 'regression':
        logging.info('Running RegPredictions')
        output_dictionary = RegPredictions(
            eval_dataloader, 
            model, 
            device=device, 
            cube_flag=True
        )
    else:
        logging.info('Running AllPredictions')
        output_dictionary = AllPredictions(
            eval_dataloader, 
            model, 
            device=device, 
            cube_flag=True
        )
    # Now set up maps
    # here we subtract 1 for background visualization
    label_map_size = eval_dataloader.dataset.label_map_size
    gt_map = np.zeros((label_map_size, label_map_size)) - 1
    pred_map = np.zeros((label_map_size, label_map_size)) - 1
    corr_map = np.zeros((label_map_size, label_map_size)) - 1

    for i in range(len(output_dictionary['predictions'])):
        map_index = tuple(output_dictionary['map_index'][i])
        gt_map[map_index] = output_dictionary['labels'][i]
        pred_map[map_index] = output_dictionary['predictions'][i]
        corr_map[map_index] = output_dictionary['predictions'][i] == output_dictionary['labels'][i]
    
    output_dictionary.update({
        'label_map': gt_map,
        'pred_map': pred_map,
        'correct_mask': corr_map,
    })
    return output_dictionary

def calc_regression_metrics(labels, predictions):
    if type(labels) == torch.Tensor:
        labels = labels.numpy()
    if type(predictions) == torch.Tensor:
        predictions = predictions.numpy()
    # assume that everything was computed on raw abundances
    labels = np.log10(labels) + 12.0
    predictions = np.log10(predictions) + 12.0

    MSE = np.sum((labels - predictions)**2.0)/len(labels)
    MAE = np.sum(np.abs(labels - predictions))/len(labels)
    MAPE = np.sum(np.abs((labels - predictions)/labels))/len(labels)
    out = {'logMSE':MSE, 'logMAE':MAE, 'logMAPE':MAPE}
    
    # do a PPM version
    labels = pow(10, labels - 6.0)
    predictions = pow(10, predictions - 6.0)

    MSE = np.sum((labels - predictions)**2.0)/len(labels)
    MAE = np.sum(np.abs(labels - predictions))/len(labels)
    MAPE = np.sum(np.abs((labels - predictions)/labels))/len(labels)
    out.update({'ppmMSE':MSE, 'ppmMAE':MAE, 'ppmMAPE':MAPE})
    return out