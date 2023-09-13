from tqdm import tqdm
import torchmetrics
import numpy as np
import torch
from HSI_SSFTT import get_SSFTT_model
import pathlib
from utils import get_OH_bins_and_labels, get_index_to_name
from data_fns import(
    HSICubeDataset,
    HSIPatchDataset,
    BinLogOHLabels,
    ssftt_data,
    repatchSSFTT,
    map_collate_fn,
    default_collate_fn
)
data_dir = pathlib.Path('/qfs/projects/gumby/data/manga')

def bin_OH_map(input, OH_key='default'):
    OH_bins, _ = get_OH_bins_and_labels(OH_key)
    output = np.digitize(input, bins=OH_bins)
    if type(input) is np.ma.core.MaskedArray:
        output = np.ma.array(output, mask=input.mask)
    return output

def get_test_dataloader(split_dir='OH_1', patch_norm='global', default_patch_size=9, OH_key='default',
                        spatial_patch_size=7, normalize=False, min_wave_ind=0, max_wave_ind=2040, 
                        batch_size=512, num_workers=0, easy_splits=False):
    """Returns dataloader for test split of input split_dir
    """

    if (spatial_patch_size != default_patch_size) | ((min_wave_ind,max_wave_ind) != (0,2040)):
        preprocess_fn = repatchSSFTT(
            current_patch_size=default_patch_size,
            new_patch_size=spatial_patch_size,
            wave_inds=(min_wave_ind, max_wave_ind),
            normalize=normalize
        )
    else:
        preprocess_fn = ssftt_data(normalize=normalize)
    # instantiate datasets
    split_str = '_easy' if easy_splits else ''
    csv_dir = data_dir / 'splits' / split_dir
    test_data = HSIPatchDataset(
        csv_dir / f'test-{patch_norm}{split_str}.csv',
        preprocess_fn=preprocess_fn,
        label_fn=BinLogOHLabels(OH_key=OH_key),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=default_collate_fn,
        num_workers=num_workers
    )
    return test_dataloader

def load_trained_SSFTT_model(model_weights_dir=None, device=torch.device('cuda:0'), num_classes=6, 
                             spatial_patch_size=7, min_wave_ind=0, max_wave_ind=2040, **kwargs):
    """Returns SSFTT model with input specs and loads weights
    """
    if model_weights_dir is None:
        model_weights_dir = './'
    if type(model_weights_dir) is str:
        model_weights_dir = pathlib.Path(model_weights_dir)
    
    model = get_SSFTT_model(
        num_classes=num_classes,
        min_wave_ind=min_wave_ind,
        max_wave_ind=max_wave_ind,
        spatial_patch_size=spatial_patch_size,
        **kwargs    
    )
    model.load_state_dict(torch.load(model_weights_dir / 'best_wts.pt', map_location=device))
    return model

def predictions(dataloader, model, device=torch.cuda.current_device(), progress=True):
    """
    Aggregate predictions of a model on a dataloader.
    """
    model.to(device)
    model.eval()
    acc = 0

    if progress:
        dl = tqdm(
            enumerate(dataloader), 
            total=len(dataloader), 
            desc="collecting predictions"
        )
    else:
        dl = enumerate(dataloader)
    
    with torch.no_grad():
        for i, (x, y) in dl:
            if type(x) is dict:
                x = {k: v.to(device) for k, v in x.items()}
            else:
                x = x.to(device)
            l = model(x)
            if type(l) is not torch.Tensor:
                l = l['out']
            yhat = torch.argmax(torch.nn.functional.softmax(l, dim=1), dim=1)
            y = y.to(device)
            acc += (1.0*(yhat == y)).mean()

            if i == 0:
                preds = yhat
                true = y
            else:
                preds = torch.cat((preds, yhat))
                true = torch.cat((true, y))
            
            if progress:
                dl.set_description(f'collecting predictions. acc: {acc:.3f}')
    
    acc /= len(dataloader)
    acc = acc.cpu().item()

    if progress:
        dl.set_description(f'collecting predictions. acc: {acc:.3f}')
    return acc, preds.cpu(), true.cpu()


def datacube_evaluator(fits_file, model_weights_dir=None, normalize=False, 
                       patch_norm='global', label_task='logOH', OH_key='default',
                       default_patch_size=9, spatial_patch_size=7,
                       min_wave_ind=0, max_wave_ind=2040, num_workers=0,
                       device=torch.cuda.current_device(), batch_size=512, **kwargs):
    '''Evaluates input fits file and creates spatial maps for labels + preds
    '''

    if (spatial_patch_size != default_patch_size) | ((min_wave_ind,max_wave_ind) != (0,2040)):
        preprocess_fn = repatchSSFTT(
            current_patch_size=default_patch_size,
            new_patch_size=spatial_patch_size,
            wave_inds=(min_wave_ind, max_wave_ind),
            normalize=normalize
        )
    else:
        preprocess_fn = ssftt_data(normalize=normalize)
    
    eval_data = HSICubeDataset(
        fits_file,
        preprocess_fn=preprocess_fn,
        label_fn=BinLogOHLabels(OH_key=OH_key),
        patch_size=spatial_patch_size,
        patch_norm=patch_norm,
        label_task=label_task,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=map_collate_fn,
        num_workers=num_workers
    )

    model = load_trained_SSFTT_model(model_weights_dir=model_weights_dir, device=device, **kwargs)
    model.to(device)
    model.eval()
    acc = 0
    output_predictions = None
    output_labels = None
    output_inds = None
    print('Evaluating model...')
    with torch.no_grad():
        for batch, target, map_index in eval_dataloader:
            batch = batch.to(device)
            output = model(batch)
            predictions = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
            target = target.to(device)
            acc += (1.0*(predictions == target)).mean()

            if output_predictions is None:
                output_predictions = predictions
                output_labels = target
                output_inds = map_index
            else:
                output_predictions = torch.cat((output_predictions, predictions))
                output_labels = torch.cat((output_labels, target))
                output_inds += map_index

    acc /= len(eval_dataloader)
    acc = acc.cpu().item()

    # here we add 1 to the classes so there is background class for visualizing
    output_predictions = output_predictions.cpu().numpy()
    output_labels = output_labels.cpu().numpy()

    # here we subtract 1 for background visualization
    gt_map = np.zeros((eval_data.label_map_size, eval_data.label_map_size)) - 1
    pred_map = np.zeros((eval_data.label_map_size, eval_data.label_map_size)) - 1
    corr_map = np.zeros((eval_data.label_map_size, eval_data.label_map_size)) - 1

    for i in range(len(output_predictions)):
        gt_map[output_inds[i]] = output_labels[i]
        pred_map[output_inds[i]] = output_predictions[i]
        corr_map[output_inds[i]] = output_predictions[i] == output_labels[i]
    
    output_dictionary = {
        'label_map': gt_map,
        'pred_map': pred_map,
        'correct_mask': corr_map,
        'accuracy': acc,
        'predictions':output_predictions,
        'labels':output_labels,
        'map_index':np.array(output_inds)
    }
    labs = np.ma.masked_array(gt_map, mask=(gt_map == -1)).compressed()
    preds = np.ma.masked_array(pred_map, mask=(pred_map == -1), fill_value=0).compressed()

    metrics = calc_eval_metrics(labs, preds,OH_key=OH_key)
    output_dictionary.update(metrics)
    return output_dictionary

def calc_eval_metrics(labels, predictions, OH_key='default'):
    if type(labels) == torch.Tensor:
        labels = labels.numpy()
    if type(predictions) == torch.Tensor:
        predictions = predictions.numpy()
    accuracy = np.sum(labels == predictions)/len(labels)
    index_to_name = get_index_to_name(OH_key)
    labels = np.array(list(map(float, list(map(index_to_name.get,labels)))))
    predictions = np.array(list(map(float, list(map(index_to_name.get,predictions)))))
    MSE = np.sum((labels - predictions)**2.0)/len(labels)
    MAE = np.sum(np.abs(labels - predictions))/len(labels)
    MAPE = np.sum(np.abs(pow(10,labels) - pow(10,predictions))/pow(10,labels))/len(labels)
    out = {'accuracy':accuracy,'MSE':MSE, 'MAE':MAE, 'MAPE':MAPE}
    return out