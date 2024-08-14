import torch
import pathlib
from eval_utils import(
    get_test_dataloader,
    load_trained_model,
    AllPredictions,
    RegPredictions
)
from plotting import plot_tm_confusion_matrix
import pickle
from comparison_fns import eval_alt_split
from utils import get_index_to_name, get_num_classes
data_dir = pathlib.Path('/gscratch/scrubbed/mmckay18/DATA/')
results_dir = pathlib.Path('/gscratch/scrubbed/mmckay18/DATA/weights/')

def quick_eval(
    arch_type:str='simplenet',
    model_suff:str='base',
    split_dir:str='OH_2',
    OH_key:str='extra',
    batch_size=1024,
    device=torch.cuda.current_device(),
    task='classification',
    regression_norm='minmax',
    confidence_flag=False,
    confidence_file='./tmp.txt',
    use_local_data=True,
    base_results_dir=None,
    save_figure=True
    ):
    if base_results_dir is None:
        base_results_dir = results_dir
    if model_suff is None:
        model_name = f'{split_dir}'
    else:
        model_name = f'{split_dir}_{model_suff}'
    easy_splits = True if 'easy' in model_name else False
    print(f'---- {arch_type} / {model_name} ----------')
    num_classes = get_num_classes(OH_key)
    if task == 'regression':
        num_classes=1
    model_weights_dir = base_results_dir / arch_type / model_name
    test_dataloader = get_test_dataloader(
        arch_type=arch_type,
        split_dir=split_dir,
        patch_norm='global',
        spatial_patch_size=7,
        normalize=False,
        OH_key=OH_key,
        min_wave_ind=0,
        max_wave_ind=2040,
        task=task,
        regression_norm=regression_norm,
        batch_size=batch_size,
        num_workers=4,
        easy_splits=easy_splits,
        use_local_data=use_local_data,
    )
    # load up trained model
    model = load_trained_model(
        arch_type=arch_type,
        model_weights_dir=model_weights_dir, 
        spatial_patch_size=7, 
        device=device,
        num_classes=num_classes
    )
    if task == 'regression':
        output_dict = RegPredictions(
            test_dataloader, 
            model, 
            device=device
        )
        # these are calculated on the raw abundance
        print('Raw O/H')
        print(f'\tMAE:{output_dict["MAE"]: >8.1e}')
        print(f'\tMAPE: {output_dict["MAPE"]*100.0: >4.1f}%')
        # these are calculated on the 12+log values
        print('12+logO/H')
        print(f'\tMAE:{output_dict["logMAE"]: >8.1e}')
        print(f'\tMAPE: {output_dict["logMAPE"]*100.0: >4.1f}%')
        # these are calculated on the ppm values
        print('O/H ppm')
        print(f'\tMAE:{output_dict["ppmMAE"]: >8.0f}')
        print(f'\tMAPE: {output_dict["ppmMAPE"]*100.0: >4.1f}%')
        # save to file -------------------------
        output_file = model_weights_dir / 'test_split_metrics.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return output_dict
    else:
        output_dict = AllPredictions(
            test_dataloader, 
            model, 
            device=device, 
            OH_key=OH_key,
            confidence_flag=confidence_flag,
            confidence_file=confidence_file
        )
        print(f'\tAccuracy: {output_dict["accuracy"]*100.0: >4.1f}%')
        print(f'\tMAPE:     {output_dict["MAPE"]*100.0: >4.1f}%')

        index_to_name = get_index_to_name(OH_key, log=True)
        class_list = [v for _,v in index_to_name.items()]
        # Generate and plot confusion matrix, accuracy 
        plot_tm_confusion_matrix(
            output_dict['CM'],
            class_list=class_list,
            save_figure=save_figure,
#             output_path=f'./figs/confusion/{arch_type}_{model_name}.png',
            output_path=f'/gscratch/astro/mmckay18/FIGURES/{arch_type}_{model_name}.png',
            normalized=True,
        )
        # calculate metrics for FWHM ratio -----
        alt_metrics = eval_alt_split(
            split_dir, 
            label_task='N2', 
            easy_splits=easy_splits, 
            patch_norm='global',
            OH_key=OH_key,
        )
        
        # save to file -------------------------
        out = {'DL': output_dict, 'FWHM':alt_metrics}
        output_file = model_weights_dir / 'test_split_metrics.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        return