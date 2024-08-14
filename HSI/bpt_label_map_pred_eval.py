import torch
from convenience import quick_eval

# device = torch.device('cuda:0')

# arch_type = 'ssftt/BPT/base'
# model_suff = 'base'
# split_dir='BPT'
# OH_key='BPT'

# quick_eval(
#     arch_type=arch_type,
#     model_suff=model_suff,
#     split_dir=split_dir,
#     OH_key=OH_key,
#     device=device,
#     batch_size=1024,
#     device=torch.cuda.current_device(),
#     task='classification',
#     regression_norm='minmax',
#     confidence_flag=False,
#     confidence_file='./tmp.txt',
#     use_local_data=True,
#     base_results_dir=None,
#     save_figure=True
# )

# DATACUBE EVALUATION
import sys
import torch
import numpy as np
import pathlib
import logging
import sys
import pandas as pd
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from utils import plateid_to_fits_file
from eval_utils import(
    datacube_evaluator
)
from comparison_fns import visualize_gal_labels
data_dir = pathlib.Path('/gscratch/scrubbed/mmckay18/DATA/')
results_dir = pathlib.Path('/gscratch/astro/mmckay18/DATA/weights/')

############

device = torch.device('cuda:0')

# arch_type = 'ssftt/BPT/base'
arch_type = 'ssftt'
OH_key='BPT'
model_name = 'BPT'
metric = 'MAPE'

train_df = pd.read_csv('/gscratch/scrubbed/mmckay18/DATA/splits/BPT/test_fits.csv')

# Sample 10 files from the DataFrame
train_df = train_df.sample(3)
list_of_ids = []
# for idx, fits_file in enumerate(train_df['/gscratch/scrubbed/mmckay18/DATA/raw/9892/12703/manga-9892-12703-LOGCUBE-SPX-MILESHC-MASTARSSP.fits.gz']):
for idx, fits_file in enumerate(train_df.iloc[:, 0]):
    plate_ifu = '-'.join(fits_file.split('/')[-3:-1])
    print(plate_ifu)
    list_of_ids.append(plate_ifu)

for plate_id in list_of_ids:
    fits_file = plateid_to_fits_file(plate_id)
    # output is a dictionary with predictions and ground truth labels
    output = datacube_evaluator(
        fits_file,
        arch_type=arch_type,
        label_task='BPT',
        model_weights_dir=results_dir / 'ssftt/BPT/base/' / model_name,
        OH_key='BPT',
        device=device,
        num_workers=4,
    )
    print(f'---- {plate_id} ------')
    print(f"Accuracy: {output['accuracy']:.3f}")
    print(f"Acc (Top2): {output['Top2Accuracy']:.3f}") 
    print(f"IOU: {output['IOU']:.3f}\n")
    
    
    # make figure -----------
    visualize_gal_labels(
        plate_id, 
        output, 
        title=f'{plate_id} {metric}: {arch_type}/{model_name}',
        OH_key='BPT', 
        metric=metric,
        task='classification',
        save_fig=True,
#         OH_log=False,
#         output_path=f'./figs/{model_name}/{arch_type}_{metric}_{plate_id}.png',
        output_path=f'/gscratch/astro/mmckay18/FIGURES/{arch_type}_{metric}_{plate_id}.pdf'
#         cb_label='log( O / H )'
    )