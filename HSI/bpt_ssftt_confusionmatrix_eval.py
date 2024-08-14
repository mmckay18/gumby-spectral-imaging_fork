import random
import numpy as np
import pandas as pd
import pathlib
from astropy.table import Table
import argparse
from fits_utils import BPT_diagnostic
import matplotlib.pyplot as plt
from download_utils import *
from fits_utils import *
import seaborn as sns
from utils import get_OH_bins_and_labels, get_index_to_name, get_BPT_bins_and_labels
from utils import *
from vis_fns import *
from plotting import plot_training_curve
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
from convenience import quick_eval
import sys
import torch
import numpy as np
import pathlib
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
from utils import plateid_to_fits_file
from eval_utils import(
    datacube_evaluator
)
from comparison_fns import visualize_gal_labels




# arch_type = 'ssftt'
# model_weights_dir = 
device = torch.device('cuda:0')
# # load up trained model
# model = load_trained_model(
#     arch_type=arch_type,
#     model_weights_dir=model_weights_dir, 
#     spatial_patch_size=9, 
#     device=device,
#     num_classes=3
# )
# print(







#### Try quick eval but not working for me 

# arch_type = 'ssftt'
# model_suff = 'BPT'
# split_dir='BPT'
# OH_key='BPT'

# quick_eval(
#     arch_type=arch_type,
#     model_suff=model_suff,
#     split_dir=split_dir,
#     OH_key=OH_key,
#     device=device,
# )

# quick_eval(
#     arch_type:'ssftt',
#     model_suff:str='base',
#     split_dir:str='BPT',
#     OH_key:str='BPT',
#     batch_size=1024,
#     device=torch.cuda.current_device(),
#     task='classification',
#     regression_norm='minmax',
#     confidence_flag=False,
#     confidence_file='./tmp.txt',
#     use_local_data=False,
#     base_results_dir='/gscratch/astro/mmckay18/DATA/weights/ssftt/BPT/base/BPT/',
#     save_figure=True
# )

quick_eval(
    arch_type='ssftt/BPT/',
    model_suff='easy',
    split_dir='BPT',
    OH_key='BPT',
    batch_size=1024,
    device=device,
    task='classification',
    regression_norm='minmax',
    confidence_flag=False,
    confidence_file='./tmp.txt',
#     use_local_data=False,
    base_results_dir=pathlib.Path('/gscratch/astro/mmckay18/DATA/weights/'),
    save_figure=True
)

