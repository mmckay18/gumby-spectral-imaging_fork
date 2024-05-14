'''
python make_alt_labels.py --split_dir=OH_2 --patch_norm=global --label_task=N2 --source_label_task=logOH
'''
import pathlib
import pandas as pd
import numpy as np
import argparse
import glob

from utils import (
    get_cube_path,
    get_patch_dir,
    get_label_path
)
from fits_utils import (
    process_map
)

def npy_datapoint_loader(path):
    return np.load(path)

data_dir = pathlib.Path('/qfs/projects/thidwick/manga')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dir', '-s', type=str, default='OH_1',
                        help='subdirectory: OH_1')
    parser.add_argument('--label_task', '-lt', type=str, default='logOH',
                        help='labels are BPT or logOH')
    parser.add_argument('--source_label_task', '-slt', type=str, default='logOH',
                        help='labels are BPT or logOH')
    parser.add_argument('--patch_norm', '-pn', type=str, default='global',
                        help='type of patch normalization: None, spatial, global')
    args = parser.parse_args()

    # need to generate list of fits files that go in each of these
    # usually with data_split_char.ipynb
    csv_dir = data_dir / 'splits' / args.split_dir
    file_list_dict = {
        'train': csv_dir / f'train_fits.csv',
        'val': csv_dir / f'val_fits.csv',
        'test': csv_dir / f'test_fits.csv',
    }
    for split_key, input_fits_csv in file_list_dict.items():
        print(f'{split_key}...')
        list_of_fits_files = pd.read_csv(
            input_fits_csv, 
            header=None, 
            names=['fits_file']
        )['fits_file'].to_list()
        # CSV file containing list of all patches
        output_patch_csv = csv_dir / f'{split_key}-{args.patch_norm}_{args.label_task}.csv'
        print(f'output CSV is:\n\t{output_patch_csv}')

        n_fits = len(list_of_fits_files)
        for i, fits_file in enumerate(list_of_fits_files):
            print(f"{i+1}/{n_fits}...")
            print(f"\t{fits_file}")

            # set up everything
            cube_file = get_cube_path(
                fits_file, 
                label_task=args.label_task
            )
            # this one references the source label task (logOH)
            patch_save_dir = get_patch_dir(
                fits_file, 
                patch_norm=args.patch_norm,
                label_task=args.source_label_task
            )
            label_path = get_label_path(
                fits_file, 
                label_task=args.label_task
            )
            label_path = pathlib.Path(label_path)
            if not label_path.parent.exists():
                print(f'Made directory: {label_path.parent}')
                label_path.parent.mkdir(exist_ok=True, parents=True)
            label_path = str(label_path)
            
            # NOTE: this will try and download the fits file if necessary
            print(f'\tMaking label map:\n\t{label_path}')
            try:
                process_map(fits_file, label_path, label_task=args.label_task)
            except:
                print('\t\tsomething bad happened')
                continue
            
            # generate patches
            list_of_files = glob.glob(f'{patch_save_dir}/*.npy')
            if len(list_of_files) > 0:

                label_map = npy_datapoint_loader(label_path)
                label_map = np.ma.masked_array(label_map, mask=(label_map == 0), fill_value=0)
                
                list_of_patches = []
                for patch_path in list_of_files:
                    map_inds = tuple([int(c) for c in patch_path.split('_')[-2].split('-')])
                    this_dict = {
                        'data': patch_path,
                        'label': label_map[map_inds]
                    }
                    list_of_patches.append(this_dict)

                print('\tSaving to dataframe.')
                df = pd.DataFrame(list_of_patches)
                if i == 0:
                    df.to_csv(output_patch_csv, header=True, index=False)
                else:
                    df.to_csv(output_patch_csv, mode='a', header=False, index=False)
        print(f'Saved to CSV:\n\t{output_patch_csv}')
    
    
    # now for easy splits-----
    # first read in all of the patch data with new labels
    split_list = ['train', 'val', 'test']
    filepaths = [csv_dir / f'{split}-{args.patch_norm}_{args.label_task}.csv' for split in split_list]
    label_df = pd.concat((pd.read_csv(f) for f in filepaths), ignore_index=True)

    # now iterate through the easy splits
    for split in split_list:
        input_csv = csv_dir / f'{split}-{args.patch_norm}_easy.csv'
        output_csv = csv_dir / f'{split}-{args.patch_norm}_{args.label_task}_easy.csv'

        patch_df = pd.read_csv(input_csv)
        df = pd.merge(patch_df, label_df, on='data', suffixes=('_old','_new'))
        df = df.rename(columns={'label_new':'label'})
        df = df.drop(columns='label_old')
        df.to_csv(output_csv, index=False, header=True)
        print(f'Saved to:\n\t{output_csv}')
