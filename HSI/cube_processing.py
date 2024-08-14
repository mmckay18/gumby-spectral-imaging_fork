'''
python cube_processing.py --split_dir=OH_1 --patch_size=9 --patch_norm=global --label_task=logOH --glob_patches --splits=test
python cube_processing.py --split_dir=OH_2 --patch_size=9 --patch_norm=global --label_task=logOH --glob_patches --splits=train --reverse
'''
import pathlib
import pandas as pd
import argparse
import glob

from hsi_proc_fns import (
    patchify_cube
)
from utils import (
    get_cube_path,
    get_patch_dir,
    get_label_path
)
from fits_utils import (
    process_cube,
    process_map
)
data_dir = pathlib.Path('/gscratch/scrubbed/mmckay18/DATA')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dir', '-s', type=str, default='OH_1',
                        help='subdirectory: OH_1, OH_2')
    parser.add_argument('--label_task', '-lt', type=str, default='logOH',
                        help='labels are BPT, logOH, N2')
    parser.add_argument('--splits', '-spl', type=str, default='all',
                        help='which splits to process')
    parser.add_argument('--patch_size', '-ps', type=int, default=7,
                        help='Size of spatial patches (e.g. 7x7)')
    parser.add_argument('--overwrite_cube', action='store_true', default=False)
    parser.add_argument('--patch_norm', '-pn', type=str, default='global',
                        help='type of patch normalization: None, spatial, global')
    parser.add_argument('--overwrite_patches',
                        action='store_true', default=False)
    parser.add_argument('--reverse',
                        action='store_true', default=False)
    parser.add_argument('--glob_patches', action='store_true', default=False,
                        help="use directory list of patches instead of creating from cube file")
    args = parser.parse_args()

    # need to generate list of fits files that go in each of these
    # usually with data_split_char.ipynb
    csv_dir = data_dir / 'splits' / args.split_dir
    if args.splits=='all':
        file_list_dict = {
            'train': csv_dir / f'train_fits.csv',
            'val': csv_dir / f'val_fits.csv',
            'test': csv_dir / f'test_fits.csv',
        }
    elif args.splits == 'test':
        file_list_dict = {
            'val': csv_dir / f'val_fits.csv',
            'test': csv_dir / f'test_fits.csv',
        }
    elif args.splits == 'train':
        file_list_dict = {
            'train': csv_dir / f'train_fits.csv',
        }
    for split_key, input_fits_csv in file_list_dict.items():
        print(f'{split_key}...')
        list_of_fits_files = pd.read_csv(
            input_fits_csv, 
            header=None, 
            names=['fits_file']
        )['fits_file'].to_list()
        # CSV file containing list of all patches
        output_patch_csv = csv_dir / f'{split_key}-{args.patch_norm}.csv'
        print(f'patch_norm is {args.patch_norm} \n {output_patch_csv}')

        n_fits = len(list_of_fits_files)
        if args.reverse:
            list_of_fits_files = list_of_fits_files[::-1]
        for i, fits_file in enumerate(list_of_fits_files):
            print(f"{i+1}/{n_fits}...")
            print(f"\t{fits_file}")

            # set up everything
            cube_file = get_cube_path(
                fits_file, 
                label_task=args.label_task
            )
            patch_save_dir = get_patch_dir(
                fits_file, 
                patch_norm=args.patch_norm,
                label_task=args.label_task
            )
            label_path = get_label_path(
                fits_file, 
                label_task=args.label_task
            )

            if not cube_file.parent.exists():
                cube_file.parent.mkdir(exist_ok=True, parents=True)
            if not patch_save_dir.exists():
                patch_save_dir.mkdir(exist_ok=True, parents=True)
            
            # make h5 cube if it doesn't exist
            if (not cube_file.exists()) or (args.overwrite_cube):
                # NOTE: this will try and download the logcube fits file if necessary
                print(f'\tGenerating new cube:\n\t{cube_file}')
                try:
#                     print('Running process_cube')
                    process_cube(fits_file, cube_file)
                except:
                    print('\t\tsomething bad happened')
                    continue
            else:
                print(f'\tCube already exists!\n\t{cube_file}')
            
            # make labels, if needed
            if not pathlib.Path(label_path).exists():
                # NOTE: this will try and download the fits file if necessary
                print(f'\tMaking label map:\n\t{label_path}')
                try:
                    process_map(fits_file, label_path, label_task=args.label_task)
                    print('\t Process Map Completed')
                except:
                    print('\t\tsomething bad happened')
                    continue
            
            # generate patches
            list_of_files = glob.glob(f'{patch_save_dir}/*.npy')
            if args.glob_patches & (len(list_of_files) > 0):
                print(f'\tUsing directory list of patches:\n\t{patch_save_dir}')
                list_of_patches = []
                for patch_path in list_of_files:
                    this_dict = {
                        'data': patch_path, # path to patch
                        'label': patch_path.strip('.npy').split('_')[-1], # integer label
                    }
                    list_of_patches.append(this_dict)
            else:
                print(f'\tGenerating patches:\n\t{patch_save_dir}')
                list_of_patches = patchify_cube(
                    cube_file,
                    label_path,
                    patch_size=args.patch_size,
                    patch_norm=args.patch_norm,
                    patch_save_dir=patch_save_dir
                )
            print('\tSaving to dataframe.')
            df = pd.DataFrame(list_of_patches)
            if i == 0:
                df.to_csv(output_patch_csv, header=True, index=False)
            else:
                df.to_csv(output_patch_csv, mode='a',
                          header=False, index=False)
