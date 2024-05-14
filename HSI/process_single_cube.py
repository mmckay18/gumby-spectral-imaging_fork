'''
python process_single_cube.py --patch_size=9 --patch_norm=global --label_task=logOH --plateid=11872-12704
'''
import pathlib
import argparse
import glob
from hsi_proc_fns import (
    patchify_cube
)
from utils import (
    plateid_to_fits_file,
    get_cube_path,
    get_patch_dir,
    get_label_path
)
from fits_utils import (
    process_cube,
    process_map
)
data_dir = pathlib.Path('/qfs/projects/thidwick/manga')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fits_file', '-s', type=str, default='',
                        help='path to fits file')
    parser.add_argument('--plateid', '-pid', type=str, default='',
                        help='plate-ifuid')
    parser.add_argument('--label_task', '-lt', type=str, default='logOH',
                        help='labels are BPT or logOH')
    parser.add_argument('--patch_size', '-ps', type=int, default=7,
                        help='Size of spatial patches (e.g. 7x7)')
    parser.add_argument('--overwrite_cube', action='store_true', default=False)
    parser.add_argument('--patch_norm', '-pn', type=str, default='global',
                        help='type of patch normalization: None, spatial, global')
    parser.add_argument('--overwrite_patches',
                        action='store_true', default=False)
    parser.add_argument('--glob_patches', action='store_true', default=False,
                        help="use directory list of patches instead of creating from cube file")
    args = parser.parse_args()
    
    fits_file=args.fits_file
    if args.plateid != '':
        fits_file = plateid_to_fits_file(args.plateid)
    
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
            process_cube(fits_file, cube_file)
        except:
            print('something bad happened')
            exit()
    else:
        print(f'\tCube already exists!\n\t{cube_file}')
    
    # make labels, if needed
    if not pathlib.Path(label_path).exists():
        # NOTE: this will try and download the MAP fits file if necessary
        print(f'\tMaking label map:\n\t{label_path}')
        try:
            process_map(fits_file, label_path, label_task=args.label_task)
        except:
            print('something bad happened')
            exit()
    
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
