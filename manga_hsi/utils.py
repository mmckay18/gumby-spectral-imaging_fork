"""
h5 datacubes: data_dir / processed / cubes / plate / {plate-id}.h5
numpy patches: data_dir / processed / patches / label_task / patch_norm / plate / {plate-id}.npy
label maps: data_dir / processed / labels / label_task / {plate-id}.npy
"""

import pathlib

data_dir = pathlib.Path('/qfs/projects/gumby/data/manga')
results_dir = pathlib.Path('/qfs/projects/gumby/results/weights/manga/')

OH_bin_dict = {
    'default':{
        'OH_bins':[8.40, 8.44, 8.48, 8.52, 8.56],
        'OH_labels':['8.40', '8.44', '8.48', '8.52', '8.56', '8.60']
    },
    'extra':{
        'OH_bins':[8.34, 8.4, 8.44, 8.47, 8.5, 8.52, 8.55],
        'OH_labels':['8.34', '8.40', '8.44', '8.47', '8.50', '8.52', '8.55', '8.64']
    }
}

def get_num_classes(key):
    num_classes = len(OH_bin_dict[key]['OH_labels'])
    return num_classes

def get_index_to_name(key, log=False):
    '''returns index_to_name dictionary
    '''
    label_map = {k:v for k,v in enumerate(OH_bin_dict[key]['OH_labels'])}
    if log:
        for key,val in label_map.items():
            label_map[key] = f'{float(val)-12:.2f}'
    return label_map

def get_OH_bins_and_labels(key):
    '''returns (bins, labels)
    '''
    return OH_bin_dict[key]['OH_bins'], OH_bin_dict[key]['OH_labels']

def plateid_to_fits_file(plateid):
    '''go from '8550-9102' to fits_file
    '''
    plate,ifuid = plateid.split('-')
    filename = f'manga-{plate}-{ifuid}-LOGCUBE-SPX-MILESHC-MASTARSSP.fits.gz'
    fits_file = data_dir / 'raw' / plate / ifuid / filename
    return fits_file.as_posix()

def get_label_path(fits_file, label_task='logOH'):
    """Returns path to GT label map (.npy file)
    default: /qfs/projects/manga/data/manga/processed/labels/logOH
    """
    assert label_task in ['BPT','logOH', 'N2']
    plate_ifu = '-'.join(fits_file.split('/')[-3:-1])
    # this has ground truth labels
    label_file = str(data_dir / f'processed/labels/{label_task}/{plate_ifu}.npy')
    return label_file

def get_label_path_from_cube(cube_file, label_task='logOH'):
    """Returns path to GT label map (image)
    default: /qfs/projects/manga/data/manga/processed/labels/logOH
    """
    assert label_task in ['BPT','logOH','N2']
    label_dir = data_dir / f'processed/labels/{label_task}'
    plate_ifu = f'{cube_file.parent.relative_to(cube_file.parent.parent)}-{cube_file.stem}'

    # this has ground truth labels
    label_file = str( label_dir / f'{plate_ifu}.npy')
    return label_file

def get_cube_path(fits_file, label_task='logOH'):
    """Get h5 cube path for input fits file
    INPUTS:
        cube_file (str): path to manga fits file
        output_dir (str or Path obj): where to save output H5 file
            default (/qfs/projects/manga/data/manga/processed/cubes/)
            files saved to output_dir/{plate}/{ifu}.h5
    """
    cube_dir = data_dir / f"processed/cubes"
    plate_ifu = '-'.join(fits_file.split('/')[-3:-1])
    output_path = cube_dir / f"{plate_ifu.replace('-','/')}.h5"
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def get_patch_dir_from_cube_file(cube_file, patch_norm="global", label_task='logOH'):
    """Get output path for input fits file
    INPUTS:
        cube_file (str): path to manga fits file
        /qfs/projects/manga/data/manga/processed/cubes/{plate}/{ifu}.h5

        /qfs/projects/manga/data/manga/processed/patches/logOH/global/{plate_id}/xxx.npy
    """
    patch_dir = data_dir / f"processed/patches/{label_task}/{patch_norm}/"
    if type(cube_file) is str:
        cube_file = pathlib.Path(cube_file)
    plate_ifu = f'{cube_file.parent.relative_to(cube_file.parent.parent)}-{cube_file.stem}'
    output_path = patch_dir / plate_ifu
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def get_patch_dir(fits_file, patch_norm="global", label_task='logOH'):
    """Get output path for input fits file
    INPUTS:
        cube_file (str): path to manga fits file
        output_dir (str or Path obj): where to save output H5 file
            default (/qfs/projects/manga/data/manga/processed/cubes/)
            files saved to output_dir/{plate}/{ifu}.h5
    """
    patch_dir = data_dir / f"processed/patches/{label_task}/{patch_norm}/"
    plate_ifu = '-'.join(fits_file.split('/')[-3:-1])
    output_path = patch_dir / plate_ifu
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path
