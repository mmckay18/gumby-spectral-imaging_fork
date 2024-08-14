import numpy as np
import pickle
from PIL import Image
import h5py

def h5_datapoint_loader(path):
    """returns (labels, cube)
    path: pathlib.Path object
    """
    hf = h5py.File(path, 'r')
    flux_cube = np.array(hf.get('cube'))
    #label_map = np.array(hf.get('labels'))
    hf.close()
    return flux_cube

def masked_h5_datapoint_loader(path):
    """returns (labels, cube)
    path: pathlib.Path object
    """
    hf = h5py.File(path, 'r')
    flux_cube = np.array(hf.get('cube'))
    #label_map = np.array(hf.get('labels'))
    hf.close()
    flux_cube = np.ma.masked_array(flux_cube, mask=(flux_cube <= 0.0), fill_value=0.0)
    return flux_cube

def normalize_datacube(hsi_image, norm='spectral'):
    '''Performs a normalization of the input image.
    input: 
        hsi_image: masked datacube
        norm (str): how to normalize cube
            if 'spatial', normalizes image at each wavelength (i.e. spatial map normalization)
            if 'global', normalizes entire cube
            if None, normalizes spectrum for each pixel (default)
    output: normalized datacube
    '''
    if type(hsi_image) is not np.ma.core.MaskedArray:
        hsi_image = np.ma.masked_array(hsi_image, mask=(hsi_image <= 0.0), fill_value=0.0)
    
    if norm == 'spatial':
        cube_mins = hsi_image.min(axis=(0,1), keepdims=True)
        cube_maxs = hsi_image.max(axis=(0,1), keepdims=True)
    elif norm == 'global':
        cube_mins = hsi_image.min(keepdims=True)
        cube_maxs = hsi_image.max(keepdims=True)
    else:
        cube_mins = hsi_image.min(axis=2, keepdims=True)
        cube_maxs = hsi_image.max(axis=2, keepdims=True)
    output = (hsi_image - cube_mins) / (cube_maxs - cube_mins)
    return output.filled(0.0)

def pad_hsi_image(input_hsi_image, patch_size=5, use_mirror=False, verbose=False):
    '''adds padding to input HSI image to allow for patch_size extractions.
    INPUTS:
        input_hsi_image: masked datacube (HxWxC)
        patch_size (int): size of patch to extract (size X size)
        use_mirror (bool): if True >> will pad cube with spectra from opposite side of cube
        verbose (bool): if True >> will print statements
    OUTPUTS:
        padded_hsi: H+2*(patch_size//2) x W+2*(patch_size//2) x C
    '''
    height, width, num_bands = input_hsi_image.shape
    padding=patch_size//2
    #padded_hsi=np.ma.zeros((height+(2*padding), width+(2*padding), num_bands),dtype=float, fill_value=0.0)
    padded_hsi=np.zeros((height+(2*padding), width+(2*padding), num_bands),dtype=float)
    # central region: input image
    padded_hsi[padding:(padding+height),padding:(padding+width),:] = input_hsi_image

    if use_mirror:
        # left mirror
        for i in range(padding):
            padded_hsi[padding:(height+padding),i,:]=input_hsi_image[:,padding-i-1,:]
        # right mirror
        for i in range(padding):
            padded_hsi[padding:(height+padding),width+padding+i,:]=input_hsi_image[:,width-1-i,:]
        # top mirror
        for i in range(padding):
            padded_hsi[i,:,:]=padded_hsi[padding*2-i-1,:,:]
        # bottom mirror
        for i in range(padding):
            padded_hsi[height+padding+i,:,:]=padded_hsi[height+padding-1-i,:,:]
    if verbose:
        print("**************************************************")
        print("patch is : {}".format(patch_size))
        print("padded image shape : [{0},{1},{2}]".format(padded_hsi.shape[0],padded_hsi.shape[1],padded_hsi.shape[2]))
        print("**************************************************")
    return padded_hsi

def grab_spatial_patch(padded_image, cube_labels, patch_index, patch_size=7):
    """grab patch of datacube surrounding input patch_index.
    INPUTS:
        padded_image: padded datacube  (H+2*padding x W+2*padding x C)
        cube_labels: labels with original cube dimensions (H,W)
        patch_index: tuple of (x,y) index to use for central pixel
        patch_size: size of patch to extract
    OUTPUTS:
        temp_image: (patch_size x patch_size x C)
    """
    x,y = patch_index
    padding=patch_size//2
    temp_image = padded_image[
        x+padding:(x+padding+patch_size),
        y+padding:(y+padding+patch_size),
        :
    ]
    # only need label of central pixel
    temp_label = cube_labels[patch_index]
    if type(temp_label) is not int:
        temp_label = f"{temp_label:.2f}"
    return temp_image, temp_label

def patchify_cube(cube_file, label_path, patch_size=7, 
                  patch_save_dir=None, patch_norm='global'):
    """Read in h5 file and generate patches
    """
    print('Start patchify')
    # (74, 74), (74,74,2040)
    flux_cube = h5_datapoint_loader(cube_file)
#     print(flux_cube.shape)
    
    if '.npy' in label_path:
#         print(f'\tLabel Path: {label_path}')
        label_map = np.load(label_path)
#         print(label_map.shape)
    else:
        label_map = np.array(Image.open(label_path))[..., 0]
    
    masked_labels = np.ma.masked_array(label_map, mask=(label_map == 0), fill_value=0)
    # normalize datacube
    if patch_norm == 'PCA':
        norm_cube = None
    norm_cube = normalize_datacube(flux_cube, norm=patch_norm)
    # pad image
    padded_cube = pad_hsi_image(norm_cube, patch_size=patch_size)

    # go through patchification
    global_size = label_map.shape[0]
    list_to_save = []
    for patch_index in np.ndindex(label_map.shape):
        # only look at non-masked pixels
        # if the pixel is masked, it will have a "mask" attribute
        # otherwise, it's just an integer
        if not hasattr(masked_labels[patch_index],'mask'):
            this_patch, patch_lab = grab_spatial_patch(
                padded_cube, 
                masked_labels, 
                patch_index, 
                patch_size=patch_size
            )

            # this is weird, 1/1,000,000 is off by 1 in patch size...just don't add them to CSV
            if this_patch.shape != (patch_size,patch_size,flux_cube.shape[-1]):
                continue
            # save patch to file
            # /path/to/dir/{plate-IFU}_size_x-y_label.npy
            patch_str = '-'.join([str(i) for i in patch_index])
            output_patch_path = patch_save_dir / f"{patch_save_dir.name}_{global_size}_{patch_str}_{patch_lab}.npy"
#             print(output_patch_path)
            np.save(output_patch_path, this_patch) # can't save masked arrays yet
            # output path and label for CSV
            this_dict = {
                'data': output_patch_path, # path to patch
                'label': patch_lab, # integer label
            }
            list_to_save.append(this_dict)
    return list_to_save