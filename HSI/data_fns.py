import torch
import glob
import pathlib
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Callable
import logging

from utils import get_OH_bins_and_labels
from hsi_proc_fns import h5_datapoint_loader, masked_h5_datapoint_loader, patchify_cube
from utils import(
    get_cube_path, 
    get_label_path,
    get_label_path_from_cube,
    get_patch_dir_from_cube_file
)
from aug_utils import CosineSimAug

def npy_datapoint_loader(path):
    return np.load(path)

def stack_preprocessing_fn(datapoint):
    datapoint = torch.Tensor(datapoint)
    return datapoint.reshape(datapoint.shape[0]*datapoint.shape[0], datapoint.shape[-1])

def ssftt_label(this_label):
    """Take in single label (dim 1) and output tensor
    NOTE: this is where we had to fix the labels so that they start at 0
    """
    this_label = torch.Tensor([this_label-1]).type(torch.LongTensor)
    return this_label

class preprocData(torch.nn.Module):
    def __init__(self, normalize=False, mean=0.0162, std=0.0409, add_extra_dim=True) :
        '''
        Initialize data processing function so that patches can be normalized
        default mean and std were calculated over the bpt train/val splits
        '''
        super(preprocData, self).__init__()
        self.normalize=normalize
        self.mean = mean
        self.std = std
        self.add_extra_dim = add_extra_dim

    def forward(self, this_patch):
        """Take in single patch (dim HxWxC) and output (1, C, H, W) 
        """
        # (H x W x C) > (C x H x W)
        this_patch = this_patch.transpose(2,0,1)
    
        # normalize, if needed
        if self.normalize:
            this_patch = (this_patch - self.mean) / self.std
        
        # Create Tensors
        this_patch = torch.from_numpy(this_patch).type(torch.FloatTensor)

        if self.add_extra_dim:
            # 1, C, H, W 
            # note: model requires weird extra "1" dimension
            this_patch = this_patch.unsqueeze(0)
        return this_patch

class repatchData(torch.nn.Module):
    def __init__(self, current_patch_size=7, new_patch_size=7, wave_inds=(0,2040), normalize=False, 
                 mean=0.0162, std=0.0409, add_extra_dim=True) :
        '''Reduces patch size and crops wavelength dimension
        Foward pass does the needed reshaping for SSFTT
        '''
        super(repatchData, self).__init__()
        self.new_patch_size=new_patch_size
        self.add_extra_dim = add_extra_dim
        size_diff = (current_patch_size - new_patch_size)//2
        if current_patch_size == new_patch_size:
            self.spatial_slice = Ellipsis
        else:
            self.spatial_slice = slice(size_diff,-size_diff)
        self.wave_slice = slice(*wave_inds)
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def forward(self, patch):
        # [new H, new W, new C]
        #if not self.spatial_slice == self.wave_slice:
        patch = patch[self.spatial_slice, self.spatial_slice, self.wave_slice]
        # (H x W x C) > (C x H x W)
        patch = patch.transpose(2,0,1)

        if self.normalize:
            patch = (patch - self.mean)/self.std
        
        # Create Tensors
        patch = torch.from_numpy(patch).type(torch.FloatTensor)

        # 1, C, H, W 
        # note: model requires weird extra "1" dimension
        if self.add_extra_dim:
            patch = patch.unsqueeze(0)
        if self.new_patch_size == 1:
            # (batch, 1, C)
            patch = patch.unsqueeze(0)
            patch = patch.squeeze(-1).squeeze(-1)
        return patch

def default_collate_fn(batch):
    '''Default collate function
    '''
    images = [image for image, _ in batch]
    images = torch.stack(images)
    targets = torch.LongTensor([target for _, target in batch])
    return images, targets

def map_collate_fn(batch):
    '''Collate fn for HSICubeDataset evaluation
    returns data, labels, map_inds
    '''
    datapoints = [datapoint for datapoint, _, _ in batch]
    datapoints = torch.stack(datapoints)
    targets = torch.LongTensor([target for _, target,_ in batch])

    inds = [ind for _,_,ind in batch]
    return datapoints, targets, inds

def concat_collate_fn(batch):
    '''Default collate function
    '''
    batch = torch.concat(batch)
    return batch

class BinLogOHLabels(torch.nn.Module):
    def __init__(self,
                 OH_key: str='default',
                encoder_type: str='int') :
        '''Bins OH values & encodes labels; returns (labels, encoded_lables)
        encoder_type = ['int','ordinal','one_hot']
        '''
        super(BinLogOHLabels, self).__init__()
        OH_bins, OH_labels = get_OH_bins_and_labels(OH_key)
        assert(OH_bins is not None)
        assert(len(OH_labels) == (len(OH_bins)+1))
        assert(encoder_type in ['int','ordinal','one_hot'])

        self.OH_bins = OH_bins
        self.OH_labels = OH_labels
        self.encoder_type = encoder_type
        self.name_to_index = self._get_encoder_dict()
    
    def _get_encoder_dict(self):
        """sets name_to_index dictionary for given encoder type"""
        if self.encoder_type == 'ordinal':
            name_to_index = {label:[1 if j <= i else 0 for j in range(len(self.OH_labels))] for i,label in enumerate(self.OH_labels)}
        elif self.encoder_type == 'one_hot':
            name_to_index = {label:[1 if j == i else 0 for j in range(len(self.OH_labels))] for i,label in enumerate(self.OH_labels)}
        else:
            name_to_index = {label:i for i,label in enumerate(self.OH_labels)}
        return name_to_index

    def forward(self, input):
        """input values, output (labels, encoded_labels)"""
        encoded_output = np.digitize(input, bins=self.OH_bins)
        encoded_output = torch.from_numpy(encoded_output)
        return encoded_output

class HSIPatchDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        datapoint_loader: Callable=npy_datapoint_loader,
        preprocess_fn: Callable=preprocData(normalize=False),
        label_fn: Callable=BinLogOHLabels(OH_key='extra'),
        augment_data:bool=False,
        augment_fn: Callable=CosineSimAug(max_deg=5, p=0.3, spec_p=1.0),
        use_local_data:bool=False,
        local_dir:str='/raid/byle431'
    ):
        """ Used with CSV of patches and labels
        annotations_file: csv file with colums (patch path, label) for all datapoints in split
        datapoint_loader: function to use to for datapoint I/O (i.e. load npy)
        augment_fn: function used to apply data augmentation to datapoints (only used if augment_data=True)
        preprocess_fn: function to preprocess data with (default ssftt data)
        label_fn: function to preprocess/encode labels (i.e. turn numerical value into class bin)
        use_local_data: CSV paths point to /qfs/*, this flag tells the model to use local_dir as prefix
        local_dir: path prefix for data
        """
        self.annotations_file = annotations_file
        df = pd.read_csv(annotations_file)
        self.df = df
        self.data = df["data"]
        if use_local_data:
            self.data = df["data"].map(lambda x: x.replace('/qfs/projects/thidwick',local_dir))
        # these are numerical values (i.e. 12+logOH)
        self.labels = df['label'].map(float)
        # this is the mapped label (i.e. class=1 for classification or a normalized number for regression)
        self.encoded_labels = label_fn(self.labels)
        self.datapoint_loader = datapoint_loader
        self.preprocess_fn = preprocess_fn
        self.label_fn = label_fn
        self.augment_data = augment_data
        self.augment_fn = augment_fn
        self.rng = np.random.default_rng()
    
    def get_labels(self):
        """This is used for the imabalanced dataset sampler"""
        return self.encoded_labels

    def __len__(self):
        """Returns the length of the dataset. This is the number of datapoints in the dataset.
        :return: The length of the dataset
        :rtype: int
        """
        length = len(self.data)
        return length

    def __getitem__(self, index):
        """Takes in a datapoint index and performs most of the data pipeline process
        and then passes the datapoint back.

        :param index: int or tuple, int if training classically, tuple if training in a fewshot manner
        :type index: int or tuple
        :return: tuple consisting of a datapoint and its labels.
        :rtype: tuple
        """
        # encoded labels are mapped (e.g. integer values); self.labels is raw value
        label = self.encoded_labels[index]
        datapoint = self.data[index]

        # run through dataloader functions
        if self.datapoint_loader is not None:
            datapoint = self.datapoint_loader(datapoint)
        
        # add data augmentation
        if self.augment_data:
            if isinstance(self.augment_fn,list):
                #select random choice of index and then assign self.augment_fn[index of choice](datapoint)
                choice = self.rng.integers(0,len(self.augment_fn))
                datapoint = self.augment_fn[choice](datapoint)

            else:
                datapoint = self.augment_fn(datapoint)
        
        # run through preprocess functions
        if self.preprocess_fn is not None:
            datapoint = self.preprocess_fn(datapoint)
        
        return datapoint, label

class HSICubeDataset(Dataset):
    def __init__(
        self,
        cube_file=None,
        datapoint_loader: Callable=npy_datapoint_loader,
        preprocess_fn: Callable=preprocData(normalize=False),
        label_fn: Callable=BinLogOHLabels(OH_key='extra'),
        patch_size=7,
        patch_norm: str='global',
        label_task: str='logOH',
        use_local_data:bool=False,
        local_dir:str='/raid/byle431'
    ):
        """ To run evaluation on whole cube
        cube_file: can be fits_file or cube_file
        datapoint_loader: function to use to load in data
        preprocess_fn: function to preprocess data with
        label_fn: function to preprocess/ecode labels
        NOTE: use map_collate_fn for dataloaders
        """
        # inputs can be str or pathlib, or fits/cube
        if type(cube_file) is str:
            cube_file = pathlib.Path(cube_file)
        if cube_file.suffix == '.gz':
            cube_file = get_cube_path(
                str(cube_file), 
                label_task=label_task
            )
        if not cube_file.exists():
            raise FileExistsError
        
        # get labels and patch directory
        label_path = get_label_path_from_cube(
            cube_file, 
            label_task=label_task
        )
        patch_dir = get_patch_dir_from_cube_file(
            cube_file, 
            patch_norm=patch_norm, 
            label_task=label_task
        )
        if use_local_data:
            patch_dir = pathlib.Path(str(patch_dir).replace('/qfs/projects/thidwick',local_dir))

        if not patch_dir.exists():
            raise FileExistsError
        
        list_of_patches = glob.glob(f'{patch_dir}/*.npy')
        if len(list_of_patches) == 0:
            logging.info('Patching datacube...')
            list_of_patch_dicts = patchify_cube(
                cube_file, 
                label_path,
                patch_size=patch_size, 
                patch_norm=patch_norm,
                patch_save_dir=patch_dir,
            )
            list_of_patches = [d['data'] for d in list_of_patch_dicts]
            logging.info('...done.')
        else:
            logging.info(f'Found {len(list_of_patches)} npy files in {patch_dir}')
        
        # save as attributes
        self.cube_file = cube_file
        self.patch_dir = patch_dir
        self.label_task = label_task
        # path to patch file
        self.data = list_of_patches
        # label of patch
        
        self.labels = np.array([float(self.data[i].strip('.npy').split('_')[-1]) for i in range(len(self.data))])
        self.encoded_labels = label_fn(self.labels)
        # index of patch in label map
        self.map_inds = [tuple([int(c) for c in self.data[i].split('_')[-2].split('-')]) for i in range(len(self.data))]
        # so we know how to re-index
        self.label_map_size = int(self.data[0].split('_')[1])

        self.datapoint_loader = datapoint_loader
        self.preprocess_fn = preprocess_fn
        self.label_fn = label_fn

    def __len__(self):
        """Returns the length of the dataset. This is the number of datapoints in the dataset.
        :return: The length of the dataset
        :rtype: int
        """
        length = len(self.data)
        return length

    def __getitem__(self, index):
        """Takes in an index and returns (datapoint, label, map_index)

        :param index: int or tuple, int if training classically, tuple if training in a fewshot manner
        :type index: int or tuple
        :return: tuple consisting of a datapoint and its labels.
        :rtype: tuple
        """
        label = self.encoded_labels[index]
        datapoint = self.data[index]
        map_index = self.map_inds[index]

        # run through dataloader functions
        if self.datapoint_loader is not None:
            datapoint = self.datapoint_loader(datapoint)
        
        # run through preprocess functions
        if self.preprocess_fn is not None:
            datapoint = self.preprocess_fn(datapoint)
        
        return datapoint, label, map_index

class NormalizeCube(torch.nn.Module):
    def __init__(self, norm=None) :
        '''Used as preprocess function to normalize datacube
        '''
        super(NormalizeCube, self).__init__()
        self.norm=norm
    def forward(self, cube):
        '''Performs a normalization of the input image.
        input: 
            hsi_image: masked datacube
            norm (str): how to normalize cube
                if 'spatial', normalizes image at each wavelength (i.e. spatial map normalization)
                if 'global', normalizes entire cube
                if None, normalizes spectrum for each pixel (default)
        output: normalized datacube
        '''
        if self.norm == 'spatial':
            cube_mins = cube.min(axis=(0,1), keepdims=True)
            cube_maxs = cube.max(axis=(0,1), keepdims=True)
        elif self.norm == 'global':
            cube_mins = cube.min(keepdims=True)
            cube_maxs = cube.max(keepdims=True)
        else:
            cube_mins = cube.min(axis=2, keepdims=True)
            cube_maxs = cube.max(axis=2, keepdims=True)
        output = (cube - cube_mins) / (cube_maxs - cube_mins)
        return output

class H5cubeDataset(Dataset):
    def __init__(
        self,
        input_fits_csv,
        datapoint_loader: Callable=masked_h5_datapoint_loader,
        preprocess_fn: Callable=NormalizeCube(norm='global'),
        label_task: str='logOH',
        flatten=True,
        wave_inds=(0,2040),
        verbose=False
    ):
        """NOTE: not done yet, can't stack cubes of diff dim
        annotations_file: csv/pkl/hd5 file with patches and labels
        datapoint_loader: function to use to load in data -- not used right now
        label_loader: function to use to load in labels -- not used right now
        preprocess_fn: function to preprocess data with
        label_preprocess_fn: function to preprocess labels with
        verbose: print more outputs
        """
        if type(input_fits_csv) == list:
            fits_files = []
            for this_csv in input_fits_csv:
                fits_files += pd.read_csv(this_csv, header=None, names=['fits_file'])['fits_file'].to_list()
        else:
            fits_files = pd.read_csv(input_fits_csv, header=None, names=['fits_file'])['fits_file'].to_list()
        self.data = [get_cube_path(fits_file, label_task=label_task) for fits_file in fits_files]
        self.label_files = [get_label_path(fits_file=f, label_task=label_task) for f in fits_files]
        self.datapoint_loader = datapoint_loader
        self.preprocess_fn = preprocess_fn
        self.flatten=flatten
        self.verbose = verbose
        self.wave_inds=wave_inds
        self.reindex = wave_inds != (0,2040)
        self.wave_slice = slice(*wave_inds)

    def __len__(self):
        """Returns the length of the dataset. This is the number of datapoints in the dataset.
        :return: The length of the dataset
        :rtype: int
        """
        length = len(self.data)
        return length

    def __getitem__(self, index):
        """Takes in a datapoint index and performs most of the data pipeline process
        and then passes the datapoint back.

        :param index: int or tuple, int if training classically, tuple if training in a fewshot manner
        :type index: int or tuple
        :return: tuple consisting of a datapoint and its labels.
        :rtype: tuple
        """
        datapoint = self.data[index]

        if self.verbose:
            logging.info(datapoint, type(datapoint))

        # run through dataloader functions
        if self.datapoint_loader is not None:
            _, datapoint = self.datapoint_loader(datapoint)

        if self.preprocess_fn is not None:
            datapoint = self.preprocess_fn(datapoint)
        
        if type(datapoint) is np.ma.core.MaskedArray:
            datapoint = datapoint.filled(0.0)
        return datapoint

import pickle
from hsi_proc_fns import(grab_spatial_patch, pad_hsi_image)
class PatchProcessing(torch.nn.Module):
    def __init__(self, split_dir='bpt', patch_size=7, label_task='logOH',
                 patch_norm='global', wave_inds=(0,2040)) :
        '''
        Process input with pre-fit PCA transform
        '''
        super(PatchProcessing, self).__init__()
        self.split_dir = split_dir
        self.patch_norm = patch_norm
        self.patch_size = patch_size
        self.label_task = label_task

        if self.patch_norm == 'PCA':
            self.normalize_fn = self.pca_normalize
            with open(f'./data/{split_dir}_PCA.pkl', 'rb') as f:
                this_dict = pickle.load(f)
            self.pca_estimator = this_dict['pca_estimator']
            self.mean = this_dict['mean']
            self.std = this_dict['std']
        else:
            self.normalize_fn = self.minmax_normalize

    def pca_normalize(self, batch):
        """Take in cube (H, W, C) and output (H, W, N_COMPONENTS) 
        """
        if self.reindex:
            batch = batch[...,self.wave_slice]
        # (H, W, C) > (H*W, C)
        original_shape = batch.shape
        batch = batch.reshape(original_shape[0]*original_shape[0],original_shape[-1])
        # normalize to mean and std of full dataset
        batch = (batch - self.mean) / self.std
        # transform data with pre-fit PCA 
        batch = self.pca_estimator.transform(batch)
        # (H*W, C) > (H, W, N_COMP)
        batch = batch.reshape(original_shape[0], original_shape[1], batch.shape[-1])
        # normalize
        mins = batch.min(axis=(0,1), keepdims=True)
        maxs = batch.max(axis=(0,1), keepdims=True)
        batch = (batch - mins) / (maxs - mins)
        return batch
    def minmax_normalize(self, batch):
        '''Performs a normalization of the input image.
        input: 
            hsi_image: masked datacube
            norm (str): how to normalize cube
                if 'spatial', normalizes image at each wavelength (i.e. spatial map normalization)
                if 'global', normalizes entire cube
                if 'spectral', normalizes spectrum for each pixel (default)
        output: normalized datacube
        '''
        if self.reindex:
            batch = batch[...,self.wave_slice]
        if type(batch) is not np.ma.core.MaskedArray:
            batch = np.ma.masked_array(batch, mask=(batch <= 0.0), fill_value=0.0)
        if self.patch_norm == 'spatial':
            cube_mins = batch.min(axis=(0,1), keepdims=True)
            cube_maxs = batch.max(axis=(0,1), keepdims=True)
        elif self.patch_norm == 'global':
            cube_mins = batch.min(keepdims=True)
            cube_maxs = batch.max(keepdims=True)
        else:
            cube_mins = batch.min(axis=2, keepdims=True)
            cube_maxs = batch.max(axis=2, keepdims=True)
        batch = (batch - cube_mins) / (cube_maxs - cube_mins)
        return batch.filled(0.0)
    
    def forward(self, cube_file, save_dir):
        # open cube and labels
        flux_cube = h5_datapoint_loader(cube_file)
        # open labels
        label_path = get_label_path_from_cube(cube_file, label_task=self.label_task)
        label_map = npy_datapoint_loader(label_path)
        # mask labels, process datacube
        masked_labels = np.ma.masked_array(label_map, mask=(label_map == 0), fill_value=0)
        norm_cube = self.normalize_fn(flux_cube)
        padded_cube = pad_hsi_image(norm_cube, patch_size=self.patch_size)
        # go through patchification
        global_size = label_map.shape[0]
        list_to_save = []
        for patch_index in np.ndindex(label_map.shape):
            # only look at non-masked pixels
            # if the pixel is masked, it will have a "mask" attribute (otherwise, it's an integer)
            if not hasattr(masked_labels[patch_index],'mask'):
                this_patch, patch_lab = grab_spatial_patch(
                    padded_cube, 
                    masked_labels, 
                    patch_index, 
                    patch_size=self.patch_size
                )
                # this is weird, 1/1,000,000 is off by 1 in patch size...just don't add them to CSV
                if this_patch.shape != (self.patch_size, self.patch_size, norm_cube.shape[-1]):
                    continue
                # save patch to file
                # /path/to/dir/{plate-IFU}_size_x-y_label.npy
                patch_str = '-'.join([str(i) for i in patch_index])
                output_patch_path = save_dir / f"{save_dir.name}_{global_size}_{patch_str}_{patch_lab}.npy"
                np.save(output_patch_path, this_patch)
                # output path and label for CSV
                this_dict = {
                    'data': output_patch_path, # path to patch
                    'label': patch_lab, # integer label
                }
                list_to_save.append(this_dict)
        return list_to_save