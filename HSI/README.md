# SSFTT HSI model for MaNGA data

Code for training and evaluating a deep learning model on MaNGA spectral data.

We gratefully acknowledge the use of the Spectral-Spatial Feature Tokenization Transformer model (Sun et al. 2022; [github](https://github.com/zgr6010/HSI_SSFTT)) in `HSI_SSFTT.py`, the SimpleNet architecture (Hasanpour et al. 2022; [github](https://github.com/Coderx7/SimpleNet_Pytorch)) in `HSI_SIMPLENET.py` and the MaNGA Data Analysis Pipeline ([github](https://github.com/sdss/mangadap)) in `fits_utils.py`. 

### MaNGA data processing
These scripts use a standalone conda environment built from the MaNGA DAP instructions [here](https://github.com/sdss/mangadap#installation). 
- `cube_processing.py`: this is a wrapper for calling the data processing pipeline, which consists of (1) cube processing functions and map/label processing functions contained in "fits_utils.py", (2) cube patchifying and normalization contained in "hsi_proc_fns.py", and (3) writing split CSVs.
  - `process_single_cube.py`: This is the same as above, but only runs on a single input fits file
  - `cube_patch_processing.py`: This is the same as above, but skips the fits file processing so it does not require any of the astronomy or manga data analysis pipeline packages.
- `fits_utils.py`: Contains "process_cube" and "process_map". "process_cube" reads in a MaNGA fits file and writes a H5 cube. This function de-redshifts and resamples the datacube to a fixed wavelength grid. "process_map" makes BPT/logOH labels and saves to png/npy.
- `download_utils.py`: for downloading missing datacubes

## Overview
The remaining scripts use a conda environment built for CUDA11.4 with python 3.10 and pytorch 1.13, specified in `hsi.yml` (see also `conda_env.md`).

### Utilities
- `utils.py`: functions for filepaths (fits file to label file etc.)
- `data_split_char.ipynb`: creating and visualizing data splits.
  - `create_data_splits.py`, `create_easy_data_splits.py` are scripted alternatives.

### Torch data processing
- `hsi_proc_fns.py`: Contains functions that normalize datacubes and divide cubes into patches.
- `calc_data_stats.py`: for generating PCA representations of datacubes

### Model and Dataloaders
- `HSI_SSFTT.py`: This defines the SSFTT model architecture. Fiducial model returned from function get_SSFTT_model.
- `HSI_SIMPLENET.py`: This defines the SimpleNet model architecture.
- `data_fns.py`: This contains dataset/dataloader functions and any preprocessing/collate functions.
- `aug_utils.py`: Data augmentations for dataloader.

### Training
- `run_training.py`: this is the main training script.

### Evaluation
- `demo.ipynb`: visualize predictions on a single datacube; look at training curves and evaluate test splits with trained model.
- `eval_utils.py`: includes a prediction function, a function to load a trained model, a function that runs a trained model on a single data cube, and a function that runs a trained model on the full test split.
- `vis_fns.py`: Code that creates graphics to compare predicted and ground truth maps for a datacube. 

## Data Structure
`DATA_DIR = /path/to/your/data/manga`  
- h5 datacubes: `DATA_DIR / processed / cubes / plate / {plate-id}.h5`  
- numpy patches: `DATA_DIR / processed / patches / label_task / patch_norm / plate-id / {plate-id}.npy`  
- label maps: `DATA_DIR / processed / labels / label_task / {plate-id}.npy`  
- raw fits files: `DATA_DIR / raw / plate / id / {plate-id}.fits.gz`  