# SSFTT HSI model for MaNGA data
## Utilities
- `utils.py`: functions for filepaths (fits file to label file etc.)
- `data_split_char.ipynb`: creating and visualizing data splits.
  - `create_data_splits.py`, `create_easy_data_splits.py` are scripted alternatives.

## MaNGA data processing
- `cube_processing.py`: this is a wrapper for calling the data processing pipeline, which consists of (1) cube processing functions and map/label processing functions contained in "fits_utils.py", (2) cube patchifying and normalization contained in "hsi_proc_fns.py", and (3) writing split CSVs.
  - `process_single_cube.py`: This is the same as above, but only runs on a single input fits file
  - `cube_patch_processing.py`: This is the same as above, but skips the fits file processing so it does not require any of the astronomy/manga DAP packages.
- `fits_utils.py`: Contains "process_cube" and "process_map". "process_cube" reads in a MaNGA fits file and writes a H5 cube. This function de-redshifts and resamples the datacube to a fixed wavelength grid. "process_map" makes BPT/logOH labels and saves to png/npy.
- `hsi_proc_fns.py`: Contains functions that normalize datacubes and divide cubes into patches.
- `calc_data_stats.py`: for generating PCA representations of datacubes
- `download_utils.py`: for downloading missing datacubes

## Model and Dataloaders
- `HSI_SSFTT.py`: This defines the model architecture. Fiducial model returned from function get_SSFTT_model.
- `data_fns.py`: This contains dataset/dataloader functions and any preprocessing/collate functions.

## Training
- `run_SSFTT_training.py`: this is the main training script.

## Evaluating trained models
- `demo.ipynb`: visualize predictions on a single datacube; look at training curves and evaluate test splits with trained model.
- `eval_fns.py`: includes a prediction function, a function to load a trained model, a function that runs a trained model on a single data cube, and a function that runs a trained model on the full test split.
- `vis_fns.py`: Code that creates graphics to compare predicted and ground truth maps for a datacube.

# Data structure
- patch_norm: {'global','spatial','spectral','PCA'}  
- label_task: {'BPT','logOH'}  

### Data Locations
`DATA_DIR = /qfs/projects/gumby/data/manga`  
- h5 datacubes: `DATA_DIR / processed / cubes / plate / {plate-id}.h5`  
- numpy patches: `DATA_DIR / processed / patches / label_task / patch_norm / plate-id / {plate-id}.npy`  
- label maps: `DATA_DIR / processed / labels / label_task / {plate-id}.npy`  
- raw fits files: `DATA_DIR / raw / plate / id / {plate-id}.fits.gz`  