import subprocess
from pathlib import Path

# Download manga and pipe3d catalog
# Download SDSS and Pipe3D catalog
from download_utils import *
data_dir = pathlib.Path('/gscratch/scrubbed/mmckay18/DATA/')
# Downloading data DAP Maps and Pipe3D Maps from MaNGA MPL11

# get dap and pipe3d catalogs for plateifu
print("Downloading DAPALL data...")
dap_url = "https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/dapall-v3_1_1-3.1.0.fits"
dapall_save_path = data_dir/ 'raw'/ 'dapall-v3_1_1-3.1.0.fits'  # Update this path to your specific remote directory path
download_file(dap_url, dapall_save_path)

print("Downloading Pipe3D data...")
pipe3d_url = "https://data.sdss.org/sas/dr17/manga/spectro/pipe3d/v3_1_1/3.1.1/SDSS17Pipe3D_v3_1_1.fits"
pipe3d_save_path = data_dir/ 'raw'/ 'SDSS17Pipe3D_v3_1_1.fits'  # Update this path to your specific remote directory path
download_file(pipe3d_url, pipe3d_save_path)


# Running the split and cube_processing
work_dir = '/gscratch/astro/mmckay18/gumby-spectral-imaging_fork/HSI/'

# Define the directories to check and create if they do not exist
directories = [
    Path('/gscratch/scrubbed/mmckay18/DATA/processed/labels/BPT/'),
    Path('/gscratch/scrubbed/mmckay18/DATA/splits/BPT/')
]

# Iterate over the directories and create them if they don't exist
for directory in directories:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")
    

# Create data split
### Make the split directory
# mkdir /gscratch/scrubbed/mmckay18/DATA/processed/labels/BPT/
# mkdir /gscratch/scrubbed/mmckay18/DATA/splits/BPT/
#create_data_splits_command = [
#    'python', 
#    'create_data_splits.py', 
#    '--split_dir=BPT', 
#    '--label_task=BPT', 
#    '--patch_norm=global'
#]

# Create easy data split
create_easy_data_splits_command = [
    'python', 
    'create_easy_data_splits.py', 
    '--split_dir=BPT', 
    '--label_task=BPT', 
    '--patch_norm=global'
]

# Cube processing for Training set
# !python cube_processing.py --split_dir=BPT --patch_size=9 --patch_norm=global --label_task=BPT --glob_patches --splits=train

cube_processing_train_command = [
    'python', 
    'cube_processing.py', 
    '--split_dir=BPT', 
    '--label_task=BPT', 
    '--patch_norm=global',
    '--patch_size=9',
    '--glob_patches',
    '--splits=train'
]

# Cube processing for test set
# !python cube_processing.py --split_dir=BPT --patch_size=9 --patch_norm=global --label_task=BPT --glob_patches --splits=test

cube_processing_test_command = [
    'python', 
    'cube_processing.py', 
    '--split_dir=BPT', 
    '--label_task=BPT', 
    '--patch_norm=global',
    '--patch_size=9',
    '--glob_patches',
    '--splits=test'
]

command_order = [
                # create_data_splits_command, 
                 cube_processing_train_command, 
                 cube_processing_test_command,
                 create_easy_data_splits_command]

for command in command_order:
    print(f"Running: {command[1]}")
    result = subprocess.run(command, capture_output=True, text=True, cwd=work_dir)
    print(f"{command[1]}_STDOUT:\n", result.stdout)
    print(f"{command[1]}_ERROR:\n", result.stderr)
