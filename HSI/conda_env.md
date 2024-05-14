# Setting up a conda environment on DGX with CUDA 11.4
*doublerainbow, ohmahgerd, harlemshake*  

### Anaconda/miniconda installation
Make sure you have anaconda/miniconda installed, on these machines you can install to `/raid/{userid}`, which is slightly faster. Installation scripts should already exist in /raid/, you may need to create your own userid directory.
``` bash
cd /raid
# if for some reason the install script doesn't exist
# wget  https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh
# Click through user agreement, then type "yes"
# CHANGE INSTALL LOCATION: /raid/userid/anaconda3
# you can finish with conda init, but note it might mess up any customized bashrc env management
```

### Conda environment
The specific versions of pytorch compatible with 11.4 were determined from this page: https://pytorch.org/get-started/previous-versions/#linux-and-windows-10
``` bash
# conda environment
conda create --name gumby python=3.10
conda activate gumby
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# test this is True: python -c "import torch; print(torch.cuda.is_available())"
conda install pandas scipy seaborn tqdm h5py
conda install -c conda-forge torchmetrics torchinfo einops
# only needed for using notebooks
conda install jupyterlab ipython ipywidgets
conda install -c conda-forge scikit-learn
conda install scikit-image
```