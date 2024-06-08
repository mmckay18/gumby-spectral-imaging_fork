#!/usr/bin/env python
import pathlib
from astropy.table import Table
import urllib.request
import shutil
from multiprocessing.pool import ThreadPool
from time import time as timer
import os
import pandas as pd
# from download_utils import download_single_cube
from download_utils import *


pw_file = "/gscratch/astro/mmckay18/gumby-spectral-imaging_fork/HSI/sdss_manga_dapall_data/sdss_pw.txt"  # SDSS Collab password file

# Downloading data DAP Maps and Pipe3D Maps from MaNGA MPL11

# DAP Maps for BBRD sample
# Read in csv with BreakBRD plateifu
df = pd.read_csv(
    "/gscratch/astro/mmckay18/gumby-spectral-imaging_fork/HSI/sdss_manga_dapall_data/drpall_random_subsample.csv"
)

# Interate through the plateifu and download MPL11 DAP Maps
plateifu_list = df["plateifu"]
key = 'LOGCUBE' # 'LOGCUBE' or 'MAPS'
for plateifu in plateifu_list[:10]:
    plate, ifu = plateifu.split("-")
    print(plate, ifu)
    print(f'rsync://sdss@dtn01.sdss.utah.edu/dr17/manga/spectro/analysis/v3_1_1/3.1.0/SPX-MILESHC-MASTARSSP/{plate}/{ifu}/*manga-{plate}-{ifu}-{key}-SPX-MILESHC-MASTARSSP.fits.gz')

    # Retrieves DAP Maps
    os.system(f"rsync -avz --password-file {pw_file} rsync://sdss@dtn01.sdss.utah.edu/dr17/manga/spectro/analysis/v3_1_1/3.1.0/SPX-MILESHC-MASTARSSP/{plate}/{ifu}/*manga-{plate}-{ifu}-{key}-SPX-MILESHC-MASTARSSP.fits.gz /gscratch/astro/mmckay18/DATA/raw/{plate}/{ifu}/")


#     # https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/SPX-MILESHC-MASTARSSP/10001/12701/manga-10001-12701-LOGCUBE-SPX-MILESHC-MASTARSSP.fits.gz


# # From Nell download_utils.py
# # fits_file = '/gscratch/astro/mmckay18/DATA/raw/10001/12701/manga-10001-12701-LOGCUBE-SPX-MILESHC-MASTARSSP.fits.gz'
#     fits_file = f'manga-{plate}-{ifu}-{key}-SPX-MILESHC-MASTARSSP.fits.gz'
#     print(fits_file)
#     download_single_cube(fits_file, key='LOGCUBE')