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




# # SDSS Collab password file
# # pw_file = "/gscratch/astro/mmckay18/gumby-spectral-imaging_fork/HSI/sdss_manga_dapall_data/sdss_pw.txt"  

# # Downloading data DAP Maps and Pipe3D Maps from MaNGA MPL11

# # get dap and pipe3d catalogs for plateifu
# dap_url = "https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/dapall-v3_1_1-3.1.0.fits"
# dapall_save_path = "/gscratch/astro/mmckay18/DATA/dapall-v3_1_1-3.1.0.fits"  # Update this path to your specific remote directory path
# download_file(dap_url, dapall_save_path)
# dapall_fits_path = "/gscratch/astro/mmckay18/DATA/dapall-v3_1_1-3.1.0.fits"
# dapall_df = fits_to_dataframe(dapall_fits_path)
# dapall_df.to_csv("/gscratch/astro/mmckay18/gumby-spectral-imaging_fork/HSI/sdss_manga_dapall_data/drpall_subsample.csv", index=False)
# dap_df = pd.read_csv("/gscratch/astro/mmckay18/gumby-spectral-imaging_fork/HSI/sdss_manga_dapall_data/drpall_random_subsample.csv", nrows=5)
# print(dap_df['plateifu'].to_list())

# plateifu_list = dap_df['plateifu'].to_list()
# with open('/gscratch/astro/mmckay18/DATA/plateifu_list.txt', 'w') as file:
#     for plateifu in plateifu_list:
#         file.write(plateifu + '\n')


# pipe3d_url = "https://data.sdss.org/sas/dr17/manga/spectro/pipe3d/v2_4_3/2.4.3/pipe3d-v2_4_3-2.4.3.fits"

# pipe3d_save_path = "/gscratch/astro/mmckay18/DATA/pipe3d-v2_4_3-2.4.3.fits"  # Update this path to your specific remote directory path
# download_file(pipe3d_url, pipe3d_save_path)
# # Paths to the FITS files
# pipe3d_fits_path = "/gscratch/astro/mmckay18/DATA/pipe3d-v2_4_3-2.4.3.fits"
# # Convert to DataFrames
# pipe3d_df = fits_to_dataframe(pipe3d_fits_path)
# # Paths to the FITS files
# pipe3d_fits_path = "/gscratch/astro/mmckay18/DATA/pipe3d-v2_4_3-2.4.3.fits"
# # Convert to DataFrames
# pipe3d_df = fits_to_dataframe(pipe3d_fits_path)
# # Read in the drpall data
# pipe3d_df.to_csv("/gscratch/astro/mmckay18/gumby-spectral-imaging_fork/HSI/sdss_manga_dapall_data/pipe3d_subsample.csv", index=False)
# pipe3d_df = pd.read_csv("/gscratch/astro/mmckay18/gumby-spectral-imaging_fork/HSI/sdss_manga_dapall_data/pipe3d_subsample.csv", nrows=5)
# print(pipe3d_df['plateifu'].to_list())

# plateifu_list = pipe3d_df['plateifu'].to_list()
# with open('/gscratch/astro/mmckay18/DATA/plateifu_list.txt', 'w') as file:
#     for plateifu in plateifu_list:
#         file.write(plateifu + '\n')

df = pd.read_csv(
    "/gscratch/astro/mmckay18/gumby-spectral-imaging_fork/HSI/sdss_manga_dapall_data/drpall_random_subsample.csv"
)

# Interate through the plateifu and download MPL11 DAP Maps
plateifu_list = df["plateifu"].to_list()
# print(plateifu_list[:5])
key = 'LOGCUBE' # 'LOGCUBE' or 'MAPS'
for plateifu in plateifu_list[:3000]:
    plateifu = plateifu.strip(" \t")
    plate, ifu = plateifu.split("-")[0].replace(" ", ""), plateifu.split("-")[1].replace(" ", "")
    file_url, save_path = get_url(plateifu, key=key)
    # print(file_url)
    # print(save_path)
    results = fetch_and_save(file_url, save_path)

#     # # Make directory for cube and maps
#     # print('Make directory for cube and maps')
#     # if not os.path.exists(f"/gscratch/astro/mmckay18/DATA/raw/{plate}/{ifu}/"):
#     #     os.makedirs(f"/gscratch/astro/mmckay18/DATA/raw/{plate}/{ifu}/", exist_ok=True)
#     # else:
#     #     print(f"/gscratch/astro/mmckay18/DATA/raw/{plate}/{ifu}/ already exists")
    
#     # # Retrieves DAP Maps
#     # os.system(f"rsync -avz --password-file {pw_file} --no-motd rsync://data.sdss.org/dr17/manga/spectro/analysis/v3_1_1/3.1.0/SPX-MILESHC-MASTARSSP/{plate}/{ifu}/manga-{plate}-{ifu}-{key}-SPX-MILESHC-MASTARSSP.fits.gz /gscratch/astro/mmckay18/DATA/raw/{plate}/{ifu}/")

#     # https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/SPX-MILESHC-MASTARSSP/10001/12701/manga-10001-12701-LOGCUBE-SPX-MILESHC-MASTARSSP.fits.gz


# # fits_file = '/gscratch/astro/mmckay18/DATA/raw/10001/12701/manga-10001-12701-LOGCUBE-SPX-MILESHC-MASTARSSP.fits.gz'
#     # fits_file = f'manga-{plate}-{ifu}-{key}-SPX-MILESHC-MASTARSSP.fits.gz'
#     # print(fits_file)
#     # download_single_cube(fits_file, key='LOGCUBE')


# Clean directory tree
remove_empty_dirs(path="/gscratch/astro/mmckay18/DATA/raw/")