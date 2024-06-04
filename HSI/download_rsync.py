#!/usr/bin/env python

import os
import pandas as pd


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
for plateifu in plateifu_list:
    plate, ifu = plateifu.split("-")

    # Retrieves DAP Maps
    # os.system(
    #     "rsync -avz --password-file {} rsync://sdss@dtn01.sdss.utah.edu/sas/mangawork/manga/spectro/analysis/MPL-11/HYB10-MILESHC-MASTARSSP/{}/{}/*manga-{}-{}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz /Volumes/lil_oynx/bbrd_dapfits".format(
    #         pw_file, plate, ifu, plate, ifu
    #     )
    # )

    os.system(
        f"rsync -avz --password-file {pw_file} rsync://sdss@dtn01.sdss.utah.edu/dr17/manga/spectro/analysis/v3_1_1/3.1.0/SPX-MILESHC-MASTARSSP/{plate}/{ifu}/*manga-{plate}-{ifu}-{key}-SPX-MILESHC-MASTARSSP.fits.gz /gscratch/astro/mmckay18/DATA/raw/{plate}/{ifu}/"
    )