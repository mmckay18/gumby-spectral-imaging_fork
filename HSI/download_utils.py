import pathlib
from astropy.table import Table
import urllib.request
import shutil
from multiprocessing.pool import ThreadPool
from time import time as timer
import os
from astropy.io import fits
import pandas as pd
#from gumby_utils.constants import data_dir, thread_limit
#thread_limit(1)

data_dir = pathlib.Path('/gscratch/scrubbed/mmckay18/DATA')
fits_dir = data_dir / "raw"

def get_url(PLATEIFU, key='MAPS'):
    '''
    Returns  url to fetch and local save path
    for input row['PLATEIFU']
    key = 'MAPS' or 'LOGCUBE'
    '''
    print(f"manga-{PLATEIFU}-{key}-SPX-MILESHC-MASTARSSP.fits.gz")
    file_stem = f"manga-{PLATEIFU}-{key}-SPX-MILESHC-MASTARSSP.fits.gz"
    print('file_stem')
    save_path = fits_dir / f"{PLATEIFU.replace('-','/')}" / file_stem
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = "https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/SPX-MILESHC-MASTARSSP"
    url_str = f"{base_url}/{str(save_path.relative_to(fits_dir))}"
    return url_str, save_path

def fetch_and_save(url, save_path, overwrite=True):
    '''
    fetches file at url and saves to save_path
    '''
    if overwrite or not save_path.exists():
        try:
            with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            return url, response.read(), None
        except Exception as e:
            return url, None, e
    else:
        return url, True, None

def download_single_cube(fits_file, key='LOGCUBE'):
    print('Downloading single cube function')
    print(f'{fits_file}')
    plateifu = '-'.join(fits_file.split('/')[-3:-1])
    print(f'PLATEIFU: {plateifu}')
    output = get_url(plateifu, key=key)
    print(f'URL to download: {output}')
    results = fetch_and_save(*output) # changed *output
    print(f'Fetch results: {results}')
    # print(output, results)
    return results


def remove_empty_dirs(path):
    # Walk through the directory tree, from bottom to top
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            # Check if the directory is empty
            if not os.listdir(dir_path):
                # Remove the empty directory
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")


def download_file(url, save_path):
    """
    Download a file from a specified URL to a target directory.

    Parameters:
    - url: The URL of the file to download.
    - save_path: The full path (including filename) where the file will be saved.
    """
    try:
        # Download the file from `url` and save it locally under `save_path`
        urllib.request.urlretrieve(url, save_path)
        print(f"File downloaded successfully and saved as {save_path}")
    except Exception as e:
        print(f"Failed to download the file. Error: {e}")


def fits_to_dataframe(fits_file_path):
    """
    Read a FITS file and convert its data to a pandas DataFrame.

    Parameters:
    - fits_file_path: Path to the FITS file.

    Returns:
    - DataFrame containing the data from the FITS file.
    """
    with fits.open(fits_file_path) as hdul:
        # Assuming the data is in the first extension (change if needed)
        data = hdul[1].data
        # Convert to a pandas DataFrame
        df = pd.DataFrame(data)
    return df