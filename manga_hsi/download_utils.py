import pathlib
from astropy.table import Table
import urllib.request
import shutil
from multiprocessing.pool import ThreadPool
from time import time as timer
#from gumby_utils.constants import data_dir, thread_limit
#thread_limit(1)

data_dir = pathlib.Path('/qfs/projects/gumby/data/manga')
fits_dir = data_dir / "raw"

def get_url(PLATEIFU, key='MAPS'):
    '''
    Returns  url to fetch and local save path
    for input row['PLATEIFU']
    key = 'MAPS' or 'LOGCUBE'
    '''
    file_stem = f"manga-{PLATEIFU}-{key}-SPX-MILESHC-MASTARSSP.fits.gz"
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
    plateifu = '-'.join(fits_file.split('/')[-3:-1])
    output = get_url(plateifu, key=key)
    results = fetch_and_save(*output)
    return results