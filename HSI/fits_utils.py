import numpy as np
import pandas as pd
import pathlib
import h5py
import subprocess
from PIL import Image
from astropy.io import fits
from astropy.table import Table

# This is a bitmask handling object from the DAP source code
# https://github.com/sdss/mangadap
from mangadap.dapfits import DAPCubeBitMask
from mangadap.util.sampling import Resample
from mangadap.util.fileio import channel_dictionary, channel_units
from download_utils import download_single_cube

data_dir = pathlib.Path('/gscratch/scrubbed/mmckay18/DATA')
alt_dir = pathlib.Path('/gscratch/scrubbed/mmckay18/DATA')

# load up manga catalog once -----------------------------------------
manga_catalogue_path = data_dir / "raw/dapall-v3_1_1-3.1.0.fits"
dat = Table.read(manga_catalogue_path, format="fits", hdu=1)
# this will get rid of multi-dim columns (ie. array of fluxes in a column)
names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
cat_df = dat[names].to_pandas()
cat_df["PLATEIFU"] = cat_df["PLATEIFU"].apply(lambda x: x.decode())
# ---------------------------------------------------------------------


def process_cube(
    fits_file,
    cube_file,
    new_wave_range=[5000.0, 8000.0],
    new_num_wave=2040,
    verbose=True,
):
    """Read in fits file, process, and write datacube to H5 file.
    INPUTS:
        cube_file (str): path to manga fits file
        new_wave_range: wavelength range to preserve
        new_num_wave: number of wavelength bins
        verbose (bool): verbosity of printed outputs
    """
#     print('', f'{fits_file}, {type(fits_file)}') #debugging
#     fits_file = str(fits_file)
#     plate_ifu = "-".join(fits_file.split("/")[-3:-1])
    print(fits_file)
#     print('48')
    plate_ifu = "-".join(str(fits_file).split("/")[-3:-1])
    z = cat_df["NSA_Z"][cat_df["PLATEIFU"] == plate_ifu].item()
    fits_file = pathlib.Path(fits_file)
#     print(f'\tPlateIFU: {plate_ifu}, z: {z} \n\t Path: {fits_file}') # debugging
    if not fits_file.exists():
#         print('54')
        print(f'\tNo LOGCUBE fits file exists \n\t{fits_file}')
        fits_file.parent.mkdir(parents=True, exist_ok=True)
        copy_from_path = alt_dir / str(fits_file.relative_to(data_dir))
#         print(f'\tCopy_from_path: {copy_from_path}') # debugging
        if copy_from_path.exists():
            print('\tCopying from alt dir')
            copy_result = subprocess.run([f'cp {copy_from_path.as_posix()} {fits_file}'], stdout=subprocess.PIPE, shell=True)
        else:
            print(f'\tDownloading')
#             # Myles Fix -
#             print(f'\tGet URL from plateifu: {plate_ifu}, {type(plate_ifu)}')
#             url_str, save_path = get_url(PLATEIFU=plate_ifu, key='LOGCUBE')
#             print(f'\tURL : {url_str}')
#             print(f'\tsave : {save_path}')
            
#             print('Fetch and Save')
#             fetch_and_save(url_str, fits_file)
#             download_file(url_str, save_path)
            
            print('\t\tDownloading Single Cube')
            dl_results = download_single_cube(str(fits_file), key="LOGCUBE")  # Issue but dont know what
            print(f'\tDownload results: {dl_results}')
    
    # read in data
    print(f'\n\t Start Bitmask')
    fits_file = str(fits_file)
    print(f'\tFITSFILE: {fits_file}, {type(fits_file)}')
    hdu_cube = fits.open(fits_file)
    print('83')

    #label_map = np.array(Image.open(label_file))[..., 0]
    # Declare the bitmask object to mask selected pixels
    bm = DAPCubeBitMask()
    try:
        wave = hdu_cube["WAVE"].data  # vaccuum wavelength; angstroms
    except KeyError:
        # try downloading again
        print('\tTrying to download LOGCUBE fits file again...')
        dl_results = download_single_cube(fits_file, key="LOGCUBE")
        hdu_cube = fits.open(fits_file)
        wave = hdu_cube["WAVE"].data  # vaccuum wavelength; angstroms

    # units: 1e-17 erg/s/cm2/ang/spaxel
    # dimensions: (x, y, 4563)
    # NOTE: fixed
    flux_cube = np.ma.MaskedArray(
        hdu_cube["FLUX"].data.transpose(1, 2, 0),
        mask=bm.flagged(
            hdu_cube["MASK"].data.transpose(1, 2, 0),
            ["IGNORED", "FLUXINVALID", "IVARINVALID", "ARTIFACT"],
        ),
    )
    nspec = np.product(flux_cube.shape[0:2])
    old_num_wave = flux_cube.shape[-1]
    new_shape = (*flux_cube.shape[0:2], new_num_wave)  # (x,y,2041)
    # resample flux onto consistent wavelength grid
    r = Resample(
        flux_cube.reshape(nspec, old_num_wave).filled(
            0.0
        ),  # (x*y, lam) # data values to resample
        x=wave / (1.0 + z),  # abscissa coordinates
        # starting and ending value for centers of first and last flux_cube
        newRange=new_wave_range,
        newpix=new_num_wave,  # number of wavelength bins
        # input is logarithmically binned (coords are geometric center of wavelength bins)
        inLog=True,
        newLog=True,  # output vector should be logarithmically binned
    )
    new_wave = r.outx
    new_flux_cube = r.outy.reshape(new_shape)

    # write to file
    hf = h5py.File(cube_file, "w")
    hf.create_dataset("cube", data=new_flux_cube)
    hf.create_dataset("wave", data=new_wave)
    #hf.create_dataset("labels", data=label_map)
    hf.close()
    if verbose:
        print(f"Saved to:\n\t{cube_file}")
    return

def process_map(fits_file, label_file, save_file=True, label_task='logOH', diagnostic="O3N2"):
    """Generates RGB image with GT labels
    calls "generate_BPT_labels" or "generate_logOH_labels"
    diagnostic is for logOH task, can be O3N2, N2, N2O2...
    """
    print('\n\t\tStarr proessing map')
    fits_file = fits_file.replace("LOGCUBE", "MAPS")
    print(f'\tFITS File Map: {fits_file}')

    if not pathlib.Path(fits_file).exists():
        print(f'No MAP fits file exists!\n\t{fits_file}')
        copy_from_path = alt_dir / str(pathlib.Path(fits_file).relative_to(data_dir))
        if copy_from_path.exists():
            print('\tCopying from alt dir')
            copy_result = subprocess.run([f'cp {copy_from_path.as_posix()} {fits_file}'], stdout=subprocess.PIPE, shell=True)
        else:
            print('\tDownloading')
            dl_results = download_single_cube(fits_file, key="MAPS")
    
    # Read in fits file from RAW directory
    try:
        hdu_maps = fits.open(fits_file)
    except OSError:
        print('Downloading MAPS fits file...')
        dl_results = download_single_cube(fits_file, key="MAPS")
        hdu_maps = fits.open(fits_file)

    if label_task == 'BPT':
        print('\tMaking BPT using generate_BPT_labels')
        label_spaxels = generate_BPT_labels(hdu_maps)
        label_spaxels = np.nan_to_num(label_spaxels, nan=0.0, posinf=0.0, neginf=0.0)
#         print(f'Label: {np.unique(label_spaxels, return_counts=True)}')
#         im = Image.fromarray(label_spaxels.filled()).convert("RGB")
#         print(im)
#         if save_file:
#             print(label_file)
#             im.save(label_file)
        # Myles Fix
        print(f'Label_filepath {label_file}, {type(label_file)}')
        try:
            np.save(label_file, label_spaxels)
            print(f"Array saved successfully to {output_patch_path}")
        except OSError as e:
            print(f"OS error occurred: {e}")
        except Exception as e:
            print(f"Error occurred: {e}")
    
#         print(f'Label Spaxel Shape numby {label_spaxels.shape}, {label_spaxels.mean}')
        return label_spaxels
    elif label_task == 'logOH':
        label_spaxels = generate_logOH_labels(hdu_maps, diagnostic='O3N2')
        label_spaxels = label_spaxels.round(decimals=3).filled(0.0)
        np.save(label_file, label_spaxels)
        return label_spaxels
    elif label_task == 'N2':
        label_spaxels = generate_logOH_labels(hdu_maps, diagnostic='N2')
        label_spaxels = label_spaxels.round(decimals=3).filled(0.0)
        np.save(label_file, label_spaxels)
        return label_spaxels
    else:
        raise ValueError("label_task must be in ['BPT', 'logOH', 'N2']")

# BPT labels ----------------------------------------------------------
def BPT_diagnostic(log_NII_Ha):
    # SF vs composite: Kauffman 03
    SF_lim = (0.61 / (log_NII_Ha - 0.05)) + 1.3
    # composite vs AGN: Kewley 06
    AGN_lim = (0.61 / (log_NII_Ha - 0.47)) + 1.19
    return SF_lim, AGN_lim

# def generate_BPT_labels(hdu_maps):
#     print('/nStart generate_BPT_labels')
#     emlc = channel_dictionary(hdu_maps, "EMLINE_GFLUX")
#     mask_ext = hdu_maps["EMLINE_GFLUX"].header["QUALDATA"]

#     halpha_flux = np.ma.MaskedArray(
#         hdu_maps["EMLINE_GFLUX"].data[emlc["Ha-6564"]],
#         mask=hdu_maps[mask_ext].data[emlc["Ha-6564"]] > 0,
#     )
#     hbeta_flux = np.ma.MaskedArray(
#         hdu_maps["EMLINE_GFLUX"].data[emlc["Hb-4862"]],
#         mask=hdu_maps[mask_ext].data[emlc["Hb-4862"]] > 0,
#     )
#     n2_flux = np.ma.MaskedArray(
#         hdu_maps["EMLINE_GFLUX"].data[emlc["NII-6585"]],
#         mask=hdu_maps[mask_ext].data[emlc["NII-6585"]] > 0,
#     )
#     o3_flux = np.ma.MaskedArray(
#         hdu_maps["EMLINE_GFLUX"].data[emlc["OIII-5008"]],
#         mask=hdu_maps[mask_ext].data[emlc["OIII-5008"]] > 0,
#     )
#     hdu_maps.close()

#     # Compute Ratios
#     log_OIII_Hb = np.log10(o3_flux / hbeta_flux)
#     log_NII_Ha = np.log10(n2_flux / halpha_flux)
#     output_mask = log_OIII_Hb.mask & log_NII_Ha.mask

#     # Calculate the estimated value for OIII/Hbeta map from NII/Halpha
#     SF_limit, AGN_limit = BPT_diagnostic(log_NII_Ha)

#     # label map
#     bpt_spaxels = np.zeros_like(log_OIII_Hb)
#     bpt_spaxels[log_OIII_Hb <= SF_limit] = 1
#     bpt_spaxels[(log_OIII_Hb > SF_limit) & (log_OIII_Hb <= AGN_limit)] = 2
#     bpt_spaxels[log_OIII_Hb > AGN_limit] = 3
#     bpt_spaxels = np.ma.MaskedArray(
#         bpt_spaxels, 
#         mask=output_mask, 
#         fill_value=0.0
#     )
    
    
#     # Plot
#     print('Plotting BPT')
#     print(f'label Map info \n\t{type(bpt_spaxels)} \n\t{bpt_spaxels.shape} \n\t{bpt_spaxels.max} \n\t{bpt_spaxels.min}')
# #     plt.imshow(output_mask, vmin=0, vmax=3)
# #     plt.show()
    
    
#     print('Finished Making BPT labels!')
#     return bpt_spaxels
def generate_BPT_labels(hdu_maps):
    print('\nStart generate_BPT_labels')

    # Retrieve channel dictionary for emission line fluxes
    emlc = channel_dictionary(hdu_maps, "EMLINE_GFLUX")
    mask_ext = hdu_maps["EMLINE_GFLUX"].header["QUALDATA"]

    # Function to create masked arrays for emission lines
    def create_masked_array(line_key):
        return np.ma.MaskedArray(
            hdu_maps["EMLINE_GFLUX"].data[emlc[line_key]],
            mask=hdu_maps[mask_ext].data[emlc[line_key]] > 0,
        )

    # Create masked arrays for each required emission line
    halpha_flux = create_masked_array("Ha-6564")
    hbeta_flux = create_masked_array("Hb-4862")
    n2_flux = create_masked_array("NII-6585")
    o3_flux = create_masked_array("OIII-5008")

    # Close HDU maps to free resources
    hdu_maps.close()

    # Convert masked arrays to regular ndarrays filled with NaN for masked values
    halpha_flux = halpha_flux.filled(np.nan)
    hbeta_flux = hbeta_flux.filled(np.nan)
    n2_flux = n2_flux.filled(np.nan)
    o3_flux = o3_flux.filled(np.nan)

    # Compute logarithmic ratios, handling divide by zero issues
    with np.errstate(divide='ignore', invalid='ignore'):
        log_OIII_Hb = np.log10(np.divide(o3_flux, hbeta_flux, where=hbeta_flux!=0))
        log_NII_Ha = np.log10(np.divide(n2_flux, halpha_flux, where=halpha_flux!=0))

    # Mask invalid entries and convert to regular ndarrays
    log_OIII_Hb = np.ma.masked_invalid(log_OIII_Hb).filled(np.nan)
    log_NII_Ha = np.ma.masked_invalid(log_NII_Ha).filled(np.nan)

    # Combine masks for output
    output_mask = np.isnan(log_OIII_Hb) & np.isnan(log_NII_Ha)

    # Calculate the estimated value for OIII/Hbeta map from NII/Halpha
    SF_limit, AGN_limit = BPT_diagnostic(log_NII_Ha)

    # Generate BPT spaxel labels
    bpt_spaxels = np.zeros_like(log_OIII_Hb)
    bpt_spaxels[log_OIII_Hb <= SF_limit] = 1
    bpt_spaxels[(log_OIII_Hb > SF_limit) & (log_OIII_Hb <= AGN_limit)] = 2
    bpt_spaxels[log_OIII_Hb > AGN_limit] = 3

    # Apply the mask to the BPT spaxels and set fill value to 0
    bpt_spaxels = np.where(output_mask, 0, bpt_spaxels)
    # Check for problem values
    
    # Remove problematic values else OSError: problem writing element 1024 to file may occur
    bpt_spaxels = np.nan_to_num(bpt_spaxels, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f'Has nan: {np.isnan(bpt_spaxels)} +inf: {np.isposinf(bpt_spaxels)} -inf: {np.isneginf(bpt_spaxels)}')
    
    return bpt_spaxels

# Metallicity labels ----------------------------------------------------------

def N2O2_diagnostic(NII_6583, OII_3727, OII_3729):
    # Kewley and Dopita - 2002 - Using Strong Lines to Estimate Abundances in Extra (KD02)
    # NII/OII diagnostic [N II] λ6584/[O II] λ3727,3729
    # Note: Independent of ionization parameter, Z>=8.6 for a reliable abundance
    N2O2 = np.log10(NII_6583 / (OII_3727 + OII_3729))
    OH = (
        np.log10(1.54020 + (1.26602 * N2O2) +
                 (0.167977 * N2O2**2)) + 8.93
    )  # KD02 (eq 5&7) [Z = log(O/H) +12]
    return OH

def O3N2_diagnostic(OIII_5008, NII_6583, Halpha, Hbeta):
    # Marino et al. 2013 O3N2 diagnostic
    # O3N2 diagnostic [OIII]λ5007/Hbeta * Halpha/[NII]λ6583
    O3N2 = np.log10((OIII_5008/Hbeta) * (Halpha/NII_6583))
    OH = O3N2
    #PP04:  8.73 - 0.32*O3N2 
    # 8.533 - (0.214 * O3N2)
    OH = 8.505 - (0.221 * O3N2) 
    return OH


def N2_diagnostic(NII_6583, Halpha):
    # Marino et al. 2013
    # N2 diagnostic log([NII]λ6583/Halpha)
    N2 = np.log10(NII_6583 / Halpha)
    # PP04
    OH = 8.73 + (0.462 * N2)
    # CALIFA HII regions
    #OH = 8.667 + (0.455 * N2)
    return OH


def generate_logOH_labels(hdu_maps, diagnostic="O3N2"):
    """Generates integer labels for logOH bins
    """
    emlc = channel_dictionary(hdu_maps, "EMLINE_GFLUX")
    mask_ext = hdu_maps["EMLINE_GFLUX"].header["QUALDATA"]

    halpha_flux = np.ma.MaskedArray(
        hdu_maps["EMLINE_GFLUX"].data[emlc["Ha-6564"]],
        mask=hdu_maps[mask_ext].data[emlc["Ha-6564"]] > 0,
    )
    hbeta_flux = np.ma.MaskedArray(
        hdu_maps["EMLINE_GFLUX"].data[emlc["Hb-4862"]],
        mask=hdu_maps[mask_ext].data[emlc["Hb-4862"]] > 0,
    )
    n2_flux = np.ma.MaskedArray(
        hdu_maps["EMLINE_GFLUX"].data[emlc["NII-6585"]],
        mask=hdu_maps[mask_ext].data[emlc["NII-6585"]] > 0,
    )
    o3_flux = np.ma.MaskedArray(
        hdu_maps["EMLINE_GFLUX"].data[emlc["OIII-5008"]],
        mask=hdu_maps[mask_ext].data[emlc["OIII-5008"]] > 0,
    )
    hdu_maps.close()

    # Metallicity Maps:
    if diagnostic == 'O3N2':
        OH_map = O3N2_diagnostic(o3_flux, n2_flux, 
                                 halpha_flux, hbeta_flux)
    elif diagnostic == 'N2':
        OH_map = N2_diagnostic(n2_flux, halpha_flux)
    return OH_map

