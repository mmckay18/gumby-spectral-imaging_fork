'''Needs astropy environment for fits tables
'''
import random
import numpy as np
import pandas as pd
import pathlib
from astropy.table import Table
import argparse
from fits_utils import BPT_diagnostic

data_dir = pathlib.Path('/qfs/projects/gumby/data/manga')

manga_catalogue_path = data_dir / 'raw/SDSS17Pipe3D_v3_1_1.fits'
dat = Table.read(manga_catalogue_path, format='fits', hdu=1)
cat_df = dat.to_pandas()
cat_df['PLATEIFU'] = cat_df['plateifu'].apply(lambda x: x.decode())
cat_df['IFUSIZE'] = cat_df['PLATEIFU'].map(lambda x: x.split('-')[-1][:-2])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ------- data parameters ----------------
    parser.add_argument('--split_dir', '-s', type=str, default='OH_1',
                        help='where CSVs live for train/test/val')
    parser.add_argument('--label_task', '-lt', type=str, default='logOH',
                        help='labels are BPT or logOH')
    parser.add_argument('--patch_norm', '-pn', type=str, default='global',
                        help='type of patches: spectral, spatial, global, PCA')
    parser.add_argument('--normalize', action='store_true', default=False)
    # ------- model parameters ----------------
    args = parser.parse_args()
    
    # select centrally SF galaxies
    OIII_key = f'log_OIII_Hb_cen'
    NII_key = f'log_NII_Ha_cen'
    tmp_SF, tmp_AGN = BPT_diagnostic(cat_df[NII_key])
    cat_df.loc[cat_df[OIII_key] < tmp_SF,'BPT_flag'] = 1
    cat_df.loc[(cat_df[OIII_key] >= tmp_SF) & (cat_df['log_OIII_Hb_cen'] < tmp_AGN),'BPT_flag'] = 2
    cat_df.loc[(cat_df[OIII_key] >= tmp_AGN),'BPT_flag'] = 3
    cat_df.loc[(cat_df[OIII_key] > 1.5),'BPT_flag'] = np.nan
    cat_df.loc[(cat_df[OIII_key] < -1.5),'BPT_flag'] = np.nan
    cat_df.loc[(cat_df[NII_key] > 1.0),'BPT_flag'] = np.nan
    cat_df.loc[(cat_df[NII_key] < -1.5),'BPT_flag'] = np.nan

    # using galaxies that are within 1st and 99th percentile in each of these properties
    cols_of_interest = ['EW_Ha_cen']
    condition_list = [((cat_df[key] > cat_df[key].quantile(0.05)) & (cat_df[key] < cat_df[key].quantile(0.95)))for key in cols_of_interest]

    # make cuts on BPT, SFR, Av, stellar mass, central metallicity, IFU size, inclination
    condition_list += [(cat_df['BPT_flag'] == 1)]
    condition_list += [(cat_df['log_SFR_Ha'] > 0.0)]
    condition_list += [(cat_df['log_Mass'] > 10.25)]
    condition_list += [(cat_df['log_Mass'] <= 11.25)]
    condition_list += [(cat_df['OH_Pet04_O3N2_Re_fit'] <= 8.8)]
    condition_list += [(cat_df['Av_gas_Re'] < 2.0)]
    condition_list += [(cat_df['IFUSIZE'] == '127') | (cat_df['IFUSIZE'] == '91') | (cat_df['IFUSIZE'] == '61')]
    condition_list += [(cat_df['nsa_inclination'] <= 30.0)]
    inds = np.logical_and.reduce(condition_list)

    # make dataframe of galaxies
    df = cat_df[inds].copy()
    df.reset_index(inplace=True)
    print(len(cat_df), len(df))

    # create metallicity and mass bins to sample from
    cols_to_bin = ['log_NII_Ha_cen']#,'log_OIII_Hb_cen']
    for this_key in cols_to_bin:
        q = 3
        labels = [f'{this_key}_{i}' for i in range(q)]
        df[f'{this_key}_bins'] = pd.qcut(df[this_key], q=q, labels=labels)

    cols_to_keep = cols_of_interest + ['PLATEIFU','nsa_redshift']

    # want a similar number of IFU sizes in each split
    group_by_cols = [f'{this_key}_bins' for this_key in cols_to_bin] + ['IFUSIZE']
    grouper = df.groupby(by=group_by_cols)
    list_of_dicts = []
    min_gals = 8
    max_gals = 50
    for group, grouped_df in grouper:

        ngals = len(grouped_df)
        if ngals > max_gals:
            tmp_df = grouped_df.sample(n=max_gals)
            print(f'Sampling N={max_gals} from group')
        elif ngals < min_gals:
            print(f'Group too small ({ngals})' )
            continue
        else:
            print(f'Splitting {ngals} into train/val/test')
            tmp_df = grouped_df
        
        ginds = tmp_df.index.tolist()
        random.shuffle(ginds)
        n_train = int(ngals*0.8)
        n_val = max(1,int(ngals*0.1))

        train_inds = ginds[0:n_train]
        val_inds = ginds[n_train:n_train+n_val]
        test_inds = ginds[n_train+n_val:n_train+n_val+n_val]
        #for i in range(len(tmp_df)):
        for i in ginds:
            if i in train_inds:
                this_dict = {'split':'train'}
                this_dict.update(df.iloc[i][cols_to_keep].to_dict())
                list_of_dicts.append(this_dict)
            #elif (i >= n_train) & (i < (n_train + n_val)):
            elif i in val_inds:
                this_dict = {'split':'val'}
                this_dict.update(df.iloc[i][cols_to_keep].to_dict())
                list_of_dicts.append(this_dict)
            elif i in test_inds:
                this_dict = {'split':'test'}
                this_dict.update(df.iloc[i][cols_to_keep].to_dict())
                list_of_dicts.append(this_dict)

    new_df = pd.DataFrame(list_of_dicts)
    print(f"N={len(new_df[new_df.split == 'test'])}  / {len(new_df[new_df.split == 'val'])} / {len(new_df[new_df.split == 'train'])}", f"({len(new_df[new_df.split == 'val'])*1.0/len(new_df):.3})")    
    
    split_list = ['train', 'val', 'test']
    new_df['fits_file'] = new_df['PLATEIFU'].map(lambda x: data_dir / 'raw' / x.split('-')[0] / x.split('-')[1] / f'manga-{x}-LOGCUBE-SPX-MILESHC-MASTARSSP.fits.gz')
    csv_dir = data_dir / 'splits' / args.split_dir
    for split in split_list:
        inds = new_df['split'] == split
        print(f'{split.title()}: {len(new_df[inds])}')
        output_csv = csv_dir / f'{split}_fits.csv'
        new_df[inds]['fits_file'].to_csv(output_csv, index=False, header=False)