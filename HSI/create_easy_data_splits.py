import pandas as pd
import pathlib
import argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

data_dir = pathlib.Path('/qfs/projects/gumby/data/manga')
def bin_OH_labels(input, OH_bins = [8.40, 8.44, 8.48, 8.52, 8.56]):
    output = np.digitize(input, bins=OH_bins)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ------- data parameters ----------------
    parser.add_argument('--split_dir', '-s', type=str, default='OH_1',
                        help='where CSVs live for train/test/val')
    parser.add_argument('--patch_norm', '-pn', type=str, default='global',
                        help='type of patches: spectral, spatial, global, PCA')
    parser.add_argument('--label_task', '-lt', type=str, default='logOH',
                        help='logOH or BPT')
    # ------- model parameters ----------------
    args = parser.parse_args()

    csv_dir = data_dir / 'splits' / args.split_dir
    split_list = ['train', 'val', 'test']
    filepaths = [csv_dir / f'{split}-{args.patch_norm}.csv' for split in split_list]

    df = pd.concat((pd.read_csv(f) for f in filepaths), ignore_index=True)
    
    if args.label_task == 'logOH':
        df['bin_label'] = bin_OH_labels(df['label'])
    else:
        df['bin_label'] = df['label'].copy()

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, dub_index in split.split(df, df.bin_label):
        train_df = df.iloc[train_index]
        train_df.reset_index(inplace=True, drop=True)
        train_df[['data','label']].to_csv(csv_dir / f'train-{args.patch_norm}_easy.csv', header=True, index=False)

        dub_df = df.iloc[dub_index]
        dub_df.reset_index(inplace=True, drop=True)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        for val_index, test_index in split.split(dub_df, dub_df.bin_label):
            val_df = dub_df.iloc[val_index]
            val_df.reset_index(inplace=True, drop=True)
            val_df[['data','label']].to_csv(csv_dir / f'val-{args.patch_norm}_easy.csv', header=True, index=False)
            
            test_df = dub_df.iloc[test_index]
            test_df.reset_index(inplace=True, drop=True)
            test_df[['data','label']].to_csv(csv_dir / f'test-{args.patch_norm}_easy.csv', header=True, index=False)
