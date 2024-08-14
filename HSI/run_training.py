"""
ENV: hsi.yml

example call:
    python run_training.py --out=OH_1_base --split_dir=OH_1 -e=200 --gpu=6 -b=1024 --num_workers=4
    python run_training.py --out=OH_2_easy --split_dir=OH_2  --OH_key=extra -e=100 --gpu=4 -b=1024 --num_workers=4 --easy_splits
"""
import os
import sys
import argparse
import pathlib
import logging
import time

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.sampler import RandomSampler

from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError

# local directory imports
from training import train, Acc
from utils import get_num_classes
from aug_utils import augmentation_fn_dict

from HSI_SIMPLENET import SimpleNet,SpectraNet
from HSI_SSFTT import get_SSFTT_model

from regression import normalizeOxygenLabels, reg_collate_fn
from data_fns import (
    HSIPatchDataset,
    BinLogOHLabels,
    repatchData,
    preprocData,
    default_collate_fn
)

import warnings
warnings.filterwarnings('ignore', 'invalid value encountered')

data_dir = pathlib.Path('/gscratch/scrubbed/mmckay18/DATA')
results_dir = pathlib.Path('/gscratch/scrubbed/mmckay18/DATA/weights/')
random_seed = 2147483647

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ------- data parameters ----------------
    parser.add_argument('--split_dir', '-s', type=str, default='OH_1',
                        help='subdirectory: OH_1 (small), OH_2 (big)')
    parser.add_argument('--label_task', '-lt', type=str, default='logOH',
                        help='labels are BPT or logOH')
    parser.add_argument('--OH_key', '-OH', type=str, default='extra',
                        help='default=6 classes, extra=8 classes')
    parser.add_argument('--patch_norm', '-pt', type=str, default='global',
                        help='type of patches: spectral, spatial, global, PCA')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='normalize to dataset-wide values')
    parser.add_argument('--easy_splits', action='store_true', default=False,
                        help='use data-science splits (train/val include different patches from same images)')
    parser.add_argument('--use_local_data', action='store_true', default=False)
    parser.add_argument('--local_dir', '-ld', type=str, default='/raid/byle431',
                        help='local location of data')
    # ------- model parameters ----------------
    parser.add_argument('--arch_type', '-mn', type=str, default='ssftt',
                        help='simplenet, spectranet (1d-simplenet), ssftt')
    parser.add_argument('--min_wave_ind', type=int, default=0,
                        help='minimum index in wavelength array')
    parser.add_argument('--max_wave_ind', type=int, default=2040,
                        help='maximum index in wavelength array')
    parser.add_argument('--default_patch_size', default=9, type=int,
                        help='7x7 pixel patch')
    parser.add_argument('--spatial_patch_size', default=7, type=int,
                        help='7x7 pixel patch')
    # properties specific to ssftt
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout on spectral transformer')
    parser.add_argument('--emb_dropout', default=0.1, type=float,
                        help='dropout for embedding tokenization')
    # args for regression models
    parser.add_argument('--regression', action='store_true', default=False)
    parser.add_argument('--regression_norm', type=str, default='scaled_max',
                        help='none, mean, minmax, scaledmax, strict, clip')
    parser.add_argument('--regression_loss', type=str, default='huber',
                        help='huber, MSE')
    # ------- data augmentation ----------------
    parser.add_argument('--aug_fn', '-af', type=str, default='cosine',
                        help='cosine, awg, peak, trsltup, trsltdown, compress, stretch, smooth, all, cossmo, strcom, peatup')
    parser.add_argument('--aug_data', action='store_true', default=False)
    # ------- training parameters ----------------
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='output',
                        help='Unique string for output: results_dir / arch_type / {split_dir} / {out}')
    parser.add_argument('--optimizer', '-opt', type=str, default='adam',
                        help='adam, sgd')
    parser.add_argument('--batch_size', '-b', type=int, default=1024,
                        help='Number of images in each mini-batch')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', type=float, default=1.e-3,
                        help='Initial learning rate')
    parser.add_argument('--patience', '-p', type=int, default=50,
                        help='number of epochs to wait before changing lr')
    parser.add_argument('--save_interval', '-sv', type=int, default=2,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--sampler_type', default='normal',
                        help='normal or imbalanced')
    parser.add_argument('--distributed', action='store_true', default=False)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    start_time = time.time()
    
    # ---- distributed training ----------
    rank: int = None
    ws: int = None
    if args.distributed:
        rank = int(os.environ["LOCAL_RANK"])
        ws = int(os.environ['WORLD_SIZE'])
        logging.info(f'rank {rank} of worldsize {ws}')
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl")
    log_process = (not args.distributed or rank == 0)

    # ----- data -------------------------
    num_classes = get_num_classes(args.OH_key)
    split_str = 'easy' if args.easy_splits else 'base'
    out_dir = results_dir / args.arch_type / args.split_dir / split_str / args.out
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Model Output: {out_dir}')

    if args.arch_type == 'spectranet':
        args.spatial_patch_size = 1

    # any modifiers to the existing patches
    if (args.spatial_patch_size != args.default_patch_size) | ((args.min_wave_ind, args.max_wave_ind) != (0, 2040)):
        preprocess_fn = repatchData(
            current_patch_size=args.default_patch_size,
            new_patch_size=args.spatial_patch_size,
            wave_inds=(args.min_wave_ind, args.max_wave_ind),
            normalize=args.normalize,
            add_extra_dim=args.arch_type == 'ssftt',
        )
    else:
        preprocess_fn = preprocData(
            normalize=args.normalize,
            add_extra_dim=args.arch_type == 'ssftt',
        )
    
    # augmentation
    if args.aug_data:
        logging.info(f'Augmentation: {args.aug_fn}')
        augment_fn = augmentation_fn_dict[args.aug_fn]
    else:
        logging.info(f'No data augmentation.')
        augment_fn = None
    
    # regression
    if args.regression:
        logging.info(f'REGRESSION problem. Setting up loss, metric, and collate functions.')
        num_classes=1
        label_fn = normalizeOxygenLabels(normalization=args.regression_norm)
        loss_type='regression'
        if args.regression_loss == 'huber':
            loss_fn = torch.nn.HuberLoss()
        elif args.regression_loss == 'mse':
            loss_fn = torch.nn.MSELoss()
        metric_function = MeanAbsolutePercentageError()
        collate_fn = reg_collate_fn
    else:
        logging.info(f'CLASSIFICATION problem. Setting up loss, metric, and collate functions.')
        label_fn = BinLogOHLabels(OH_key=args.OH_key)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_type='classification'
        collate_fn = default_collate_fn
        metric_function = Acc()
    
    # ----- data ----------------------
    split_str = '_easy' if args.easy_splits else ''
    # instantiate datasets from patch CSV
    csv_dir = data_dir / 'splits' / args.split_dir
    train_data = HSIPatchDataset(
        csv_dir / f'train-{args.patch_norm}{split_str}.csv',
        preprocess_fn=preprocess_fn,
        label_fn=label_fn,
        augment_data=args.aug_data,
        augment_fn=augment_fn,
        use_local_data=args.use_local_data,
        local_dir=args.local_dir,
    )
    val_data = HSIPatchDataset(
        csv_dir / f'val-{args.patch_norm}{split_str}.csv',
        preprocess_fn=preprocess_fn,
        label_fn=label_fn,
        use_local_data=args.use_local_data,
        local_dir=args.local_dir,
    )
    # select sampler
    if args.sampler_type == 'imbalanced':
        from torchsampler import ImbalancedDatasetSampler
        logging.info(f'Using imbalanced sampler.')
        train_sampler = ImbalancedDatasetSampler(train_data)
    else:
        logging.info(f'Using random sampler.')
        train_sampler = RandomSampler(train_data)

    if args.distributed:
        datasets = [train_data, val_data]
        samplers = [
            torch.utils.data.DistributedSampler(
                dataset,
                rank=rank,
                shuffle=True,
                drop_last=False
            ) for dataset in datasets]
        train_dataloader, val_dataloader = [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=sampler
            )
            for dataset, sampler in zip(datasets, samplers)
        ]
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=train_sampler,
            generator=torch.Generator().manual_seed(random_seed),
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )

    # ----- model -------------------------
    if args.arch_type == 'ssftt':
        logging.info('Using SSFTT')
        model = get_SSFTT_model(
            num_classes=num_classes,
            spatial_patch_size=args.spatial_patch_size,
            min_wave_ind=args.min_wave_ind,
            max_wave_ind=args.max_wave_ind,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout
        )
    elif args.arch_type == 'simplenet':
        logging.info(f'Using SimpleNet')
        model = SimpleNet(
            num_classes=num_classes,
            in_chans=args.max_wave_ind - args.min_wave_ind,
        )
    elif args.arch_type == 'spectranet':
        logging.info(f'Using SpectraNet')
        model = SpectraNet(
            num_classes=num_classes
        )
    
    if args.distributed:
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        device = None
        args.lr *= args.batch_size/8
    else:
        device = torch.device(f'cuda:{args.gpu}')
        model = model.to(device)
    
    # ----- train -------------------------
    train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer=args.optimizer,
        num_epochs=args.epoch,
        save_interval=args.save_interval,
        save_weights=True,
        device=device,
        distributed=args.distributed,
        rank=rank,
        lr=args.lr,
        min_lr=1e-6,
        decay=1e-5,
        loss_function=loss_fn,
        loss_type=loss_type,
        performance_metric=metric_function,
        out_dir=out_dir,
        patience=args.patience,
        use_amp=False,
    )
    if args.distributed:
        dist.destroy_process_group()

    end_time = time.time()
    run_time = end_time - start_time
    logging.info(f'run time: {run_time}')
    with open('runtimes.txt','a') as f:
        f.write(f'{out_dir.name},{run_time}')
