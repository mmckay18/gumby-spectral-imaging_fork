"""
ENV: regular torch/gumby environment

# doublerainbow (16GB) - batch can be 2048
# immortanjoe (48GB) - batch can be (atleast) 8192 (4096 without amp)

example call:
    python run_SSFTT_training.py --out=OH_1_base --split_dir=OH_1 -e=200 --gpu=6 -b=1024 --num_workers=4
    python run_SSFTT_training.py --out=OH_2_easy --split_dir=OH_2  --OH_key=extra -e=100 --gpu=4 -b=1024 --num_workers=4 --easy_splits
"""
import argparse
import torch
import os
import pathlib

from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.sampler import RandomSampler
from torchsampler import ImbalancedDatasetSampler

from training import train

from HSI_SSFTT import get_SSFTT_model
from data_fns import (
    HSIPatchDataset,
    BinLogOHLabels,
    repatchSSFTT,
    ssftt_data,
    default_collate_fn
)
from utils import get_num_classes

data_dir = pathlib.Path('/qfs/projects/gumby/data/manga')
results_dir = pathlib.Path('/qfs/projects/gumby/results/weights/manga/')

random_seed = 2147483647

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ------- data parameters ----------------
    parser.add_argument('--split_dir', '-s', type=str, default='OH_1',
                        help='subdirectory: OH_1, OH_2...')
    parser.add_argument('--label_task', '-lt', type=str, default='logOH',
                        help='labels are BPT or logOH')
    parser.add_argument('--OH_key', '-OH', type=str, default='default',
                        help='default, extra, ...')
    parser.add_argument('--patch_norm', '-pt', type=str, default='global',
                        help='type of patches: spectral, spatial, global, PCA')
    parser.add_argument('--normalize', action='store_true', default=False)
    # ------- model parameters ----------------
    parser.add_argument('--num_classes', '-nc', type=int, default=6,
                        help='Number of model classes')
    parser.add_argument('--min_wave_ind', type=int, default=0,
                        help='minimum index in wavelength array')
    parser.add_argument('--max_wave_ind', type=int, default=2040,
                        help='maximum index in wavelength array')
    parser.add_argument('--default_patch_size', default=9, type=int,
                        help='7x7 pixel patch')
    parser.add_argument('--spatial_patch_size', default=7, type=int,
                        help='7x7 pixel patch')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout on spectral transformer')
    parser.add_argument('--emb_dropout', default=0.1, type=float,
                        help='dropout for embedding tokenization')
    # ------- training parameters ----------------
    parser.add_argument('--save_interval', '-sv', type=int, default=10,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='/output/',
                        help='Directory path to save')
    parser.add_argument('--sampler_type', default='normal',
                        help='normal or imbalanced')
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
    parser.add_argument('--easy_splits', action='store_true', default=False)
    parser.add_argument('--distributed', action='store_true', default=False)
    args = parser.parse_args()

    # ---- distributed training ----------
    rank: int = None
    ws: int = None
    if args.distributed:
        rank = int(os.environ["LOCAL_RANK"])
        ws = int(os.environ['WORLD_SIZE'])
        print(f'rank {rank} of worldsize {ws}')
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl")
    log_process = (not args.distributed or rank == 0)

    # ----- data -------------------------
    num_classes = get_num_classes(args.OH_key)

    if args.patch_norm == 'PCA':
        args.max_wave_ind = 30
    # any modifiers to the existing patches
    if (args.spatial_patch_size != args.default_patch_size) | ((args.min_wave_ind, args.max_wave_ind) != (0, 2040)):
        preprocess_fn = repatchSSFTT(
            current_patch_size=args.default_patch_size,
            new_patch_size=args.spatial_patch_size,
            wave_inds=(args.min_wave_ind, args.max_wave_ind),
            normalize=args.normalize,
        )
    else:
        preprocess_fn = ssftt_data(normalize=args.normalize)

    split_str = '_easy' if args.easy_splits else ''
    # instantiate datasets from patch CSV
    csv_dir = data_dir / 'splits' / args.split_dir
    train_data = HSIPatchDataset(
        csv_dir / f'train-{args.patch_norm}{split_str}.csv',
        preprocess_fn=preprocess_fn,
        label_fn=BinLogOHLabels(OH_key=args.OH_key),
    )
    val_data = HSIPatchDataset(
        csv_dir / f'val-{args.patch_norm}{split_str}.csv',
        preprocess_fn=preprocess_fn,
        label_fn=BinLogOHLabels(OH_key=args.OH_key),
    )
    # select sampler
    if args.sampler_type == 'imbalanced':
        train_sampler = ImbalancedDatasetSampler(train_data)
    else:
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
            collate_fn=default_collate_fn,
            num_workers=args.num_workers,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=default_collate_fn,
            num_workers=args.num_workers,
        )

    # ----- model -------------------------
    model = get_SSFTT_model(
        num_classes=num_classes,
        spatial_patch_size=args.spatial_patch_size,
        min_wave_ind=args.min_wave_ind,
        max_wave_ind=args.max_wave_ind,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout
    )
    if args.distributed:
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        device = None
    else:
        device = torch.device(f'cuda:{args.gpu}')
        model = model.to(device)
    # ----- train -------------------------
    if args.distributed:
        train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer=args.optimizer,
            num_epochs=args.epoch,
            save_interval=args.save_interval,
            save_weights=True,
            device=None,
            distributed=args.distributed,
            rank=rank,
            lr=args.lr*(args.batch_size/8),
            min_lr=1e-6,
            decay=1e-5,
            out_dir= results_dir / args.out,
            patience=(25),
            use_amp=False,
        )
        dist.destroy_process_group()
    else:
        train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer=args.optimizer,
            num_epochs=args.epoch,
            save_interval=args.save_interval,
            save_weights=True,
            device=device,
            lr=args.lr, #1e-4*(args.batch_size/16),
            min_lr=1e-6,
            decay=1e-5,
            out_dir=results_dir / args.out,
            patience=args.patience,#(args.epoch/4),
            use_amp=False,
        )
