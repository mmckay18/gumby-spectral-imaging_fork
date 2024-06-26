import os, re
import pathlib
import torch
from torch import optim, nn
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from copy import deepcopy
from tqdm.auto import tqdm
from typing import Callable
from dataclasses import dataclass
import logging

@dataclass(eq=False)
class Acc(nn.Module):
    dim: int = -1
    def __post_init__(self):
        super().__init__()
    def forward(self, yhat, ytrue):
        _, preds = torch.max(yhat, dim=self.dim)
        return (1.0*(preds == ytrue)).mean()

def train(model: nn.Module, 
    train_dataloader, 
    eval_dataloader,
    optimizer:str = 'sgd', 
    decay:float=1e-4,
    momentum:float=0.9,
    lr:float=1e-3, 
    num_epochs: int = 1,
    patience:int=10,
    patience_factor: float = 0.1,
    min_lr: float = 1e-6,
    disable_progress_bar:bool=False,
    save_weights:bool=False,
    save_interval:int = None, 
    loss_function:nn.Module=nn.CrossEntropyLoss(),
    loss_type:str='classification',
    performance_metric: Callable = Acc(),
    device:torch.device or int=None,
    distributed:bool=False,
    rank:int=None,
    out_dir:str or pathlib.Path ='./output',
    eval_only: bool=False,
    use_amp=True,
    clip_gradients=False,
    clip_value=1.0,
    **kwargs):
    """The classical training loop

    BREAKING CHANGE: no longer "distributes" model/dataloaders if distributed. Needs
    to be passed an already distributed model/dataloaders.

    :param model: The object that will be trained
    :type model: torch
    :param train_dataloader: The data that the model will use for training
    :type train_dataloader: torch.utils.data.Dataset
    :param eval_dataloader: The data that the model will use for validation
    :type eval_dataloader: torch.utils.data.Dataset
    :param optimizer: The optimizer function, defaults to SGD
    :type optimizer: str, optional
    :param decay: Amount of weight decay. Defaults to zero.
    :type decay: float, optional
    :param momentum: momentum for SGD
    :type momentum: float
    :param lr: The learning rate, defaults to 1e-3
    :type lr: float, optional
    :param num_epochs: The number of epochs to loop over, defaults to 10
    :type num_epochs: int, optional
    :param patience: how long to wait before dropping lr
    :type patience: int
    :param disable_progress_bar: Indicates whether or not the user wants to
        disable to progress bar shown when the model is running, defaults to
        False
    :type disable_progress_bar: bool, optional
    :param save_weights: Tells the model to save the weights or not, defaults
        to False
    :type save_weights: bool, optional
    :param save_interval: Indicates how frequently the model saves a weights
        file, defaults to None
    :type save_interval: int, optional
    :param loss_function: The loss function, defaults to None
    :type loss_function: function, optional
    :param device: gpu to use
    :type device: torch.device
    :param distributed: whether to use torch.distributed
    :type distributed: bool
    :param rank: if distributed, which gpu to target
    :type rank: int 
    :return: None
    :rtype: None
    """
    if distributed and device:
        raise RuntimeError(f'args distributed and device are mut. excl.')
    if (not distributed) and (device == None):
        device = torch.cuda.current_device()
    # Only print/save output in one of the distributed processes
    log_process = (not distributed) or (distributed and rank==0)
    if log_process:
        logging.info(f'storing results in {out_dir}')
        if os.path.exists(out_dir) == False:
            pathlib.Path(out_dir).mkdir(parents=True)
        with open(os.path.join(out_dir, 'log.txt'), 'w') as f:
            # Low-tech way of printing a summary of training parameters 
            message = f'run of classical train with arguments:\n{locals()}\n'
            f.write(message)
            # message = f'train dataset summary: {train_dataloader.dataset.__str__()}\neval dataset summary: {eval_dataloader.dataset.__str__()}\n'
            # f.write(message)
            # message = f'model summary:\n{model.__str__()}\n'
            # f.write(message)
    # In the distributed case, set device = rank and ensure the model is on the
    # appropriate device 
    # actually, since we are assuming the model is already ddp-ed, don't push
    # the model?
    if distributed:
        device = rank
        # actually, since we are assuming the model is already ddp-ed, don't push
        # the model?
        # model.to(device)
    else:
        model.to(device)
        loss_function.to(device)
        if isinstance(performance_metric, nn.Module):
            performance_metric.to(device)
    # In both optimizers below, we only apply weight decay to the weights :-)
    if optimizer=='sgd':
        optimizer = optim.SGD([
            {'params': [x[1] for x in model.named_parameters() if re.search('weight', x[0])!= None], 'weight_decay': decay},
            {'params': [x[1] for x in model.named_parameters() if re.search('weight', x[0]) == None], 'weight_decay': 0.0}
            ], lr=lr, momentum=momentum)
    elif optimizer == 'adam':
        optimizer = optim.Adam([
            {'params': [x[1] for x in model.named_parameters() if re.search('weight', x[0])!= None], 'weight_decay': decay},
            {'params': [x[1] for x in model.named_parameters() if re.search('weight', x[0]) == None], 'weight_decay': 0.0}
            ], lr=lr, betas=(0.9, 0.999))
    # reduce LR when metric stops increasing (like an accuracy)
    lr_mode = 'max'
    # reduce LR when metric stops decreasing (like MSE)
    if loss_type == 'regression':
        lr_mode = 'min'

    # drop lr by factor of patience_factor if val acc hasn't improved (relatively) by 1% for last ``patience``
    # epochs. There could be a more optimal strategy, this was chosen arbitrarily
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        mode=lr_mode, 
        factor=patience_factor, 
        patience=patience, 
        threshold=0.01, 
        threshold_mode='rel', 
        min_lr=min_lr
    )
    # amp stuff
    scaler = GradScaler(enabled=use_amp)
    # Helper values for keeping track of best weights 
    best_metric = 0.0
    if loss_type == 'regression':
        best_metric = 10000.0
    best_wts = deepcopy(model.state_dict())
    # Keep track of learning curves 
    train_curve_loss = torch.zeros((num_epochs,), dtype=torch.float).to(device)
    train_curve_metric = torch.zeros((num_epochs,), dtype=torch.float).to(device)
    val_curve_metric = torch.zeros((num_epochs,), dtype=torch.float).to(device)
    val_curve_loss = torch.zeros((num_epochs,), dtype=torch.float).to(device)
    lr_curve = torch.zeros((num_epochs,), dtype=torch.float).to(device)
    # We need these to synchronize loss and accuracy between processes  
    if distributed:
        train_loss = torch.tensor(0.0).cuda(device)
        val_metric = torch.tensor(0.0).cuda(device)
        learn_rate = torch.tensor(0.0).cuda(device)    

    def save_stuff(out_dir, train_curve_metric, train_curve_loss, 
        val_curve_metric, val_curve_loss, 
        lr_curve, best_wts,
        epoch_index: int = None):
        # Make sure we only save from one distributed process (paranoia)
        if log_process:
            if epoch_index: # trim these arrays
                (train_curve_metric, train_curve_loss, 
                 val_curve_metric, val_curve_loss, 
                 lr_curve) = (x[:epoch_index+1] 
                                for x in (train_curve_metric, train_curve_loss, 
                                    val_curve_metric, val_curve_loss, 
                                    lr_curve)
                                )
            torch.save(train_curve_metric.cpu(), os.path.join(out_dir, 'train_curve_metric.pt'))
            torch.save(train_curve_loss.cpu(), os.path.join(out_dir, 'train_curve_loss.pt'))
            torch.save(val_curve_metric.cpu(), os.path.join(out_dir, 'val_curve_metric.pt'))
            torch.save(val_curve_loss.cpu(), os.path.join(out_dir, 'val_curve_loss.pt'))
            torch.save(lr_curve.cpu(), os.path.join(out_dir, 'lr_curve.pt'))
            torch.save(best_wts, os.path.join(out_dir, 'best_wts.pt'))
            final_wts = deepcopy(model.state_dict())
            torch.save(final_wts, os.path.join(out_dir, 'final_wts.pt'))
            msg = f'saved learning curves and weights to {out_dir}'
            logging.info(msg)

    # Outer epoch loop
    epoch_desc = f'Epoch'
    epoch_loop = range(num_epochs)
    # Only display progress from one distributed process 
    if (not disable_progress_bar) and log_process:
        epoch_loop = tqdm(epoch_loop, total=num_epochs, 
            desc=epoch_desc, disable=disable_progress_bar)
    for epoch_index in epoch_loop:
        spacer = ' ' * (5 - len(str(epoch_index)))
        train_desc = f'Training model - Iteration: ' \
                     f'{epoch_index}' + spacer
        eval_desc = f'Evaluating model - Iteration: ' \
                    f'{epoch_index}' + spacer
        if not eval_only:
            # This is the start of the training loop
            train_gen = train_dataloader
            if (not disable_progress_bar) and log_process:
                train_gen = tqdm(
                        train_gen,
                        # total=len(train_dataloader),
                        desc=train_desc,
                        disable=disable_progress_bar
                    )
            model.train()
            # In the distributed case, sync up samplers 
            if distributed and isinstance(train_dataloader, DataLoader):
                train_dataloader.sampler.set_epoch(epoch_index)
            for sample, labels in train_gen:
                optimizer.zero_grad()
                if type(sample) is dict:
                    sample = {k: v.to(device) for k, v in sample.items()}
                else:
                    sample = sample.to(device)
                labels = labels.to(device)
                with autocast(enabled=use_amp):
                    output = model(sample)
                    # temporary fix to work with torch segmentation models
                    if type(output) is not torch.Tensor:
                        output = output['out']
                    if loss_type=='regression':
                        output = output.squeeze(1)
                    loss = loss_function(output, labels)
                    train_metric = performance_metric(output, labels)
                # amp backprop
                scaler.scale(loss).backward()
                if clip_gradients:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                scaler.step(optimizer)
                scaler.update()
                train_curve_loss[epoch_index] += loss.item()
                train_curve_metric[epoch_index] += train_metric.item()
                if (not disable_progress_bar) and log_process:
                    train_gen.set_description(f'{train_desc}, loss: {loss.item():.4g}, metric: {train_metric.item():.4g}')
            train_curve_loss[epoch_index] /= len(train_dataloader)
            train_curve_metric[epoch_index] /= len(train_dataloader)
        
        # Perform the evaluation step of the training loop
        eval_gen = eval_dataloader
        if (not disable_progress_bar) and log_process:
            eval_gen = tqdm(
                    eval_gen,
                    # total=len(eval_dataloader),
                    desc=eval_desc,
                    disable=disable_progress_bar
                )
        model.eval()
        # In the distributed case, sync up samplers 
        if distributed and isinstance(eval_dataloader, DataLoader):
            eval_dataloader.sampler.set_epoch(epoch_index)
        for sample, labels in eval_gen:
            with torch.no_grad():
                if type(sample) is dict:
                    sample = {k: v.to(device) for k, v in sample.items()}
                else:
                    sample = sample.to(device)
                labels = labels.to(device)
                with autocast():
                    output = model(sample)
                    if type(output) is not torch.Tensor:
                        output = output['out']   
                    if loss_type=='regression':
                        output = output.squeeze(1)             
                    loss = loss_function(output, labels)
                    eval_metric = performance_metric(output, labels)
            val_curve_loss[epoch_index] += loss.item()
            val_curve_metric[epoch_index] += eval_metric
            if (not disable_progress_bar) and log_process:
                eval_gen.set_description(f'{eval_desc}, loss: {loss.item():.4g}, metric: {eval_metric.item():.4g}')
        val_curve_loss[epoch_index] /= len(eval_dataloader)
        val_curve_metric[epoch_index] /= len(eval_dataloader)

        lr_curve[epoch_index] = torch.tensor([p['lr'] for p in optimizer.param_groups]).mean()
        # Most complicated piece of distributed case: we need to synchronize
        # loss and accuracy across processes, to obtain the correct loss and
        # validation curves. 
        # NOTE: throwing in a barrier here. Probably hurts efficiency, but some
        # of these puppies are timing out and crashing distributed training.
        if distributed:
            dist.barrier() # make sure processes are synced before these collective ops.
            train_loss = train_curve_loss[epoch_index]
            dist.all_reduce(train_loss.cuda(device), op = dist.ReduceOp.SUM)
            train_curve_loss[epoch_index] = train_loss/torch.cuda.device_count()
            val_metric = val_curve_metric[epoch_index]
            dist.all_reduce(val_metric.cuda(device), op=dist.ReduceOp.SUM)
            val_curve_metric[epoch_index] = val_metric/torch.cuda.device_count()
            lr_curve[epoch_index] = torch.tensor([p['lr'] for p in optimizer.param_groups]).mean()
            learn_rate = lr_curve[epoch_index]
            dist.all_reduce(learn_rate.cuda(device), op=dist.ReduceOp.SUM)
            lr_curve[epoch_index] = learn_rate/torch.cuda.device_count()
        # Check if current validation accuracy is the best so far (if
        # distributed, only need to do this in the logging process)
        if log_process:
            # accuracy INCREASES for normal tasks
            improvement_conditional = val_curve_metric[epoch_index] > best_metric
            # for regression: better model has SMALLER val acc
            if loss_type == 'regression':
                improvement_conditional = val_curve_metric[epoch_index] < best_metric
            
            if improvement_conditional:
                # If so, update best accuracy and best weights 
                best_metric = val_curve_metric[epoch_index]
                best_wts = deepcopy(model.state_dict())
                better_val_loss = True
            else:
                better_val_loss = False
        # Test reduce on plateau criterion, step the lr scheduler. I *think*
        # this should be done in all processes (based on examples at
        # https://github.com/pytorch/examples/blob/main/imagenet/main.py, but there
        # is literally no doc on the topic)
        scheduler.step(val_curve_metric[epoch_index])
        # Save weights at specified interval and if we just hit a new best
        if (save_weights and log_process):
            if (epoch_index % save_interval == 0) or better_val_loss:
                save_stuff(out_dir, train_curve_metric, train_curve_loss, val_curve_metric, val_curve_loss, lr_curve, best_wts)
        # print info
        if log_process:
            msg =  f'{epoch_desc}: {epoch_index}, train loss: {train_curve_loss[epoch_index]:.4g}, ' \
                + f'val metric: {val_curve_metric[epoch_index]:.4g}, better? {better_val_loss:.4g}, lr: {lr_curve[epoch_index]:.4g}'
            if not disable_progress_bar:
                epoch_loop.set_description(msg)
            else:
                logging.info(msg)
        
        # for some reason this isn't triggering...
        lr_condition = torch.allclose(
            lr_curve[epoch_index].to(device), 
            torch.tensor([l for l in scheduler.min_lrs]).mean().to(device))
        if lr_condition:
            if log_process:
                save_stuff(out_dir, train_curve_metric, train_curve_loss,  
                    val_curve_metric, val_curve_loss, lr_curve, best_wts)
                err_msg = f'hit minimum lr of {torch.tensor([l for l in scheduler.min_lrs]).mean():.4g}'
                err_msg = err_msg + f'best val metric: {val_curve_metric.max().item():.4g}'
                logging.info(err_msg)
            break  

    # ensure we save after loop ends ...
    if (save_weights and log_process):
        save_stuff(out_dir, train_curve_metric, train_curve_loss, val_curve_metric, val_curve_loss, lr_curve, best_wts)
    # Load up the best weights, in case this is being used interactively 
    # only makes sense if not distributed (??)
    if not distributed:
        model.load_state_dict(best_wts)
    if eval_only:
        return val_curve_metric
    else:
        return None
