from argparse import Namespace
from typing import Dict, Tuple

import torch
import wandb
from torch import nn, Tensor
from torch.cuda import device
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import dod_utils as cddod
from pytorch_references.detection.utils import MetricLogger, SmoothedValue


def train_one_epoch(
        model: nn.Module,
        optimizer: Optimizer,
        device: device,
        epoch: int,
        dl_source: DataLoader,
        dl_target: DataLoader = None,
        scaler: GradScaler = None,
        wandb_logger: wandb = None,
        args: Namespace = None,
) -> Tuple[Dict[str, Tensor], int, MetricLogger]:
    """
    References: https://github.com/pytorch/vision/blob/main/references/detection/engine.py
    https://github.com/VisionLearningGroup/DA_Detection/blob/master/trainval_net_global_local.py
    Trains one epoch
    Args:
        model (nn.Module): model used for training - fasterRCNN with FPN/FPA/RA
        optimizer (nn.Module):
        device: (device or int): 'cpu' or 'cuda'
        epoch (int): current epoch number
        dl_source (Dataloader): dataloader for iterating the source dataset
        dl_target (Dataloader): dataloader for iterating the target dataset (default=None)
        scaler (GradScaler): for gradient scaling when using automatic mixed precision training
        wandb_logger (wandb): weight and bias logger for logging
        args (Namespace): arguments passed

    Returns:
        loss_reduced_dict (dict): combined losses from different modules
        local_iter_count (int): number of iterations in the current epoch
        metric_logger (MetricLogger): logger with logging values

    """
    use_fpa = args.fpa
    use_ra = args.ra
    iters_per_epoch = len(dl_source)
    print_freq = args.print_freq
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    loss_dict_source = {}
    target = False

    dl_iter_source = iter(dl_source)
    dl_iter_target = None

    if use_fpa:
        target = True
        dl_iter_target = iter(dl_target)

    lr_scheduler = cddod.warmup_training(epoch, optimizer=optimizer, data_loader=dl_source)
    local_iter_count = 0
    for _ in metric_logger.log_every(range(iters_per_epoch), print_freq, header):

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # ------------------ SOURCE -------------------
            images_s, targets_s = cddod.next_iter(dl_source, dl_iter_source, device=device)
            loss_dict_source = model(images_s, targets_s, context='source')
            # loss without any additional modules, i.e. without FPA and RA
            losses, module_loss_dict_s = basemodel_loss(loss_dict=loss_dict_source, context='source', detach=False)
            # feature pyramid alignment loss, region alignment loss or both
            module_loss_source = fpa_ra_loss(
                use_fpa=use_fpa,
                use_ra=use_ra,
                context='source',
                loss_dict=module_loss_dict_s,
                args=args
            )
            # total_source_loss
            losses += module_loss_source

            # ------------------ TARGET ---------------------
            if target:
                images_t, targets_t = cddod.next_iter(dl_target, dl_iter_target, device=device)
                loss_dict_target = model(images_t, targets_t, context='target')
                # removing bbox regression and classifier from the computational graph
                # and returning only the module losses in a dictionary
                _, module_loss_dict_t = basemodel_loss(loss_dict=loss_dict_target, context='target', detach=True)
                # module losses of the target
                module_loss_target = fpa_ra_loss(
                    use_fpa=use_fpa,
                    use_ra=use_ra,
                    context='target',
                    loss_dict=module_loss_dict_t,
                    args=args
                )
                losses += module_loss_target

        loss_value = losses.item()
        if target:
            loss_dict_source.update({**module_loss_dict_t})

        cddod.check_isfinite(loss_value, loss_dict=loss_dict_source)
        cddod.step_optimizer(optimizer, losses=losses, scaler=scaler)

        if lr_scheduler is not None:
            lr_scheduler.step()

        local_iter_count += 1
        if wandb_logger is not None:
            wandb_logger.log(
                {**loss_dict_source, 'lr': optimizer.param_groups[0]["lr"], 'iter_count': local_iter_count}
            )
        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"], **loss_dict_source)

    return loss_dict_source, local_iter_count, metric_logger


def fpa_ra_loss(
        use_fpa: bool,
        use_ra: bool,
        loss_dict: Dict[str, Tensor],
        context: str,
        args: Namespace
) -> Tensor:
    module_loss = torch.tensor(0.0, device='cuda', dtype=torch.float32)
    if use_fpa:
        module_loss += args.lambda1 * loss_dict[f'loss_domain_cls_{context}']
        if use_ra:
            module_loss += args.lambda2 * loss_dict[f'loss_region_align_{context}']
    return module_loss


def basemodel_loss(loss_dict: dict, context: str, detach=False) -> Tuple[Tensor, Dict[str, Tensor]]:
    base_dict_loss = {}
    module_dict_loss = {}
    for key, loss in loss_dict.items():
        if key not in [f'loss_domain_cls_{context}', f'loss_region_align_{context}']:
            if detach:
                loss.detach_()
            else:
                base_dict_loss.update({key: loss})
        else:
            module_dict_loss.update({key: loss})
    base_loss = sum(base_dict_loss.values())
    return base_loss, module_dict_loss
