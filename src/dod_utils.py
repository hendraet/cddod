import copy
import logging
import math
import os
import random
import sys
import warnings
from argparse import Namespace
from logging import Logger
from pathlib import Path
from typing import Tuple, List, Iterator, Dict

import numpy
import numpy as np
import torch
import wandb
from torch import nn, Tensor
from torch.cuda import device
from torch.cuda.amp import GradScaler
from torch.nn import Parameter
from torch.optim import lr_scheduler, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader

from dod_base_dataset import DODDataset
from pytorch_references.detection import _utils, utils
from pytorch_references.detection import transforms as T
from pytorch_references.detection.transforms import Compose

logger = logging.getLogger(__name__)


def mkdir(path: str) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)


def make_logger(filename: str, datefmt: str = '%m-%d-%Y %H:%M:%S', file_dir: Path = None) -> Logger:
    """
    for logging experiment logs
    Args:
        datefmt (str): date and time format
        filename (str): filename/path of the logfile
        file_dir (Path): log file directory path

    Returns:
        logger (Logger): for recording logs
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{file_dir}/{filename}')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter_fh = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s', datefmt=datefmt)
    formatter_ch = logging.Formatter('%(asctime)s  %(message)s', datefmt=datefmt)
    file_handler.setFormatter(formatter_fh)
    console_handler.setFormatter(formatter_ch)
    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train: bool, strong_augment: bool = None) -> Compose:
    """
    Args:
        train (bool): training phase or not
        strong_augment (bool): whether to use strong augmentation on images while training
    Returns:
        Compose with a list a transforms
    """
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
        if strong_augment:
            transforms.append(T.RandomPhotometricDistort())
            transforms.append(T.ScaleJitter((200, 200)))
            transforms.append(T.RandomZoomOut())
            transforms.append(T.RandomGaussianBlurring())
            transforms.append(T.RandomGrayscale())
            transforms.append(T.RandomPosterize(bits=2))
            transforms.append(T.FixedSizeCrop(size=(200, 200)))
            transforms.append(T.RandomEqualize())
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float32))
    return T.Compose(transforms)


def set_deterministic(deterministic: bool = True, seed: int = 42) -> None:
    """For reproducibility of the experiments"""
    if deterministic:
        warnings.warn(
            'You have chosen to seed training. This will turn on the CUDNN '
            'deterministic setting, which could slow down your training considerably!'
            ' You may see unexpected behavior when restarting from checkpoints.'
        )
        # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
        # cudnn seed settings are slower and more reproducible, else faster and less reproducible
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # for multiple GPUs, exception safe
        torch.cuda.manual_seed_all(seed)
        # using deterministic algorithms
        # torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        # reference: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        #            https://pytorch.org/docs/stable/notes/randomness.html
        # Disabling cudnn auto-tuner to find the best algorithm to use for your hardware.
        # Enabling it is good whenever the input sizes for the network do not vary. This way,
        # cudnn will look for the optimal set of algorithms for that particular configuration
        # (which takes some time). This usually leads to faster runtime.
        # NOTE: But if your input sizes changes at each iteration, then cudnn will benchmark
        # every time a new size appears, possibly leading to worse runtime performances.
        torch.backends.cudnn.benchmark = False


def wandb_logger(project_name: str, name: str, job_type: str, group: str, config: Namespace) -> wandb:
    """
    instantiating wand_logger and storing the config file
    Args:
        project_name (str): name of the project
        name (str): name of the experiment,
        job_type (str): type of the experiment (for example training or evaluation)
        group (str) : for grouping experiments in weight and biases user interface.
        config (Namespace): arguments for the experiments'

    Returns:
        weight and biases (wandb) logger
    """
    wb_logger = wandb
    wb_logger.init(project=project_name, name=name, group=group, job_type=job_type, config=vars(config))
    return wb_logger


def get_datasets(
        root_dir_source: str,
        root_dir_target: str = None,
        img_extension_s: str = 'jpg',
        img_extension_t: str = None,
        txt_file_s: str = 'trainval',
        txt_file_t: str = None,
        txt_file_eval: str = 'test',
        augment: bool = False,
        classes: List[str] = None
) -> Tuple[DODDataset, DODDataset, DODDataset]:
    """
    Args:
        root_dir_source (str): directory/filepath of the source images
        root_dir_target (str): directory/filepath of the target images
        img_extension_s (str): source file image extension (jpg/png)
        img_extension_t (str): target file image extension (jpg/png)
        txt_file_s (str): text file with source images names
        txt_file_t (str): text file with target images names
        txt_file_eval (str): text file with evaluation dataset names (source/target)
        augment (bool): whether to use augmentation or note (default=False)
        classes (bool): whether to reduce classes from six to three

    Returns:
        a tuple [DODDataset, DODDataset, DODDataset] DODDataset dataset class
    """
    eval_img_extension = img_extension_s
    target_dataset = None
    source_dataset = DODDataset(
        root_dir=root_dir_source,
        bounding_box=True,
        image_set=txt_file_s,
        image_file_ext=img_extension_s,
        transforms=get_transform(train=True, strong_augment=augment),
        classes=classes
    )

    if root_dir_target is not None:
        root_dir_target = root_dir_target
        target_dataset = DODDataset(
            root_dir=root_dir_target,
            bounding_box=False,
            image_set=txt_file_t,
            image_file_ext=img_extension_t,
            transforms=get_transform(train=True, strong_augment=False),
            classes=classes
        )
        eval_img_extension = img_extension_t

    eval_dataset = DODDataset(
        root_dir=root_dir_source if root_dir_target is None else root_dir_target,
        bounding_box=True,
        image_set=txt_file_eval,
        image_file_ext=eval_img_extension,
        transforms=get_transform(train=False, strong_augment=False),
        classes=classes
    )

    return source_dataset, target_dataset, eval_dataset


def get_dataloaders(
        dataset_source: DODDataset,
        dataset_eval: DODDataset,
        dataset_target: DODDataset = None,
        batch_size: int = 1,
        collate_fn=collate_fn,
        num_workers: int = 4,
        dd_parallel: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    dl_target = None

    if dd_parallel:
        sampler_s = torch.utils.data.distributed.DistributedSampler(dataset_source)
        sampler_t = torch.utils.data.distributed.DistributedSampler(dataset_target)
        num_workers = 0
    else:
        sampler_s = None
        sampler_t = None

    if dataset_target is not None:
        dl_target = torch.utils.data.DataLoader(
            dataset=dataset_target,
            batch_size=batch_size,
            sampler=sampler_t,
            shuffle=(sampler_t is None),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    dl_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        sampler=sampler_s,
        shuffle=(sampler_s is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    dl_eval = torch.utils.data.DataLoader(
        dataset=dataset_eval,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return dl_source, dl_target, dl_eval


def next_iter(
        data_loader: DataLoader,
        data_loader_iter: Iterator[DataLoader],
        device: device
) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
    try:
        data = next(data_loader_iter)
    except StopIteration:
        data_loader_iter = iter(data_loader)
        data = next(data_loader_iter)

    images = list(image.to(device) for image in data[0])
    targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]

    return images, targets


def print_dataset_len(source_ds: DODDataset, target_ds: DODDataset, eval_ds: DODDataset) -> None:
    logger.info(f"Length of the training (source) dataset : {source_ds.__len__()}")
    if target_ds is not None:
        logger.info(f"Length of the training (target) dataset : {target_ds.__len__()}")
        logger.info(f"Length of the evaluation (target) dataset : {eval_ds.__len__()}")
    else:
        logger.info(f"Length of the evaluation (source) dataset : {eval_ds.__len__()}")


# All the codes following are adapted/copied from PyTorch detection module.
# Reference https://github.com/pytorch/vision/tree/c02d6ce17644fc3b1de0f983c497d66f33974fc6/references/detection
def set_normalized_weight_decay(
        model: nn.Module,
        norm_weight_decay: float = None,
        weight_decay: float = None
) -> List[Parameter]:
    """
    Args:
        model (nn.Module): model for training
        norm_weight_decay (float) : weight decay for normalization layers
        weight_decay: weight decay

    Returns:
        parameters (List[Parameter]): list of trainable parameters
    """
    if norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = _utils.split_normalization_params(model)
        wd_groups = [norm_weight_decay, weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    return parameters


def set_optimizer(
        optimizer_name: str,
        parameters: List[Parameter],
        lr: float,
        momentum: float,
        weight_decay: float
) -> Optimizer:
    """
    Args:
        optimizer_name (str): optimizer for training (SGD/AdamW)
        parameters (List[Parameter]): trainable parameters
        lr (float): learning rate
        momentum (float): for accelerating optimization process
        weight_decay (float): for regularization

    Returns:
        an optimizer (Optimizer)
    """
    opt_name = optimizer_name.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {optimizer_name}. Only SGD and AdamW are supported.")
    return optimizer


def set_lr_scheduler(
        lr_scheduler: str,
        optimizer: Optimizer,
        lr_steps: Tuple,
        lr_gamma: float,
        epochs: int
) -> _LRScheduler:
    """
    learning rate scheduler
    Args:
        lr_scheduler (str): name of the learning rate scheduler (multisteplr/cosine annealing learning rate)
        optimizer (Optimizer): Stochastic gradient descent/AdamW
        lr_steps (tuple): steps at which learning rate has to be changed
        lr_gamma (float): the factor at which learning rate has to be changed
        epochs (int): total of times a full forward-backward pass on the complete dataset
    """
    lr_scheduler = lr_scheduler.lower()
    if lr_scheduler == "multisteplr":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)
    elif lr_scheduler == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )


def resume_from_checkpoint(
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        amp: bool,
        scaler: GradScaler,
        args: Namespace,
        resume: bool = False
):
    """
    For resuming training from a saved checkpoint
    Args:
        model (nn.Module): model for training
        optimizer (Optimizer): SGD/AdamW
        lr_scheduler (_LRScheduler): learning rate scheduler multistepLR/cosine annealing
        amp (bool): for automatic mixed precision training --float16
        scaler (GradScaler): for scaling gradients -- improves convergence for networks with float16 gradients by minimizing gradient underflow.
        args (Namespace): arguments
        resume (bool): for resuming training from a checkpoint
    """
    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        # for mixed precision
        if amp:
            scaler.load_state_dict(checkpoint["scaler"])


def check_isfinite(loss_value: float, loss_dict: Dict):
    """
    Check whether the loss value is going to infinity while training.
    if it explodes, the training will stop and exit.
    Args:
        loss_value (float): loss value
        loss_dict (Dict): dictionary of losses
    """
    if not math.isfinite(loss_value):
        logger.info(f"Loss is {loss_value}, stopping training")
        logger.info(loss_dict)
        sys.exit(1)


def step_optimizer(optimizer: Optimizer, losses: Tensor, scaler: GradScaler):
    """
    For optimizer taking a step during training after backward pass
    Args:
        optimizer (Optimizer): optimizer used in the training (SGD/AdamW)
        losses (Tensor.float): loss value in after a forward pass
        scaler (GradScaler): for scaling gradient if automatic fixed precision is used (float16)
    """
    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        losses.backward()
        optimizer.step()


def warmup_training(epoch: int, data_loader: DataLoader, optimizer: Optimizer) -> _LRScheduler:
    """
    For warming up training. First 1000 iterations are used for warmup
    Args:
        epoch (int): current training epoch
        data_loader (DataLoader): dataloader
        optimizer (Optimizer): Optimizer used for optimization (SGD/AdamW)

    Returns:
        Linear learning rate scheduler
    """
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
        return lr_scheduler


def save_checkpoint(
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: lr_scheduler,
        scaler: GradScaler,
        epoch: int,
        args: Namespace
):
    """

    Args:
        model (nn.Model): model to be saved
        optimizer (Optimizer): optimizer used in the training (SGD/AdamW)
        lr_scheduler (_LRScheduler): learning rate scheduler
        scaler (GradScaler): gradient scaling for automatic fixed precision
        epoch (int): current training epoch
        args (Namespace): parameters given by the user at the start of the experiment
    """
    if args.output_dir:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": args,
            "epoch": epoch,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        logger.info(f"Saving checkpoint: epoch {epoch}")
        utils.save_on_master(checkpoint, Path(args.output_dir, f"model_{epoch}.pth"))
