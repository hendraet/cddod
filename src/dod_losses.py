from typing import List

import torch
from torch import Tensor

from default_config import config as cfg


def fpa_dc_loss(inputs: List[Tensor], context: str = 'source', weight: bool = cfg.FPA_WEIGHT) -> Tensor:
    """
    Calculate the pixel loss from the four domain classifiers attached
    to the Feature Pyramid Alignment module and returns
    target or source loss

    references: https://github.com/kailigo/cddod,
                https://github.com/VisionLearningGroup/DA_Detection
    Args:
        inputs (List[Tensor]): feature map from feature pyramid layers
        context (str): whether to use compute loss for source or target
        weight (bool): whether to weight the loss

    Returns:
        mean_loss (float): total loss of the domain classifiers (source or target)
    """

    loss = torch.zeros(len(inputs), device='cuda')
    for i in range(len(inputs)):
        pred = inputs[i]
        if context == 'source':
            loss[i] = torch.mean(torch.log(pred))
        elif context == 'target':
            loss[i] = torch.mean(torch.log(1 - pred))
        else:
            ValueError("Enter context as 'source' or 'target'")
    if weight:
        mod_factor = torch.tensor(list(cfg.FPA_MOD_WEIGHTS), dtype=torch.float32, device='cuda')
        loss = loss * mod_factor
    return -torch.mean(loss)


def focal_loss(
        inputs: Tensor,
        alpha: float = cfg.ALPHA,
        gamma: float = cfg.GAMMA,
        context: str = 'source'
) -> Tensor:
    """
        Calculate the focal loss for source or target
        references: https://github.com/kailigo/cddod,
                    https://github.com/VisionLearningGroup/DA_Detection
        Args:
            inputs (Tensor): feature map from feature pyramid layers
            context (str): whether to use compute loss for source or target
            alpha (float): weight balance between positive and negative samples'
            gamma (float): if gamma is large, lower is the loss for well-classified examples,
                           so attention of the model is towards ‘hard-to-classify examples and vice-versa
                           gamma = 0, cross-entropy loss, equal preference for hard and easy to classify samples
        Returns:
            mean_loss (float):  loss of RA domain classifier(source or target)
        """
    p = torch.sigmoid(inputs)
    batch_loss = 0
    if context == 'source':
        ds = p
        log_p = torch.log(ds)
        batch_loss = (torch.pow((1 - ds), gamma)) * log_p
    elif context == 'target':
        dt = 1 - p
        log_p = torch.log(dt)
        batch_loss = (torch.pow((1 - dt), gamma)) * log_p
    else:
        ValueError("Enter context as 'source' or 'target'")

    batch_loss *= alpha
    return -torch.mean(batch_loss)


def local_classifier_loss(
        inputs: List[Tensor],
        context: float = 'source',
        weight: bool = cfg.WEIGHT,
        mse: bool = cfg.MSE
) -> Tensor.float:
    """
    Calculate domain classifier loss (swda)
    references: https://github.com/VisionLearningGroup/DA_Detection
    Args:
        inputs (List[Tensor]):
        context (str): whether using target or source image for loss calculation (default='source')
        weight (bool): hyperparameter for weighing the loss from the domain classifier (default=True)
        mse (bool): whether to use mean square loss or not (default=True)

    Returns:
        local domain classifier loss (float)
    """
    loss = torch.zeros(len(inputs), device='cuda')
    for i in range(len(inputs)):
        if context == 'source':
            if mse:
                loss[i] = torch.mean(inputs[i] ** 2)
            else:
                loss[i] = torch.mean(torch.log(inputs[i]))
        elif context == 'target':
            if mse:
                loss[i] = (1 - torch.mean(inputs[i])) ** 2
            else:
                loss[i] = torch.mean(torch.log(1 - inputs[i]))
    if weight:
        mod_factor = torch.tensor([0.1, 0.1, 1], dtype=torch.float32, device='cuda')
        loss = loss * mod_factor
    return -torch.mean(loss)
