from typing import Dict

import torch
from torchvision.models.resnet import resnet101 as _resnet101, ResNet

from default_config import config as cfg


def resnet101(pretrained: bool = False, **kwargs: Dict) -> ResNet:
    """
    Args:
        pretrained (bool): load pretrained weights into the model
        **kwargs (Dict): arguments
    Returns:
        ResNet101 with pretrained weights
    """
    if pretrained:
        model = _resnet101()
        checkpoint = cfg.RESNET_PRETRAINED_DIR
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        return model
    return _resnet101(pretrained=pretrained, **kwargs)
