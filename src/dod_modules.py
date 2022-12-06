from typing import OrderedDict, Tuple, List

import torch
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from base_model import resnet101
from default_config import config as cfg
from domain_classifiers import FPADomainClassifier, RADomainClassifier, grad_reverse


class Resnet101withFPN(torch.nn.Module):
    """""Implements Feature Pyramid Network on Resnet101"""

    def __init__(
            self,
            in_channels: Tuple[int] = cfg.IN_CHANNELS,
            out_channels: int = cfg.OUT_CHANNELS,
            stages: Tuple[int] = cfg.STAGES
    ) -> None:
        """
        Args:
            in_channels (tuple): channel dimensions of the stages.
            out_channels (int): channel dimensions of the feature pyramid network output
            stages (tuple) : stages in a ResNet101 from which feature maps have to be extracted.
                A stage refers to the layers producing output maps of the same size.
        """

        super(Resnet101withFPN, self).__init__()

        m = resnet101(pretrained=True)
        # Stages (layers producing output maps of the same size) from Resnet101 to be returned,
        return_nodes = {f'layer{k}': str(v) for v, k in enumerate(list(stages))}
        # extract last feature map from the four stages of the ResNet101
        self.backbone = create_feature_extractor(model=m, return_nodes=return_nodes)
        # this attribute is necessary for generalized rcnn
        self.out_channels = out_channels
        self.fpn = FeaturePyramidNetwork(in_channels_list=list(in_channels), out_channels=out_channels)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        return x


class FPA(torch.nn.Module):
    """Feature pyramid alignment (FPA) module"""

    def __init__(self):
        super(FPA, self).__init__()
        for i in range(4):
            setattr(self, f'dc{i + 1}', FPADomainClassifier())

    def forward(self, x: OrderedDict[str, Tensor]) -> List[Tensor]:
        x1, x2, x3, x4 = x.values()
        # in each of these domain classifiers grad_reverse (GRL layer)
        # reverses the gradient during backward pass
        dc1_out = self.dc1(grad_reverse(x1))
        dc2_out = self.dc2(grad_reverse(x2))
        dc3_out = self.dc3(grad_reverse(x3))
        dc4_out = self.dc4(grad_reverse(x4))
        return [dc1_out, dc2_out, dc3_out, dc4_out]


class RegionAlignment(torch.nn.Module):
    """Region Alignment module in custom FasterRCNN"""
    def __init__(self, in_channels: int):
        super(RegionAlignment, self).__init__()
        # two domain/classes -- source or target
        self.ra_dc = RADomainClassifier(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        ra_predictions = self.ra_dc(x)
        return ra_predictions
