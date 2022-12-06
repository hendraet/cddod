import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Union

import torch
from torch import nn, Tensor

from dod_losses import fpa_dc_loss


class CustomGeneralizedRCNN(nn.Module):
    """
    Implements a custom generalized rcnn for document object detection
    Adapted from: https://github.com/pytorch/vision/blob/c02d6ce17644fc3b1de0f983c497d66f33974fc6/torchvision/models/detection/generalized_rcnn.py
    Two submodules, namely, feature pyramid alignment (FPA)
    and Region alignment (RA) are added
    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        fpa (nn.Module): for feature pyramid pixel alignment
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone: nn.Module,
                 fpa: nn.Module,
                 rpn: nn.Module,
                 roi_heads: nn.Module,
                 use_fpa: bool,
                 transform: nn.Module) -> None:
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.fpa = fpa
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False
        self.use_fpa = use_fpa

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images: List[Tensor], targets: List[Dict[str, Tensor]] = None, context: str = None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
            context (str): whether the images are coming from source or target
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training:
            if targets is None:
                assert (False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        assert len(boxes.shape) == 2 and boxes.shape[-1] == 4
                        assert f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}."
                    else:
                        assert False, f"Expected target boxes to be of type Tensor, got {type(boxes)}."

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            assert f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}"
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    assert False, f"All bounding boxes should have positive height and width. Found invalid box {degen_bb} for target at index {target_idx}."

        features = self.backbone(images.tensors)
        losses = {}
        if self.training and self.use_fpa:
            domain_pixels = self.fpa(features)
            domain_classifier_loss = fpa_dc_loss(domain_pixels, context=context)
            losses.update({f'loss_domain_cls_{context}': domain_classifier_loss})

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.rpn(images, features, targets)

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets, context=context)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
