from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign

from default_config import config as cfg
from dod_generalized_rcnn import CustomGeneralizedRCNN
from dod_modules import FPA, RegionAlignment
from dod_roi_heads import CustomRoIHeads


class FasterRCNNcddod(CustomGeneralizedRCNN):
    """
    Adapted from: https://github.com/pytorch/vision/blob/c02d6ce17644fc3b1de0f983c497d66f33974fc6/torchvision/models/detection/faster_rcnn.py
    Additional modules are the introduction of feature pyramid alignment and region alignment
    Implements a custom model of Faster R-CNN with domain classifiers for domain adaptation learning.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.
    The behavior of the model changes depending on whether it is training or evaluation mode.
    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        """

    def __init__(
            self,
            backbone: nn.Module,
            num_classes: int = None,
            # transform parameters
            min_size: int = cfg.MIN_SIZE,  # 800
            max_size: int = cfg.MAX_SIZE,  # 1333 original 1200 last used
            image_mean: Tuple[float, float, float] = cfg.IMAGE_MEAN,
            image_std: Tuple[float, float, float] = cfg.IMAGE_STD,
            # RPN parameters
            rpn_anchor_generator: AnchorGenerator = None,
            rpn_head: RPNHead = None,
            rpn_pre_nms_top_n_train: int = cfg.RPN_PRE_NMS_TOP_N_TRAIN,  # 2000
            rpn_pre_nms_top_n_test: int = cfg.RPN_PRE_NMS_TOP_N_TEST,  # 1000
            rpn_post_nms_top_n_train: int = cfg.RPN_POST_NMS_TOP_N_TRAIN,
            rpn_post_nms_top_n_test: int = cfg.RPN_POST_NMS_TOP_N_TEST,  # 1000
            rpn_nms_thresh: float = cfg.RPN_NMS_THRESH,
            rpn_fg_iou_thresh: float = 0.7,
            rpn_bg_iou_thresh: float = 0.3,
            rpn_batch_size_per_image: int = 256,
            rpn_positive_fraction: float = 0.5,
            rpn_score_thresh: float = 0.0,
            # Box parameters
            box_roi_pool: MultiScaleRoIAlign = None,
            box_head: nn.Module = None,
            box_predictor: nn.Module = None,
            box_score_thresh: float = 0.05,
            box_nms_thresh: float = 0.5,
            box_detections_per_img: int = 100,
            box_fg_iou_thresh: float = 0.5,
            box_bg_iou_thresh: float = 0.5,
            box_batch_size_per_image: int = 512,
            box_positive_fraction: float = 0.25,
            bbox_reg_weights: Tuple[float, float, float, float] = None,
            use_fpa: bool = False,
            fpa: FPA = None,
            use_ra: bool = False,
            region_alignment: RegionAlignment = None,
            representation_size: int = 1024,
            **kwargs: dict,
    ) -> None:

        if use_fpa and fpa is None:
            fpa = FPA()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels specifying"
                "the number of output channels (assumed to be the same for all the levels)"
            )
        if rpn_anchor_generator is not None and not isinstance(rpn_anchor_generator, AnchorGenerator):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator "
                f"or None instead of {type(rpn_anchor_generator)}"
            )
        if box_roi_pool is not None and not isinstance(box_roi_pool, MultiScaleRoIAlign):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            rpn_anchor_generator = self._default_anchorgen()
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)

        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        if box_predictor is None:
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        if use_ra and region_alignment is None:
            region_alignment = RegionAlignment(representation_size)

        roi_heads = CustomRoIHeads(
            # Box
            fg_iou_thresh=box_fg_iou_thresh,
            bg_iou_thresh=box_bg_iou_thresh,
            batch_size_per_image=box_batch_size_per_image,
            positive_fraction=box_positive_fraction,
            bbox_reg_weights=bbox_reg_weights,
            score_thresh=box_score_thresh,
            nms_thresh=box_nms_thresh,
            detections_per_img=box_detections_per_img,
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            region_alignment=region_alignment,
            use_ra=use_ra,
        )

        transform = GeneralizedRCNNTransform(min_size, max_size, list(image_mean), list(image_std), **kwargs)

        super().__init__(backbone, fpa, rpn, roi_heads, use_fpa, transform)

    @staticmethod
    def _default_anchorgen() -> AnchorGenerator:
        anchor_sizes = ((64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0, 3.0),) * len(anchor_sizes)
        return AnchorGenerator(anchor_sizes, aspect_ratios)


class TwoMLPHead(nn.Module):
    """
    Copied from: https://github.com/pytorch/vision/blob/c02d6ce17644fc3b1de0f983c497d66f33974fc6/torchvision/models/detection/faster_rcnn.py
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels: int, representation_size: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Copied from: https://github.com/pytorch/vision/blob/c02d6ce17644fc3b1de0f983c497d66f33974fc6/torchvision/models/detection/faster_rcnn.py
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x: Tensor) -> Tuple[float, List]:
        if x.dim() == 4:
            torch._assert(
                x.shape[2:] == (1, 1),
                f"x has the wrong shape, expecting the last two dimensions to be (1,1) instead of {x.shape[2:]}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
