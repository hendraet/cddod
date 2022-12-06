from pathlib import Path
from typing import Callable, Optional, Tuple, Dict
from xml.etree.ElementTree import parse as et_parse

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class DODDataset(Dataset):
    """
    Dataset class for document object detection (DOD)
    Adapted from: https://github.com/VisionLearningGroup/DA_Detection
        https://github.com/pytorch/vision/blob/c02d6ce17644fc3b1de0f983c497d66f33974fc6/torchvision/datasets/voc.py
    """

    def __init__(
            self,
            root_dir: str,
            bounding_box: bool = False,
            image_set: str = "train",
            classes: Tuple[str] = None,
            transforms: Optional[Callable] = None,
            image_file_ext: str = None,
    ):
        super().__init__()
        self.bounding_box = bounding_box
        self.transforms = transforms
        self.classes = classes
        root_dir_path = Path(root_dir)
        if not root_dir_path.is_dir():
            raise RuntimeError("Root directory does not exists")
        # text file with image file names
        txt_file = root_dir_path / "ImageSets" / "Main" / f"{image_set}.txt"
        # reading the filenames into a list
        with open(txt_file) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = root_dir_path / "JPEGImages"
        annotations_dir = root_dir_path / "Annotations"

        self.image_paths = []
        self.annotation_paths = []
        for file_name in file_names:
            image_path = image_dir / f'{file_name}.{image_file_ext}'
            if bounding_box:
                annotation_path = annotations_dir / f"{file_name}.xml"
                if image_path.is_file() and annotation_path.is_file():
                    self.image_paths.append(image_path)
                    self.annotation_paths.append(annotation_path)
            else:
                self.image_paths.append(image_path)
        self.class_to_ind = {cls: i for i, cls in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.image_paths[index]).convert("RGB")
        if self.bounding_box:
            target = self.parse_pascal_annotations(index)
        else:
            target = self.make_target_annotations_zero(index)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    @staticmethod
    def make_target_annotations_zero(index: int) -> Dict[str, Tensor]:
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": torch.zeros((1,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64)
        }
        return target

    def parse_pascal_annotations(self, index: int) -> Dict[str, Tensor]:
        """
        Parse the annotated files and extract the bounding boxes, labels, area etc
        """
        filename = self.annotation_paths[index]
        tree = et_parse(filename)
        root = tree.getroot()
        objs = root.findall('object')

        boxes = []
        box_area = []
        is_crowd = []
        classes = []

        for obj in objs:
            bbox = obj.find('bndbox')
            # extracting bounding boxes coordinates
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            # area of the bounding box
            area = (x2 - x1) * (y2 - y1)
            # checking bounding boxes are crowded
            crowd = obj.find('difficult')
            difficult = int(crowd.text) if crowd else 0

            # extracting class labels
            cls_text = obj.find('name').text.lower().strip()
            if cls_text not in self.classes:
                cls_text = 'text'
            else:
                if cls_text == 'footnote':
                    cls_text = 'text'
            # converting to label
            cls = self.class_to_ind[cls_text]
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                is_crowd.append(difficult)
                classes.append(cls)
                box_area.append(area)

        # convert everything to torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes, dtype=torch.int64)
        area = torch.as_tensor(box_area, dtype=torch.float32)
        is_crowd = torch.as_tensor(is_crowd, dtype=torch.uint8)
        image_id = torch.tensor([index])

        return {
            'boxes': boxes,
            'labels': classes,
            'iscrowd': is_crowd,
            'area': area,
            'image_id': image_id
        }
