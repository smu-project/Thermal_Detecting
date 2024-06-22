from __future__ import print_function, division
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO
import cv2
import torch

class KittiDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None, target_transform = None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.target_transform = target_transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {'BACKGROUND':0}
        self.coco_labels         = {0:0}
        self.coco_labels_inverse = {0:0}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        self.image = {}

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # img = self.load_image(idx)
        img = self.load_image(idx)
        img = img.astype(np.float32)
        annot = self.load_annotations(idx)

        boxes = annot[:,:4]
        labels = annot[:,4]

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])

        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)
        boxes = boxes.astype(np.float32)
        labels = labels.astype(np.int64)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return img, boxes, labels

    def get_image(self, idx):
        img = self.load_image(idx)
        return img

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        return img