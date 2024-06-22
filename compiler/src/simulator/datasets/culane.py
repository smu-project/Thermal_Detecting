import os

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from ..transforms import lane_transforms as tf


def coord_ip_to_op(x, y, scale):
    # (1640, 590) --> (1664, 590) --> (1664, 590-14=576) --> (1664/scale, 576/scale)
    if x is not None:
        x = x*1664./1640.
        x = int(x/scale)
    if y is not None:
        y = int((y-14)/scale)
    return x, y


def generateAFs(label):
    # creating AF arrays
    num_lanes = np.amax(label)
    VAF = np.zeros((label.shape[0], label.shape[1], 2))
    HAF = np.zeros((label.shape[0], label.shape[1], 2))

    # loop over each lane
    for l in range(1, num_lanes+1):
        # initialize previous row/cols
        prev_cols = np.array([], dtype=np.int64)
        prev_row = label.shape[0]

        # parse row by row, from second last to first
        for row in range(label.shape[0]-1, -1, -1):
            cols = np.where(label[row, :] == l)[0] # get fg cols

            # get horizontal vector
            for c in cols:
                if c < np.mean(cols):
                    HAF[row, c, 0] = 1.0 # point to right
                elif c > np.mean(cols):
                    HAF[row, c, 0] = -1.0 # point to left
                else:
                    HAF[row, c, 0] = 0.0 # point to left

            # check if both previous cols and current cols are non-empty
            if prev_cols.size == 0: # if no previous row/cols, update and continue
                prev_cols = cols
                prev_row = row
                continue
            if cols.size == 0: # if no current cols, continue
                continue
            col = np.mean(cols) # calculate mean

            # get vertical vector
            for c in prev_cols:
                # calculate location direction vector
                vec = np.array([col - c, row - prev_row], dtype=np.float32)
                # unit normalize
                vec = vec / np.linalg.norm(vec)
                VAF[prev_row, c, 0] = vec[0]
                VAF[prev_row, c, 1] = vec[1]

            # update previous row/cols with current row/cols
            prev_cols = cols
            prev_row = row

    return VAF, HAF


class CULane(Dataset):
    def __init__(self, path, image_set='train', random_transforms=False, quantized=False):
        super(CULane, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.input_size = (288, 832) # original image res: (590, 1640) -> (590-14, 1640+24)/2
        self.output_scale = 0.25
        self.samp_factor = 2./self.output_scale
        self.data_dir_path = path
        self.image_set = image_set
        self.random_transforms = random_transforms

        self.ignore_label = 255

        if quantized:
            self.transforms = transforms.Compose([
                tf.GroupRandomScale(size=(0.5, 0.5)),
                tf.GroupNormalize(mean=((0., 0., 0.,), (0,)), std=((1./255., 1./255., 1./255.), (1,))),
                tf.GroupNormalize(mean=((128., 128., 128.,), (0,)), std=((1., 1., 1.), (1,))),
            ])
        else:
            self.transforms = transforms.Compose([
                tf.GroupRandomScale(size=(0.5, 0.5)),
                tf.GroupNormalize(mean=((0., 0., 0.,), (0,)), std=((256./255., 256./255., 256./255.), (1,))),
                tf.GroupNormalize(mean=((.5, .5, .5,), (0,)), std=((.5, .5, .5), (1,))),
            ])

        self.create_index()

    def create_index(self):
        self.img_list = []
        self.seg_list = []

        self.out_list = []

        listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))
        if not os.path.exists(listfile):
            raise FileNotFoundError("List file doesn't exist. Label has to be generated! ...")

        with open(listfile) as f:
            for line in f:
                l = line.strip()
                self.img_list.append(os.path.join(self.data_dir_path, l[1:]))  # l[1:]  get rid of the first '/' so as for os.path.join
                self.out_list.append(l[1:-3] + 'lines.txt')

                if self.image_set == 'test':
                    self.seg_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16_test', l[1:-3] + 'png'))
                else:
                    self.seg_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16', l[1:-3] + 'png'))

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx]).astype(np.float32)/255. # (H, W, 3)
        img = cv2.resize(img[14:, :, :], (1664, 576), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if os.path.exists(self.seg_list[idx]):
            seg = cv2.imread(self.seg_list[idx], cv2.IMREAD_UNCHANGED) # (H, W)
            seg = np.tile(seg[..., np.newaxis], (1, 1, 3)) # (H, W, 3)
            seg = cv2.resize(seg[14:, :, :], (1664, 576), interpolation=cv2.INTER_NEAREST)
            img, seg = self.transforms((img, seg))
            seg = cv2.resize(seg, None, fx=self.output_scale, fy=self.output_scale, interpolation=cv2.INTER_NEAREST)
            # create binary mask
            mask = seg[:, :, 0].copy()
            mask[seg[:, :, 0] >= 1] = 1
            mask[seg[:, :, 0] == self.ignore_label] = self.ignore_label
            # create AFs
            seg_wo_ignore = seg[:, :, 0].copy()
            seg_wo_ignore[seg_wo_ignore == self.ignore_label] = 0
            vaf, haf = generateAFs(seg_wo_ignore.astype(np.long))
            af = np.concatenate((vaf, haf[:, :, 0:1]), axis=2)

            # convert all outputs to torch tensors
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
            seg = torch.from_numpy(seg[:, :, 0]).contiguous().long().unsqueeze(0)
            mask = torch.from_numpy(mask).contiguous().float().unsqueeze(0)
            af = torch.from_numpy(af).permute(2, 0, 1).contiguous().float()
        else: # if labels not available, set ground truth tensors to nan values
            img, _ = self.transforms((img, img))
            # convert all outputs to torch tensors
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
            seg, mask, af = torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan'))

        return img, seg

    def get_path(self, idx):
        return self.out_list[idx]

    def __len__(self):
        return len(self.img_list)
