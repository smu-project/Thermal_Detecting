import torch
import torch.nn as nn

from torch.nn.modules.utils import _pair

from ..utils import box_utils


class DetectPostProcess(torch.nn.Module):
    def __init__(self, anchor, num_class, var=0.125, th_conf=0.5, th_iou=0.5,
                 background=True, logistic='softmax'):
        super().__init__()

        self.num_anchor = anchor.size(0)
        self.register_buffer('anchor', anchor)

        self.num_class = num_class

        self.var = _pair(var)
        self.th_conf = th_conf
        self.th_iou = th_iou

        self.background = background
        self.logistic = logistic
        self.regression = self.build_regression(logistic)

    def forward(self, conf, loc):
        cls_len = self.num_class

        if self.background:
            cls_len = cls_len + 1

        try:
            conf = conf[..., 0:cls_len]
            loc = loc[..., 0:4]
        except:
            conf = loc[..., 0:cls_len]
            loc = conf[..., 0:4]

        batch_size = conf.view(-1, self.num_anchor, cls_len).size(0)

        loc = loc.view(batch_size, self.num_anchor, -1)
        score = conf.view(batch_size, self.num_anchor, -1)

        if self.regression:
            score = self.regression(score)

        box = self.decode(loc, self.anchor)

        batches = self.regular_nms(score, box)

        return batches

    def decode(self, loc, anchor):
        anchor = anchor.unsqueeze(0)

        # delta_x * anchor_width + anchor_x
        encoded_xy = loc[..., 0:2] * self.var[0]
        xy = encoded_xy * anchor[..., 2:4] + anchor[..., 0:2]

        # exp(delta_w) * anchor_width
        encoded_wh = loc[..., 2:4] * self.var[1]
        wh = torch.exp(encoded_wh) * anchor[..., 2:4]

        return torch.cat((xy - wh/2., xy + wh/2.), dim=2)

    def extra_repr(self):
        s = ('var={var}, th_conf={th_conf}, th_iou={th_iou}')
        return s.format(**self.__dict__)

    @staticmethod
    def build_regression(logistic="softmax"):
        regression = None

        if logistic == 'softmax':
            regression = nn.Softmax(dim=-1)
        elif logistic == 'sigmoid':
            regression = nn.Sigmoid()

        return regression

    def regular_nms(self, score, box):
        batches = []

        batch_size = score.size(0)

        cls_len = self.num_class
        if self.background:
            cls_len = cls_len + 1

        for b in range(0, batch_size):
            classes = []

            for i in range(self.background, cls_len):
                mask = score[b][:, i] >= self.th_conf

                _box = box[b][mask]
                _score = score[b][mask, i]

                idx = box_utils.nms(_box, _score, self.th_iou)
                objs = torch.cat((_box[idx], _score[idx].unsqueeze(1)), 1)

                classes.append(objs.tolist())

            batches.append(classes)

        return batches
