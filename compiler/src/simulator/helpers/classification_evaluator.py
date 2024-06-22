import os
import sys
import multiprocessing
import torch

from torch.utils.data import DataLoader

from ..datasets import load_dataset
from ..transforms import detector_transforms as transforms
from ..utils import Accuracy

from enlight import quantizations


class ClassificationEvaluator:
    def __init__(self, args, model, eval_cnt=-1):
        self.args = args

        self.model = model

        self.dataset = self.init_dataset(model)
        self.dataloader = self.init_dataloader()

        if eval_cnt > 0:
            self.data_cnt = eval_cnt
        else:
            self.data_cnt = len(self.dataset)
        
        self.metric = Accuracy(topk=args.topk)

        self.has_background = args.has_background

        self.score = 0

    def __call__(self):
        args = self.args

        self.model.eval()
        self.metric.reset()

        cnt = 0

        for i, batch in enumerate(self.dataloader):
            x, y, _ = batch

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

            with torch.no_grad():
                y_ = self.model(x)

                if self.has_background:
                    y_ = y_[..., 1:]

                self.match(y_, y)

            if cnt > self.data_cnt:
                break

            cnt += x.size(0)

        print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

        accuracy = self.metric.get_result()
        self.save_score(accuracy, args.topk)

        # print results
        print("accuracy = %.3f" % accuracy)

    def save_score(self, score, topk):
        info = {}

        info['has_score'] = True
        if topk == 5:
            info['top5'] = score
        elif topk == 1:
            info['top1'] = score
        else:
            print([f"[INFO] Saving score if only k = 1 or 5 in topk (current k = {topk})"])

        info['evaluation_dataset'] = self.args.dataset

        self.model.set_info(info)

        self.score = score

    def get_score(self):
        return self.score
    
    def get_dataset(self):
        return self.args.dataset

    def match(self, y_, y):
        self.metric.match(y_, y)

    def init_dataset(self, model):
        args = self.args

        if args.force_resize is None:
            size = model.get_input_size()[-2:]
        else:
            size = args.force_resize

        t = []

        t.append(transforms.Resize(size))

        if args.crop:
            t.append(transforms.CenterCrop(args.crop))

        if model.is_quantized():
            t.append(transforms.TensorFromImage())

            if not model.is_fused_normalization():
                raise Exception("should fuse input normalization for quantized network")
        else:
            t.append(transforms.ToTensor())

            mean = model.get_normalization_mean()
            std = model.get_normalization_std()

            if not model.is_fused_normalization():
                t.append(transforms.Normalize(mean, std))
            else:
                t.append(transforms.QuantFriendlyNormalize(
                    scale=quantizations.input_scale)
                )

        t = transforms.Compose(t)

        labels = model.get_class_label()

        dataset = load_dataset(args,
                               image_set='test', 
                               download=args.download, 
                               transforms=t,
                               class_label=labels)

        return dataset

    def init_dataloader(self):
        args = self.args

        if args.num_workers < 0:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = args.num_workers

        return DataLoader(self.dataset,
                          pin_memory=True,
                          batch_size=args.batch_size,
                          num_workers=num_workers,
                          collate_fn=self.collate)

    @staticmethod
    def collate(batch):
        imgs = []
        targets = []
        org_imgs = []

        for (img, target, org_img) in batch:
            imgs.append(img)
            targets.append(target)
            org_imgs.append(org_img)

        return torch.stack(imgs, 0), torch.stack(targets, 0), org_imgs
