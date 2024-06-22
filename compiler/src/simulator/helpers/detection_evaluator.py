from inspect import classify_class_attrs
import os
import multiprocessing
import torch
import copy

from torch.utils.data import DataLoader

from .coco_mAP import CocomAP
from ..utils import MeanAp

from ..datasets import load_dataset
from ..transforms import detector_transforms as transforms

from simulator.post_process import *

from enlight import nn

from enlight import quantizations


class DetectionEvaluator:
    def __init__(self, args, model, skip_measure_metric=False, eval_cnt=-1):
        self.args = args

        self.model = model
        model.set_post_process_threshold(args.th_conf, args.th_iou)


        self.dataset = self.init_dataset(model)
        self.dataloader = self.init_dataloader()

        if eval_cnt > 0:
            self.data_cnt = eval_cnt
        else:
            self.data_cnt = len(self.dataset)
        self.dataset_type = self.get_dataset_type(self.args.dataset)
        self.mAP = self.get_mAP_metric(self.dataset_type)

        self.imported_post_process = self.get_post_process()

        self.score = 0.
        self.skip_measure_metric = skip_measure_metric

    def __call__(self, quiet=False):
        args = self.args

        self.model.eval()
        self.mAP.init()

        cnt = 0

        for i, batch in enumerate(self.dataloader):
            x = y = img_ids = img_sizes = None
            if self.dataset_type == 'COCO':
                x, y, img_ids, img_sizes = batch
            else:
                x, y = batch

            if cnt >= self.data_cnt:
                break

            if torch.cuda.is_available():
                x = x.cuda()

            if not quiet:
                print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

            with torch.no_grad():
                y_ = self.model(x)

                if self.imported_post_process is not None:
                    y_ = self.imported_post_process(*y_)

                if not self.skip_measure_metric:
                    self.match(y_, y, img_ids, img_sizes)


            cnt += x.size(0)

        if not quiet:
            print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

        if not self.skip_measure_metric:
            mAP = self.print_mAP(quiet=quiet)
        else:
            mAP = -1

        return mAP

    def get_dataset_type(self, dataset):
        if dataset[0:4] == 'COCO':
            dataset_type = 'COCO'
        else:
            dataset_type = 'VOC'
        return dataset_type


    def get_mAP_metric(self, dataset_type):
        if dataset_type == 'COCO':
            mAP = CocomAP(self.dataset.classes, self.dataset)
        else:
            mAP = MeanAp(len(self.dataset.classes))
        return mAP

    def print_mAP(self, quiet=False):
        if self.dataset_type == 'VOC':
            mAP, aps = self.mAP.calc_mean_ap()

            for cls, ap in enumerate(aps):
                cls_name = self.dataset.decode_class(cls)

                if not quiet:
                    print("AP(%s) = %.3f" % (cls_name, ap))

            if not quiet:
                print("mAP = %.3f" % mAP)

            self.mAP.reset()
        else:
            self.mAP.calc_mean_ap()
            self.mAP.print_eval()

            # Iou=0.50:0.95 | area= All | maxDets=100
            mAP = self.mAP.get_AP_AllAverage()

        self.save_score(mAP)

        return mAP

    def save_score(self, score):
        info = {}

        info['has_score'] = True
        info['mAP'] = score
        info['evaluation_dataset'] = self.args.dataset

        self.model.set_info(info)

        self.score = score

    def get_score(self):
        return self.score

    def get_dataset(self):
        return self.args.dataset

    def match(self, y_, y, img_ids, img_sizes):
        if self.dataset_type == 'VOC':
            for a, b in zip(y_, y):
                self.mAP.match(a, b)
        else:
            self.mAP.collect_pred_data(y, y_, img_ids, img_sizes)

    def init_dataset(self, model):
        args = self.args

        # In case of COCO dataset, yolo evaluation has only 80 class label,
        # instead of 90.
        is_yolo = model.is_yolo_post_process() or (self.model.info['num_class'] == 80)

        if args.force_resize is None:
            size = model.get_input_size()[-2:]
        else:
            size = args.force_resize

        t = []
        if args.enable_letterbox:
            t.append(transforms.LetterBox())

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
                               is_yolo=is_yolo,
                               class_label = labels)

        return dataset

    def init_dataloader(self):
        args = self.args

        if args.num_workers < 0:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = args.num_workers
        
        collate = self.get_collate(self.args.dataset)

        return DataLoader(self.dataset,
                          pin_memory=True,
                          batch_size=args.batch_size,
                          num_workers=num_workers,
                          collate_fn=collate)
    
    def get_post_process(self):
        args = self.args
        model = self.model

        if args.import_post_process is None:
            return None

        with open(args.anchor_file, "rb") as f:
            bytebuffer = bytearray(f.read())
            anchor = torch.tensor(bytebuffer).view(-1, 4).float() / 256.0

        opts = model.get_optimization()

        if 'PostProcessParameterFold' in opts:
            var = 1.
        else:
            var = args.variance

        post_process_layer = DetectPostProcess(anchor=anchor,
                                               num_class=args.num_class,
                                               var=var,
                                               th_conf=args.th_conf,
                                               th_iou=args.th_iou,
                                               background=not args.no_background,
                                               logistic=args.logistic
                                               )

        if torch.cuda.is_available():
            post_process_layer = post_process_layer.cuda()

        return post_process_layer

    def get_collate(self, dataset_type):
        if dataset_type == 'COCO':
            collate = self.collate_coco
        else:
            collate = self.collate_voc
        return collate

    @staticmethod
    def collate_voc(batch):
        imgs = []
        targets = []

        for (img, target) in batch:
            imgs.append(img)
            targets.append(target)

        return torch.stack(imgs, 0), targets

    @staticmethod
    def collate_coco(batch):
        imgs = []
        targets = []
        img_ids = []
        img_sizes = []

        for (img, target, img_id, img_size) in batch:
            imgs.append(img)
            targets.append(target)
            img_ids.append(img_id)
            img_sizes.append(img_size)

        return torch.stack(imgs, 0), targets, img_ids, img_sizes
