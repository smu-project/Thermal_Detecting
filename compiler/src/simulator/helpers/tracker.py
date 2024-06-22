import os
import sys
import multiprocessing
import torch

from torch.utils.data import DataLoader

from ..datasets import load_dataset
from ..transforms import detector_transforms as transforms

from enlight import quantizations


class Tracker:
    def __init__(self, args, model, is_regress=False):
        self.args = args

        self.model = model

        self.is_regress = is_regress

        self.dataset = self.init_dataset()
        self.dataloader = self.init_dataloader()

        self.data_cnt = len(self.dataset)

    def __call__(self):
        self.model.eval()

        if self.args.enable_track:
            self.model.track(per_channel=self.args.track_per_channel)

        if self.args.collect_histogram:
            self.model.collect_histogram()

        num_input = self.model.get_num_input_layer()

        cnt = 0

        for i, x in enumerate(self.dataloader):
            if torch.cuda.is_available():
                if num_input > 1:
                    x = [x.cuda() for _ in range(num_input)]
                else:
                    x = x.cuda()

            print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

            with torch.no_grad():
                self.model(x)

            if num_input > 1:
                cnt += x[0].size(0)
            else:
                cnt += x.size(0)

            if self.args.num_images > 0 and cnt > self.args.num_images :
                break

        if self.args.collect_histogram:
            print(f'Generate activation threshold with histogram...', end='\r')
            self.model.generate_act_threshold()
            print(f'Generate activation threshold with histogram... Done', end='\n')

        print("[%d / %d]" % (cnt, self.data_cnt), end='\n')

    def init_dataset(self):
        args = self.args

        n, c, h, w = self.model.get_input_size()

        t = []
        if args.enable_letterbox:
            t.append(transforms.LetterBox())

        t.extend([transforms.Resize((h, w)),
                  transforms.ToTensor()])
        
        if self.is_regress:
            t.append(transforms.ExpandChannel(c))
        else:
            if args.disable_fuse_normalization:
                t.append(transforms.Normalize(args.mean, args.std))
            else:
                t.append(transforms.QuantFriendlyNormalize(
                    scale=quantizations.input_scale)
                )

        t = transforms.Compose(t)

        dataset = load_dataset(args,
                               image_set=args.image_set, 
                               download=args.download, 
                               transforms=t)

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

    def get_num_images(self):
        return self.data_cnt
    
    @staticmethod
    def collate(batch):
        imgs = []

        for b in batch:
            img = b[0]

            imgs.append(img)

        return torch.stack(imgs, 0)

