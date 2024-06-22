import os
import multiprocessing
import torch

from torch.utils.data import DataLoader
from ..datasets import CULane
from ..post_process.laneaf_post_process import laneaf_post_process


class LaneAfEvaluator:
    def __init__(self, args, model, eval_cnt=-1):
        self.args = args

        self.model = model

        self.dataset = self.init_dataset(model)
        self.dataloader = self.init_dataloader()

        if eval_cnt > 0:
            self.data_cnt = eval_cnt
        else:
            self.data_cnt = len(self.dataset)

    def __call__(self):
        self.model.eval()

        cnt = 0

        for i, batch in enumerate(self.dataloader):
            img, input_seg = batch
            path = self.dataloader.dataset.get_path(i)

            if torch.cuda.is_available():
                img = img.cuda()
                input_seg = input_seg.cuda()

            print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

            with torch.no_grad():
                y = self.model(img)

                lanes = laneaf_post_process(y[0], y[1], y[2], input_seg)

                self.print_lanes(path, lanes)

            if cnt > self.data_cnt:
                break

            cnt += img.size(0)

        print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

    def print_lanes(self, path, lanes):
        output_dir = self.args.output

        path = path
        d = os.path.dirname(path)

        if not os.path.exists(os.path.join(output_dir, d)):
            os.makedirs(os.path.join(output_dir, d))

        with open(os.path.join(output_dir, path), 'w') as f:
            f.write('\n'.join(' '.join(map(str, _lane)) for _lane in lanes))

    def init_dataset(self, model):
        args = self.args

        quantized = model.is_quantized()

        dataset = CULane(os.path.join(args.dataset_root, 'CULane'),
                         image_set='test',
                         quantized=quantized)

        return dataset

    def init_dataloader(self):
        args = self.args

        if args.num_workers < 0:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = args.num_workers

        return DataLoader(self.dataset,
                          pin_memory=True,
                          batch_size=1,
                          num_workers=num_workers)

    @staticmethod
    def collate(batch):
        imgs = []
        segs = []
        masks = []
        afs = []

        for (img, seg, mask, af) in batch:
            imgs.append(img)
            segs.append(seg)

            masks.append(mask)
            afs.append(af)

        return torch.stack(imgs, 0), torch.stack(segs, 0), masks, afs
