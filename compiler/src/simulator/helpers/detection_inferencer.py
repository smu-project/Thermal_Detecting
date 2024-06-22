import torch

from enlight import nn
from enlight import quantizations

from ..utils.box_utils import draw_object_box
from ..datasets.custom_image import CustomImage
from ..transforms import detector_transforms as transforms

from ..utils import get_file_names, build_dir


class DetectionInferencer:
    def __init__(self, args, model):
        self.args = args

        self.model = model
        model.set_post_process_threshold(args.th_conf, args.th_iou)

        self.dataset = self.init_dataset(model)

        self.data_cnt = len(self.dataset)

        self.dump_dir = build_dir(args.model, args.result_root)

    def __call__(self):
        args = self.args

        file_names = get_file_names(self.dump_dir, args.inputs)

        self.model.eval()

        cnt = 0

        for i, imgs in enumerate(self.dataset):
            x, _, org = imgs

            if torch.cuda.is_available():
                x = x.cuda()

            print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

            with torch.no_grad():
                self.show_results(org, self.model(x.unsqueeze(0)), file_name=file_names[i])

            cnt += 1

        print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

    def get_class_labels(self):
        return self.model.get_class_label()

    def init_dataset(self, model):
        args = self.args

        t = []

        if args.force_resize is None:
            size = model.get_input_size()[-2:]
        else:
            size = args.force_resize

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

        return CustomImage(args.inputs, transforms=t)

    def show_results(self, org, results, file_name=''):
        objs = []
        for _cls, _objs in enumerate(results[0]):
            if not _objs:
                continue
        
            class_labels = self.get_class_labels()
            if class_labels is not None:
                num_labels = len(class_labels)

                if _cls > num_labels:
                    print(f"[WARN] class index ({_cls}) is greather than number of labels ({num_labels}), so put just class index instead label")
                    label = str(_cls)
                else:
                    label = class_labels[_cls].rstrip('\n')
            else:
                label = str(_cls)
        
            for _obj in _objs:
                _obj.append(label)
                _obj.append(_cls)
                objs.append(_obj)

        img = draw_object_box(org, objs, file_name=file_name)

        if self.args.enable_show:
            img.show()