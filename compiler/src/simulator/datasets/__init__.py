import os

from torchvision.datasets import ImageNet
from .voc_detection import VOCDetection
from .coco_detection import CocoDetection
from .imagenet import ImageNet
from .waymo_detection import WAYMODetection
from .culane import CULane
from .custom_image import CustomImage
from .custom_detection import CustomDetection


def load_dataset(args, image_set='train', download=True,
                 transforms=None, is_yolo=False, class_label=None):
    dataset = args.dataset
    root = args.dataset_root
    if dataset == 'Custom_detection':

        return CustomDetection(root,transforms,class_label)

    elif dataset[0:3] == 'VOC':
        if dataset == 'VOC2007':
            year = ('2007',)
        elif dataset == 'VOC2012':
            year = ('2012',)
        elif image_set == 'test':
            year = ('2007',)
        else:
            year = ('2007', '2012')

        return VOCDetection(root,
                            year=year,
                            image_set=image_set,
                            download=download,
                            transforms=transforms,
                            )

    elif dataset[0:4] == 'COCO':
        if dataset == "COCO2014":
            year = ('2014',)
        elif dataset == "COCO2017":
            year = ('2017',)
        elif image_set == 'test':
            image_set = 'val'
            year = ('2017',)
        else:
            year = ('2014', '2017')

        return CocoDetection(root, year,
                             image_set=image_set,
                             transforms=transforms,
                             is_yolo=is_yolo,
                             download=download)
    elif dataset == 'ImageNet':
        if image_set == 'test':
            image_set = 'val'
        else:
            raise NotImplementedError()

        return ImageNet(os.path.join(root, 'imagenet'),
                        image_set=image_set,
                        download=download,
                        transforms=transforms
                        )

    elif dataset == 'WAYMO':
        if image_set == 'test':
            image_set = 'val'
        else:
            raise NotImplementedError()

        return WAYMODetection(os.path.join(root, 'waymo'),
                              image_set=image_set,
                              transforms=transforms
                              )

    elif dataset == 'CULane':
        if image_set not in ('train', 'val', 'test'):
            raise NotImplementedError()

        return CULane(os.path.join(root, 'CULane'),
                      image_set=image_set
                      )

    elif dataset == 'Custom_classification':
        return CustomImage(root,
                           transforms=transforms,
                           num_images=args.num_images,
                           class_label=class_label
                           )
    elif dataset == 'Custom':
        return CustomImage(root,
                           transforms=transforms,
                           num_images=args.num_images,
                           class_label=class_label
                           )

    raise Exception("Unknown dataset %s" % dataset)
