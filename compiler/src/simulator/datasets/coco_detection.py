from genericpath import isfile
import os

from torchvision.datasets import CocoDetection as CocoDetection_
import wget
import zipfile
class_80 = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', \
            'bus', 'train','truck','boat','trafficlight', 'firehydrant', \
            'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', \
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat',
            'baseballglove', 'skateboard', 'surfboard', 'tennisracket', 'bottle', \
            'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', \
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', \
            'pizza', 'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed', \
            'diningtable', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', \
            'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', \
            'book', 'clock',  'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush')

class_91 = ('person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'trafficlight',
            'firehydrant', 'streetsign', 'stopsign', 'parkingmeter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
            'umbrella', 'shoe', 'eyeglasses', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball',
            'kite', 'baseballbat', 'baseballglove', 'skateboard',
            'surfboard', 'tennisracket', 'bottle', 'plate',
            'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'pottedplant', 'bed', 'mirror', 'diningtable',
            'window', 'desk', 'toilet', 'door', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cellphone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'blender',
            'book', 'clock', 'vase', 'scissors', 'teddybear',
            'hairdrier', 'toothbrush', 'hairbrush')

image_url = "http://images.cocodataset.org/zips/val2017.zip"
annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except:
        print(" Error: Creatin directory. " + directory)

class CocoDetection:
    def __init__(self,
                 root,
                 years=('2017',),
                 image_set='train',
                 transforms=None,
                 is_yolo=False,
                 download=False):

        self.classes = class_80 if is_yolo else class_91
        self.datasets = []
        self.cnt_per_dataset = []
        self.is_yolo = is_yolo

        file_root = 'COCO/'
        if download:
        
            createFolder(os.path.join(root,file_root)) #make COCO dir
            image_zip = os.path.join(root,file_root,image_url.split("/")[-1])
            annotation_zip = os.path.join(root,file_root,annotation_url.split("/")[-1])
            print(image_zip)
            if not os.path.isfile(image_zip):
                wget.download(image_url,image_zip)
            with zipfile.ZipFile(image_zip, 'r') as existing_zip:
                existing_zip.extractall(os.path.join(root,file_root))  

            if not os.path.isfile(annotation_zip):
                wget.download(annotation_url,annotation_zip)
            with zipfile.ZipFile(annotation_zip, 'r') as existing_zip:
                existing_zip.extractall(os.path.join(root,file_root))


        if image_set not in ('train', 'val', 'test'):
            raise Exception('unknown image set {}'.format(image_set))

        if not isinstance(years, list) and not isinstance(years, tuple):
            years = (years,)

        for year in years:
            image_path = os.path.join(root, file_root + image_set+year)
            self.ann_file = os.path.join(root, file_root + "annotations/instances_"
                                    + image_set+year + ".json")

            d = CocoDetection_(image_path, annFile=self.ann_file)

            self.datasets.append(d)
            self.cnt_per_dataset.append(len(d))

        self.transforms = transforms

    def __getitem__(self, index):
        for dataset, cnt in zip(self.datasets, self.cnt_per_dataset):
            if index < cnt:
                img, objs = dataset[index]
                img_id = dataset.ids[index]
                break

            index -= cnt

        size = img.size

        w = float(size[0])
        h = float(size[1])

        target = []

        for obj in objs:
            label = obj['category_id'] - 1
            if self.is_yolo:
                label = self.classes.index(class_91[label])

            x1 = float(obj['bbox'][0]) / w
            y1 = float(obj['bbox'][1]) / h
            x2 = float(obj['bbox'][0] + obj['bbox'][2]) / w
            y2 = float(obj['bbox'][1] + obj['bbox'][3]) / h

            target.append([x1, y1, x2, y2, label])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, img_id, size

    def __len__(self):
        return sum(self.cnt_per_dataset)
