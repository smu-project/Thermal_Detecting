import os

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from ..utils.box_utils import draw_object_box

classes = ('vehicle', 'pedestrian', 'sign', 'cyclist')


class WAYMODetection(Dataset):

    def __init__(self,
                 root,
                 image_set='val',
                 transforms=None,
                 ):

        self.root = root
        self.image_set = image_set
        self.classes = classes

        image_list = os.path.join(self.root, image_set + '.txt')

        self.ids = []
        for line in open(image_list):
            name, _ = os.path.splitext(line)

            self.ids.append(name)

        self.transforms = transforms

    def __getitem__(self, index):
        idx = os.path.join(self.root, self.image_set, self.ids[index])

        img = self.get_image(idx)
        target = self.get_target(idx)

        if False:
            img_ = draw_object_box(img, target)
            img_.show()

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_image(self, name):
        img = Image.open(name + '.jpg').convert('RGB')

        return img

    def get_target(self, name):
        targets = []

        with open(name + '.txt', 'r') as f:
            lines = f.readlines()

            for line in lines:
                l = line.split() 

                label = int(l[0])

                # the 'sign' class is not exist in network,
                # but the number of label in dataset is '3'
                if label == 3:
                    label = 2

                box = l[1:]

                # Normalized xywh to pixel xyxy format
                center_x = float(box[0])
                center_y = float(box[1])
                width = float(box[2])
                height = float(box[3])

                x_min = center_x - width/2
                y_min = center_y - height/2
                x_max = center_x + width/2
                y_max = center_y + height/2

                targets.append([x_min, y_min, x_max, y_max, label])

        return targets

    def __len__(self):
        return len(self.ids)

    def decode_class(self, class_id):
        return self.classes[class_id]
