import os
import glob

import numpy as np

from PIL import Image

import xml.etree.ElementTree as ET

def read_xml(xml_path,class_name,eval_mode =1):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    image_name = root.find('filename')
    
    size = root.find('size')
    image_width = float(size.find('width').text)
    image_height = float(size.find('height').text)
    bboxes = []
    classes = []
    for obj in root.findall('object'):
        difficult = obj.find('difficult')
        label = obj.find('name').text.replace(" ","")
        bbox = obj.find('bndbox')
        
        bbox_xmin = float(bbox.find('xmin').text.split('.')[0])
        bbox_ymin = float(bbox.find('ymin').text.split('.')[0])
        bbox_xmax = float(bbox.find('xmax').text.split('.')[0])
        bbox_ymax = float(bbox.find('ymax').text.split('.')[0])
        if difficult is not None:
            difficult = int(obj.find('difficult').text)
            if((difficult == 1) and (eval_mode == 1)):
                continue
        
        if (bbox_xmax - bbox_xmin) == 0 or (bbox_ymax - bbox_ymin) == 0:
            continue
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(label)
    return image_width, image_height, bboxes, classes

def read_txt(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]
    
class CustomDetection:
    def __init__(self, dataset_root, transforms, class_label):

        if dataset_root[-1] == '/':
            root_dir = dataset_root[:-1]
        else:
            root_dir = dataset_root
        self.image_dir = root_dir +'/image/'
        self.xml_dir = root_dir +'/xml/'
        if not os.path.isdir(self.image_dir):
            raise Exception("There is no image dir")
        if not os.path.isdir(self.xml_dir):
            raise Exception("There is no xml dir")  
        print(f"class label : {class_label}")
        self.classes = class_label
        self.class_dic = {name : index for index, name in enumerate(self.classes)}      
        self.image_ids = [xml_name.replace('.xml', '') for xml_name in os.listdir(self.xml_dir)]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
        
        target = []
        
        width, height, bboxes, classes = read_xml(self.xml_dir + image_id + '.xml',self.classes) 

        for (xmin, ymin, xmax, ymax), class_name in zip(bboxes, classes):

            x1 = float(xmin) / width
            y1 = float(ymin) / height
            x2 = float(xmax) / width
            y2 = float(ymax) / height

            target.append([x1, y1, x2, y2, self.class_dic[class_name]])

        if self.transforms is not None:
            image, target = self.transforms(image, target)


        return image, target

    def decode_class(self, class_id):
        return self.classes[class_id]
