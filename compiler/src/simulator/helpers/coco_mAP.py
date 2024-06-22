from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

from collections import defaultdict
import simulator.datasets.coco_detection as coco_detection

import numpy as np

import time
import copy

import json
from tempfile import NamedTemporaryFile

class SubCocoEval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt = cocoGt              # ground truth COCO API
        self.cocoDt = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image, category evaluation
        self.eval = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = SubParams(iouType=iouType)  # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


class SubParams(Params):
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05))
                                   + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01))
                                   + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2],
                        [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []

        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05))
                                   + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01))
                                   + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2],
                        [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79,
                                        .79, .72, .72, .62, .62, 1.07,
                                        1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None


class CocomAP:
    def __init__(self, classes, dataset):
        self.classes = classes
        ann_file = dataset.ann_file
        self.cocoGt = COCO(ann_file)

    def init(self):
        self.img_ids = []
        self.target = []

    def collect_pred_data(self, y, y_, img_ids, img_sizes):
        for gts, pds, img_id, img_size in zip(y, y_, img_ids, img_sizes):
            self.img_ids.append(img_id)
            for class_val, objs in enumerate(pds):
                for obj in objs:
                    if len(self.classes) == 80:
                        class_id = coco_detection.class_91.index((coco_detection.class_80[class_val]))
                    else:
                        class_id = class_val
                    img_w, img_h = img_size
                    x1 = (obj[0] * img_w)
                    y1 = (obj[1] * img_h)
                    box_w = (obj[2] * img_w) - x1
                    box_h = (obj[3] * img_h) - y1

                    bbox = [x1, y1, box_w, box_h]
                    self.target.append({
                        'score': obj[4],
                        'category_id': class_id + 1,
                        'bbox': bbox,
                        'image_id': img_id,
                        })

    def print_eval(self):
        self.cocoEval.summarize()

    def calc_mean_ap(self):
        with NamedTemporaryFile(suffix='.json') as tf:
            content = json.dumps(self.target).encode(encoding='utf-8')
            tf.write(content)
            res_path = tf.name
            self.cocoDt = self.cocoGt.loadRes(res_path)
        self.cocoEval = SubCocoEval(self.cocoGt, self.cocoDt, 'bbox')
        self.cocoEval.params.imgIds  = self.img_ids
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()

    def get_AP_AllAverage(self):
        return self.cocoEval.stats[0]
