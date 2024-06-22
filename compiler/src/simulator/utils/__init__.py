from .trainer_callback import TrainerCallback
from .mean_ap import MeanAp
from .accuracy import Accuracy

import os


def build_dir(model_name, dump_root, model_type='detector'):
    model_name = os.path.basename(model_name)
    model_name, _ = os.path.splitext(model_name)

    dump_root = os.path.join(dump_root, model_type, model_name)
    
    os.path.exists(dump_root) or os.makedirs(dump_root)

    return dump_root


def get_file_names(dump_root, input_names=[]):
    names = []

    for input_name in input_names:
        input_name = os.path.basename(input_name)
        input_name, _ = os.path.splitext(input_name)
        names.append(dump_root + "/" + input_name)

    return names