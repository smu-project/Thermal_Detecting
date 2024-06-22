import os, sys

import argparse
import torch

from enlight import apply_model_config, export_model_config
from enlight import EnlightParser, Writer
from enlight import Optimizer, optimizations
from enlight import Quantizer

from simulator.helpers import DetectionEvaluator, ClassificationEvaluator

from enlight import optimizations


def check_model_info(net):
    if net.is_object_detection():
        if not net.has_detection_layer():
            print('No exist detection layer', file=sys.stderr)

            sys.exit(1)


def prepare_model(args):
    if args.model is None:
        print(f'[Error] put model file ex) ssdlite_300.enlight', file=sys.stderr)

        sys.exit(1)

    if not os.path.exists(args.model):
        print(f'{args.model} is not exist', file=sys.stderr)

        sys.exit(1)

    with EnlightParser() as parser:
        net = parser.parse(args.model)

    check_model_info(net)

    opts = []

    if args.import_post_process:
        opts.append(optimizations.OmitPostProcess())

    if len(opts) > 0:
        with Optimizer(opts) as optimizer:
            net = optimizer.optimize(net)

    net.mark_output_layer()

    if torch.cuda.is_available():
        net = net.cuda()

    return net


def show_arguments(args):
    args_dict = vars(args)
    
    for k, v in args_dict.items():
        # do not show items of developer only
        if k in ['model_config_root', 'export_model_config']:
            continue

        print(f'{k:40s}: {v}')

    print('')


def check_arguments(args, model=None):
    print("")
    print("Checking arguments...", end='\r')

    dataset = args.dataset
    class_label = model.get_class_label()

    run_flag = True

    if dataset == 'None':
        print("[INFO] network evaluation is not supported")
        print("[INFO] only support VOC, COCO, ImageNet")

        run_flag = False

    elif dataset == 'Custom':
        if model.is_classification():
            if class_label is None:
                print("[Error] Custom dataset is available in evaluator if labels of class exists")
                print("[Error] Convert network to enlight format including proper labels of classes")

                run_flag = False
        else:
            print("[Error] Custom dataset is available in evaluator if network type is class")
            print("[Error] Check network's type is class or not for evaluation")

            run_flag = False

    if not run_flag:
        sys.exit(0)

    print("Checking arguments... done")

def set_net_info(args, net):
    info = dict()

    # nothing

    net.set_info(info)


def save_model(net, args):
    filename = args.output

    net = net.cpu()

    set_net_info(args, net)

    path_dir = os.path.dirname(filename) or '.'

    os.makedirs(path_dir, exist_ok=True)

    with open(filename, 'wb') as f:
        writer = Writer()
        writer.write(net, f)

def save_instance(inst, args):
    import pickle

    # shoule be same as f_dump_inst in test_script/test_sdk.py
    filename = './temp/dumped_inst.bin'

    path_dir = os.path.dirname(filename) or '.'

    os.makedirs(path_dir, exist_ok=True)

    with open(filename, 'wb') as f:
        pickle.dump(inst, f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('model',
                        help='Detector model file (ex: ssdlite.enlight)')
    
    # model config

    parser.add_argument('--model-config', type=str, default='auto',
                        help='model configuration file')

    parser.add_argument('--output', '-o', type=str,
                        default=None,
                        help='output filename')
    parser.add_argument('--dataset', default='VOC',
                        choices=['VOC', 'COCO', 'ImageNet', 'WAYMO', 'CULane', 'Custom_classification', 'Custom_detection' ,'None'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--dataset-root', default='downloads',
                        help='Dataset root directory path')
    parser.add_argument('--download', default=False,
                        action='store_true',
                        help='Download dataset')
    parser.add_argument('--batch-size', default=4, type=int,
                        help='Batch size for training')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of workers used in dataloading')

    # input transform
    parser.add_argument('--enable-letterbox', default=False, 
                        action='store_true',
                        help='Enable letterboxing image')
    parser.add_argument('--force-resize', default=None, type=int,
                        help='resize specific size \
                              if resize is none, resize with input shape')
    parser.add_argument('--crop', default=None, type=int,
                        help='crop specific size')

    # object detection
    parser.add_argument('--th-iou', default=0.5, type=float,
                        help='IOU Threshold (only object detection)')
    parser.add_argument('--th-conf', default=0.05, type=float,
                        help='Confidence Threshold (only object detection)')

    # import post process (Only SSD)
    parser.add_argument('--import-post-process', default=None, 
                        action='store_true',
                        help='import sample code for post-process instead of build-in function in SDK (only SSD)')
    parser.add_argument('--anchor-file', default=None, type=str,
                        help='anchor file if args.import_post_process is True')
    parser.add_argument('--num-class', default=20, type=int,
                        help='the number of class if args.import_post_process is True')
    parser.add_argument('--variance', nargs=2, type=float,
                        default=(0.125, 0.125),
                        help='variance of x and y if args.import_post_process is True')
    parser.add_argument('--no-background',
                        action='store_true',
                        help='Background class not exist if args.import_post_process is True')
    parser.add_argument('--logistic', type=str, default='softmax',
                        choices=['softmax', 'sigmoid', 'none'],
                        help='Post process logistic function if args.import_post_process is True')

    # classficiation
    parser.add_argument('--has-background', default=False,
                        action='store_true',
                        help='Existence of background class (only classification)')
    parser.add_argument('--topk', type=int,
                        default=5,
                        help='topk... (only classification)')

    # Below options is only for Openedges developer
    parser.add_argument('--model-config-root', type=str, default='./input',
                        help=argparse.SUPPRESS)
    parser.add_argument('--export-model-config', default=None, type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument('--num-images', default=-1, type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument('--dump-inst', default=False, action='store_true',
                        help=argparse.SUPPRESS)


    tool_name = 'evaluator'
    # override model configurations
    parser = apply_model_config(tool_name,
                              parser_handler=parser)

    args = parser.parse_args()

    print("*"*80)
    print("Evaluate network. start.")
    print("*"*80)

    #show_arguments(args)
    
    if args.export_model_config is not None:
        export_model_config(args,
                            file=args.export_model_config,
                            tool_name=tool_name)

    # prepare model
    model = prepare_model(args)

    # check arguments with network
    check_arguments(args, model)

    # validate dataset & print result
    if model.is_object_detection():
        evaluator = DetectionEvaluator(args, model, eval_cnt=args.num_images)
    elif model.is_classification():
        evaluator = ClassificationEvaluator(args, model, eval_cnt=args.num_images)
    elif model.is_lane_detection():
        print("[Error] Not implemented evaluator of lane detection yet ", file=sys.stderr)

        sys.exit(1)
    elif model.is_unknown_type():
        print("[INFO] model is defined with 'unknown' type, so do nothing")
        sys.exit(0)
    else:
        print(f"[Error] model has undefined network-type, {model.get_type()}", file=sys.stderr)

        sys.exit(1)

    evaluator()

    if args.output:
        save_model(model, args)

    if args.dump_inst:
        save_instance(evaluator, args)

    print("")
    print("Evaluator done.")


if __name__ == "__main__":
    main()

    sys.exit(0)
