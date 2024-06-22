import os, sys

import argparse
import torch

from enlight import get_config, apply_model_config, export_model_config, get_spec
from enlight import Optimizer, Writer, load_network
from enlight import optimizations
from enlight import AnalyzeCompatibility
from enlight.exceptions import CompatibilityError

from simulator.helpers import Tracker

import json

import logging


def create_logger(filename):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")

    filehandler = logging.FileHandler(filename, "w")
    stdohandler = logging.StreamHandler(sys.stdout)

    filehandler.setFormatter(formatter)
    stdohandler.setFormatter(formatter)

    logger.addHandler(filehandler)
    logger.addHandler(stdohandler)

    return logger


def prepare_model(args):
    if args.preset == "enlight":
        args.add_detection_post_process = 'auto'

        args.mean = (0.5, 0.5, 0.5)
        args.std = (0.5, 0.5, 0.5)

    if args.preset == "darknet":
        if args.weight is None:
            weight = args.model[0:args.model.rfind('.')] + '.weights'

            if os.path.exists(weight):
                args.weight = weight

        args.mean = (0.0, 0.0, 0.0)
        args.std = (1.0, 1.0, 1.0)

    if args.add_detection_post_process == 'auto':
        anchor = args.model[0:args.model.rfind('.')] + '.anchor'

        if os.path.exists(anchor):
            args.add_detection_post_process = anchor
        else:
            args.add_detection_post_process = None

    net = load_network(args.model, args.weight)

    net.mark_conf_layer()

    if args.add_detection_post_process:
        add_detection_post_process(net, args)

    hw_config = get_config(args.hw_config)

    opts = Optimizer.get_default_opts(hw_config)

    if args.omit_post_process:
        opts.append(optimizations.OmitPostProcess())

    is_overriding_input = args.force_input and len(args.force_input)
    is_overriding_output = args.force_output and len(args.force_output)

    if is_overriding_input:
        input_layer_names = args.force_input
        opts.append(optimizations.ForceLayerAsInput(input_layer_names))

    if is_overriding_output:
        output_layer_names = args.force_output
        opts.append(optimizations.ForceLayerAsOutput(output_layer_names))

    if args.enable_yolo_output_decomposing:
        opts.append(optimizations.PostProcessSrcConvTransform())

    if not args.disable_fuse_normalization and \
        not is_overriding_input and \
        args.input_ch_pad != "repeat":
        opts.append(optimizations.InputQNormalizeFold(
            mean=args.mean,
            std=args.std))

    if not args.disable_compensation_fuse_norm:
        opts.append(optimizations.ExposePadLayer(mean=args.mean))

    opts.append(optimizations.DecomposeActivation())
    opts.extend(Optimizer.get_quantization_aware_opts(hw_config))

    optimizer = Optimizer(opts)

    net = optimizer.optimize(net)

    net.make_up_layers_name()
    net.mark_output_layer()

    return net


def add_detection_post_process(net, args):
    with open(args.add_detection_post_process, "rb") as f:
        bytebuffer = bytearray(f.read())
        anchor = torch.tensor(bytebuffer).view(-1, 4).float() / 256.0

    net.add_detection_post_process(anchor, args.num_class,
                                   order=args.output_order,
                                   var=args.variance,
                                   background=not args.no_background,
                                   logistic=args.logistic)
    
    info = dict()

    info['has_detection_layer'] = True
    net.set_info(info)


def track_model(net, args, input_ch_pad=None):
    if torch.cuda.is_available():
        net = net.cuda()

    # validate dataset & print result
    is_padded_input_ch = False
    if input_ch_pad == 'repeat':
        is_padded_input_ch = True
    tracker = Tracker(args, net, is_padded_input_ch)
    tracker()

    return tracker


def process_for_quantization(net, args):
    prologue_opts = [optimizations.ComposeActivation(),
                    optimizations.FuseLayers()]
    epilogue_opts = [optimizations.DecomposeActivation(keep_layer_name=True)]

    optimization_applied = True
    applied_layer = None

    while(True):
        opts = optimizations.LayerChannelEqualization(applied_layer=applied_layer)

        optimizer = Optimizer(prologue_opts + [opts] + epilogue_opts)
        
        with torch.no_grad():
            net = optimizer.optimize(net.cpu())

        optimization_applied = not opts.is_activated()

        if not optimization_applied:
            break

        applied_layer = opts.get_applied_layer()
        print(f'[INFO] converter.py - {applied_layer}')

        # args.track_per_channel = False
        net.reset_track_data()

        tracker = track_model(net, args, args.input_ch_pad)

    return net


def get_class_label(file):
    with open(file, 'r') as f:
        labels = f.readlines()

    labels = [label.strip() for label in labels]

    if len(labels) == 0:
        raise Exception("can't find class labels")

    return labels

def set_net_info(net, tracker, args):
    if args.dataset in 'Custom':
        dataset = '{}_{}'.format(args.dataset, args.num_images)
    else:
        dataset = '{}_{}'.format(args.dataset, args.image_set)

    info = dict()

    info['model'] = args.model
    info['type'] = args.type

    is_yolo = net.is_yolo_post_process()
    post_process = net.get_post_process_layer()

    if post_process is not None:
        num_class = post_process.num_class

        if args.num_class > 0 and num_class != args.num_class:
            if is_yolo:
                print(f"[INFO] Number of class ({num_class}) is pre-defined in darknet cfg, so ignore --num-class ({args.num_class})")
            else:
                print(f"[WARN] Mismatch number of class ({args.num_class}) and class of post process ({num_class})")
                print(f"[WARN] so, ignore argument about --num-class")
    else:
        num_class = args.num_class

    info['num_class'] = num_class

    if args.class_labels is not None:
        class_labels = get_class_label(args.class_labels)

        num_class_labels = len(class_labels)

        if not is_yolo and args.num_class != num_class_labels:
            print(f"[WARN] Mismatch number of class ({args.num_class}) and labels ({num_class_labels})")
            print("")
    else:
        class_labels = None

    info['class_labels'] = class_labels

    info['is_tracked'] = args.enable_track
    info['has_histogram'] = args.collect_histogram

    if info['is_tracked'] or info['has_histogram']:
        info['track_dataset'] = dataset

        if args.num_images < 0:
            num_images = tracker.get_num_images()
        else:
            num_images = args.num_images
        info['num_images'] = num_images

    info['is_fused_normalization'] = not args.disable_fuse_normalization
    info['norm_mean'] = args.mean
    info['norm_std'] = args.std

    net.set_info(info)


def save_model(net, tracker, args):
    if args.output is None:
        filename = args.model[0:args.model.rfind('.')] + '.enlight'
    else:
        filename = args.output

    net = net.cpu()

    set_net_info(net, tracker, args)

    path_dir = os.path.dirname(filename) or '.'

    os.path.exists(path_dir) or os.makedirs(path_dir, exist_ok=True)

    with open(filename, 'wb') as f:
        writer = Writer()
        writer.write(net, f)

    if args.dump_stats:
        filename = args.model[0:args.model.rfind('.')] + '_stat.json'
        net.save_stats(filename)


def show_arguments(args):
    args_dict = vars(args)

    for k, v in args_dict.items():
        # do not show items of developer only
        if k in ['preset', 'collect_histogram', 'force_input', \
                'hw_config', 'model_config_root', 'export_model_config',
                'enable_yolo_output_decomposing', 'disable_fuse_normalization',
                'disable_compensation_fuse_norm']:
            continue

        print(f'{k:40s}: {v}')

    print('')


def check_arguments(args, model=None):
    print("")
    print("Checking arguments...", end='\r')

    file_list = [args.model,
                 args.weight,
                 args.add_detection_post_process,
                 args.class_labels,
                 args.dataset_root]

    for f in file_list:
        if f is None:
            continue

        if not os.path.exists(f):
            print(f'{f} is not exist', file=sys.stderr)

            sys.exit(1)

    network_type = args.type
    num_class = args.num_class

    if model is None:
        if network_type is None:
            print(f"[Error] --type is mandatory, choose from 'obj', 'class', 'lanedet', 'unknown'", file=sys.stderr)

            sys.exit(1)

    else:
        exist_yolo_post_process = model.is_yolo_post_process()
        exist_post_process = model.get_post_process_layer()

        if network_type == 'obj':
            if not exist_post_process:
                print(f"[Error] if --type is 'obj', post process layer should exist", file=sys.stderr)

                sys.exit(1)

            elif exist_yolo_post_process:
                if num_class != -1:
                    print(f"[Error] if network is based yolo, --num-class can't to be set", file=sys.stderr)

                    sys.exit(1)
            else:
                if num_class == -1:
                    print("[Error] --num-class should be set proper value if --type is 'obj'", file=sys.stderr)

                    sys.exit(1)

    print("Checking arguments... done")


def check_compatibility(args, net):
    if args.disable_checking_compatibility:
        return

    log_root_dir = args.compatibility_out_root
    os.path.exists(log_root_dir) or os.makedirs(log_root_dir, exist_ok=True)

    out_fp = os.path.basename(args.model)
    out_fp, _ = os.path.splitext(out_fp)
    out_fp = f"{log_root_dir}/{out_fp}_compatibility.log"

    logger = create_logger(out_fp)
    enlight_npu_spec = get_spec(args.compatibility_list)

    checker = AnalyzeCompatibility(enlight_npu_spec, logger)

    try:
        checker.check(net)
    except CompatibilityError as e:
        print(e, file=sys.stderr)
        args.enable_track = False

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        pass


def main():
    #make download directory
    createFolder("./downloads")
    parser = argparse.ArgumentParser(description='convert network')

    # model parameter
    parser.add_argument('model', default=None,
                        help='model file (ex: ssdlite.onnx)')

    # output name
    parser.add_argument('--output', '-o', type=str,
                        default=None,
                        help='output filename (default: auto)')

    # model config
    parser.add_argument('--model-config', type=str, default='auto',
                        help='model configuration file')

    parser.add_argument('--type', default=None,
                        choices=['obj', 'class', 'lanedet', 'unknown'], type=str,
                        help='purpose of model, \
                              obj = object detection, \
                              class = classification, \
                              lane = lane detection')

    parser.add_argument('--weight', default=None,
                        help='weight file (for darknet weight)')

    parser.add_argument('--mean', nargs=3, type=float,
                        default=(0.486, 0.456, 0.406),
                        help='mean for normalizing')
    parser.add_argument('--std', nargs=3, type=float,
                        default=(0.229, 0.224, 0.225),
                        help='std for normalizing')

    parser.add_argument('--add-detection-post-process', type=str, default=None,
                        help='anchor file')
    parser.add_argument('--num-class', type=int, default=-1,
                        help='number of class')
    parser.add_argument('--class-labels', type=str,
                        default=None,
                        help='class label file')
    parser.add_argument('--omit-post-process',
                        default=False,
                        action='store_true')
    parser.add_argument('--output-order', type=str,
                        choices=['lc', 'cl', 'auto'],
                        default='auto',
                        help='specify output activation order in location and confidence')
    parser.add_argument('--variance', nargs=2, type=float,
                        default=(0.125, 0.125),
                        help='variance of x and y')
    parser.add_argument('--no-background',
                        action='store_true',
                        help='Background class not exist')
    parser.add_argument('--logistic', type=str, default='softmax',
                        choices=['softmax', 'sigmoid', 'none'],
                        help="Post process logistic function")

    parser.add_argument('--force-output', type=str, nargs="*",
                        help="Names of layer to be output, ex) Conv_1 Conv_2")

    # data loader
    parser.add_argument('--dataset', default='VOC2007',
                        choices=['ImageNet', 'VOC', 'VOC2007', 'VOC2012', 'WAYMO', 'Custom', 'COCO', 'CULane'],
                        type=str, help='Dataset name')
    parser.add_argument('--dataset-root', default='downloads',
                        help='Dataset root directory path')
    parser.add_argument('--image-set', default='test', type=str,
                        choices=['test', 'train', 'val', 'trainval'],
                        help='Dataset image set')
    parser.add_argument('--download', default=False,
                        action='store_true',
                        help='Download dataset')
    parser.add_argument('--batch-size', default=4, type=int,
                        help='Batch size for training')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--enable-letterbox', default=False,
                        action='store_true',
                        help='Enable letterboxing image')

    # track parameter
    parser.add_argument('--enable-track', default=False,
                        action='store_true',
                        help='Enable tracking min/max of each layer')
    parser.add_argument('--num-images', default=1000, type=int,
                        help='Number of images for track if custom datasets')
    parser.add_argument('--dump-stats', default=False,
                        action='store_true',
                        help='dump statistics to json file')

    # Checking Layer Compatibility
    parser.add_argument('--compatibility-out-root', default='./log/compatibility_results', type=str,
                         help='Directory records compatibility with ENLIGHT NPU')

    # Below options is only for Openedges developer
    parser.add_argument('--model-config-root', type=str, default='./input',
                        help=argparse.SUPPRESS)
    parser.add_argument('--export-model-config', default=None, type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument('--preset', default=None,
                        choices=['enlight', 'darknet'],
                        help=argparse.SUPPRESS)
    parser.add_argument('--collect-histogram', default=False,
                        action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--force-input', type=str, nargs="*",
                        help=argparse.SUPPRESS)
    parser.add_argument('--hw-config', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--compatibility-list', default=None, type=str,
                         help=argparse.SUPPRESS)
    parser.add_argument('--disable-checking-compatibility', default=False,
                        action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--input-ch-pad', default=None,
                        choices=['repeat'],
                        help=argparse.SUPPRESS)
    
    ## for quantization
    parser.add_argument('--disable-compensation-fuse-norm', default=True,
                        action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--disable-fuse-normalization', default=False,
                        action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--enable-yolo-output-decomposing', default=False,
                        action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--track-per-channel', default=False,
                        action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--enable-channel-equalization', default=False,
                        action='store_true',
                        help=argparse.SUPPRESS)


    tool_name = 'converter'
    # override model configurations
    parser = apply_model_config(tool_name,
                                parser_handler=parser)


    args = parser.parse_args()

    print("*"*80)
    print("   Converting to enlight format. start.")
    print("*"*80)

    #show_arguments(args)

    if args.export_model_config is not None:
        export_model_config(args,
                            file=args.export_model_config,
                            tool_name=tool_name)

    check_arguments(args)

    # prepare model
    model = prepare_model(args)

    # check arguments with network
    check_arguments(args, model)

    # check compatibilty
    check_compatibility(args, model)

    if args.enable_track or args.collect_histogram:
        tracker = track_model(model, args, args.input_ch_pad)

        if args.track_per_channel and args.enable_channel_equalization:
            model = process_for_quantization(model, args)
    else:
        tracker = None

    # save model
    save_model(model, tracker, args)

    print("")
    print("Converter. done.")


if __name__ == "__main__":
    main()

    sys.exit(0)
