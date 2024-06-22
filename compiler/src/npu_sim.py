import os, sys
import argparse

import logging

import torch

from enlight import get_config, apply_model_config
from enlight import EnlightParser, Writer
from enlight import Optimizer, optimizations

from simulator.utils.box_utils import draw_object_box, draw_class
from simulator.transforms import detector_transforms as transforms
from simulator.datasets import CustomImage

from simulator.utils.npu_utils import NPUDataGenerator


def create_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename, mode='w')
    logger.addHandler(file_handler)

    return logger

def check_model_info(net):
    if not net.is_quantized():
        print("[Error] network should be quantized", file=sys.stderr)

        sys.exit(1)


def get_imageloader(args, model):
    if args.force_resize is None:
        size = model.get_input_size()[-2:]
    else:
        size = args.force_resize

    t = []

    if args.enable_letterbox:
        t.append(transforms.LetterBox())

    t.append(transforms.Resize(size, use_cv2=args.use_cv2))

    if args.crop:
        t.append(transforms.CenterCrop(args.crop))

    t = transforms.Compose(t)

    return CustomImage(args.input, 
                       transforms=t,
                       use_cv2=args.use_cv2)


def evaluation_npu(model, args, hw_config):
    if args.perf_file is None:
        fname = args.model[0:args.model.rfind('.')] + '_perf.csv'
    else:
        fname = args.perf_file

    logger = create_logger(fname)

    opts = optimizations.PerformanceGenerator(hw_config['wbuf_size'],
                                              hw_config['gbuf_size'],
                                              hw_config['bias_width'],
                                              hw_config['align_activation'],
                                              hw_config['num_core'],
                                              hw_config['hw_version'],
                                              logger=logger)

    try:
        with torch.no_grad():
            with Optimizer(opts=[opts]) as optimizer:
                optimizer.optimize(model.cpu())

        perf = opts.get_results()

        bw = perf['bandwidth']

        load_act = bw['load_act']
        save_act = bw['save_act']
        load_param = bw['load_param']
        total = bw['total']

        if logger is not None:
            logger.info(f'')
            logger.info(f"SUM(Bytes) {load_act} {load_param} {save_act} {total}")
            logger.info(f"SUM(MBytes) {load_act/1024./1024.:.2f} {load_param/1024./1024.:.2f} {save_act/1024./1024.:.2f} {total/1024./1024.:.2f}")

        print("")
        print(f"========= Bandwidth Summary ==========")
        print(f"LOAD_ACT : {bw['load_act']/1024/1024:.2f} Mbytes")
        print(f"SAVE_ACT : {bw['save_act']/1024/1024:.2f} Mbytes")
        print(f"LOAD_WGT : {bw['load_param']/1024/1024:.2f} Mbytes")
        print(f"TOTAL    : {bw['total']/1024/1024:.2f} Mbytes")
        print("")

        print(f"Save results: {fname}")
        print("")

    except:
        print("[WARN] NPU simulation run successfully, but can't show NPU performance some reason")


def prepare_model(args, hw_config):
    if args.model is None:
        print(f'[Error] put model file ex) ssdlite_300_quantized.enlight', file=sys.stderr)

        sys.exit(1)

    if not os.path.exists(args.model):
        print(f'{args.model} is not exist', file=sys.stderr)

        sys.exit(1)

    with EnlightParser() as parser:
        net = parser.parse(args.model)

    check_model_info(net)

    hw_config = get_config(args.hw_config)

    if args.image_format == 'ARGB':
        opts = [optimizations.ExpandFirstConvLayer()]

        with Optimizer(opts=opts) as optimizer:
            net = optimizer.optimize(net)

    if not args.disable_opts:
        prologue_opts = [optimizations.ComposeActivation(),
                         optimizations.ComposeSinglePattern(),
                         optimizations.FuseLayers()]

        with Optimizer(opts=prologue_opts) as optimizer:
            net = optimizer.optimize(net)

        opts = []

        if args.load_group_structure is not None:
            opts.extend(Optimizer.get_customize_group_opts(hw_config,
                                                           file=args.load_group_structure))
        else:
            opts.extend(Optimizer.get_delimiter_opts(hw_config, args.disable_group_layer))

        if not args.disable_partition and args.save_group_structure is None:
            opts.extend(Optimizer.get_partition_opts(hw_config))

        with Optimizer(opts=opts) as optimizer:
            net = optimizer.optimize(net)

        if args.save_group_structure is not None:
            save_group_structure(net, args)

        epilogue_opts = [optimizations.DecomposeActivation(keep_layer_name=False),
                         optimizations.DecomposeSinglePattern()]

        # bypass delimiter layer for v1 code generator
        if hw_config['hw_version'] == 1:
            epilogue_opts.append(optimizations.OmitDelimiter())

        with Optimizer(opts=epilogue_opts) as optimizer:
            net = optimizer.optimize(net)

    if torch.cuda.is_available():
        net = net.cuda()

    net.mark_input_layer()
    net.mark_output_layer()

    net.set_post_process_threshold(args.th_conf, args.th_iou)

    net.eval()

    return net


def save_group_structure(model, args):
    filename = args.save_group_structure

    model.save_group_structure(filename)


def save_model(net, args):
    filename = args.output

    net = net.cpu()

    path_dir = os.path.dirname(filename) or '.'

    os.makedirs(path_dir, exist_ok=True)

    with open(args.output, 'wb') as f:
        writer = Writer()
        writer.write(net, f)


def show_arguments(args):
    args_dict = vars(args)
    
    for k, v in args_dict.items():
        # do not show items of developer only
        if k in ['disalbe-opts', 'disable_partition', 'load_group_structure', \
                'save_group_structure', 'hw_config', 'model_config_root', 'disable_opts',
                'dump_all', 'disable_group_layer', 'debug']:
            continue

        print(f'{k:40s}: {v}')

    print('')


# for FPGA verification
def generate_npu_input(img, image_format, data_generator, input_ch_pad):
    '''
        img : resized RGB with PIL object
    '''

    t = []

    t.append(transforms.TensorFromImage(is_input=False))

    if image_format == 'RGB':
        pass
    elif image_format == 'ARGB':
        t.append(transforms.ColorSpaceConvert('RGBToARGB'))
    elif image_format == 'IYU2':
        t.append(transforms.ColorSpaceConvert('RGBToIYU2'))
    else:
        raise Exception(f'[Error] Not supported image format {image_format}')

    if input_ch_pad == 'repeat':
        _, c, _, _ = data_generator.model.get_input_size()
        t.append(transforms.ExpandChannel(c))

    t = transforms.Compose(t)

    data_generator.generate_npu_input(img, transforms=t)


def prepare_qsim_input(image_format, data_generator, input_ch_pad):
    t = []

    if image_format == 'IYU2':
        t.append(transforms.ColorSpaceConvert('IYU2ToRGB'))
    
    t.append(transforms.MakeNPUInput())

    if input_ch_pad == "repeat":
        _, c, _, _ = data_generator.model.get_input_size()
        t.append(transforms.ExpandChannel(c))

    t = transforms.Compose(t)

    return data_generator.get_npu_input(transforms=t)


def show_results(img, results, model_type, class_labels, enable_show, file_name=''):
    if class_labels is not None:
        num_labels = len(class_labels)

    if model_type == 'obj':
        # print results
        objs = []
        for _cls, _objs in enumerate(results[0]):
            if not _objs:
                continue

            if class_labels is not None:
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
        
        img = draw_object_box(img, objs, file_name=file_name)

        if enable_show:
            img.show()
    elif model_type == 'class':
        shape = list(results.shape)

        if not (len(shape) > 2 and shape[2] == 1 and shape[3] == 1):
            print("")
            print(f"[INFO] Simulation success. But activation shape (N, C, 1, 1) or (N, C) is only available (current shape is ({list(shape)})")
            print("")

            return

        _, results_ = results.topk(1, 1)
        results_ = results_.flatten()

        _cls = results_.item()

        if class_labels is not None:
            if _cls > num_labels:
                print(f"[WARN] class index ({_cls}) is greather than number of labels ({num_labels}), so put just class index instead label")
                label = str(_cls)
            else:
                label = class_labels[results_]
        else:
            label = str(_cls)

        img = draw_class(img, label, _cls, file_name=file_name)

        if enable_show:
            img.show()
    elif model_type == 'lanedet':
        print("")
        print("[INFO] Inference done, lane detection post-processing is not implemented")
        print("")
    else:
        print("")
        print("[INFO] Inference done, {} type model has no post-processing".format(model_type))
        print("")

def inference(args, model, x):
    if torch.cuda.is_available():
        x = x.cuda()

    return model(x)


def single_run(args, model, x, img, data_generator):
    '''
        x : resized image (PIL)
        img : org image (PIL) for rendering 
    '''

    image_format = args.image_format

    generate_npu_input(x, image_format, data_generator, args.input_ch_pad)

    x = prepare_qsim_input(image_format, data_generator, args.input_ch_pad)

    with torch.no_grad():
        results = inference(args,
                            model, 
                            x)

    if args.debug:
        show_results(img, results,
                    model.get_type(),
                    model.get_class_label(),
                    args.enable_show,
                    data_generator.get_input_path())
    else:
        try:
            show_results(img, results,
                        model.get_type(),
                        model.get_class_label(),
                        args.enable_show,
                        data_generator.get_input_path())

        except Exception as e:
            print("")
            print(f"Error reason: {e}")
            print("[WARN] Simulation success, but can't show results properly becuase of some unexpected error")
            print("")

    data_generator.unregister_hook()
    data_generator.gather()


def main():
    parser = argparse.ArgumentParser(description='npu simulation')
    parser.add_argument('--model', default='ssdlite.enlight',
                        help='Detector model file')
    # model config

    parser.add_argument('--model-config', type=str, default='auto',
                        help='model configuration file')

    parser.add_argument('--output', '-o', type=str,
                        default=None,
                        help='output filename')

    parser.add_argument('input',
                        help='Input image path')

    parser.add_argument('--image-format', default='RGB', choices=['ARGB', 'RGB', 'IYU2'],
                        type=str, help='Input image Format')

    parser.add_argument('--th-conf', default=0.5, type=float,
                        help='Confidence Threshold')
    parser.add_argument('--th-iou', default=0.5, type=float,
                        help='IOU Threshold')

    # FPGA verification
    parser.add_argument('--dump', default=False,
                        action='store_true',
                        help='dump input & output activation for NPU verification')
    parser.add_argument('--dump-root', default="./dump",
                        help='root for dump data')

    # Input transform
    parser.add_argument('--enable-letterbox', default=False, 
                        action='store_true',
                        help='Enable letterboxing image')
    parser.add_argument('--force-resize', default=None, type=int,
                        help='resize specific size \
                              if resize is none, resize with input shape')
    parser.add_argument('--crop', default=None, type=int,
                        help='crop specific size')
    parser.add_argument('--use-cv2', default=False,
                        action='store_true',
                        help='use open-cv package for image pre-processing')

    parser.add_argument('--enable-show', default=False, action='store_true',
                        help='show results with image viewer')

    # NPU performance
    parser.add_argument('--show-perf', default=False, action='store_true',
                        help='show results with npu performance data')

    parser.add_argument('--perf-file', type=str, default=None, 
                        help='filename of performance results')

    # Below options is only for Openedges developer
    parser.add_argument('--model-config-root', type=str, default='./input',
                        help=argparse.SUPPRESS)
    parser.add_argument('--disable-opts', default=False, action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--disable-partition', default=False, action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--disable-group-layer', default=False, action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--load-group-structure', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--save-group-structure', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--dump-all', default=False,
                        action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--hw-config', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--debug', default=False, action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--input-ch-pad', default=None,
                        choices=['repeat'],
                        help=argparse.SUPPRESS)


    tool_name = 'simulator'
    # override model configurations
    parser = apply_model_config(tool_name,
                                parser_handler=parser)

    args = parser.parse_args()

    print("*"*80)
    print("   NPU simulation. start.")
    print("*"*80)

    #show_arguments(args)

    hw_config = get_config(args.hw_config)

    model = prepare_model(args, hw_config)

    data_generator = NPUDataGenerator(model,
                                      hw_version=hw_config['hw_version'],
                                      dump_all_layers=args.dump_all)

    image_loader = get_imageloader(args, model)


    for x, _, img in image_loader:
        # use rgb
        input_file = image_loader.get_fname()

        data_generator.make_dump_dir(model=args.model,
                                     input=input_file,
                                     dump_root=args.dump_root)

        if args.dump or args.dump_all:
            data_generator.prepare()

        single_run(args, model, x, img, data_generator)

    if args.show_perf and not args.disable_opts:
        evaluation_npu(model, args, hw_config)

    if args.output:
        save_model(model, args)

    print("")
    print("NPU simulator. done.")


if __name__ == "__main__":
    main()

    sys.exit(0)
