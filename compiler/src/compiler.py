import argparse
import torch
import os
import sys
import traceback
import copy

from enlight import get_config, apply_model_config
from enlight import Optimizer, optimizations, Writer, load_network
from enlight.npu import Validator as Validator_V10
from enlight.npu import CodeGenerator as CodeEmitter_V10

from enlight import CodeEmitter as CodeEmitter_V20
from enlight.code_emission import Validator as Validator_V20


def prepare_model(args, target, validation_disable=False):
    hw_config = get_config(config_file=args.hw_config)

    _hw_config = copy.deepcopy(hw_config)
    if args.batch_mode:
        _hw_config['num_core'] = 1

    Validator = Validator_V10 if target == 'NPUV10' else Validator_V20
    hl_validator = Validator(target=target, low_level=False)
    ll_validator = Validator(target=target, low_level=True)

    net = load_network(args.model)

    if not args.opts_disable:
        # 1. Start prologue opts
        prologue_opts = [optimizations.ComposeActivation(),
                         optimizations.ComposeSinglePattern(),
                         optimizations.FuseLayers()]

        with Optimizer(opts=prologue_opts) as optimizer:
            net = optimizer.optimize(net)

        # 2. Start group type opts
        group_type_opts = []

        if args.load_group_structure is not None:
            fmt = "Loading group structure file path : {}\n"
            print(fmt.format(args.load_group_structure))
            group_type_opts.extend(
                    Optimizer.get_customize_group_opts(
                        _hw_config,
                        file=args.load_group_structure))
        else:
            group_type_opts.extend(Optimizer.get_delimiter_opts(_hw_config))

        with Optimizer(opts=group_type_opts) as optimizer:
            net = optimizer.optimize(net)

        if args.save_group_structure is not None:
            save_group_structure(net, args)

        # 3. Start partitioning opts
        partitioning_opts = []
        partitioning_opts.extend(Optimizer.get_partition_opts(_hw_config))

        with Optimizer(opts=partitioning_opts) as optimizer:
            net = optimizer.optimize(net)

        # 4. Start epilogue opts
        epilogue_opts = [optimizations.DecomposeActivation(keep_layer_name=False),
                         optimizations.DecomposeSinglePattern()]

        # bypass delimiter layer for v1 code generator
        if _hw_config['hw_version'] == 1:
            epilogue_opts.append(optimizations.OmitDelimiter())

        with Optimizer(opts=epilogue_opts) as optimizer:
            net = optimizer.optimize(net)

    if not validation_disable:
        hl_validator(net)

    return net


def save_group_structure(model, args):
    filename = args.save_group_structure

    model.save_group_structure(filename)


def save_model(net, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fname = os.path.join(out_dir, 'network.out.enlight')

    net = net.cpu()

    with open(fname, 'wb') as f:
        writer = Writer()
        writer.write(net, f)


def show_arguments(args):
    args_dict = vars(args)

    for k, v in args_dict.items():
        # do not show items of developer only
        if k in ['debug',
                 'hw_config',
                 'im2col_disable',
                 'validation_disable',
                 'opts_disable',
                 'model_config_root',
                 'load_group_structure',
                 'save_group_structure']:
            continue

        print(f'{k:40s}: {v}')

    print('')


def main():

    parser = argparse.ArgumentParser(description='Elight NPU code generator')

    # model parameter
    parser.add_argument('model',
                        help='model file (ex: ssdlite.enlight)')

    # output dump root_dir
    parser.add_argument('--dump-root', type=str,
                        default='./output_code',
                        help='output dump root dir, default:./output_code')

    parser.add_argument('--model-config', type=str, default='auto',
                        help='model configuration file')

    parser.add_argument('--th-conf', default=0.5, type=float,
                        help='Confidence Threshold')
    parser.add_argument('--th-iou', default=0.5, type=float,
                        help='IOU Threshold')

    parser.add_argument('--batch-mode', default=False,
                        action='store_true')

    parser.add_argument('--batch-size', default=2, type=int,
                        help=argparse.SUPPRESS)

    # Below options is only for Openedges developer
    # debug mode
    # model config: search config in model-config-root w/ network name.
    parser.add_argument('--model-config-root', type=str,
                        default='./input',
                        help=argparse.SUPPRESS)

    parser.add_argument('--debug', default=False,
                        action='store_true', help=argparse.SUPPRESS)

    parser.add_argument('--hw-config', type=str, default=None,
                        help=argparse.SUPPRESS)

    # NPU V1 only
    parser.add_argument('--im2col-disable', default=False,
                        action='store_true', help=argparse.SUPPRESS)

    parser.add_argument('--validation-disable', default=False,
                        action='store_true', help=argparse.SUPPRESS)

    # Options related with network graph
    parser.add_argument('--opts-disable', default=False,
                        action='store_true', help=argparse.SUPPRESS)

    parser.add_argument('--load-group-structure', type=str, default=None,
                        help=argparse.SUPPRESS)

    parser.add_argument('--save-group-structure', type=str, default=None,
                        help=argparse.SUPPRESS)

    tool_name = 'compiler'
    # override model configurations
    parser = apply_model_config(tool_name,
                                parser_handler=parser)

    args = parser.parse_args()

    print("*"*80)
    print(" Compiling network. start.")
    print("*"*80)

    #show_arguments(args)

    basename = os.path.basename(args.model)
    basename = os.path.splitext(basename)[0]
    output_dir = os.path.join(args.dump_root, basename)
    print('{:40s}: {}'.format('output_dir', output_dir))

    hw_config = get_config(args.hw_config)

    target = 'NPUV20' if hw_config['hw_version'] == 2 else 'NPUV10'

    if target == 'NPUV20' and args.im2col_disable:
        raise Exception("im2col_disable=True only for NPU V1 option")

    # prepare model
    model = prepare_model(args, target,
                          validation_disable=args.validation_disable)

    # save model
    if args.debug:
        save_model(model, os.path.dirname(args.model))

    # code gen
    if target == 'NPUV10':
        code_gen = CodeEmitter_V10(output_dir=output_dir,
                                   im2col_off=args.im2col_disable,
                                   th_conf=args.th_conf,
                                   th_iou=args.th_iou)
    else:
        core_num = hw_config['num_core']

        if args.batch_mode and core_num == 1:
            print("")
            print('[OptionErr] In batch_mode, core_num should be lager than 1')
            print("Compiler. done.")
            print("")
            sys.exit(1)


        code_gen = CodeEmitter_V20(model=model,
                                   network_name=args.model,
                                   output_dir=output_dir,
                                   core_num=core_num,
                                   batch_mode=args.batch_mode,
                                   batch_size=args.batch_size,
                                   th_conf=args.th_conf,
                                   th_iou=args.th_iou,
                                   version=target)


    compile_out = '{}/compile_debug.log'.format(output_dir)

    with open(compile_out, 'w') as f:

        try:
            code_gen(model, 
                     debug=args.debug)

            fmt = \
            r'''
                NPU compiler: done sucessfully.
            '''
            msg = fmt.format(args.model)

            print(msg)
            f.write('{}'.format(msg))

        except Exception as e:

            # fmt = '\033[38;5;208m{}\033[0;0m'
            # print(fmt.format(e))

            f.write('{}'.format(e))

            call_stack = traceback.format_exc()

            f.write('{}'.format(call_stack))

            if args.debug:
                print(call_stack)

            fmt = \
            r'''
                {}
                Compiler: error occurs.
                    Please contact ENLIGHT NPU SW Toolkit developer
                    with error messages file: {}
            '''
            msg = fmt.format(e, compile_out)

            f.write('{}'.format(msg))

            msg = '\033[38;5;208m{}\033[0;0m'.format(msg)
            sys.exit(msg)

    print("")
    print("Compiler. done.")


if __name__ == "__main__":
    main()
