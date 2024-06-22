import os, sys
import argparse

from enlight import get_config, apply_model_config, export_model_config
from enlight import load_network, Writer, optimizations
from enlight import EnlightParser, Optimizer, Quantizer
from enlight.sanity_checker_for_custom_qparam import SanityCheckerForCustomQParam
from enlight import quantizations


def check_model_info(args, net):
    if not net.is_tracked():
        print('No exist activation statistics data', file=sys.stderr)

        sys.exit(1)

    if args.use_histogram:
        if not net.has_histogram():
            print('No exist activation histogram data', file=sys.stderr)

            sys.exit(1)


def prepare_model(args):
    if args.model is None:
        print(f'[Error] put model file ex) ssdlite_300.enlight', file=sys.stderr)

        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f'{args.model} is not exist', file=sys.stderr)

        sys.exit(1)

    net = load_network(args.model)

    net.mark_input_layer()
    net.mark_last_conv2d()
    net.mark_conf_layer()

    check_model_info(args, net)

    opts = [optimizations.BatchormalizeFold(),
            optimizations.FuseConstantEltwLayer()]

    with Optimizer(opts) as optimizer:
        net = optimizer.optimize(net)

    qparams = generate_qparams(args)

    if args.scale_2n:
        quantizer = quantizations.QuantSymmetricPow2
    else:
        quantizer = quantizations.QuantSymmetric

    net_name = args.model[0:args.model.rfind('.')]

    if (args.scales_file or args.qbits_file) is not None and args.enable_sanity_checker:
        sanity_checker = SanityCheckerForCustomQParam(
                                                      args.scales_file,
                                                      args.qbits_file,
                                                      args.custom_qparam_type,
                                                      net_name,
                                                      net)
        print("Start sanity check for custom quantization")
        sanity_checker()

    with Quantizer(quantizer=quantizer,
                   qparam=qparams,
                   stats_file=args.stats_file,
                   scales_file=args.scales_file,
                   qbits_file=args.qbits_file,
                   verbose=args.verbose,
                   custom_qparam_type=args.custom_qparam_type,
                   overwrite_concat_qscale=args.overwrite_concat_qscale) as quantizer:
        net = quantizer.quantize(net)

    # dump scale before optimization
    if args.dump_scales:
        net.save_scales(net_name + '_scale.json', args.dump_type)
    if args.dump_qbits:
        net.save_qbits(net_name + '_qbit.json', args.dump_type)

    hw_config = get_config(config_file=args.hw_config)

    if not args.disable_hw_friendly_opt:
        opts = [optimizations.MakeCompilerFriendly(hw_version=hw_config['hw_version'])]
        with Optimizer(opts) as optimizer:
            net = optimizer.optimize(net)

    return net


def save_model(net, args):
    if args.output is None:
        filename = args.model[0:args.model.rfind('.')] + '_quantized.enlight'
    else:
        filename = args.output

    net = net.cpu()

    path_dir = os.path.dirname(filename) or '.'

    os.makedirs(path_dir, exist_ok=True)

    with open(filename, 'wb') as f:
        writer = Writer()
        writer.write(net, f)


def show_arguments(args):

    args_dict = vars(args)
    
    for k, v in args_dict.items():
        # do not show items of developer only
        if k in ['disable_hw_friendly_opt', 'use_histogram', 'verbose', \
                'hw_config', 'model_config_root', 'export_model_config']:
            continue

        print(f'{k:40s}: {v}')

    print('')

def generate_qparams(args):
    if args.m_std_ratio is None:
        args.m_std_ratio = 1.0 if args.scale_2n else 1.6

    if args.m_std_8 is None:
        args.m_std_8 = 11 if args.scale_2n else 10

    qparams = {}

    qparams['m_std_8'] = args.m_std_8
    qparams['m_std_4'] = args.m_std_4
    qparams['m_std_ratio'] = args.m_std_ratio
    qparams['iter_weight_mean_correction'] = args.iter_weight_mean_correction
    qparams['clip_min_max'] = not args.disable_clip_min_max
    qparams['quantize_post_process'] = args.quantize_post_process
    qparams['use_histogram'] = args.use_histogram

    return qparams

def main():
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

    # quantization information
    parser.add_argument('--stats-file', default=None, type=str,
                        help='load activation statistics')
    parser.add_argument('--scales-file', default=None, type=str,
                        help='load scales with file')
    parser.add_argument('--qbits-file', default=None, type=str,
                        help='load quantization bit information with file')
    parser.add_argument('--dump-scales', default=False,
                        action='store_true',
                        help='dump quantization scale to json file')
    parser.add_argument('--dump-qbits', default=False,
                        action='store_true',
                        help='dump quantization bit information to json file')
    parser.add_argument('--dump-type',default='enlight_name',
                        choices=['enlight_name', 'origin_name', 'origin_id'],
                        help='decision quantization information json key type')

    # custom quantization option
    parser.add_argument('--custom-qparam-type', default='enlight_name',
                        choices=['enlight_name', 'origin_name', 'origin_id'],
                        type=str, help='Custom quantization parameter key value type')
    parser.add_argument('--enable-sanity-checker', default=False,
                        action='store_true',
                        help='Enable sanity checker for custom quantization')
    parser.add_argument('--overwrite-concat-qscale', default=False,
                        action='store_true',
                        help='Overwrite custom concat and source conv layers quantization scale to enlight quantization scale')

    # quantization parameter
    parser.add_argument('--scale-2n', default=False,
                        action='store_true',
                        help='symmetric quantization with 2^n scale')
    parser.add_argument('--m-std-8', default=11,  type=float,
                        help='parameter for generating maximum value for scale (only 8-bit quantization)')
    parser.add_argument('--m-std-4', default=5,  type=float,
                        help='parameter for generating maximum value for scale (only 4-bit quantization)')
    parser.add_argument('--m-std-ratio', default=1.0,  type=float,
                        help='ratio of m_std between Conv and Conv+activation')
    parser.add_argument('--iter-weight-mean-correction', default=0,  type=int,
                        help='Number of iteration by weight mean correction')
    parser.add_argument('--disable-clip-min_max', default=False,
                        action='store_true',
                        help='clip maximum value for scale with min/max if m_std > 0')
    parser.add_argument('--quantize-post-process', default=False,
                        action='store_true',
                        help='quantization post process layer if exist')

    # Below options is only for Openedges developer
    parser.add_argument('--model-config-root', type=str, default='./input',
                        help=argparse.SUPPRESS)
    parser.add_argument('--export-model-config', default=None, type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument('--disable-hw-friendly-opt', default=False,
                        action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--use-histogram', default=False,
                        action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--verbose', default=False, action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--hw-config', type=str, default=None,
                        help=argparse.SUPPRESS)

    tool_name = 'quantizer'
    # override model configurations
    parser = apply_model_config(tool_name,
                              parser_handler=parser)

    args = parser.parse_args()

    print("*"*80)
    print("   Quantization. start")
    print("*"*80)

    #show_arguments(args)
    
    if args.export_model_config is not None:
        export_model_config(args,
                            file=args.export_model_config,
                            tool_name=tool_name)

    # prepare model
    model = prepare_model(args)

    # save model
    save_model(model, args)

    print("")
    print("Quantizer done.")


if __name__ == "__main__":
    main()

    sys.exit(0)
