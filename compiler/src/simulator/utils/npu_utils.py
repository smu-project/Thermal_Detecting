import os
import shutil
import tarfile

import torch

from enlight import nn


class NPUDataGenerator():
    def __init__(self, model=None, is_regress=False, hw_version=1, dump_all_layers=False):
        self.model = model

        self.is_regress = is_regress
        self.hw_version = hw_version

        self.dump_all_layers = dump_all_layers if not is_regress else False

        self.npu_input = None

        self.model_bname = ''
        self.input_bname = ''
        self.dump_dir = ''
        self.dump_dir_post_process = ''

        self.npu_align = dict()

        self.hooks_registered = []

        self.gen_hw_configuration()

    def gen_hw_configuration(self):
        self.npu_align['IN'] = 16

        if self.hw_version == 1:
            self.npu_align['IACT'] = 2
            self.npu_align['OACT'] = 1
        elif self.hw_version >= 2.:
            self.npu_align['IACT'] = 1
            self.npu_align['OACT'] = 1
        else:
            raise Exception(f'[Error] Not supported NPU HW version [{self.hw_version}]')

    def make_dump_dir(self, model='', input='', dump_root='./'):
        if model != '':
            model_name = model
            model_name = os.path.basename(model_name)
            model_name, _ = os.path.splitext(model_name)
        else:
            model_name = ''

        dump_dir = os.path.join(dump_root, model_name)
        os.path.exists(dump_dir) or os.makedirs(dump_dir)

        if not self.is_regress:
            input_name = input
            input_name = os.path.basename(input_name)
            input_name, _ = os.path.splitext(input_name)
        else:
            # Not make directory having name input 
            input_name = ''

        if model_name != '':
            dump_dir = os.path.join(dump_root, model_name, input_name)
            os.path.exists(dump_dir) or os.makedirs(dump_dir)

        if model_name != '' and not self.is_regress:
            post_process_dir = "post_process/python_data"
            post_process_dir = "{}/{}".format(dump_dir, post_process_dir)

            os.path.exists(post_process_dir) or os.makedirs(post_process_dir)

            self.dump_dir_post_process = post_process_dir

        self.dump_dir = dump_dir
        self.model_bname = model_name
        self.input_bname = input_name

        self.input = input

        print('\nSave results (dump) path : {}\n'.format(self.get_dump_root()))

    def get_dump_root(self):
        return self.dump_dir

    def get_input_path(self):
        file_name = os.path.join(self.dump_dir, self.input_bname)
    
        return file_name

    def prepare(self):
        def dump_activation_4d(m, inp, out, in_bits=8, out_bits=8):
            align = self.npu_align
            is_regress = self.is_regress

            is_input = m.is_input if not is_regress else False
            is_output = m.is_output if not is_regress else False

            input_word_align = False if is_input else True
            input_align = align['IN'] if is_input else align['IACT']
            output_align = align['OACT']

            # if fused activation is already input qmodule
            if hasattr(m, 'act') and isinstance(m.act, nn.q.QModule):
                m.is_input = m.act.is_input

            # input
            if m.is_input or not is_regress:
                if len(inp) > 1:
                    for i, inp_act in enumerate(inp):
                        fn = self.get_name(layer=m.name,
                                           param='IACT',
                                           multi_input=True,
                                           multi_idx=i)

                        print_act_mem(inp[i], fn,
                                      word_align=input_word_align,
                                      align=input_align,
                                      num_bits=in_bits)
                else:
                    fn = self.get_name(layer=m.name,
                                       param='IACT')

                    print_act_mem(inp[0], fn,
                                    word_align=input_word_align,
                                    align=input_align,
                                    num_bits=in_bits)

            # output
            fn = self.get_name(layer=m.name,
                               param='OACT')

            print_act_mem(out, fn,
                          align=output_align,
                          output_mode=is_output,
                          num_bits=out_bits)

        def dump_virtual_dma(m, inp, out):
            align = self.npu_align
            is_regress = self.is_regress

            dma_mode = m.mode

            is_input = True if dma_mode == 'input' else False
            is_output = True if dma_mode == 'output' else False

            input_word_align = False if is_input else True
            input_align = align['IN'] if is_input else align['IACT']

            if m.is_input or not is_regress:
                # input
                fn = self.get_name(m.name, param='IACT')
                print_act_mem(inp[0], fn,
                            word_align=input_word_align,
                            align=input_align)

            # output
            fn = self.get_name(layer=m.name,
                               param='OACT')

            print_act_mem(out, fn,
                          align=OACT_ALIGN,
                          output_mode=is_output,
                          im2_col=m.im2col)

        def dump_parameter(m):
            # weight
            dw = self.is_dw(m)

            fn = self.get_name(m.name, param='W')
            print_w_mem(m.weight.data, dw, fn,
                        num_bits=m.qbit)

            # bias
            if m.bias is not None:
                quantizer = m.quantizer

                b_int, b_trunc = quantizer.convert_nbits_bias(m.bias)

                m_int = quantizer.get_scale('m_int')
                trunc = quantizer.get_scale('hw_trunc')

                fn = self.get_name(m.name, param='B')
                print_b_mem(b_int,
                            b_trunc,
                            trunc,
                            m_int,
                            fn,
                            self.hw_version)
            else:
                raise Exception("")

        def dump_registered_tensor(m):
            dump_buffer = m.dump_buffer
            post_process_dir = self.dump_dir_post_process

            for k, v in dump_buffer.items():
                fn = k
                fn_dict = v

                dump_format = fn_dict['format']
                dump_tensor = fn_dict['data']

                file_path = "{}/{}.bin".format(post_process_dir, fn)
    
                with open(file_path, "wb") as f:
                    dump_tensor = dump_tensor.data.cpu().numpy().astype(dump_format)
                    dump_tensor.tofile(f)

        def hook_layer(m, inp, out):
            if isinstance(m, nn.q.QSigmoid):
                # [FIXME]
                # Sigmoid can be output layer instead of concat
                fn = self.get_name(m.name,
                                   param='OACT')

                if out.dim() == 3 and m.is_output:
                    print_sig_mem(out, fn, output_mode=True)
                else:
                    dump_activation_4d(m, inp, out)
            else:
                dump_activation_4d(m, inp, out)

        def hook_QConv2d(m, inp, out):
            qbit = m.qbit if self.is_regress else 8

            dump_activation_4d(m, inp, out, in_bits=qbit)
            dump_parameter(m)

        def hook_QConcat(m, inp, out):
            # Only dump output activation
            fn = self.get_name(m.name,
                               param='OACT')

            print_concat_mem(out, fn, output_mode=m.is_output)

        def hook_VirtualDma(m, inp, out):
            dump_virtual_dma(m, inp, out)

        def hook_VirtualBitConverter(m, inp, out):
            dump_activation_4d(m, inp, out, in_bits=8, out_bits=4)

        def hook_Postprocess(m, inp, out):
            dump_registered_tensor(m)

        model = self.model

        for m in model.modules():
            # Dump not nested instance only
            if hasattr(m, 'name') and m.name is None:
                continue

            if not hasattr(m, 'is_output'):
                continue

            if m.is_output or self.dump_all_layers:
                if isinstance(m, nn.q.QMaxPool2d) or \
                    isinstance(m, nn.q.QMaxPool2d) or \
                    isinstance(m, nn.q.QAvgPool2d) or \
                    isinstance(m, nn.q.QGlobalAvgPool2d) or \
                    isinstance(m, nn.q.QAdd) or \
                    isinstance(m, nn.q.QMul) or \
                    isinstance(m, nn.q.QMulConst) or \
                    isinstance(m, nn.q.Resize) or \
                    isinstance(m, nn.q.QSigmoid) or \
                    (isinstance(m, nn.VirtualScaler) and \
                     not isinstance(m, nn.VirtualBitConverter)):

                    self.register_hook(m, hook_layer)

                elif isinstance(m, nn.q.QConv2d):
                    self.register_hook(m, hook_QConv2d)

                elif isinstance(m, nn.q.QConcat):
                    self.register_hook(m, hook_QConcat)

                elif isinstance(m, nn.VirtualDma):
                    self.register_hook(m, hook_VirtualDma)

                elif isinstance(m, nn.VirtualBitConverter):
                    self.register_hook(m, hook_VirtualBitConverter)

                elif isinstance(m, nn.post_process.QYoloDetectionPostProcess) or \
                        isinstance(m, nn.post_process.YoloDetectionPostProcess) or \
                        isinstance(m, nn.post_process.QDetectPostProcess) or \
                        isinstance(m, nn.post_process.DetectPostProcess):
                    m.debug_flag = True

                    self.register_hook(m, hook_Postprocess)

    def generate_npu_input(self, x, transforms=None, fname='input'):
        if transforms is not None:
            y, _ = transforms(x, [])

        fn = self.get_name(fname,
                           param='IACT')

        print_act_mem(y.unsqueeze(0),
                      fn, word_align=False, align=self.npu_align['IN'])

        self.npu_input = y

    def get_npu_input(self, transforms=None):
        x = self.npu_input

        if transforms is not None:
            y, _ = transforms(x, [])

        return y.unsqueeze(0)
    
    def get_name(self, layer, param='IACT', multi_input=False, multi_idx=0):
        ''' 
            param in ['IACT', 'OACT', 'W', 'B']
        '''

        name = 'test' if self.is_regress else layer

        if param == 'IACT':
            ext = '.ia'

            if multi_input:
                ext = ext + str(multi_idx)

        elif param == 'OACT':
            ext = '.oa'
        elif param == 'W':
            ext = '.w'
        elif param == 'B':
            ext = '.b'
        
        ext = ext + '.bin'

        full_name = '{}/{}{}'.format(self.dump_dir, name, ext)

        return full_name

    def is_dw(self, inst):
        in_channels = inst.in_channels
        groups = inst.groups
        out_channels = inst.out_channels

        dw = (in_channels == groups) and (in_channels == out_channels)

        if dw and in_channels == 1:
            dw = False

        return dw

    def register_hook(self, inst, hook):
        handle = inst.register_forward_hook(hook)

        self.hooks_registered.append(handle)

    def unregister_hook(self):
        for handle in self.hooks_registered:
            handle.remove()

    def gather(self):
        shutil.copy(self.input, self.dump_dir)

def print_concat_mem(acts, fn, little_end=True, word_align=True, output_mode=False):
    if acts.dim() == 4:
        print_act_mem(acts, fn, output_mode=output_mode)
    else:
        try:
            b, x, c = acts.shape
        except:
            # linear layer
            b, c = acts.shape
            x = 1

            aligned_c = (c + 31) & (~31)
            zero_acts = torch.zeros(b, aligned_c)

            zero_acts[..., :c] = acts[..., :c]

            acts = zero_acts

            c = aligned_c

        if c % 32:
            raise Exception("Not allowed case in Concat layer output activation")

        acts = acts.view(b, x, -1, 32)   # b,x,c/N,N

        if not output_mode:
            acts = acts.permute(0, 2, 1, 3)  # b,c/N,x,N

        dump_act = acts.data.cpu().numpy().astype('int8')

        with open(fn, "wb") as f:
            dump_act.tofile(f)


def print_sig_mem(acts, fn, little_end=True, word_align=True, output_mode=False):
    try:
        b, x, c = acts.shape
    except:
        # linear layer
        b, c = acts.shape
        x = 1

        aligned_c = (c + 31) & (~31)
        zero_acts = torch.zeros(b, aligned_c)

        zero_acts[..., :c] = acts[..., :c]

        acts = zero_acts

        c = aligned_c

    if c % 32:
        raise Exception("Not allowed case in Concat layer output activation")

    acts = acts.view(b, x, -1, 32)   # b,x,c/N,N

    dump_act = acts.data.cpu().numpy().astype('int8')

    with open(fn, "wb") as f:
        dump_act.tofile(f)


def print_act_mem(acts, fn, little_end=True, word_align=True, align=1, output_mode=False, im2_col=False,
                  num_bits=8):
    if not acts.shape:
        return

    align_ch = 32 if num_bits == 8 else 64

    b, c, y, x = acts.shape

    if word_align:
        new = torch.zeros(b, c, y, ALIGN_N(x, align))
        new[..., 0:x] = acts[..., 0:x]
    else:
        new = acts

    act_align_w = new.permute(0, 2, 3, 1)         # b,y,x,c
    b, y, x, c = act_align_w.shape

    if im2_col:
        act_align_w = act_align_w.reshape(b, y, 1, -1)
        b, y, x, c = act_align_w.shape

    if word_align:
        new = torch.zeros(b, y, x, ALIGN_N(c, align_ch))
        new[..., 0:c] = act_align_w[..., 0:c]

        act_align_w_c = new.view(b, y, x, -1, align_ch)   # b,y,x,c/N,N

        if not output_mode:
            act_align_w_c = act_align_w_c.permute(0, 3, 1, 2, 4)  # b,c/N,y,x,N

    dump_act = act_align_w_c if word_align else act_align_w.unsqueeze(dim=1)
    if num_bits == 4:
        dump_act = pack_byte_format(dump_act)

    # 16-byte align for RGB
    if not word_align:
        b, cn, y, _, _ = dump_act.shape
        dump_act = dump_act.reshape(b, cn, y, -1)
        size = dump_act.shape[-1]

        new = torch.zeros(b, cn, y, ALIGN_N(size, align))
        new[..., 0:size] = dump_act[..., 0:size]
        dump_act = new

    dump_act = dump_act.data.cpu().numpy().astype('int8')

    with open(fn, "wb") as f:
        dump_act.tofile(f)


def print_w_mem(w, dw, fn, little_end=True, word_align=True, num_bits=8): # RSKC
    align_ch = 32 if num_bits == 8 else 64

    w_rskc = w.permute(2, 3, 1, 0) if dw else w.permute(2, 3, 0, 1)

    r, s, k, c = w_rskc.shape

    if word_align:
        new = torch.zeros(r, s, k, ALIGN_N(c, align_ch))
        new[..., 0:c] = w_rskc[..., 0:c]

        w_align = new.view(r, s, k, -1, align_ch)   # R S K C -> R S K C/N N
        w_align = w_align.permute(3, 0, 1, 2, 4)  #         -> C/N R S K N

    dump_w = w_align if word_align else w.unsqueeze(dim=0)
    if num_bits == 4:
        dump_w = pack_byte_format(dump_w)

    dump_w = dump_w.data.cpu().numpy().astype('int8')

    with open(fn, "wb") as f:
        dump_w.tofile(f)


def print_b_mem(params, scale, trunc, m_int,
                fn, hw_version=1, little_end=True, word_align=True):
    trunc = trunc.unsqueeze(0)
    m_int = m_int.unsqueeze(0)

    if (trunc.dim() != 1):
        size = params.size(dim=0)
        trunc = trunc.view(size)
        m_int = m_int.view(size)

    size = trunc.size(dim=0)

    if size == 1: # per-layer
        trunc = trunc.repeat(params.size(dim=0))
        scale = scale.repeat(params.size(dim=0))
        m_int = m_int.repeat(params.size(dim=0))
        size = params.size(dim=0)

    if hw_version == 1:
        dump_bias = ((params.int() & 0b11111111) << 8)   \
                    | ((scale.int() & 0b1111) << 4) \
                    | (trunc.int() & 0b1111)
    else:
        dump_bias = ((m_int.int() & 0b11111111) << 8) \
                    | ((params.int() & 0b11111111) << 24) \
                    | ((scale.int() & 0b1111) << 20) \
                    | ((trunc.int() & 0b1111) << 16)

    if word_align:
        if size % 32 is not 0:
            dummy = torch.zeros([32 - (size % 32)], device=params.device)

            dump_bias = torch.cat((dump_bias.float(), dummy), 0)

        dump_bias = dump_bias.view(-1, 2, 16)
    else:
        dump_bias = dump_bias.unsqueeze(dim=0).unsqueeze(dim=0)

    dump_bias = dump_bias.data.cpu().numpy()

    if hw_version == 1:
        dump_bias = dump_bias.astype('uint16')
    else:
        dump_bias = dump_bias.astype('uint32')

    with open(fn, "wb") as f:
        dump_bias.tofile(f)


def ALIGN_N(val, align):
    align = align - 1

    return (val + align) & (~align)


def pack_byte_format(org, num_bits=4):
    b, c, y, x, N = org.shape
    new = torch.zeros(b, c, y, x, N//2)

    mask = 2**num_bits - 1

    for i in range(N//2):
        idx = 2*i

        low = org[:, :, :, :, idx].int() & mask
        high = org[:, :, :, :, idx+1].int() & mask
        high = high << 4

        new[:, :, :, :, i] = low | high

    return new
