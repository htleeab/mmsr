import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import CONV_LAYERS
from mmcv.utils import print_log
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

import deform_conv_ext

VISUALIZE_OFFSET = False # offset visualization will slow down the program
if VISUALIZE_OFFSET:
    import os, glob
    import torchvision
    OFFSET_IMG_DIR = '../video/offset'
    VISUALIZE_OFFSET_PER_GROUP = True


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1,
                deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError(f'Expected 4D tensor as input, got {input.dim()}' 'D tensor instead.')
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, offset, weight)

        output = input.new_empty(
            DeformConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] % cur_im2col_step) == 0, 'im2col step must divide batchsize'
            deform_conv_ext.deform_conv_forward(input, weight, offset, output,
                                                ctx.bufs_[0], ctx.bufs_[1], weight.size(3),
                                                weight.size(2), ctx.stride[1], ctx.stride[0],
                                                ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                                                ctx.dilation[0], ctx.groups, ctx.deformable_groups,
                                                cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] % cur_im2col_step) == 0, 'im2col step must divide batchsize'

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_ext.deform_conv_backward_input(
                    input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0],
                    weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1],
                    ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups,
                    ctx.deformable_groups, cur_im2col_step)

            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_ext.deform_conv_backward_parameters(
                    input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1],
                    weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1],
                    ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups,
                    ctx.deformable_groups, 1, cur_im2col_step)

        return (grad_input, grad_offset, grad_weight, None, None, None, None, None)

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be '
                             f'{"x".join(map(str, output_size))})')
        return output_size


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_ext.modulated_deform_conv_forward(input, weight, bias, ctx._bufs[0], offset,
                                                      mask, output, ctx._bufs[1], weight.shape[2],
                                                      weight.shape[3], ctx.stride, ctx.stride,
                                                      ctx.padding, ctx.padding, ctx.dilation,
                                                      ctx.dilation, ctx.groups,
                                                      ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_ext.modulated_deform_conv_backward(
            input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight,
            grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3],
            ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
            ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None,
                None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation *
                                                  (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation *
                                                (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


deform_conv = DeformConvFunction.apply
modulated_deform_conv = ModulatedDeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()

        assert not bias
        assert in_channels % groups == 0, \
            f'in_channels {in_channels} is not divisible by groups {groups}'
        assert out_channels % groups == 0, \
            f'out_channels {out_channels} is not divisible ' \
            f'by groups {groups}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        # To fix an assert error in deform_conv_cuda.cpp:128
        # input image is smaller than kernel
        input_pad = (x.size(2) < self.kernel_size[0] or x.size(3) < self.kernel_size[1])
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
        out = deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation,
                          self.groups, self.deformable_groups)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) - pad_w].contiguous()
        return out


@CONV_LAYERS.register_module('DCN')
class DeformConvPack(DeformConv):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            dilation=_pair(self.dilation), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation,
                           self.groups, self.deformable_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, DeformConvPack loads previous benchmark models.
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(prefix[:-1] +
                                                                           '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix + 'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                         '_offset.bias')

        if version is not None and version > 1:
            print_log(f'DeformConvPack {prefix.rstrip(".")} is upgraded to '
                      'version 2.', logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride,
                                     self.padding, self.dilation, self.groups,
                                     self.deformable_groups)


@CONV_LAYERS.register_module('DCNv2')
class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2
    
    def __init__(self, *args, extra_offset_mask=False, layer_name='', **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.extra_offset_mask = extra_offset_mask
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            dilation=_pair(self.dilation), bias=True)
        self.init_offset()
        if VISUALIZE_OFFSET:
            self.offset_output_folder = os.path.join(OFFSET_IMG_DIR, layer_name)
            os.makedirs(self.offset_output_folder)
            if VISUALIZE_OFFSET_PER_GROUP:
                for i in range(0, self.deformable_groups):
                    os.makedirs(os.path.join(self.offset_output_folder, str(i)))

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        if self.extra_offset_mask:
            # x = [input, features]
            out = self.conv_offset(x[1])
            x = x[0]
        else:
            out = self.conv_offset(x)
        # out [1, C, H, W], C= deformable group x 3 x kernal_H x kernal_W, 3-> x,y,m
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 100:
            logger.warning('Offset mean is {}, larger than 100.'.format(offset_mean))
        if VISUALIZE_OFFSET:
            _, C, H, W = out.shape
            offset_view = offset.view([self.deformable_groups, 3, 3, 2, H, W])
            offset_max = torch.max(torch.abs(offset))
            # simply display the max of all channels 8 groups *3*3 (kernel size)
            # offset is groups x kernel x kernel x 2 x H, W, guess from modulated_deformable_im2col_gpu_kernel
            offset_x = torch.max(torch.abs(offset_view[:, :, :, 0, :, :]).view([-1, H, W]), dim=0)[0] / H
            offset_y = torch.max(torch.abs(offset_view[:, :, :, 1, :, :]).view([-1, H, W]), dim=0)[0] / W
            # Warning
            offset_x_max_all = torch.max(offset_x)
            offset_y_max_all = torch.max(offset_y)
            if offset_x_max_all > H or offset_y_max_all > W:
                logger.warning(
                    f'Offset exceed image shape: {offset.shape}\t- offset: {offset_mean:>8.4f}\t({offset_x_max_all:>8.4f},{offset_y_max_all:>8.4f})'
                )
            ## save image
            padding = torch.zeros([H, W], dtype=offset.dtype).to(offset.get_device())
            offset_tensor_img = torch.stack([offset_x, padding, offset_y])
            # simply append via counting frame
            # Note: Each frame output N=5 offset image because PCD run N times per frame
            next_index = len(glob.glob(os.path.join(self.offset_output_folder, '*.png')))
            torchvision.utils.save_image(
                offset_tensor_img, os.path.join(self.offset_output_folder, f'{next_index:05d}.png'))
            if VISUALIZE_OFFSET_PER_GROUP:
                for group_idx in range(0, self.deformable_groups):
                    group_out_folder = os.path.join(self.offset_output_folder, str(group_idx))
                    offset_x = torch.max(torch.abs(offset_view[group_idx, :, :, 0, :, :]).view([-1, H, W]), dim=0)[0] / H
                    offset_y = torch.max(torch.abs(offset_view[group_idx, :, :, 1, :, :]).view([-1, H, W]), dim=0)[0] / W
                    offset_tensor_img = torch.stack([offset_x, padding, offset_y])
                    torchvision.utils.save_image(
                        offset_tensor_img,
                        os.path.join(group_out_folder,
                                     f'{len(os.listdir(group_out_folder)):05d}.png')
                    )

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride,
                                     self.padding, self.dilation, self.groups,
                                     self.deformable_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, ModulatedDeformConvPack
            # loads previous benchmark models.
            if prefix + 'conv_offset.weight' not in state_dict:
                if prefix[:-1] + '_offset.weight' in state_dict:
                    state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(prefix[:-1] +
                                                                               '_offset.weight')
                elif prefix + 'conv_offset_mask.weight' in state_dict:
                    state_dict[prefix +
                               'conv_offset.weight'] = state_dict.pop(prefix +
                                                                      'conv_offset_mask.weight')
            if prefix + 'conv_offset.bias' not in state_dict:
                if prefix[:-1] + '_offset.bias' in state_dict:
                    state_dict[prefix + 'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                             '_offset.bias')
                elif prefix + 'conv_offset_mask.bias' in state_dict:
                    state_dict[prefix +
                               'conv_offset.bias'] = state_dict.pop(prefix +
                                                                    'conv_offset_mask.bias')

        if version is not None and version > 1:
            print_log(f'ModulatedDeformConvPack {prefix.rstrip(".")} is upgraded to '
                      'version 2.', logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)
