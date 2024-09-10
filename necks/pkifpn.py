# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmengine.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@MODELS.register_module()
class CAFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        atto=[],
        SC=[],
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()



        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.CA1 = FeatureSelection_CoordAtt(in_channels[0], out_channels)
        self.CA2 = FeatureSelection_CoordAtt(in_channels[1], out_channels)
        self.CA3 = FeatureSelection_CoordAtt(in_channels[2], out_channels)
        self.CA4 = FeatureSelection_CoordAtt(in_channels[3], out_channels)
        atto.append(self.CA1)
        atto.append(self.CA2)
        atto.append(self.CA3)
        atto.append(self.CA4)
        self.atto = atto

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)


            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i == len(self.lateral_convs) - 1:
                att = inputs[i + self.start_level]
                laterals.append(self.atto[i](att))
            else:
                laterals.append(lateral_conv(inputs[i + self.start_level]))

        # # build laterals
        # laterals = [
        #     lateral_conv(inputs[i + self.start_level])
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureSelection_CoordAtt(nn.Module):
    def __init__(x):

        return x


from typing import Optional, Union, Sequence
class ConvFFN(BaseModule):
    """Multi-layer perceptron implemented with ConvModule"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            hidden_channels_scale: float = 4.0,
            hidden_kernel_size: int = 3,
            dropout_rate: float = 0.,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * hidden_channels_scale)

        self.ffn_layers = nn.Sequential(
            BCHW2BHWC(),
            nn.LayerNorm(in_channels),
            BHWC2BCHW(),
            ConvModule(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(hidden_channels, hidden_channels, kernel_size=hidden_kernel_size, stride=1,
                       padding=hidden_kernel_size // 2, groups=hidden_channels,
                       norm_cfg=norm_cfg, act_cfg=None),
            GSiLU(),
            nn.Dropout(dropout_rate),
            ConvModule(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Dropout(dropout_rate),
        )
        self.add_identity = add_identity

    def forward(self, x):
        x = x + self.ffn_layers(x) if self.add_identity else self.ffn_layers(x)
        return x
import torch
class BCHW2BHWC(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 2, 3, 1])


class BHWC2BCHW(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 3, 1, 2])

class GSiLU(BaseModule):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))

from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks import DropPath
class PKIBlock(BaseModule):
    """Poly Kernel Inception Block"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            expansion: float = 1.0,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale: Optional[float] = 1.0,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        if norm_cfg is not None:
            self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
            self.norm2 = build_norm_layer(norm_cfg, hidden_channels)[1]
        else:
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(hidden_channels)

        self.block = InceptionBottleneck(in_channels, hidden_channels, kernel_sizes, dilations,
                                         expansion=1.0, add_identity=True,
                                         with_caa=with_caa, caa_kernel_size=caa_kernel_size,
                                         norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ffn = ConvFFN(hidden_channels, out_channels, ffn_scale, ffn_kernel_size, dropout_rate, add_identity=False,
                           norm_cfg=None, act_cfg=None)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        if self.layer_scale:
            if self.add_identity:
                x = x + self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = x + self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
        else:
            if self.add_identity:
                x = x + self.drop_path(self.block(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.block(self.norm1(x)))
                x = self.drop_path(self.ffn(self.norm2(x)))
        return x


class InceptionBottleneck(BaseModule):
    """Bottleneck with Inception module"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  autopad(kernel_sizes[0], None, dilations[0]), dilations[0],
                                  groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   autopad(kernel_sizes[1], None, dilations[1]), dilations[1],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv2 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                   autopad(kernel_sizes[2], None, dilations[2]), dilations[2],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                   autopad(kernel_sizes[3], None, dilations[3]), dilations[3],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv4 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[4], 1,
                                   autopad(kernel_sizes[4], None, dilations[4]), dilations[4],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)

        if with_caa:
            self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size, None, None)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and in_channels == out_channels

        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)

        y = x  # if there is an inplace operation of x, use y = x.clone() instead of y = x
        x = self.dw_conv(x)
        x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        x = self.pw_conv(x)
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x



def autopad(kernel_size: int, padding: int = None, dilation: int = 1):
    assert kernel_size % 2 == 1, 'if use autopad, kernel size must be odd'
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa

    Args:
        value (int, float): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class CAA(BaseModule):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor

from typing import Optional, Union, Sequence
class ConvFFN(BaseModule):
    """Multi-layer perceptron implemented with ConvModule"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            hidden_channels_scale: float = 4.0,
            hidden_kernel_size: int = 3,
            dropout_rate: float = 0.,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * hidden_channels_scale)

        self.ffn_layers = nn.Sequential(
            BCHW2BHWC(),
            nn.LayerNorm(in_channels),
            BHWC2BCHW(),
            ConvModule(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(hidden_channels, hidden_channels, kernel_size=hidden_kernel_size, stride=1,
                       padding=hidden_kernel_size // 2, groups=hidden_channels,
                       norm_cfg=norm_cfg, act_cfg=None),
            GSiLU(),
            nn.Dropout(dropout_rate),
            ConvModule(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Dropout(dropout_rate),
        )
        self.add_identity = add_identity

    def forward(self, x):
        x = x + self.ffn_layers(x) if self.add_identity else self.ffn_layers(x)
        return x
import torch
class BCHW2BHWC(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 2, 3, 1])


class BHWC2BCHW(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 3, 1, 2])

class GSiLU(BaseModule):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))

