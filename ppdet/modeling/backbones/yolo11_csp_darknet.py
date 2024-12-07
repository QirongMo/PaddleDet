# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn

from ppdet.core.workspace import register, serializable
from .csp_darknet import BaseConv, SPPFLayer
from .yolov8_csp_darknet import C2fLayer, BottleNeck
from .yolov10_csp_darknet import AttnLayer
from ..shape_spec import ShapeSpec

__all__ = ['YOLO11CSPDarkNet']


class C3kLayer(nn.Layer):
    """Implements a CSP Bottleneck module with three convolutions for enhanced feature extraction in neural networks."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=False,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super().__init__()
        self.c = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(
            in_channels, self.c, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, self.c, ksize=1, stride=1, bias=bias, act=act)
        self.conv3 = BaseConv(
            2*self.c, out_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            BottleNeck(
                self.c,
                self.c,
                shortcut=shortcut,
                kernel_sizes=(3, 3),
                expansion=1.0,
                depthwise=depthwise,
                bias=bias,
                act=act) for _ in range(num_blocks)
        ])

    def forward(self, x):
        y = [self.bottlenecks(self.conv1(x)), self.conv2(x)]
        return self.conv3(paddle.concat(y, 1))


class C3k2Layer(C2fLayer):
     def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=False,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu",
                 use_c3k=False):
        super(C3k2Layer, self).__init__(in_channels, out_channels, num_blocks= num_blocks, shortcut=shortcut,
                                expansion=expansion, depthwise=depthwise, bias=bias, act=act)
        if use_c3k:
            self.bottlenecks = nn.LayerList([
                C3kLayer(
                    self.c,
                    self.c,
                    num_blocks=2,
                    shortcut=shortcut,
                    expansion=1.0,
                    depthwise=depthwise,
                    bias=bias,
                    act=act) for _ in range(num_blocks)
            ])


class PSABlock(nn.Layer):
    """Partial self-attention layer named PSA in YOLOv10"""

    def __init__(self, embed_dim, num_heads=4, attn_ratio=0.5, act='silu', shortcut=True):
        super().__init__()
        self.attn = AttnLayer(embed_dim,
                              num_heads=num_heads,
                              attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(*[
            BaseConv(embed_dim,
                     2 * embed_dim,
                     ksize=1,
                     stride=1,
                     groups=1,
                     bias=False,
                     act=act),
            BaseConv(2 * embed_dim,
                     embed_dim,
                     ksize=1,
                     stride=1,
                     groups=1,
                     bias=False,
                     act=nn.Identity())
        ])
        self.shortcut = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.shortcut else self.attn(x)
        x = x + self.ffn(x) if self.shortcut else self.ffn(x)
        return x


class C2PSALayer(nn.Layer):
    def __init__(self, embed_dim,
                 num_blocks=1,
                 expansion=0.5,
                 act="silu"):
        super().__init__()
        self.hidden_dim = int(embed_dim * expansion)
        self.conv1 = BaseConv(embed_dim,
                              2 * self.hidden_dim,
                              ksize=1,
                              stride=1,
                              groups=1,
                              bias=False,
                              act=act)
        self.conv2 = BaseConv(2 * self.hidden_dim,
                              embed_dim,
                              ksize=1,
                              stride=1,
                              groups=1,
                              bias=False,
                              act=act)
        self.bottlenecks = nn.Sequential(*[PSABlock(self.hidden_dim, num_heads=self.hidden_dim//4, 
                                                    attn_ratio=0.5) for _ in range(num_blocks)])

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        out1, out2 = paddle.split(self.conv1(x), (self.hidden_dim, self.hidden_dim), axis=1)
        out2 = self.bottlenecks(out2)
        return self.conv2(paddle.concat((out1, out2), 1))



@register
@serializable
class YOLO11CSPDarkNet(nn.Layer):
    """
    YOLOv10 CSPDarkNet backbone.
    """
    __shared__ = ['depth_mult', 'width_mult', 'act']

    # in_channels, out_channels, num_blocks, use_c3k, c3k2_scale, add_shortcut, use_sppf, use_c2psa
    arch_settings = {
        's': [[64, 128, 2, False, 0.25, True, False, False],
               [128, 256, 2, False, 0.25, True, False, False],
               [256, 512, 2, True, 0.5, True, False, False],
               [512, 1024, 2, True, 0.5, True, True, True]],
        'm': [[64, 128, 2, True, 0.25, True, False, False],
               [128, 256, 2, True, 0.25, True, False, False],
               [256, 512, 2, True, 0.5, True, False, False],
               [512, 1024, 2, True, 0.5, True, True, True]],
    }

    def __init__(self,
                 arch='s',
                 depth_mult=1.0,
                 width_mult=1.0,
                 last_stage_ch=1024,
                 act='silu',
                 return_idx=[2, 3, 4]):
        super().__init__()
        self.return_idx = return_idx
        arch_setting = self.arch_settings[arch]
        if last_stage_ch != 1024:
            assert last_stage_ch > 0
            arch_setting[-1][1] = last_stage_ch
        _out_channels = []

        input_channels = 3
        base_channels = int(arch_setting[0][0] * width_mult)
        self.stem = BaseConv(input_channels,
                             base_channels,
                             ksize=3,
                             stride=2,
                             groups=1,
                             bias=False,
                             act=act)
        _out_channels.append(base_channels)

        layers_num = 1
        self.csp_dark_blocks = []

        for i, (in_channels, out_channels, num_blocks, use_c3k, c3k2_scale,
                add_shortcut, use_sppf, use_c2psa) in enumerate(arch_setting):
            in_channels = int(in_channels * width_mult)
            out_channels = int(out_channels * width_mult)
            _out_channels.append(out_channels)
            num_blocks = max(round(num_blocks * depth_mult), 1)
            stage = []

            conv_layer = self.add_sublayer(
                f'layers{layers_num}.stage{i + 1}.conv_layer',
                BaseConv(in_channels,
                            out_channels,
                            ksize=3,
                            stride=2,
                            groups=1,
                            bias=False,
                            act=act))
            stage.append(conv_layer)
            layers_num += 1
       
            c3k2_layer = self.add_sublayer(
                f'layers{layers_num}.stage{i + 1}.c3k2_layer',
                C3k2Layer(out_channels,
                            out_channels,
                            num_blocks=num_blocks,
                            shortcut=add_shortcut,
                            expansion=c3k2_scale,
                            depthwise=False,
                            bias=False,
                            act=act,
                            use_c3k=use_c3k))
            stage.append(c3k2_layer)
            layers_num += 1

            if use_sppf:
                sppf_layer = self.add_sublayer(
                    f'layers{layers_num}.stage{i + 1}.sppf_layer',
                    SPPFLayer(out_channels,
                              out_channels,
                              ksize=5,
                              bias=False,
                              act=act))
                stage.append(sppf_layer)
                layers_num += 1

            if use_c2psa:
                c2psa_layer = self.add_sublayer(
                    f'layers{layers_num}.stage{i + 1}.c2psa_layer',
                    C2PSALayer(out_channels, expansion=0.5, act=act))
                stage.append(c2psa_layer)
                layers_num += 1

            self.csp_dark_blocks.append(nn.Sequential(*stage))

        self._out_channels = [_out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        out = self.stem(x)
        if 0 in self.return_idx:
            outputs.append(out)
        for i, layer in enumerate(self.csp_dark_blocks):
            out = layer(out)
            if i + 1 in self.return_idx:
                outputs.append(out)
        return outputs

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=c, stride=s)
            for c, s in zip(self._out_channels, self.strides)
        ]
