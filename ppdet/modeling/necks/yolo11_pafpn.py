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
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec
from ..backbones.csp_darknet import BaseConv
from ..backbones.yolo11_csp_darknet import C3k2Layer

__all__ = ['YOLO11CSPPAN']


@register
@serializable
class YOLO11CSPPAN(nn.Layer):
    """
    YOLOv10 CSP-PAN FPN, used in YOLOv10.
    """
    __shared__ = ['depth_mult', 'act']

    def __init__(self,
                 depth_mult=1.0,
                 in_channels=[256, 512, 1024],
                 fpn_use_c3k=[False, False],
                 pan_use_c3k=[False, True],
                 act='silu'):
        super().__init__()
        self.in_channels = in_channels
        self._out_channels = in_channels

        # top-down
        self.fpn_p4 = C3k2Layer(int(in_channels[2] + in_channels[1]),
                                    int(in_channels[1]),
                                    num_blocks=round(2 * depth_mult),
                                    shortcut=False,
                                    act=act, use_c3k=fpn_use_c3k[0])

      
        self.fpn_p3 = C3k2Layer(int(in_channels[1] + in_channels[0]),
                                   int(in_channels[0]),
                                   num_blocks=round(3 * depth_mult),
                                   shortcut=False,
                                   expansion=0.5,
                                   depthwise=False,
                                   bias=False,
                                   act=act, use_c3k=fpn_use_c3k[1])
       # bottom-up
        self.down_conv2 = BaseConv(
            int(in_channels[0]), int(in_channels[0]), 3, stride=2, act=act)
        self.pan_n3 = C3k2Layer(
            int(in_channels[0] + in_channels[1]),
            int(in_channels[1]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=False,
            act=act, use_c3k=pan_use_c3k[0])

        self.down_conv1 = BaseConv(
            int(in_channels[1]), int(in_channels[1]), 3, stride=2, act=act)
        self.pan_n4 = C3k2Layer(
            int(in_channels[1] + in_channels[2]),
            int(in_channels[2]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=False,
            act=act, use_c3k=pan_use_c3k[1])

    def forward(self, feats, for_mot=False):
        [c3, c4, c5] = feats

        # top-down FPN
        up_feat1 = F.interpolate(c5, scale_factor=2., mode="nearest")
        f_concat1 = paddle.concat([up_feat1, c4], axis=1)
        f_out1 = self.fpn_p4(f_concat1)

        up_feat2 = F.interpolate(f_out1, scale_factor=2., mode="nearest")
        f_concat2 = paddle.concat([up_feat2, c3], axis=1)
        f_out0 = self.fpn_p3(f_concat2)

        # bottom-up PAN
        down_feat1 = self.down_conv1(f_out0)
        p_concat1 = paddle.concat([down_feat1, f_out1], axis=1)
        pan_out1 = self.pan_n3(p_concat1)

        down_feat2 = self.down_conv2(pan_out1)
        p_concat2 = paddle.concat([down_feat2, c5], 1)
        pan_out0 = self.pan_n4(p_concat2)

        return [f_out0, pan_out1, pan_out0]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
        }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
