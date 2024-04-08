from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
from ppocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer


def get_bias_attr(k):
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
    bias_attr = ParamAttr(initializer=initializer)
    return bias_attr


class Head(nn.Layer):
    def __init__(self, in_channels, kernel_list=[3, 2, 2], **kwargs):
        super(Head, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            weight_attr=ParamAttr(),
            bias_attr=False)
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act='relu')

        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4))
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu")
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4), )

    def forward(self, x, return_f=False):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        if return_f is True:
            f = x
        x = self.conv3(x)
        x = F.sigmoid(x)
        if return_f is True:
            return x, f
        return x


class DBHead(nn.Layer):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        self.binarize = Head(in_channels, **kwargs)
        self.thresh = Head(in_channels, **kwargs)

    def step_function(self, x, y):
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {'maps': shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
        return {'maps': y}


class LocalModule(nn.Layer):
    def __init__(self, in_c, mid_c, use_distance=True):
        super(self.__class__, self).__init__()
        self.last_3 = ConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act='relu')
        self.last_1 = nn.Conv2D(mid_c, 1, 1, 1, 0)

    def forward(self, x, init_map, distance_map):
        outf = paddle.concat([init_map, x], axis=1)
        # last Conv
        out = self.last_1(self.last_3(outf))
        return out

