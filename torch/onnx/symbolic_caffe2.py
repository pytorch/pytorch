from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils

import torch.onnx.symbolic_helper as sym_help

def linear_prepack(g, input, weight):
    return input

def linear(g, input, weight, scale, zero_point):
    return g.op("_caffe2::Int8FC", input, weight, scale, zero_point)

def conv_prepack(g, input, weight, stride, padding, dilation, groups):
    return input

def conv2d(g, input, weight, stride, padding, dilation, groups, scale, zero_point):
    return g.op("_caffe2::Int8Conv", input, weight, stride, padding, dilation, groups, scale, zero_point)

def conv2d_relu(g, input, weight, stride, padding, dilation, groups, scale, zero_point):
    return g.op("_caffe2::Int8ConvRelu", input, weight, stride, padding, dilation, groups, scale, zero_point)

def add(g, input_a, input_b, scale, zero_point):
    return g.op("_caffe2::Int8Add", input_a, input_b, scale, zero_point)

def upsample_nearest_2d(g, input, size, scale_factor, mode , align_corners):
    return g.op("_caffe2::Int8ResizeNearest", input, scale_factor, scale_factor)

def relu(g, input, scale, zero_point):
    return g.op("_caffe2::Int8Relu", input, scale, zero_point)
