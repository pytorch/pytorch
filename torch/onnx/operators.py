r"""This file provides a location for operators that help exporting
models via onnx. E.g. shape_as_tensor and reshape_from_tensor_shape
are to make all dynamic sizes operations traceble.

"""

import torch
import torch.onnx


def select(g, self, dim, index):
    slice_node = g.op("Slice", self, axes_i=[dim], starts_i=[index], ends_i=[index + 1])
    return g.op("Squeeze", slice_node, axes_i=[dim])


def _shape_as_tensor(g, input):
    return g.op('Shape', input)


@torch.onnx.symbolic_override(_shape_as_tensor)
def shape_as_tensor(x):
    return torch.LongTensor(tuple(x.shape))


def _reshape_from_tensor_shape(g, input, shape):
    return g.op('Reshape', input, shape)


@torch.onnx.symbolic_override(_reshape_from_tensor_shape)
def reshape_from_tensor_shape(x, shape):
    return x.reshape(shape.tolist())


def _batch_size_from_packed_sequence(g, packed_sequence):
    # in the onnx graph, batch_sizes is actually sequence_lengths
    _, sequence_lengths = packed_sequence
    return select(g, _shape_as_tensor(g, g.op('prim::SequenceLengthsOfPackedSequence', sequence_lengths)), 0, 0)


@torch.onnx.symbolic_override(_batch_size_from_packed_sequence)
def batch_size_from_packed_sequence(packed_sequence):
    _, batch_sizes = packed_sequence
    return batch_sizes[0]


def _constant_tensor_from_tensor_shape(g, input, val, shape):
    return g.op('ConstantFill', shape, input_as_shape_i=1)


@torch.onnx.symbolic_override(_constant_tensor_from_tensor_shape)
def constant_tensor_from_tensor_shape(input, val, shape):
    shape = tuple(int(x) for x in shape)
    x = input.data.new(*shape).fill_(val)
    return torch.autograd.Variable(x, requires_grad=False)
