import torch
import torch.onnx


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
