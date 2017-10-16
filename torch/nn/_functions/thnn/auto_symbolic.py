from torch.autograd._functions.utils import prepare_onnx_paddings


def threshold_symbolic(g, input, threshold=0, value=0, inplace=False):
    # TODO: [Export inplace]
    if threshold != 0:
        raise RuntimeError("Non-zero threshold in Threshold not supported")
    if value != 0:
        raise RuntimeError("Non-zero value in Threshold not supported")
    return g.op("Relu", input)


def leakyrelu_symbolic(g, input, negative_slope, inplace=False):
    # TODO: [Export inplace]
    return g.op("LeakyRelu", input, alpha_f=negative_slope)


def glu_symbolic(g, input, dim):
    assert input.type().sizes()[dim] % 2 == 0

    first, second = g.op('Split', input, axis_i=dim, outputs=2)
    return g.op('Mul', first, g.op('Sigmoid', second))


def softmax_symbolic(g, input):
    return g.op('Softmax', input)


def logsoftmax_symbolic(g, input):
    # TODO use logsoftmax to replace this combination.
    return g.op("Log", g.op('Softmax', input).typeAs(input))


def reflectionpad_symbolic(g, input, *params):
    mode = "reflect"
    paddings = prepare_onnx_paddings(len(input.type().sizes()), params)
    return g.op("Pad", input, paddings_i=paddings, mode_s=mode)


def replicationpad_symbolic(g, input, *params):
    mode = "edge"
    paddings = prepare_onnx_paddings(len(input.type().sizes()), params)
    return g.op("Pad", input, paddings_i=paddings, mode_s=mode)


symbolic_fns = {
    'Threshold': threshold_symbolic,
    'LeakyReLU': leakyrelu_symbolic,
    'GatedLinear': glu_symbolic,
    'Softmax': softmax_symbolic,
    'LogSoftmax': logsoftmax_symbolic,
    'ReflectionPad1d': reflectionpad_symbolic,
    'ReflectionPad2d': reflectionpad_symbolic,
    'ReplicationPad1d': replicationpad_symbolic,
    'ReplicationPad2d': replicationpad_symbolic,
    'ReplicationPad3d': replicationpad_symbolic,
}
