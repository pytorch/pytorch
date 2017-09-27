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

symbolic_fns = {
    'Threshold': threshold_symbolic,
    'LeakyReLU': leakyrelu_symbolic,
    'GatedLinear': glu_symbolic,
}
