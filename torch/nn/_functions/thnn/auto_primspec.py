def threshold_primspec(g, input, threshold=0, value=0, inplace=False):
    # TODO: [Export inplace]
    if threshold != 0:
        raise RuntimeError("Non-zero threshold in Threshold not supported")
    if value != 0:
        raise RuntimeError("Non-zero value in Threshold not supported")
    return g.op("Relu", input)


def leakyrelu_primspec(g, input, negative_slope, inplace=False):
    # TODO: [Export inplace]
    return g.op("LeakyRelu", input, alpha_f=negative_slope)


primspec_fns = {
    'Threshold': threshold_primspec,
    'LeakyReLU': leakyrelu_primspec,
}
