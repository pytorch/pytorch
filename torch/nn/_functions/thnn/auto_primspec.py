def threshold_primspec(g, input, threshold=0, value=0, inplace=False):
    if inplace or threshold != 0 or value != 0:
        return None
    r = g.appendNode(g.create("Relu", [input]))
    return r


def leakyrelu_primspec(g, input, negative_slope, inplace=False):
    if inplace:
        return None
    return g.appendNode(g.create("LeakyRelu", [input]).f_("alpha", negative_slope))


primspec_fns = {
    'Threshold': threshold_primspec,
    'LeakyReLU': leakyrelu_primspec,
}
