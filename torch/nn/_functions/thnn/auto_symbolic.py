from torch.autograd._functions.utils import prepare_onnx_paddings


def reflectionpad_symbolic(g, input, *params):
    mode = "reflect"
    paddings = prepare_onnx_paddings(len(input.type().sizes()), params)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode)


def replicationpad_symbolic(g, input, *params):
    mode = "edge"
    paddings = prepare_onnx_paddings(len(input.type().sizes()), params)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode)


symbolic_fns = {
    'ReflectionPad1d': reflectionpad_symbolic,
    'ReflectionPad2d': reflectionpad_symbolic,
    'ReplicationPad1d': replicationpad_symbolic,
    'ReplicationPad2d': replicationpad_symbolic,
    'ReplicationPad3d': replicationpad_symbolic,
}
