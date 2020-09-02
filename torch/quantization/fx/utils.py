import torch

# turn foo.bar -> ['foo', 'bar']
def _parent_name(target):
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]

def is_per_tensor(qscheme):
    return qscheme == torch.per_tensor_affine or \
        qscheme == torch.per_tensor_symmetric

def is_per_channel(qscheme):
    return qscheme == torch.per_channel_affine or \
        qscheme == torch.per_channel_symmetric

def get_per_tensor_qparams(activation_post_process):
    assert is_per_tensor(activation_post_process.qscheme), 'Only per tensor quantization is supported'
    scale, zero_point = activation_post_process.calculate_qparams()
    scale = float(scale)
    zero_point = int(zero_point)
    dtype = activation_post_process.dtype
    return scale, zero_point, dtype

# given an activation_post_process module,
# return quantize op(e.g. quantize_per_tensor) and a dictionary
# of extracted qparams from the module
def get_quantize_op_and_qparams(activation_post_process):
    scale, zero_point = activation_post_process.calculate_qparams()
    dtype = activation_post_process.dtype
    if is_per_channel(activation_post_process.qscheme):
        ch_axis = int(activation_post_process.ch_axis)
        qparams = {'_scale_': scale, '_zero_point_': zero_point, '_axis': ch_axis, '_dtype_': dtype}
        quantize_op = torch.quantize_per_channel
    else:
        scale = float(scale)
        zero_point = int(zero_point)
        qparams = {'_scale_': scale, '_zero_point_': zero_point, '_dtype_': dtype}
        quantize_op = torch.quantize_per_tensor
    return quantize_op, qparams

def quantize_node(parent_module, graph, node, activation_post_process):
    def noattr(module, qparams, i):
        for name in qparams.keys():
            if hasattr(module, name + str(i)):
                return False
            return True

    def get_next_i(module, qparams):
        i = 0
        while not noattr(module, qparams, i):
            i += 1
        return i

    quantize_op, qparams = get_quantize_op_and_qparams(activation_post_process)
    i = get_next_i(parent_module, qparams)
    inputs = [node]
    for key, value in qparams.items():
        setattr(parent_module, key + str(i), value)
        qparam_full_path = key + str(i)
        inputs.append(graph.create_node('get_param', qparam_full_path))
    return graph.create_node('call_function', quantize_op, tuple(inputs), {})
