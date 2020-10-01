import re
import torch

# turn foo.bar -> ['foo', 'bar']
def _parent_name(target):
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]

def graph_pretty_str(g, shorten=True) -> str:
    """Returns a printable representation of the ops in the graph of g.
    If shorten is True, tries to abbreviate fields.
    """
    built_in_func_re = re.compile('<built-in function (.*)>')
    built_in_meth_re = re.compile('<built-in method (.*) of type.*>')
    op_dict = {
        'placeholder': 'plchdr',
        'get_attr': 'gt_prm',
        'call_function': 'cl_fun',
        'call_module': 'cl_mod',
        'call_method': 'cl_meth',
    }

    max_lens = {}
    col_names = ("name", "op", "target", "args", "kwargs")
    for s in col_names:
        max_lens[s] = len(s)

    results = []
    for n in g.nodes:

        # activation_post_process_0 -> obs_0
        name = str(n.name)
        if shorten:
            name = name.replace("activation_post_process", "obs")

        op = str(n.op)
        # placeholder -> plchdr, and so on
        if shorten and op in op_dict:
            op = op_dict[op]

        target = str(n.target)
        # <built-in function foo> -> <bi_fun foo>, and so on
        if shorten:
            built_in_func = built_in_func_re.search(target)
            if built_in_func:
                target = f"<bi_fun {built_in_func.group(1)}>"
            built_in_meth = built_in_meth_re.search(target)
            if built_in_meth:
                target = f"<bi_meth {built_in_meth.group(1)}>"
            target = target.replace("activation_post_process", "obs")

        args = str(n.args)
        if shorten:
            args = args.replace("activation_post_process", "obs")

        kwargs = str(n.kwargs)

        # calculate maximum length of each column, so we can tabulate properly
        for k, v in zip(col_names, (name, op, target, args, kwargs)):
            max_lens[k] = max(max_lens[k], len(v))
        results.append([name, op, target, args, kwargs])

    res_str = ""
    format_str = "{:<{name}} {:<{op}} {:<{target}} {:<{args}} {:<{kwargs}}\n"
    res_str += format_str.format(*col_names, **max_lens)
    for result in results:
        res_str += format_str.format(*result, **max_lens)

    # print an exra note on abbreviations which change attribute names,
    # since users will have to un-abbreviate for further debugging
    if shorten:
        res_str += "*obs_{n} = activation_post_process_{n}\n"
    return res_str

def is_per_tensor(qscheme):
    return qscheme == torch.per_tensor_affine or \
        qscheme == torch.per_tensor_symmetric

def is_per_channel(qscheme):
    return qscheme in [torch.per_channel_affine,
                       torch.per_channel_affine_float_qparams,
                       torch.per_channel_symmetric]

def get_per_tensor_qparams(activation_post_process):
    assert is_per_tensor(activation_post_process.qscheme), 'Only per tensor quantization is supported'
    scale, zero_point = activation_post_process.calculate_qparams()
    scale = float(scale)
    zero_point = int(zero_point)
    dtype = activation_post_process.dtype
    return scale, zero_point, dtype

def get_quantize_op_and_qparams(activation_post_process):
    ''' Given an activation_post_process module,
    return quantize op(e.g. quantize_per_tensor) and a dictionary
    of extracted qparams from the module
    '''
    scale, zero_point = activation_post_process.calculate_qparams()
    dtype = activation_post_process.dtype
    if is_per_channel(activation_post_process.qscheme):
        ch_axis = int(activation_post_process.ch_axis)
        qparams = {'_scale_': scale, '_zero_point_': zero_point, '_axis_': ch_axis, '_dtype_': dtype}
        quantize_op = torch.quantize_per_channel
    else:
        scale = float(scale)
        zero_point = int(zero_point)
        qparams = {'_scale_': scale, '_zero_point_': zero_point, '_dtype_': dtype}
        quantize_op = torch.quantize_per_tensor
    return quantize_op, qparams

def quantize_node(root_module, graph, node, activation_post_process):
    ''' Add quantization nodes for given node to graph
    with the qparams calculated from activation_post_process module
    e.g. Given input `node` in `node = self.conv(x)`, insert node:
    `quantized_node = torch.quantize_per_tensor(x, self._scale_0, self._zer_point_0, self._dtype_0)`
    where self._scale_0, self._zero_point_0 and self._dtype_0 are
    calculated from `activation_post_process`
    '''
    def module_has_qparams_attr_with_index(module, qparams, i):
        for name in qparams.keys():
            if hasattr(module, name + str(i)):
                return True
        return False

    def get_next_qparams_idx(module, qparams):
        idx = 0
        while module_has_qparams_attr_with_index(module, qparams, idx):
            idx += 1
        return idx

    quantize_op, qparams = get_quantize_op_and_qparams(activation_post_process)
    idx = get_next_qparams_idx(root_module, qparams)
    inputs = [node]
    for key, value in qparams.items():
        setattr(root_module, key + str(idx), value)
        qparam_full_path = key + str(idx)
        inputs.append(graph.create_node('get_attr', qparam_full_path))
    return graph.create_node('call_function', quantize_op, tuple(inputs), {})
