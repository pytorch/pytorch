import re
import torch
from ..quant_type import QuantType, quant_type_to_str

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

def get_custom_module_class_keys(custom_config_dict, custom_config_dict_key):
    r""" Get all the unique custom module keys in the custom config dict
    e.g.
    Input:
    custom_config_dict = {
        "float_to_observed_custom_module_class": {
           "static": {
               CustomModule1: ObservedCustomModule
           },
           "dynamic": {
               CustomModule2: DynamicObservedCustomModule
           },
           "weight_only": {
               CustomModule3: WeightOnlyObservedCustomModule
           },
        },
    }

    Output:
    # extract all the keys in "static", "dynamic" and "weight_only" dict
    [CustomModule1, CustomModule2, CustomModule3]
    """
    # using set to dedup
    float_custom_module_classes = set()
    custom_module_mapping = custom_config_dict.get(custom_config_dict_key, {})
    for quant_mode in ["static", "dynamic", "weight_only"]:
        quant_mode_custom_module_config = custom_module_mapping.get(quant_mode, {})
        quant_mode_custom_module_classes = set(quant_mode_custom_module_config.keys())
        float_custom_module_classes |= quant_mode_custom_module_classes
    return list(float_custom_module_classes)

def get_swapped_custom_module_class(custom_module, custom_module_class_mapping, qconfig):
    """ Get the observed/quantized custom module class that we need
    to swap `custom_module` to
    Input:
        custom_module: input, can be an instance of either a float or observed custom module
        custom_module_class_mapping: the float to observed or observed to quantized custom module class mapping
        qconfig: qconfig configured for the custom module

    Output:
        corresponding observed/quantized custom module class for input custom module instance
    """
    quant_type = get_quant_type(qconfig)
    quant_type_str = quant_type_to_str(quant_type)
    class_mapping = custom_module_class_mapping.get(quant_type_str, {})
    assert type(custom_module) in class_mapping, "did not found corresponding observed " \
        "module class for {} in mapping: {}".format(type(custom_module), class_mapping)
    return class_mapping[type(custom_module)]

def activation_is_statically_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    statically quantized or not
    """
    assert qconfig is not None
    activation = qconfig.activation()
    return activation.dtype in [torch.quint8, torch.qint8]

def weight_dtype(qconfig):
    assert qconfig is not None
    weight = qconfig.weight()
    return weight.dtype

def weight_is_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized or not
    """
    return weight_dtype(qconfig) in [torch.quint8, torch.qint8]

def get_qconfig_dtypes(qconfig):
    r""" returns the qconfig tuple for qconfig:
    (activation_dtype, weight_dtype, activation_compute_dtype)
    """
    assert qconfig is not None
    activation = qconfig.activation()
    weight = qconfig.weight()
    compute_dtype = activation.compute_dtype if hasattr(activation, 'compute_dtype') else None
    return (activation.dtype, weight.dtype, compute_dtype)

def get_quant_type(qconfig):
    assert qconfig is not None
    activation = qconfig.activation()
    weight = qconfig.weight()
    static_dtypes = [torch.quint8, torch.qint8]
    if weight.dtype in static_dtypes:
        if activation.dtype in static_dtypes:
            return QuantType.STATIC
        elif hasattr(activation, 'compute_dtype') and activation.compute_dtype in static_dtypes:
            return QuantType.DYNAMIC
        else:
            return QuantType.WEIGHT_ONLY

    if weight.dtype == torch.float16:
        if activation.dtype == torch.float:
            return QuantType.WEIGHT_ONLY

    raise Exception("Unrecognized dtype combination in get_quant_type: activation({}),"
                    "weight({})".format(activation.dtype, weight.dtype))

def get_linear_prepack_op_for_dtype(dtype):
    if dtype == torch.float16:
        return torch.ops.quantized.linear_prepack_fp16
    elif dtype == torch.qint8:
        return torch.ops.quantized.linear_prepack
    else:
        raise Exception("can't get linear prepack op for dtype:", dtype)
