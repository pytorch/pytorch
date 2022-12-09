"""
Utils shared by different modes of quantization (eager/graph)
"""
import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qconfig_propagation_list,
    no_observer_set,
    QuantType,
)
from torch.ao.quantization.qconfig import _activation_is_memoryless
from torch.ao.quantization.quantization_mappings import (
    _get_special_act_post_process,
    _has_special_act_post_process,
)

from torch.fx import Node
from torch.nn.intrinsic import _FusedModule
from torch.nn.utils.parametrize import is_parametrized, type_before_parametrizations

__all__ = [
    "NodePattern",
    "Pattern",
    "MatchAllNode",
    "get_fqn_to_example_inputs",
]

NodePattern = Union[Tuple[Node, Node], Tuple[Node, Tuple[Node, Node]], Any]
NodePattern.__module__ = "torch.ao.quantization.utils"

# This is the Quantizer class instance from torch/quantization/fx/quantize.py.
# Define separately to prevent circular imports.
# TODO(future PR): improve this.
# make this public once fixed (can't be public as is because setting the module directly
# doesn't work)
QuantizerCls = Any

# Type for fusion patterns, it can be more complicated than the following actually,
# see pattern.md for docs
# TODO: not sure if typing supports recursive data types
Pattern = Union[
    Callable, Tuple[Callable, Callable], Tuple[Callable, Tuple[Callable, Callable]], Any
]
Pattern.__module__ = "torch.ao.quantization.utils"

# TODO: maybe rename this to MatchInputNode
class MatchAllNode:
    """ A node pattern that matches all nodes, used in defining
    fusion patterns in FX Graph Mode Quantization
    """
    pass

_module_type_list = {
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.AdaptiveAvgPool1d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.AdaptiveAvgPool3d,
    torch.nn.AvgPool1d,
    torch.nn.AvgPool2d,
    torch.nn.AvgPool3d,
    torch.nn.MaxPool1d,
    torch.nn.MaxPool2d,
    torch.nn.MaxPool3d,
    torch.nn.Identity,
    torch.nn.Hardsigmoid,
    torch.nn.Sigmoid,
    torch.nn.Tanh,
}
_func_list = {
    torch.nn.functional.adaptive_avg_pool1d,
    torch.nn.functional.adaptive_avg_pool2d,
    torch.nn.functional.adaptive_avg_pool3d,
    torch.nn.functional.elu,
    torch.nn.functional.hardswish,
    torch.nn.functional.instance_norm,
    torch.nn.functional.layer_norm,
    torch.nn.functional.leaky_relu,
    torch.nn.functional.silu,
    torch.nn.functional.mish,
    torch.nn.functional.dropout,
    torch.nn.functional.max_pool1d,
    torch.nn.functional.max_pool2d,
    torch.nn.functional.max_pool3d,
    torch.nn.functional.relu,
    torch.nn.functional.hardtanh,
    torch.nn.functional.hardtanh_,
    torch.nn.functional.hardsigmoid,
    torch.nn.functional.sigmoid,
    torch.transpose,
    torch.repeat_interleave,
    torch.sigmoid,
    torch.squeeze,
    torch.stack,
    torch.sum,
    torch.tanh,
    torch.unsqueeze,
    torch.cat,
}
_method_list = {
    torch.mean,
    'relu',
    'relu_',
    'contiguous',
    'detach',
    'detach_',
    'hardsigmoid',
    'hardsigmoid_',
    'permute',
    'repeat',
    'repeat_interleave',
    'reshape',
    'resize_',
    'shape',
    'sigmoid',
    'sigmoid_',
    'size',
    'squeeze',
    'squeeze_',
    'tanh',
    'tanh_',
    'transpose',
    'unsqueeze',
    'unsqueeze_',
    'view',
}

def _get_combined_dict(default_dict, additional_dict):
    d = default_dict.copy()
    d.update(additional_dict)
    return d

def _is_per_tensor(qscheme):
    return qscheme == torch.per_tensor_affine or \
        qscheme == torch.per_tensor_symmetric

def _is_per_channel(qscheme):
    return qscheme in [torch.per_channel_affine,
                       torch.per_channel_affine_float_qparams,
                       torch.per_channel_symmetric]

def _getattr_from_fqn(obj: Any, fqn: str) -> Any:
    """
    Given an obj and a fqn such as "foo.bar.baz", returns gm.foo.bar.baz.
    """
    return functools.reduce(getattr, fqn.split("."), obj)

def _to_underlying_dtype(qdtype):
    DTYPE_MAPPING = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
        torch.quint4x2: torch.uint8,
        torch.quint2x4: torch.uint8,
    }
    assert qdtype in DTYPE_MAPPING, "Unsupported dtype: " + qdtype
    return DTYPE_MAPPING[qdtype]

def _get_qparam_dict(observer_or_fake_quant):
    qscheme = observer_or_fake_quant.qscheme if hasattr(observer_or_fake_quant, "qscheme") else None
    dtype = observer_or_fake_quant.dtype
    qparams = {"qscheme": qscheme, "dtype": dtype}

    if not qscheme:
        return qparams

    if _is_per_tensor(qscheme):
        qscheme = torch.per_tensor_affine
    elif _is_per_channel(qscheme):
        # change symmetric to affine since we do not have symmetric
        # quantized Tensor
        if qscheme == torch.per_channel_symmetric:
            qscheme = torch.per_channel_affine
        qparams["axis"] = observer_or_fake_quant.ch_axis
    else:
        raise RuntimeError(f"Unrecognized qscheme: {qscheme}")
    # update qscheme, since we don't have symmetric quant qscheme
    # in quantized Tensor
    qparams["qscheme"] = qscheme

    scale, zero_point = observer_or_fake_quant.calculate_qparams()
    qparams["scale"] = scale
    qparams["zero_point"] = zero_point

    return qparams


def _get_swapped_custom_module_class(custom_module, custom_module_class_mapping, qconfig):
    """ Get the observed/quantized custom module class that we need
    to swap `custom_module` to
    Input:
        custom_module: input, can be an instance of either a float or observed custom module
        custom_module_class_mapping: the float to observed or observed to quantized custom module class mapping
        qconfig: qconfig configured for the custom module

    Output:
        corresponding observed/quantized custom module class for input custom module instance
    """
    quant_type = _get_quant_type(qconfig)
    class_mapping = custom_module_class_mapping.get(quant_type, {})
    assert type(custom_module) in class_mapping, "did not find corresponding observed " \
        "module class for {} in mapping: {}".format(type(custom_module), class_mapping)
    return class_mapping[type(custom_module)]

def _activation_dtype(qconfig):
    assert qconfig is not None
    activation = qconfig.activation()
    return activation.dtype

def _weight_dtype(qconfig):
    assert qconfig is not None
    weight = qconfig.weight()
    return weight.dtype

def _activation_is_statically_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized or not, this includes quantizing to quint8, qint8 and qint32 and float16
    """
    return (
        _activation_dtype(qconfig) in [torch.quint8, torch.qint8, torch.qint32, torch.float16]
        and (not _activation_is_dynamically_quantized(qconfig))
    )

def _activation_is_dynamically_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    dynamically quantized or not, this includes dynamically quantizing to
    quint8, qint8 and float16
    """
    activation_dtype, _, activation_is_dynamic = \
        _get_qconfig_dtypes(qconfig)
    return activation_is_dynamic

def _activation_is_int8_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized to int8 or not, this includes quantizing to quint8, qint8
    """
    return _activation_dtype(qconfig) in [torch.quint8, torch.qint8]

def _activation_is_int32_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized to int32 or not
    """
    return _activation_dtype(qconfig) == torch.qint32

def _weight_is_quantized(qconfig):
    """ Given a qconfig, decide if the weight needs to be
    quantized or not
    """
    return _weight_dtype(qconfig) in [torch.quint8, torch.qint8, torch.float16, torch.quint4x2]

def _weight_is_statically_quantized(qconfig):
    """ Given a qconfig, decide if the weight needs to be statically
    quantized or not
    """
    return _weight_dtype(qconfig) in [torch.quint8, torch.qint8]

def _op_is_int8_dynamically_quantized(qconfig) -> bool:
    """ Given a qconfig, returns True if this op is using int8 dynamic
    quantization
    """
    activation_dtype, weight_dtype, activation_is_dynamic = \
        _get_qconfig_dtypes(qconfig)
    return (
        _activation_dtype is torch.quint8 and
        # for now, the lines below assume fbgemm or qnnpack
        weight_dtype is torch.qint8 and
        activation_is_dynamic
    )

def _get_qconfig_dtypes(qconfig):
    r""" returns the qconfig tuple for qconfig:
    (activation_dtype, weight_dtype, activation_is_dynamic)
    """
    assert qconfig is not None
    activation = qconfig.activation()
    weight = qconfig.weight()
    act_is_dynamic = activation.is_dynamic if hasattr(activation, 'is_dynamic') else False
    return (activation.dtype, weight.dtype, act_is_dynamic)

# TODO remove this once BC no longer needed
get_qconfig_dtypes = _get_qconfig_dtypes

def _get_quant_type(qconfig):
    assert qconfig is not None
    activation = qconfig.activation()
    weight = qconfig.weight()
    static_dtypes = [torch.quint8, torch.qint8, torch.quint4x2, torch.qint32]
    if weight.dtype in static_dtypes:
        if hasattr(activation, 'is_dynamic') and activation.is_dynamic:
            return QuantType.DYNAMIC
        elif activation.dtype in static_dtypes:
            return QuantType.STATIC
        else:
            return QuantType.WEIGHT_ONLY

    if weight.dtype == torch.float16:
        if hasattr(activation, 'is_dynamic') and activation.is_dynamic:
            return QuantType.DYNAMIC
        elif activation.dtype == torch.float16:
            return QuantType.STATIC

    raise Exception("Unrecognized dtype combination in _get_quant_type: activation({}),"
                    "weight({})".format(activation.dtype, weight.dtype))

def _check_min_max_valid(min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
    """ Checks if the given minimum and maximum values are valid, meaning that
    they exist and the min value is less than the max value.
    """
    if min_val.numel() == 0 or max_val.numel() == 0:
        warnings.warn(
            "must run observer before calling calculate_qparams. " +
            "Returning default values."
        )
        return False

    if min_val.dim() == 0 or max_val.dim() == 0:
        if min_val == float("inf") and max_val == float("-inf"):
            warnings.warn(
                "must run observer before calling calculate_qparams. " +
                "Returning default values."
            )

            return False

        assert min_val <= max_val, "min {} should be less than max {}".format(
            min_val, max_val
        )
    else:
        assert torch.all(
            min_val <= max_val
        ), "min {} should be less than max {}".format(min_val, max_val)

    return True


def _calculate_qmin_qmax(quant_min: int, quant_max: int, has_customized_qrange: bool, dtype: torch.dtype,
                         reduce_range: bool) -> Tuple[int, int]:
    r"""Calculates actual qmin and qmax based on the quantization range,
    observer datatype and if range is reduced.
    """
    # TODO(jerryzh): Figure out why custom quant_min/quant_max are still adjusted.
    if has_customized_qrange:
        # This initialization here is to be resolve TorchScript compilation issues and allow
        # using of refinement to decouple initial_qmin and initial_qmax from quantization range.
        # The actual values of initial_qmin and initial_qmax will be reset below.
        if dtype == torch.qint32:
            initial_quant_min, initial_quant_max = 0, 2**31 - 1
        else:
            initial_quant_min, initial_quant_max = 0, 255
        # The following assignment of self.qmin and self.qmax to the local variables and the if check refine the
        # attribute from Optional valid integers for use, based on TorchScript's requirements.
        custom_quant_min, custom_quant_max = quant_min, quant_max
        if custom_quant_min is not None and custom_quant_max is not None:
            initial_quant_min, initial_quant_max = (
                custom_quant_min,
                custom_quant_max,
            )

        qrange_len = initial_quant_max - initial_quant_min + 1
        if dtype == torch.qint8:
            assert (
                0 < qrange_len <= 256
            ), "quantization range should be positive and not exceed the maximum bit range (=256)."
        elif dtype == torch.qint32:
            assert (
                0 < qrange_len <= 2**31
            ), "quantization range should be positive and not exceed the maximum bit range (=4294967296)."
        if reduce_range:
            quant_min, quant_max = quant_min // 2, quant_max // 2
    else:
        # Fallback onto default 8-bit qmin and qmax calculation if dynamic range is not used.
        if dtype == torch.qint8:
            if reduce_range:
                quant_min, quant_max = -64, 63
            else:
                quant_min, quant_max = -128, 127
        elif dtype == torch.quint8:
            if reduce_range:
                quant_min, quant_max = 0, 127
            else:
                quant_min, quant_max = 0, 255
        elif dtype == torch.qint32:
            quant_min, quant_max = -1 * (2 ** 31), (2 ** 31) - 1
        else:
            quant_min, quant_max = 0, 15
    return quant_min, quant_max


def _parent_name(target):
    """
    Turn 'foo.bar' into ['foo', 'bar']
    """
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]

def _has_no_children_ignoring_parametrizations(module):
    """
    Checks if module._modules is empty or
    if module is a parametrization, checks that module._modules only has
    the 'parametrizations' module
    """
    if len(module._modules) == 0:
        return True
    elif is_parametrized(module):
        return len(module._modules) == 1 and 'parametrizations' in module._modules
    else:
        return False

def _get_path_of_module(root: torch.nn.Module, submodule: torch.nn.Module) -> Optional[str]:
    """ Get the path (fully qualified name) of a submodule

    Example::

    >> class M(torch.nn.Module):
           def __init__(self):
               self.linear = torch.nn.Linear(5, 5)
           def forward(self, x):
               return self.linear(x)

    >> m = M()
    >> l = m.linear
    >> _get_path_of_module(m, l)
    "linear"
    """
    for n, p in root.named_modules():
        if submodule is p:
            return n
    return None

def _get_signature_locals(f: Callable, loc: Dict[str, Any]) -> Dict[str, Any]:
    """ Get local keyword arguments

    Example::

    >> def f(self, a, b=9):
           pass
    >> loc = {"a": 6, "c": 7}
    >> _get_signature_locals(f, loc)
    {"a": 6}
    """
    return {k: v for k, v in loc.items() if k in signature(f).parameters}

def _get_default_kwargs(f: Callable) -> "OrderedDict[str, Any]":
    """ Get all default keyword arguments from function signature

    Example::

    >> def f(self, a, b=9):
           pass
    >> _get_default_kwargs(f)
    {"b": 9}
    """
    kwargs = {}
    for name, param in signature(f).parameters.items():
        if param.default is not param.empty:
            kwargs[name] = param.default
        elif param.kind is param.VAR_POSITIONAL:
            kwargs[name] = ()
        elif param.kind is param.VAR_KEYWORD:
            kwargs[name] = {}
    return OrderedDict(kwargs)

def _normalize_kwargs(func: Callable, loc: Dict[str, Any]) -> "OrderedDict[str, Any]":
    """ Given a function and local function arguments, normalize the keyword
    arguments by filling in default arguments from function signature

    Example::

    >> def f(self, key1=3, key2=3):
           pass
    >> loc = {"key2": 6}
    >> _normalize_kwargs(f, loc)
    {"key1": 3, "key2": 6}
    """
    default_kwargs = _get_default_kwargs(func)
    local_kwargs = _get_signature_locals(func, loc)
    normalized_kwargs = default_kwargs.copy()
    for attr, val in local_kwargs.items():
        if attr in normalized_kwargs:
            # override the default keyword arguments
            normalized_kwargs[attr] = val
    return normalized_kwargs

def _get_num_pos_args(f: Callable) -> int:
    """ Get number of positional args for a function

    Example::

    >> def f(self, key1=3, key2=3):
           pass
    >> _get_num_pos_args(f)
    3
    """
    return len(getfullargspec(f).args)

def get_fqn_to_example_inputs(
    model: torch.nn.Module,
    example_inputs: Tuple[Any, ...]
) -> Dict[str, Tuple[Any, ...]]:
    """ Given a model and its example inputs, return a dictionary from
    fully qualified name of submodules to example_inputs for that submodule,
    e.g. {"linear1": (tensor1,), "linear2": (tensor2,), "sub": (tensor3,),
          "sub.linear1": (tensor4,), ...}

    Used to make quantizing submodules easier now that FX Graph Mode Quantization requries
    example inputs.

    Also works for keyword arguments with default values, we would flatten keyword
    arguments as positional arguments and fill in the missing keyword args with default
    values, e.g. if we have a forward function:
    def forward(self, x, key1=3, key2=3):
        ...

    and we call it with self.submodule(x, key2=6)
    we'll get example_inputs: (x, 3, 6)

    user can also override `key1` with positional arguments as well:
    for self.submodule(x, 5, key2=6)
    we'll get: (x, 5, 6)

    variable positional arguments and variable positional keyword arguments in forward
    function are not supported currently, so please make sure no submodules is using
    them.
    """
    root = model
    fqn_to_example_inputs = {}

    def _patched_module_call(self, *args, **kwargs):
        submodule_example_inputs = list(args).copy()
        normalized_kwargs = _normalize_kwargs(self.forward, kwargs)
        # minus 1 to skipping counting `self`
        num_args = _get_num_pos_args(self.forward) - 1
        num_to_pop = num_args - len(submodule_example_inputs)
        while num_to_pop and normalized_kwargs:
            normalized_kwargs.popitem(last=False)
            num_to_pop -= 1
        submodule_example_inputs.extend(normalized_kwargs.values())
        submodule_example_inputs_tuple = tuple(submodule_example_inputs)
        fqn = _get_path_of_module(root, self)
        if fqn is not None:
            fqn_to_example_inputs[fqn] = submodule_example_inputs_tuple
        return orig_module_call(self, *args, **kwargs)

    orig_module_call = torch.nn.Module.__call__
    torch.nn.Module.__call__ = _patched_module_call
    try:
        model(*example_inputs)
    finally:
        # restore the module call even if there is an exception
        torch.nn.Module.__call__ = orig_module_call
    return fqn_to_example_inputs

def _get_lstm_with_individually_observed_parts(
    float_lstm: torch.nn.LSTM,
    # Use Callable instead of _PartialWrapper here to avoid circular dependencies
    linear_output_obs_ctr: Optional[Callable] = None,
    sigmoid_obs_ctr: Optional[Callable] = None,
    tanh_obs_ctr: Optional[Callable] = None,
    cell_state_obs_ctr: Optional[Callable] = None,
    hidden_state_obs_ctr: Optional[Callable] = None,
) -> torch.ao.nn.quantizable.LSTM:
    """
    Return an observed `torch.ao.nn.quantizable.LSTM` created from a `torch.nn.LSTM`
    with specific observers or fake quantizes assigned to the inner ops or submodules.

    In both eager and FX graph mode quantization, `torch.ao.nn.quantizable.LSTM` is
    used as an observed custom module, which is responsible for inserting its own
    observers. By default, all inner ops inherit the parent custom module's QConfig.
    Users who wish to override this behavior may extend `torch.ao.nn.quantizable.LSTM`
    and use this helper function to customize the observer insertion logic.

    Args:
        `float_lstm`: The float LSTM module
        `linear_output_obs_ctr`: observer or fake quantize for linear outputs Wx + b,
            where W is the weight matrix, b is the bias, and x is either the inputs
            or the hidden state from the previous layer (if any)
        `sigmoid_obs_ctr`: observer or fake quantize for sigmoid activations
        `tanh_obs_ctr`: observer or fake quantize for tanh activations
        `cell_state_obs_ctr`: observer or fake quantize for the cell state
        `hidden_state_obs_ctr`: observer or fake quantize for the hidden state and
            the output

    Return:
        A `torch.ao.nn.quantizable.LSTM` with the specified observers or fake quantizes
        attached to the inner submodules.
    """
    def make_qconfig(obs_ctr: Callable) -> torch.ao.quantization.QConfig:
        """
        Make a QConfig with fixed qparams observers or fake quantizes.
        """
        if isinstance(obs_ctr(), torch.ao.quantization.FakeQuantizeBase):
            weight = torch.ao.quantization.default_weight_fake_quant
        else:
            weight = torch.ao.quantization.default_weight_observer
        return torch.ao.quantization.QConfig(activation=obs_ctr, weight=weight)

    observed_lstm = torch.ao.nn.quantizable.LSTM(
        float_lstm.input_size, float_lstm.hidden_size, float_lstm.num_layers, float_lstm.bias,
        float_lstm.batch_first, float_lstm.dropout, float_lstm.bidirectional)

    # Assign QConfigs with fixed qparams to all inner submodules
    # Module hierarchy: LSTM > _LSTMLayer > _LSTMSingleLayer (forward or backward) > LSTMCell
    for layer in observed_lstm.layers:
        inner_layers = [layer.layer_fw]
        if float_lstm.bidirectional:
            inner_layers.append(layer.layer_bw)
        for inner_layer in inner_layers:
            cell = inner_layer.cell
            if linear_output_obs_ctr is not None:
                qconfig = make_qconfig(linear_output_obs_ctr)
                cell.igates.qconfig = qconfig
                cell.hgates.qconfig = qconfig
            if sigmoid_obs_ctr is not None:
                qconfig = make_qconfig(sigmoid_obs_ctr)
                cell.input_gate.qconfig = qconfig
                cell.forget_gate.qconfig = qconfig
                cell.output_gate.qconfig = qconfig
            if tanh_obs_ctr is not None:
                cell.cell_gate.qconfig = make_qconfig(tanh_obs_ctr)
            if cell_state_obs_ctr is not None:
                cell.fgate_cx_igate_cgate.qconfig = make_qconfig(cell_state_obs_ctr)
                obs = cell_state_obs_ctr()
                if hasattr(obs, "scale") and hasattr(obs, "zero_point"):
                    cell.initial_cell_state_qparams = (obs.scale, obs.zero_point)
                cell.cell_state_dtype = obs.dtype
            if hidden_state_obs_ctr is not None:
                cell.ogate_cy.qconfig = make_qconfig(hidden_state_obs_ctr)
                obs = hidden_state_obs_ctr()
                if hasattr(obs, "scale") and hasattr(obs, "zero_point"):
                    cell.initial_hidden_state_qparams = (obs.scale, obs.zero_point)
                cell.hidden_state_dtype = obs.dtype

    # Insert the observers based on the previously attached QConfigs
    # Pass in non_leaf_module_list to prevent the observers for sigmoid/tanh from being overridden
    _add_observer_(  # type: ignore[attr-defined]
        observed_lstm,
        non_leaf_module_list=[torch.nn.Sigmoid, torch.nn.Tanh]
    )
    return observed_lstm

_DEFAULT_CUSTOM_CONFIG_DICT = {
    'float_to_observed_custom_module_class': {
        nn.LSTM: nn.quantizable.LSTM,
        nn.MultiheadAttention: nn.quantizable.MultiheadAttention,
    },
    'observed_to_quantized_custom_module_class': {
        nn.quantizable.LSTM: nn.quantized.LSTM,
        nn.quantizable.MultiheadAttention: nn.quantized.MultiheadAttention,
    }
}

def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output
    """
    return self.activation_post_process(output)

def _observer_forward_pre_hook(self, input):
    r"""Forward pre hook that calls observer on the output
    """
    return self.activation_post_process(input[0])

def _register_activation_post_process_hook(module, pre_hook=False):
    assert hasattr(module, 'activation_post_process'), \
        'Expect activation_post_process attribute already attached to the module'
    if pre_hook:
        handle = module.register_forward_pre_hook(
            _observer_forward_pre_hook, prepend=True
        )
    else:
        handle = module.register_forward_hook(
            _observer_forward_hook, prepend=True
        )


def _add_observer_(module, qconfig_propagation_list=None, non_leaf_module_list=None, device=None, custom_module_class_mapping=None):
    r"""Add observer for the leaf child of the module.

    This function insert observer module to all leaf child module that
    has a valid qconfig attribute.

    Args:
        module: input module with qconfig attributes for all the leaf modules that we want to quantize
        qconfig_propagation_list: a list of quantizable modules that will have observers added to them
            if they are leaf nodes
        device: parent device, if any
        non_leaf_module_list: list of non-leaf modules we want to add observer

    Return:
        None, module is modified inplace with added observer modules and forward_hooks
    """
    if qconfig_propagation_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()

    if custom_module_class_mapping is None:
        custom_module_class_mapping = {}

    # respect device affinity when adding observers
    if device is None:
        devices = _get_unique_devices_(module)
        assert len(devices) <= 1, (
            "_add_observer_ only works with cpu or single-device CUDA modules, "
            "but got devices {}".format(devices)
        )
        device = next(iter(devices)) if len(devices) > 0 else None

    def get_activation_post_process(qconfig, device, special_act_post_process=None):
        activation = qconfig.activation() if special_act_post_process is None else special_act_post_process()
        if device is not None:
            activation.to(device)
        return activation

    def needs_observation(m):
        return hasattr(m, 'qconfig') and m.qconfig is not None

    def insert_activation_post_process(m, special_act_post_process=None):
        """ Adds an activation post process module and register
        a pre or post hook that calls the module
        """
        # We don't insert observer/fake_quantize for DeQuantStub
        if needs_observation(m) and not isinstance(m, torch.ao.quantization.DeQuantStub):
            # observer and hook will be gone after we swap the module
            m.add_module('activation_post_process', get_activation_post_process(
                m.qconfig, device, special_act_post_process))
            # Register observer as the first entry in the hook list
            # All post forward hooks are preserved and will be executed after the observer before convert
            _register_activation_post_process_hook(m, pre_hook=_activation_is_memoryless(m.qconfig))

    for name, child in module.named_children():
        # TODO remove Dropout special after codebase stable
        if type_before_parametrizations(child) in [nn.Dropout]:
            continue
        elif type_before_parametrizations(child) in [nnq.FloatFunctional, nnq.QFunctional]:
            if needs_observation(child):
                child.activation_post_process = get_activation_post_process(child.qconfig, device)
        elif isinstance(child, _FusedModule):
            # activation_post_process are now added directly to nn.Sequentail/_FusedModule
            if needs_observation(child):
                insert_activation_post_process(child)
        elif non_leaf_module_list is not None and type_before_parametrizations(child) in non_leaf_module_list:
            if needs_observation(child):
                insert_activation_post_process(child)
        elif _has_special_act_post_process(child):
            special_act_post_process = _get_special_act_post_process(child)
            insert_activation_post_process(child, special_act_post_process)
        elif needs_observation(child) and type_before_parametrizations(child) in custom_module_class_mapping:
            observed_child = custom_module_class_mapping[type_before_parametrizations(child)].from_float(child)
            setattr(module, name, observed_child)
            # TODO: These are the modules that cannot be observed
            #       Once there are more, we should move them to a separate list
            if custom_module_class_mapping[type_before_parametrizations(child)] not in no_observer_set():
                insert_activation_post_process(observed_child)
        else:
            _add_observer_(child, qconfig_propagation_list, non_leaf_module_list, device, custom_module_class_mapping)

    # Insert observers only for leaf nodes, note that this observer is for
    # the output of the module, for input QuantStub will observe them
    if _has_no_children_ignoring_parametrizations(module) and not isinstance(module, torch.nn.Sequential) \
       and type_before_parametrizations(module) in qconfig_propagation_list:
        insert_activation_post_process(module)

def _get_unique_devices_(module):
    return {p.device for p in module.parameters()} | \
        {p.device for p in module.buffers()}

def _remove_activation_post_process(module):
    # TODO: maybe we should change activation_post_process to _activation_post_process
    # to prevent it from being used by user
    if hasattr(module, 'activation_post_process') and \
       torch.ao.quantization.is_activation_post_process(module.activation_post_process):
        delattr(module, 'activation_post_process')

    # remove activation_post_proceess pre and post hooks
    def remove_hooks(pre_hook=False):
        hook_map = module._forward_pre_hooks if pre_hook else module._forward_hooks
        observer_hook = _observer_forward_pre_hook if pre_hook else _observer_forward_hook
        handle_ids_to_remove = set()
        for handle_id, hook_fn in hook_map.items():
            if hook_fn is observer_hook:
                handle_ids_to_remove.add(handle_id)
        for handle_id in handle_ids_to_remove:
            hook_map.pop(handle_id)

    remove_hooks(pre_hook=True)
    remove_hooks(pre_hook=False)

# TODO: rename to something more general
def _remove_qconfig(module):
    r"""Clean up the qconfig left in the module so that new qconfig can be
    propagated.

    Args:
        module: module to be cleaned up
    """
    for child in module.children():
        _remove_qconfig(child)

    if hasattr(module, "qconfig"):
        del module.qconfig

    _remove_activation_post_process(module)

def _get_observer_dict(mod, target_dict, prefix=""):
    r"""Traverse the modules and save all observers into dict.
    This is mainly used for quantization accuracy debug
    Args:
        mod: the top module we want to save all observers
        prefix: the prefix for the current module
        target_dict: the dictionary used to save all the observers
    """
    def get_prefix(prefix):
        return prefix if prefix == "" else prefix + '.'

    if hasattr(mod, 'activation_post_process'):
        target_dict[get_prefix(prefix) + 'activation_post_process'] = mod.activation_post_process
    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_observer_dict(child, target_dict, module_prefix)
