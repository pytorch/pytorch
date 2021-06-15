import torch
import torch.nn as nn
from torch.fx import GraphModule
from torch.fx.graph import Node

from .utils import collect_producer_nodes, get_new_attr_name_with_prefix
from ..observer import (
    PerChannelMinMaxObserver,
    _with_args,
    ObserverBase,
)
from ..utils import check_min_max_valid

from collections import namedtuple
from typing import Dict, Any, Tuple, Optional
import warnings


class _InputEqualizationObserver(nn.Module):
    r"""Observer for tracking the running min/max values of input columns, and
    computing the quantization parameters for the overall min/max input values.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    The running minimum/maximum :math:`x_\text{min/max}` are computed in the
    same way as :class:`~torch.quantization.observer.PerChannelMinMaxObserver`,
    with the difference that the running min/max values are stored per column.

    The qparams are calculated by multiplying the min/max input column values
    with the equalization scale, reducing to find the global min/max input
    values, and then calculating in the same way as in
    :class:`~torch.quantization.observer.MinMaxObserver`

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 quant_min=None, quant_max=None, factory_kwargs=None) -> None:
        super(_InputEqualizationObserver, self).__init__()

        if qscheme not in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            raise TypeError("Input qscheme must be per-tensor")

        self.dtype = dtype
        self.qscheme = qscheme

        self.input_obs = PerChannelMinMaxObserver(ch_axis=1, dtype=dtype,
                                                  qscheme=qscheme,
                                                  quant_min=quant_min,
                                                  quant_max=quant_max,
                                                  factory_kwargs=factory_kwargs)

        self.equalization_scale = torch.empty(0)

    def forward(self, x_orig):
        # TODO: Allow for convoluational layers
        if not (x_orig.ndim == 2):
            raise ValueError("InputEqualizationObserver only supports Linear layers")

        return self.input_obs(x_orig)

    def get_input_minmax(self):
        return (self.input_obs.min_vals, self.input_obs.max_vals)

    def set_equalization_scale(self, equalization_scale):
        self.equalization_scale = equalization_scale

    def calculate_scaled_minmax(self):
        r"""
        Returns the scaled min/max inputs
        """
        if self.equalization_scale.nelement() == 0:
            warnings.warn(
                "Must call calculate_scale before calling calculate_qparams.\
                Returning default min and max input."
            )
            return torch.tensor([0]), torch.tensor([0])

        # Calculate qparams for the scaled min/max inputs
        # Scale the input by the equalization scale located at the same column
        # index
        (min_inputs, max_inputs) = self.get_input_minmax()
        min_input_scaled = torch.min(torch.mul(min_inputs, self.equalization_scale))
        max_input_scaled = torch.max(torch.mul(max_inputs, self.equalization_scale))

        return min_input_scaled, max_input_scaled

    with_args = classmethod(_with_args)


class _WeightEqualizationObserver(nn.Module):
    r"""Observer for tracking the running min/max values of weight columns and
    rows, and computing the quantization parameters for the weight rows.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    This observer is made up of 2 PerChannelMinMaxObservers
        - weight_col_obs: Used to record the running minimum and maximum of
        columns of incoming weight tensors
        - weight_row_obs: Used to record the running minimum and maximum of
        rows of incoming weight tensors

    The running minimum/maximum :math:`w_\text{min/max}` are computed in the
    same way as :class:`~torch.quantization.observer.PerChannelMinMaxObserver`.

    The qparams are calculated by multiplying the min/max weight row values
    with the inverse of the equalization scale, and then calculating in the same
    way as in :class:`~torch.quantization.observer.PerChannelMinMaxObserver`

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, quant_min=None,
                 quant_max=None, factory_kwargs=None) -> None:
        super(_WeightEqualizationObserver, self).__init__()

        self.dtype = dtype
        self.qscheme = qscheme
        self.ch_axis = 0

        self.weight_col_obs = PerChannelMinMaxObserver(ch_axis=1, dtype=dtype,
                                                       qscheme=qscheme,
                                                       quant_min=quant_min,
                                                       quant_max=quant_max,
                                                       factory_kwargs=factory_kwargs)

        self.equalization_scale = torch.empty(0)

    def forward(self, w_orig):
        # TODO: Allow for convoluational layers
        if not (w_orig.ndim == 2):
            raise ValueError("WeightEqualizationObserver only supports Linear layers")
        return self.weight_col_obs(w_orig)

    def get_weight_col_minmax(self):
        return (self.weight_col_obs.min_vals, self.weight_col_obs.max_vals)

    def set_equalization_scale(self, equalization_scale):
        self.equalization_scale = equalization_scale

    with_args = classmethod(_with_args)


def calculate_equalization_scale(input_obs: _InputEqualizationObserver,
                                 weight_obs: _WeightEqualizationObserver) -> torch.Tensor:
    r""" Calculates the equalization scale and sets the equalization_scale value
    in the observers.

    Args:
        input_obs: Observer that tracks the ranges for the input columns
        weight_obs: Observer that tracks the ranges for the weight columns
    """

    (min_inputs, max_inputs) = input_obs.get_input_minmax()
    (min_weights, max_weights) = weight_obs.get_weight_col_minmax()

    if not (check_min_max_valid(min_inputs, max_inputs) and check_min_max_valid(min_weights, max_weights)):
        return torch.tensor(1)

    if not (min_inputs.shape == min_weights.shape):
        raise ValueError(
            "Input and Weight must have the same column dimension. " +
            f"Found {min_inputs.shape} and {max_inputs.shape} instead."
        )

    equalization_scale = torch.sqrt((max_weights - min_weights) / (max_inputs - min_inputs))
    return equalization_scale


class EqualizationQConfig(namedtuple('EqualizationQConfig', ['input_activation', 'weight'])):
    """
    Describes how to quantize a layer or a part of the network specifically for
    input-weight equalization by providing settings (observer classes) for
    inputs, outputs, and weights.

    Note that EqualizationQConfig needs to contain observer **classes** (like
    MinMaxObserver) or a callable that returns instances on invocation, not the
    concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of
    the layers.

    Observer classes have usually reasonable default arguments, but they can be
    overwritten with `with_args` method (that behaves like functools.partial):

    my_qconfig = EqualizationQConfig(input_activation=_InputEqualizationObserver.with_args(dtype=torch.qint8),
                                    weight=_WeightEqualizationObserver.with_args(dtype=torch.qint8))
    """
    def __new__(cls, input_activation=torch.nn.Identity, weight=torch.nn.Identity):
        if isinstance(input_activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError("EqualizationQConfig received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        self = super(EqualizationQConfig, cls).__new__(cls, input_activation, weight)
        return self


input_equalization_observer = _InputEqualizationObserver.with_args(
    dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_equalization_observer = _WeightEqualizationObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
default_equalization_qconfig = EqualizationQConfig(input_activation=input_equalization_observer,
                                                   weight=weight_equalization_observer)

def node_supports_equalization(node: Node, modules) -> bool:
    """ Checks if the current node supports equalization
    Currently we only support nn.Linear and F.Linear layers
    """
    if node.op == 'call_module':
        return isinstance(modules[node.target], nn.Linear)
    elif node.op == 'call_function':
        return node.target == nn.functional.linear
    return False

def get_weight_eq_obs(
    input_eq_obs_node: Node,
    model: GraphModule,
    modules: Dict[str, nn.Module]
) -> Tuple[Optional[Node], Optional[_WeightEqualizationObserver]]:
    """ Gets the following weight equalization observer. There should always
    exsist a weight equalization observer after an input equalization observer.

    Returns the node containing the weight equalization observer, and the weight
    equalization observer if it has been newly created
    """

    # Find the op node that comes directly after the input equaliation observer
    op_node = None
    for user in input_eq_obs_node.users.keys():
        if node_supports_equalization(user, modules):
            op_node = user
            break

    if op_node is None:
        return None, None

    elif op_node.op == 'call_module':
        # If the next op_node is a nn.Linear layer, then it must have a
        # WeightEqualizationObserver configuration
        equalization_qconfig_map: Dict[str, Any] = model._equalization_qconfig_map  # type: ignore[assignment]
        assert(equalization_qconfig_map.get(op_node.name, None) is not None)
        weight_eq_obs = equalization_qconfig_map.get(op_node.name, None).weight()

        assert(isinstance(weight_eq_obs, _WeightEqualizationObserver))
        return op_node, weight_eq_obs

    elif op_node.op == 'call_function':
        assert(isinstance(op_node.args[1], Node))
        weight_observer_nodes = collect_producer_nodes(op_node.args[1])
        if weight_observer_nodes is None:
            return None, None

        for weight_node in weight_observer_nodes:
            if weight_node.op == 'call_module' and isinstance(weight_node.target, str) and \
               isinstance(modules[weight_node.target], _WeightEqualizationObserver):
                return weight_node, None

        return None, None

    return None, None

def get_next_input_obs(node: Node, modules: Dict[str, nn.Module]):
    """ Gets the following input observer. We can use this function for both
    obtaining the next input equalization observer and the next input
    quantization observer

    Returns the node containing the input observer, and the input observer
    """
    node = node.next
    while node.op not in ('call_module', 'call_function', 'output'):
        node = node.next
    if node.op in ('output', 'call_function'):
        return None, None

    assert(isinstance(node.target, str))
    input_obs = modules[node.target]
    return node, input_obs

def maybe_get_next_equalization_scale(node: Node, modules: Dict[str, nn.Module]) -> Optional[torch.Tensor]:
    """ If the next next node is an InputEqualizationObserver then we want to
    return its equalization scale, else we return 1

    This is used in the case where there are two connecting linear layers:
        linear1 -> LinearOutObs -> InputEqObs -> linear2
    In this case, the node given is linear1 and we want to locate the InputEqObs.
    """
    # TODO: This code seems a little hacky/hard-coded, so any suggestions for
    # how to improve it would be greatly appreciated!
    next_node, _ = get_next_input_obs(node, modules)
    _, next_next_obs = get_next_input_obs(next_node, modules)
    if isinstance(next_next_obs, _InputEqualizationObserver):
        return next_next_obs.equalization_scale
    return None

def scale_input_observer(node: Node, modules: Dict[str, nn.Module]) -> None:
    """ Scales the following input quantization observer's min/max values by
    updating the values with the scaled min/max values calculated by the input
    equalization observer
    """
    assert(isinstance(node.target, str))
    input_eq_obs = modules[node.target]
    assert(isinstance(input_eq_obs, _InputEqualizationObserver))

    input_quant_obs_node = node.args[0]
    assert(isinstance(input_quant_obs_node, Node) and isinstance(input_quant_obs_node.target, str))
    input_quant_obs = modules[input_quant_obs_node.target]

    if not isinstance(input_quant_obs, ObserverBase):
        return

    min_input_scaled, max_input_scaled = input_eq_obs.calculate_scaled_minmax()
    input_quant_obs.min_val = min_input_scaled
    input_quant_obs.max_val = max_input_scaled

def scale_weight_node(
    node: Node,
    modules: Dict[str, nn.Module],
    equalization_scale: torch.Tensor,
    next_equalization_scale: Optional[torch.Tensor],
) -> None:
    """ Scale the weights for input-weight equalization by multiplying the
    weight by 1/equalization_scale and next_equalization_scale

    Args:
        node: Current node whose weights we want to scale
        equalization_scale: Current node's calculated equalization scale
        next_equalization_scale: Next node's calculated equalization scale if
           the following node needs to be equalized, 1 otherwise
    """
    assert(isinstance(node.target, str))

    # Scale the weights for input-weight equalization
    # If the following layer needs to be equalized then we will multiply its scale
    weight = modules[node.target].weight
    assert(isinstance(weight, torch.Tensor))

    scaled_weight = torch.mul(weight, torch.reciprocal(equalization_scale))

    if next_equalization_scale is None:
        modules[node.target].weight = nn.Parameter(scaled_weight)
        return

    # Multiply the weights row wise by the next equalization scale
    new_shape = [1] * weight.ndim
    new_shape[0] = weight.size(0)
    next_equalization_scale_reshaped = torch.reshape(next_equalization_scale, new_shape)
    scaled_weight = torch.mul(scaled_weight, next_equalization_scale_reshaped)

    modules[node.target].weight = nn.Parameter(scaled_weight)

    # Multiply the bias element wise by the next equalization scale
    bias = modules[node.target].bias
    assert(isinstance(bias, torch.Tensor))

    scaled_bias = torch.mul(bias, next_equalization_scale)
    modules[node.target].bias = nn.Parameter(scaled_bias)

def scale_weight_functional(node: Node, model: GraphModule, equalization_scale: torch.Tensor):
    """ Scales the weight value for functional layers
    """

    # Find the next functional node so that we can construct the weight observer nodes
    while node.op != 'output' and node.op != 'call_function':
        node = node.next

    if node.op == 'output':
        return

    assert(isinstance(node.args[1], Node))
    weight_observer_nodes = collect_producer_nodes(node.args[1])
    if weight_observer_nodes is None:
        return

    for weight_node in weight_observer_nodes:
        # Find the node containing the weight values
        if weight_node.op == 'get_attr':
            assert(isinstance(weight_node.target, str))
            weight = model.get_buffer(weight_node.target)

            # Scale the weights for input-weight equalization
            # If the following layer needs to be equalized then we will multiply its scale
            scaled_weight = torch.mul(weight, torch.reciprocal(equalization_scale))
            # TODO: connecting functional layers?
            # scaled_weight = torch.mul(scaled_weight, next_equalization_scale)

            # Set the weight with the newly scaled weight in the original model
            # weight_node.target looks something like "linear1.w", so we want to
            # get rid of the last two characters to retrieve the path to the
            # submodule containing the weight
            model.get_submodule(weight_node.target[:-2]).w = scaled_weight

def update_obs_for_equalization(model: GraphModule, modules: Dict[str, nn.Module]) -> Dict[str, _WeightEqualizationObserver]:
    """ Update all of the observer's equalization scale. For each
    InputEqualizationObserver, we will find the location of the next
    WeightEqualizationObserver, create it, and calculate the equalization scale
    based on the two observers.

    We will then return a dictionary mapping node names to the newly created
    WeightEqualizationObserver.
    """
    weight_eq_obs_dict = {}
    for node in model.graph.nodes:
        if node.op == 'call_module' and isinstance(modules[node.target], _InputEqualizationObserver):
            input_eq_obs = modules[node.target]
            assert(isinstance(input_eq_obs, _InputEqualizationObserver))
            next_node, weight_eq_obs = get_weight_eq_obs(node, model, modules)

            if next_node is None:
                continue

            assert(isinstance(next_node.target, str))
            if weight_eq_obs is None:
                # If the equalization observer has already been created, then we
                # can extract the observer from the node itself
                weight_eq_obs = modules[next_node.target]  # type: ignore[assignment]
                assert(isinstance(weight_eq_obs, _WeightEqualizationObserver))
            else:
                weight_eq_obs(modules[next_node.target].weight)

            # Calculate and set the equalization scale values
            equalization_scale = calculate_equalization_scale(input_eq_obs, weight_eq_obs)
            input_eq_obs.set_equalization_scale(equalization_scale)
            weight_eq_obs.set_equalization_scale(equalization_scale)

            weight_eq_obs_dict[next_node.name] = weight_eq_obs

    return weight_eq_obs_dict

def convert_eq_obs(
    model: GraphModule,
    modules: Dict[str, nn.Module],
    weight_eq_obs_dict: Dict[str, _WeightEqualizationObserver],
) -> None:
    """ Removes the input equalization observers and replaces them with mul
    operators whenever applicable. Updates the input quantization observers with
    the scaled input min/max values. Scales the weights by the current and next
    equalization scales.

    Before:
                                     weight values
                                          |
                                      WeightEqObs
                                          |
                                     WeightQuantObs
                                          |
        x -> InpQuantObs -> InpEqObs -> linear -> OutQuantObs

    After:
                                              scaled weight values
                                                      |
                                                 WeightEqObs
                                                      |
       equalization scale                       WeightQuantObs
              |                                       |
        x -> mul -> InpQuantObs (scaled min/max) -> linear -> OutQuantObs
    """
    for node in model.graph.nodes:
        if node.op == 'call_module' and isinstance(modules[node.target], _InputEqualizationObserver):
            inp_quant_obs_node = node.args[0]
            prev_node = inp_quant_obs_node.args[0]

            if prev_node.op == 'call_module' and isinstance(prev_node.target, str) and \
               isinstance(modules[prev_node.target], nn.Linear):
                # If this is a connecting linear layer, we want to remove the
                # InputEqualizationObserver (current node)
                orig_users = list(node.users.keys())
                for user_node in orig_users:
                    user_node.replace_input_with(node, inp_quant_obs_node)
                # Erase the node
                model.graph.erase_node(node)
                continue

            # Update the following input quantization observer's min/max values
            scale_input_observer(node, modules)

            # Remove the InputEqualization node and add a mul operator before
            # the quantization observer node that appears before the equalization node
            # Before: x -> input_quant_obs -> input_eq_obs -> linear
            # After: x -> mul -> input_quant_obs -> linear

            # Create a node containing the equalization scale
            with model.graph.inserting_before(inp_quant_obs_node):
                get_new_eq_scale_name = get_new_attr_name_with_prefix(prev_node.name + '_equalization_scale')
                name = get_new_eq_scale_name(modules)
                setattr(model, name, modules[node.target].equalization_scale)
                eq_scale_node = model.graph.create_node('get_attr', name)

            # Create a node multiplying the input with the equalization scale
            with model.graph.inserting_after(eq_scale_node):
                inputs = (prev_node, eq_scale_node)
                mul_node = model.graph.create_node("call_function", torch.mul, inputs)

            # Set the mul nod to be the input_quant_obs_node's input instead of
            # the previous node
            inp_quant_obs_node.replace_input_with(prev_node, mul_node)

            # For all of the current node's users, replace the current node with
            # the input quantization observer node
            orig_users = list(node.users.keys())
            for user_node in orig_users:
                user_node.replace_input_with(node, inp_quant_obs_node)

            # Erase the InputEqualizationObserver node
            model.graph.erase_node(node)

        elif weight_eq_obs_dict.get(node.name, None) is not None:
            weight_eq_obs = weight_eq_obs_dict.get(node.name)
            assert(isinstance(weight_eq_obs, _WeightEqualizationObserver))
            assert(isinstance(node.target, str))

            if (modules[node.target] == weight_eq_obs):
                # If the WeightEqualizationObserver is already in the node
                # itself, this implies we are using functional layers(?) so we
                # need to scale the weights differently
                equalization_scale = weight_eq_obs.equalization_scale
                scale_weight_functional(node, model, equalization_scale)

                # Erase the weight equalization observer node
                prev_node = node.args[0]
                orig_users = list(node.users.keys())
                for user_node in orig_users:
                    user_node.replace_input_with(node, prev_node)
                model.graph.erase_node(node)
                continue

            equalization_scale = weight_eq_obs.equalization_scale

            # Scales the weights and runs the weight quantization observers
            maybe_next_equalization_scale = maybe_get_next_equalization_scale(node, modules)
            scale_weight_node(node, modules, equalization_scale, maybe_next_equalization_scale)

def _convert_equalization_ref(model: GraphModule):
    """ Reference function which applies changes needed for equalization, but
    does not quantize the nodes
    """
    modules = dict(model.named_modules(remove_duplicate=False))

    # Calculate the equalization scale, update the observers with the scaled
    # inputs, and scale the weight
    weight_eq_obs_dict = update_obs_for_equalization(model, modules)
    convert_eq_obs(model, modules, weight_eq_obs_dict)

    return GraphModule(model, model.graph)
