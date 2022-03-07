import torch
import operator
import warnings
from torch.fx import (
    GraphModule,
)
from torch.fx.graph import (
    Graph,
    Node,
)
from torch.fx.node import Argument

from ..quantize import (
    propagate_qconfig_,
)
from ..observer import (
    ObserverBase,
)
from ..qconfig import QConfigAny, is_reuse_input_qconfig
from ..qconfig_dict_utils import (
    get_flattened_qconfig_dict,
    convert_dict_to_ordered_dict,
    update_qconfig_for_qat,
)
from .qconfig_utils import (
    generate_qconfig_map,
    update_qconfig_for_fusion,
    get_standalone_module_configs,
)

from .quantization_patterns import (
    QuantizeHandler,
    CustomModuleQuantizeHandler,
    StandaloneModuleQuantizeHandler,
)

from .quantization_types import Pattern

from ._equalize import (
    is_equalization_observer,
    node_supports_equalization,
)

from .graph_module import (
    ObservedGraphModule,
    ObservedStandaloneGraphModule,
)

from .pattern_utils import (
    MatchResult,
    get_default_quant_patterns,
)

from .match_utils import (
    find_matches,
)

from ..utils import _parent_name
from .utils import (
    get_custom_module_class_keys,
    all_node_args_have_no_tensors,
    assert_and_get_unique_device,
    get_non_observable_arg_indexes_and_types,
    get_new_attr_name_with_prefix,
    NON_QUANTIZABLE_WEIGHT_OPS,
    WEIGHT_INDEX_DICT,
    BIAS_INDEX_DICT,
)

from ..quantization_mappings import (
    get_default_qat_module_mappings,
)

from torch.ao.quantization.quantize import (
    is_activation_post_process,
    convert
)

from ..utils import (
    get_combined_dict,
    get_qconfig_dtypes,
    get_swapped_custom_module_class,
    activation_is_statically_quantized,
    activation_is_int8_quantized,
)

from .backend_config.utils import (
    get_pattern_to_quantize_handlers,
    get_pattern_to_dtype_configs,
    get_pattern_to_input_type_to_index,
    get_module_to_qat_module,
)

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
from collections import defaultdict

# list of dtypes to not add observers to
DO_NOT_OBS_DTYPE_LIST = [int, float, torch.bool, None]

def is_activation_post_process_node(node: Node, modules: Dict[str, torch.nn.Module]) -> bool:
    return isinstance(node, torch.fx.Node) and node.op == "call_module" and \
        is_activation_post_process(modules[str(node.target)])

def node_arg_is_weight(node: Node, arg: Any) -> bool:
    if isinstance(node, Node) and node.op == 'call_function' and \
            node.target in WEIGHT_INDEX_DICT:
        for i, node_arg in enumerate(node.args):
            if arg is node_arg and i in \
                    WEIGHT_INDEX_DICT[node.target]:  # type: ignore[index]
                return True
        for kwarg_name, kwarg_value in node.kwargs.items():
            if kwarg_name == 'weight' and arg is kwarg_value:
                return True
    return False

def node_arg_is_bias(node: Node, arg: Any) -> bool:
    if not isinstance(node, Node) or node.op != 'call_function' or \
       node.target not in BIAS_INDEX_DICT:
        return False

    for i, node_arg in enumerate(node.args):
        if arg is node_arg and i in \
           BIAS_INDEX_DICT[node.target]:  # type: ignore[index]
            return True

    return node.kwargs.get('bias', None) is arg

def is_input_arg_dtype_supported_by_backend(
    arg: Argument,
    node: Node,
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
    dtype_config: Dict[str, torch.dtype],
) -> bool:
    """ Check if the configured qconfig for the argument
    is supported by the backend or not
    """
    if isinstance(arg, (list, tuple)):
        return all(map(lambda a: is_input_arg_dtype_supported_by_backend(a, node, node_name_to_target_dtype, dtype_config), arg))
    if not isinstance(arg, Node):
        return True
    # TODO: support check for standalone module
    is_weight = node_arg_is_weight(node, arg)
    is_bias = node_arg_is_bias(node, arg)
    is_activation = not is_weight and not is_bias
    if is_activation:
        input_activation_dtype = dtype_config.get("input_activation_dtype", None)
        return input_activation_dtype is None or \
            node_name_to_target_dtype[node.name]["input_activation_dtype"] == input_activation_dtype
    elif is_weight:
        weight_dtype = dtype_config.get("weight_dtype", None)
        return weight_dtype is None or node_name_to_target_dtype[node.name]["weight_dtype"] == weight_dtype
    else:  # bias
        bias_dtype = dtype_config.get("bias_dtype", None)
        return bias_dtype is None or node_name_to_target_dtype[node.name]["bias_dtype"] == bias_dtype

def is_output_dtype_supported_by_backend(
    node: Node,
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
    dtype_config: Dict[str, torch.dtype],
) -> bool:
    """ Check if the configured qconfig for the output
    is supported by the backend or not
    """
    output_dtype = dtype_config.get("output_dtype", None)
    return output_dtype is None or \
        output_dtype == node_name_to_target_dtype[node.name]["output_activation_dtype"]

def is_observer_in_same_graph(node, modules, node_name_to_target_dtype):
    """ Check if observer in same graph
    when the node output is not fp32 and input is 'placeholder'
    the input is assumed to be quantized, so it is observed
    in a different place rather than not observed.
    """
    node_output_dtype = get_arg_target_dtype_as_output(node, modules, node_name_to_target_dtype)
    if isinstance(node.args[0], Node):
        if node_output_dtype == torch.quint8 and node.args[0].op == 'placeholder':
            return False
    return True

def is_pattern_dtype_config_supported_by_backend(
    pattern: Optional[Pattern],
    matched_nodes: Optional[List[Node]],
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
    backend_config_dict: Optional[Dict[str, Any]]
) -> bool:
    """ Check is the dtype configuration of a pattern is supported by
    the backend or not
    """
    if backend_config_dict is None or pattern is None:
        return True
    assert matched_nodes is not None and len(matched_nodes) >= 1
    pattern_to_dtype_configs = get_pattern_to_dtype_configs(backend_config_dict)
    dtype_configs: List[Dict[str, torch.dtype]] = pattern_to_dtype_configs.get(pattern, [])

    # TODO: this only checks one input and one output, need to generalize to multiple
    # inputs/output
    input_node = matched_nodes[-1]
    output_node = matched_nodes[0]
    for dtype_config in dtype_configs:
        # check if arg dtype are supported
        supported = True
        for arg in input_node.args:
            supported = supported and \
                is_input_arg_dtype_supported_by_backend(
                    arg, input_node, node_name_to_target_dtype, dtype_config)
        for k, arg in input_node.kwargs.items():
            supported = supported and \
                is_input_arg_dtype_supported_by_backend(
                    arg, input_node, node_name_to_target_dtype, dtype_config)
        # check if output dtype is supported
        supported = supported and is_output_dtype_supported_by_backend(
            output_node, node_name_to_target_dtype, dtype_config)
        if supported:
            return True
    return False

def prepare_get_standalone_module_configs(
    node: Node,
    modules: Dict[str, torch.nn.Module],
    prepare_custom_config_dict: Dict[str, Any],
    parent_qconfig: QConfigAny,
    parent_backend_config_dict: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Returns the standalone module qconfig_dict and prepare_config_dict
    for `node`, assuming that the module pointed to by `node` is
    a standalone modules.
    """
    standalone_module_name = str(node.target)
    standalone_module_type = type(modules[standalone_module_name])  # type: ignore[index]
    sm_qconfig_dict, sm_prepare_config_dict, sm_backend_config_dict = \
        get_standalone_module_configs(standalone_module_name, standalone_module_type, prepare_custom_config_dict)
    # fallback to use parent module's qconfig if user didn't specify qconfig dict
    if sm_qconfig_dict is None:
        sm_qconfig_dict = {"": parent_qconfig}
    if sm_prepare_config_dict is None:
        sm_prepare_config_dict = {}
    # TODO: sm_backend_config_dict can fallback to use parent's backend_config_dict
    # as well, this can be added later
    if sm_backend_config_dict is None:
        sm_backend_config_dict = parent_backend_config_dict
    return sm_qconfig_dict, sm_prepare_config_dict, sm_backend_config_dict

def qat_swap_modules(
        root: torch.nn.Module,
        module_to_qat_module: Dict[Callable, Callable]) -> None:
    convert(root, mapping=module_to_qat_module, inplace=True, remove_qconfig=False)

# TODO: remove observed_op, looks like it's not used
def insert_observer(
    node: Node,
    observed_op: Node,
    observer: ObserverBase,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    """
    Attaches `observer` to `model`, and creates a node which calls
    `observer` on the output of `node`.
    """
    model_device = assert_and_get_unique_device(model)
    if model_device:
        observer.to(model_device)
    # add observer module as attribute
    if is_equalization_observer(observer):
        prefix = node.name + '_equalization_process_'
    else:
        prefix = 'activation_post_process_'
    get_new_observer_name = get_new_attr_name_with_prefix(prefix)
    observer_name = get_new_observer_name(model)
    setattr(model, observer_name, observer)
    modules[observer_name] = observer
    with graph.inserting_after(node):
        new_obs = graph.create_node(
            'call_module', observer_name, (node,), {})
    return new_obs

def get_target_activation_dtype_for_node(
    node: Node,
    qconfig: QConfigAny,
    inputs_seen_counter: int,
    outputs_seen_counter: int,
    input_quantized_idxs: List[int],
    output_quantized_idxs: List[int],
    qhandler: Optional[QuantizeHandler],
    modules: Dict[str, torch.nn.Module],
    cache_for_no_tensor_check: Dict[Node, bool],
) -> Dict[str, Optional[Union[torch.dtype, type]]]:
    """
    Returns the expected dtype of the input and output of this node after
    convert. If the value is not None, it represents the dtype of the
    Tensor. If the value is None, it means the value is not a Tensor.

    Note: this is for activations only, weight dtypes are not handled here.

    TODO(future PR, if needed): explicitly spell out the non-Tensor
    dtypes.
    """
    if node.op == 'placeholder':
        if inputs_seen_counter in input_quantized_idxs:
            return {
                "input_activation_dtype": torch.quint8,
                "output_activation_dtype": torch.quint8,
            }
        else:
            # if dtype is fp32 (default), do nothing
            # note: other dtypes are not supported
            return {
                "input_activation_dtype": torch.float,
                "output_activation_dtype": torch.float,
            }

    elif node.op in ('call_module', 'call_method', 'call_function'):
        args_have_no_tensors = \
            all_node_args_have_no_tensors(
                node, modules, cache_for_no_tensor_check)
        if args_have_no_tensors:
            return {
                "input_activation_dtype": None,
                "output_activation_dtype": None,
            }

        # TODO(future PR): consider stopping matching getitem
        is_getitem = node.op == 'call_function' and \
            node.target == operator.getitem
        if is_getitem:
            return {
                "input_activation_dtype": torch.float,
                "output_activation_dtype": torch.float,
            }

        # get qconfig to determine the eventual dtype of this node
        if qconfig is not None:
            if qhandler is not None and qhandler.input_output_observed() and qhandler.is_output_quantized(qconfig):
                act_dtype, weight_dtype, act_compute_dtype = \
                    get_qconfig_dtypes(qconfig)
                bias_dtype = torch.float16 \
                    if act_dtype == torch.float16 and weight_dtype == torch.float16 \
                    else torch.float
                return {
                    "input_activation_dtype": act_dtype,
                    "weight_dtype": weight_dtype,
                    "bias_dtype": bias_dtype,
                    "output_activation_dtype": act_dtype,
                }
        return {
            "input_activation_dtype": torch.float,
            "output_activation_dtype": torch.float,
        }

    elif node.op == 'get_attr':
        return {
            "input_activation_dtype": torch.float,
            "output_activation_dtype": torch.float,
        }

    elif node.op == 'output':
        if outputs_seen_counter in output_quantized_idxs:
            return {
                "input_activation_dtype": torch.quint8,
                "output_activation_dtype": torch.quint8
            }
        else:
            # if dtype is fp32 (default), do nothing
            # note: other dtypes are not supported
            return {
                "input_activation_dtype": torch.float,
                "output_activation_dtype": torch.float,
            }

    else:
        raise AssertionError(f'need to handle {node.format_node()}')

def get_arg_target_dtype_as_output(
    arg: Node,
    modules: Dict[str, torch.nn.Module],
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
) -> Optional[Union[torch.dtype, type]]:
    """ Get the target output activation dtype for
    the argumnet in the original graph, skipping inserted observers
    We are assuming that the observers are inserted correctly, and the dtype for
    argument in quantized graph will match what is specified by the qconfig
    """
    assert isinstance(arg, Node)
    if is_activation_post_process_node(arg, modules):
        observed_arg = arg.args[0]
        assert isinstance(observed_arg, Node), "Currently we only support observing Node"
        return node_name_to_target_dtype[observed_arg.name]["output_activation_dtype"]
    else:
        return node_name_to_target_dtype[arg.name]["output_activation_dtype"]

def get_arg_target_dtype_as_input_to_node(
    arg: Node,
    node: Node,
    modules: Dict[str, torch.nn.Module],
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
) -> Optional[Union[torch.dtype, type]]:
    """ Get the target argument dtype for the argument `arg`, as input
    to node `node`
    """
    assert isinstance(arg, Node)
    is_weight = node_arg_is_weight(node, arg)
    is_bias = node_arg_is_bias(node, arg)
    is_activation = not is_weight and not is_bias
    if is_activation:
        return node_name_to_target_dtype[node.name]["input_activation_dtype"]
    elif is_weight:
        if node.target in NON_QUANTIZABLE_WEIGHT_OPS:
            return None
        else:
            return node_name_to_target_dtype[node.name]["weight_dtype"]
    else:
        return node_name_to_target_dtype[node.name]["bias_dtype"]


def maybe_insert_input_observer_for_arg_or_kwarg(
    node: Union[Node, Any],
    arg: Argument,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
    graph: Graph,
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
    qhandler: Optional[QuantizeHandler],
    prepare_custom_config_dict: Dict[str, Any],
    backend_config_dict: Optional[Dict[str, Any]],
) -> Argument:
    """
    Given a `node` and an `arg`, inserts an input observer between
    `node` and `arg` if necessary.
    """
    # for ops such as torch.cat([x0, x1]),
    # traverse through the list
    if isinstance(arg, (list, tuple)):
        new_arg_to_return = []
        for inner_arg in arg:
            new_inner_arg = maybe_insert_input_observer_for_arg_or_kwarg(
                node, inner_arg, qconfig, model, modules,
                graph, node_name_to_target_dtype,
                qhandler,
                prepare_custom_config_dict,
                backend_config_dict)
            new_arg_to_return.append(new_inner_arg)
        return type(arg)(new_arg_to_return)

    if not isinstance(arg, Node):
        return arg
    assert isinstance(arg, Node)
    # default (no observer)
    new_arg = arg

    is_standalone_module = qhandler is not None and \
        isinstance(qhandler, StandaloneModuleQuantizeHandler)
    assert qconfig is not None
    if not is_standalone_module:
        # regular flow for most nodes, except standalone modules
        is_weight = node_arg_is_weight(node, arg)

        is_reuse_input_qconfig_ = is_reuse_input_qconfig(qconfig)

        act_post_process_ctr = qconfig.weight if is_weight else \
            qconfig.activation

        arg_as_output_target_dtype = get_arg_target_dtype_as_output(arg, modules, node_name_to_target_dtype)
        arg_as_input_target_dtype = get_arg_target_dtype_as_input_to_node(arg, node, modules, node_name_to_target_dtype)
        needs_obs = (
            # if the dtypes are different, we need an observer
            (arg_as_output_target_dtype != arg_as_input_target_dtype) and
            # except if the second dtype is float, a dequant will be inserted
            # without an observer in convert
            # TODO(future PR): change this so a placeholder is inserted for
            # future dequants, to make the logic easier to understand
            (arg_as_input_target_dtype != torch.float) and
            # if arg output dtype is in DO_NOT_OBS_DTYPE_LIST do not insert observer
            (arg_as_output_target_dtype not in DO_NOT_OBS_DTYPE_LIST) and
            # if qconfig is reuse_input qconfig, we won't insert extra observer for input
            not is_reuse_input_qconfig_
        )

    else:
        # custom flow for standalone modules
        _sm_qconfig_dict, sm_prepare_config_dict, _sm_backend_config_dict = \
            prepare_get_standalone_module_configs(
                node, modules, prepare_custom_config_dict, qconfig, backend_config_dict)

        sm_input_quantized_idxs = \
            sm_prepare_config_dict.get('input_quantized_idxs', [])
        # for args, this is set to the index of the current arg
        # for kwargs, this is left at None
        cur_input_idx = None
        for arg_idx, arg_to_check in enumerate(node.args):
            if arg_to_check is arg:
                cur_input_idx = arg_idx
                break

        if cur_input_idx is None:
            needs_obs = False
        else:
            arg_as_output_target_dtype = get_arg_target_dtype_as_output(arg, modules, node_name_to_target_dtype)
            arg_as_input_target_dtype = torch.quint8 if cur_input_idx in sm_input_quantized_idxs \
                else torch.float
            needs_obs = (
                (arg_as_output_target_dtype != arg_as_input_target_dtype) and
                (arg_as_input_target_dtype != torch.float)
            )

        act_post_process_ctr = qconfig.activation

    if needs_obs:

        new_obs_mod = act_post_process_ctr()
        existing_obs_node = None

        # Before using the new observer, check if an observer
        # of the correct type already exists. If it does, use it.
        # This prevents duplicate observer insertions if a node is
        # used by multiple nodes.
        # TODO: this is looking into how the value is used in the future
        # we should remove this
        # removing this means we insert one observer for each use, even if they
        # have the same dtype, we can have an extra pass that removes the extra observers
        for maybe_obs_node, _ in arg.users.items():
            if maybe_obs_node.op == 'call_module':
                maybe_obs_mod = modules[maybe_obs_node.target]  # type: ignore[index]
                if (
                    type(maybe_obs_mod) == type(new_obs_mod) and
                    maybe_obs_mod.dtype == arg_as_input_target_dtype
                ):
                    existing_obs_node = maybe_obs_node
                    break

        if existing_obs_node is None:
            new_obs_node = insert_observer(
                arg, node, new_obs_mod, model, modules, graph)
            # override this arg to be the observed arg
            new_arg = new_obs_node
        else:
            new_arg = existing_obs_node

    return new_arg


def maybe_insert_input_observers_for_node(
    node: Node,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
    graph: Graph,
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
    qhandler: Optional[QuantizeHandler],
    prepare_custom_config_dict: Dict[str, Any],
    backend_config_dict: Optional[Dict[str, Any]],
) -> None:
    """
    If needed, inserts observers to the input args and kwargs of `node`.
    Note: modifies `node` inplace.

    For example, if cur_node needs an observer after prev_node, we change from

      prev_node -> cur_node

    To

      prev_node -> obs -> cur_node
    """
    if qconfig is None:
        # if quantization is turned off for this node, we do not need
        # to insert input observers
        return
    assert qconfig is not None

    # Look through every input arg.  If that arg's target dtype does not
    # match the current node's target dtype, insert an observer.
    new_args = []
    for arg in node.args:
        new_arg = maybe_insert_input_observer_for_arg_or_kwarg(
            node, arg, qconfig, model, modules, graph,
            node_name_to_target_dtype,
            qhandler,
            prepare_custom_config_dict,
            backend_config_dict)
        new_args.append(new_arg)

    new_kwargs = {}
    for k, kwarg in node.kwargs.items():
        new_kwarg = maybe_insert_input_observer_for_arg_or_kwarg(
            node, kwarg, qconfig, model, modules, graph,
            node_name_to_target_dtype,
            qhandler,
            prepare_custom_config_dict,
            backend_config_dict)
        new_kwargs[k] = new_kwarg

    # assign the new args and kwargs to the node, inplace
    node.args = tuple(new_args)
    node.kwargs = new_kwargs

def maybe_insert_input_equalization_observers_for_node(
    node: Node,
    equalization_qconfig: Any,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
    graph: Graph,
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
    is_branch: bool,
) -> None:
    """
    If `node` needs to be equalized, find the input/weight observers it needs in
    `equalization_qconfig`, creates them, and inserts it into `graph`.

    If `node` does not need an equalization observer, returns None.
    """
    if equalization_qconfig is None or not node_supports_equalization(node, modules):
        return

    if is_branch:
        warnings.warn(
            f"Cannot equalize {node} because it is part of a branch."
        )
        return

    new_args = []
    for arg in node.args:
        if not isinstance(arg, Node) or node_arg_is_bias(node, arg):
            new_args.append(arg)
            continue

        is_weight = node_arg_is_weight(node, arg)

        act_eq_process_ctr = equalization_qconfig.weight if is_weight else \
            equalization_qconfig.input_activation

        new_eq_obs_mod = act_eq_process_ctr()
        new_eq_obs_node = insert_observer(
            arg, node, new_eq_obs_mod, model, modules, graph)

        new_args.append(new_eq_obs_node)

    # assign the new args and kwargs to the node, inplace
    node.args = tuple(new_args)

def maybe_insert_output_observer_for_node(
    node: Node,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
    graph: Graph,
    matches: Dict[str, MatchResult],
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
    matched_pattern: Any,
    qhandler: Optional[QuantizeHandler],
    is_qat: bool,
) -> Optional[Node]:
    """
    If `node` needs an output observer, creates it, inserts it into `graph`
    and returns it.

    If `node` does not need an output observer, returns None.
    """
    root_node, matched_nodes, pattern, qhandler, qconfig = matches.get(
        node.name, (None, None, None, None, None))

    if qhandler is None:
        return None

    assert qconfig is not None
    assert node.op != 'output', 'observer insertion for outputs is handled elsewhere'

    is_standalone_module = qhandler is not None and \
        isinstance(qhandler, StandaloneModuleQuantizeHandler)

    dtype = node_name_to_target_dtype[node.name]["output_activation_dtype"]
    should_insert_observer = \
        qhandler.should_insert_observer_for_output(
            qconfig, is_qat) and dtype not in DO_NOT_OBS_DTYPE_LIST + [torch.float]
    # TODO(future PR): move the following logic to
    # should_insert_observer_for_output
    should_insert_observer = should_insert_observer and \
        activation_is_statically_quantized(qconfig)

    # we never insert observers to output of standalone module, we assume
    # if needed, they are inserted inside the standalone module
    should_insert_observer = should_insert_observer and \
        (not is_standalone_module)

    if should_insert_observer:
        act_post_process_ctr = qconfig.activation
        if activation_is_int8_quantized(qconfig):
            act_post_process_ctr = qhandler.get_activation_ctr(
                qconfig,
                matched_pattern,
                is_qat)
        observer = act_post_process_ctr()
        new_obs = insert_observer(node, node, observer, model, modules, graph)
        return new_obs
    else:
        return None

def maybe_insert_observers_before_graph_output(
    graph_output_node: Node,
    output_quantized_idxs: List[int],
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
    qconfig_map: Dict[str, QConfigAny],
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
    graph: Graph,
) -> None:
    """
    If the output needs to be quantized and there are any nodes
    in the output which are not already observed, inserts observers
    for those nodes.
    """

    # TODO(future PR): update the output_quantized_idxs API to match
    # arbitrary data structures. There is always a single output, and
    # that output can have arbitrary nesting of values. List[int] is
    # not the right data type for this.
    assert output_quantized_idxs == [0] or output_quantized_idxs == [], \
        'unrecognized format of output_quantized_idxs'

    # Currently dequants are inserted in the convert step. So, we only
    # have to do anything if the output is hardcoded to be quantized
    if output_quantized_idxs == []:
        return
    # TODO(future PR): support more dtypes in model outputs, if necessary
    output_target_dtype = torch.quint8

    def _recursive_maybe_replace_node_with_obs(
        maybe_node: Argument,
        target_dtype: torch.dtype,
        node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
        qconfig_map: Dict[str, QConfigAny],
        model: torch.nn.Module,
        modules: Dict[str, torch.nn.Module],
        graph: Graph,
    ) -> Argument:
        """
        Navigate an arbitrary data structure of lists, tuples, dicts.
        For each container type, recurse on all inputs. Once any Node
        is found, insert an observer if needed and do not recurse further.

        For example, given a structure of

          {'foo1': [[bar1]], 'foo2': {'foo3': [[[bar3]]]}}

        we recurse down to bar1 and bar3, observe them if necessary,
        and if we inserted an observer then replace the original node
        with its observer.

        Returns the data structure with all nodes needing observation being
        replaced by their observers.
        """
        if isinstance(maybe_node, Node):
            # check dtype of this node
            this_node_dtype = get_arg_target_dtype_as_output(
                maybe_node, modules, node_name_to_target_dtype)
            if this_node_dtype != target_dtype:
                # insert observer
                qconfig = qconfig_map.get(maybe_node.name)
                # TODO(future PR): see if we need to allow specifying qconfig
                #   on output nodes, to remove the restriction below.
                assert qconfig is not None, \
                    'Quantizing the output node without a qconfig is not supported'
                observer_mod = qconfig.activation()
                observer_node = insert_observer(
                    maybe_node, maybe_node, observer_mod, model, modules, graph)
                return observer_node
            else:
                return maybe_node
        elif isinstance(maybe_node, (list, tuple)):
            results = []
            for inner_node in maybe_node:
                results.append(_recursive_maybe_replace_node_with_obs(
                    inner_node, target_dtype, node_name_to_target_dtype,
                    qconfig_map, model, modules, graph))
            if isinstance(maybe_node, list):
                return results
            else:
                return tuple(results)
        elif isinstance(maybe_node, dict):
            results_dict = {}
            for k, inner_v in maybe_node.items():
                results_dict[k] = _recursive_maybe_replace_node_with_obs(
                    inner_v, target_dtype, node_name_to_target_dtype,
                    qconfig_map, model, modules, graph)
            return results_dict
        else:
            return results

    new_args = []
    for old_arg in graph_output_node.args:
        new_args.append(
            _recursive_maybe_replace_node_with_obs(
                old_arg, output_target_dtype, node_name_to_target_dtype,
                qconfig_map, model, modules, graph))

    graph_output_node.args = tuple(new_args)  # type: ignore[assignment]


def maybe_propagate_dtype_for_node(
    node: Node,
    target_dtype: Union[torch.dtype, type],
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
    matches: Dict[str, MatchResult],
) -> None:
    """
    Assigns `target_dtype` to `node`. If `node` is a general tensor shape op
    (see GeneralTensorShapeOpQuantizeHandler in quantization_patterns.py for more details)
    also call this function recursively on
    the first argument, to propagate the dtype to the caller.
    """
    node_name_to_target_dtype[node.name]["input_activation_dtype"] = target_dtype
    node_name_to_target_dtype[node.name]["output_activation_dtype"] = target_dtype
    # if this is a copy node, propagate to first arg
    root_node, matched_nodes, pattern, qhandler, qconfig = matches.get(
        node.name, (None, None, None, None, None))
    if qhandler is not None and qhandler.is_general_tensor_shape_op():
        prev_node = node.args[0]
        if isinstance(prev_node, Node):
            maybe_propagate_dtype_for_node(
                prev_node, target_dtype, node_name_to_target_dtype, matches)

def propagate_dtypes_for_known_nodes(
    graph: Graph,
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]],
    matches: Dict[str, MatchResult],
) -> None:
    """
    Currently we assume that inputs to the graph are either `torch.float` or
    `torch.quint8`, which is not always correct. For ops such as
    `x.masked_fill(mask, value)`, we know that the dtype of  `mask` is a
    `BoolTensor`. Propagate this information throughout the graph.

    Note: not all dtypes in the graph will be correct after this pass, but a
    higher percentage of them will be correct. Hopefully in the future we can
    replace this with a better way to reason about dtypes of tensors.
    """
    for node in graph.nodes:
        non_observable_arg_dict = get_non_observable_arg_indexes_and_types(node)

        for arg_type in non_observable_arg_dict:
            non_observable_indices = non_observable_arg_dict[arg_type](node)

            for index in non_observable_indices:
                arg = node.args[index]

                # when an argument is a tuple, it does not show up as another node so we need to go through
                # all elements of the tuple manually
                if isinstance(arg, tuple) or isinstance(arg, list):
                    arg_list = list(arg)
                else:
                    arg_list = [arg]

                for cur_arg in arg_list:
                    # hard coded arguments show up but aren't `Node` typed and do not need dtype propgated
                    if isinstance(cur_arg, torch.fx.node.Node):
                        maybe_propagate_dtype_for_node(
                            cur_arg, arg_type, node_name_to_target_dtype, matches)

def maybe_make_input_output_share_observers(
    node: Node,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
) -> bool:
    """
    Ensures that we share an observer
    for all input arguments as well as the output argument. In detail, given
    a graph of

      x0 -> obs0 -> op -> x2
                  /
      x1 -> obs1 /

    where node obs0 points to observer instance observer0,
    obs1 points to observer1 and obs2 points to observer2, we make nodes obs1
    and ob2 point to observer0.
    Returns: whether the operation succeeded or not
    """
    first_arg = None
    # find the first non-Tensor arg
    for i in range(len(node.args)):
        if isinstance(node.args[i], (Node, list, tuple)):
            first_arg = node.args[i]
            break

    # if there is no non-Tensor arg, return directly
    if first_arg is None:
        return False

    if isinstance(first_arg, (list, tuple)):
        first_arg_arg = first_arg[0]
    elif isinstance(first_arg, Node):
        first_arg_arg = first_arg
    else:
        return False

    # if we have a graph such as
    #   observed_node -> non_observed_node -> cat
    # we need to navigate up to the first observer
    iteration_guard = 0
    while not is_activation_post_process_node(first_arg_arg, modules):
        if not isinstance(first_arg_arg, Node):
            return False
        # did not find an activation_post_process for the op
        if first_arg_arg.op == "placeholder":
            return False
        # trace back the args until we found the first Tensor/Node
        trace_back_node = None
        for i in range(len(first_arg_arg.args)):
            trace_back_node = first_arg_arg.args[i]
            if isinstance(trace_back_node, Node):
                break
        if trace_back_node is None:
            return False
        first_arg_arg = trace_back_node

        iteration_guard += 1
        if iteration_guard > 10000:
            raise AssertionError('Unable to find observer of previous node')

    assert isinstance(first_arg_arg, Node)
    target_to_use = first_arg_arg.target
    assert isinstance(target_to_use, str)
    obs_mod_to_use = modules[target_to_use]

    if isinstance(first_arg, (list, tuple)):
        # set all other input observer nodes to use that module
        for input_idx, input_arg in enumerate(first_arg):
            if input_idx == 0:
                continue
            iteration_guard = 0
            while not is_activation_post_process_node(input_arg, modules):
                input_arg = input_arg.args[0]
                iteration_guard += 1
                if iteration_guard > 10000:
                    raise AssertionError('Unable to find observer of previous node')

            parent_name, name = _parent_name(input_arg.target)
            setattr(modules[parent_name], name, obs_mod_to_use)

    # set the output observer node to use that module
    for output_obs_node, _ in node.users.items():
        assert is_activation_post_process_node(output_obs_node, modules)
        parent_name, name = _parent_name(output_obs_node.target)
        setattr(modules[parent_name], name, obs_mod_to_use)

    # TODO(future PR): delete the orphaned observer modules
    return True

def remove_output_observer(
        node: Node,
        model: torch.nn.Module,
        modules: Dict[str, torch.nn.Module]):
    items = list(node.users.items())
    for output_obs_node, _ in items:
        assert is_activation_post_process_node(output_obs_node, modules)
        output_obs_node.replace_all_uses_with(node)
        model.graph.erase_node(output_obs_node)  # type: ignore[union-attr, operator]

def swap_custom_module_to_observed(
        node: Node,
        qconfig: QConfigAny,
        modules: Dict[str, torch.nn.Module],
        prepare_custom_config_dict: Dict[str, Any]):
    custom_module = modules[node.target]  # type: ignore[index]
    custom_module_class_mapping = prepare_custom_config_dict.get(
        "float_to_observed_custom_module_class", {})
    observed_custom_module_class = \
        get_swapped_custom_module_class(
            custom_module, custom_module_class_mapping, qconfig)
    observed_custom_module = \
        observed_custom_module_class.from_float(custom_module)
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, observed_custom_module)

def insert_observers_for_model(
    model: GraphModule,
    modules: Dict[str, torch.nn.Module],
    matches: Dict[str, MatchResult],
    qconfig_map: Dict[str, QConfigAny],
    graph: Graph,
    prepare_custom_config_dict: Dict[str, Any],
    equalization_config_map: Dict[str, Any],
    input_quantized_idxs: List[int],
    output_quantized_idxs: List[int],
    backend_config_dict: Optional[Dict[str, Any]],
    observed_node_names: Set[str],
    is_qat: bool,
) -> Optional[Node]:
    """
    Inserts observers, using the following high level algorithm:

    For each node in the graph:
      1. determine the target dtype of this node in the quantized graph, and save
           it for future steps
      2. determine the target dtype or all args and kwargs of this node
      3. if any arg or kwarg's target dtype does not match the current node's
           dtype, insert an observer
      4. if the current node needs an output observer, insert it

    For example:

    - starting graph:
        x0 -> linear -> x1

    - observed graph after processing x0:
        x0(fp32)

    - observed graph after processing linear:
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8)

    - observed graph after processing x1:
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8) -> x1

    After a node is processed, the naive observer placement is guaranteed to be
    complete for that node and all of its predecessors. There can be future
    passes which optimize the graph by deduplicating observers, etc.
    """

    # name of Node in original FX Graph to the target dtype information
    # that's derived from qconfig for the Node, for example, if we have
    # a conv2d node that has a qconfig
    # {
    #   # information for input and bias node omitted
    #   # for getattr node
    #   # weight = getattr(self, 'weight')
    #   'weight': {
    #      'output_activation_dtype': torch.float,
    #   }
    #   # for conv2d node
    #   # conv2d = call_function[target=torch.nn.functional.conv2d](
    #   #            args=(input, weight, bias))
    #   'conv2d': {
    #       'input_activation_dtype': torch.quint8,
    #       'weight_dtype': torch.qint8,
    #       'bias_dtype': torch.float,
    #       'output_activation_dtype': torch.quint8,
    #     }
    #   }
    #
    # TODO: rename this to node_name_to_target_dtype_info
    node_name_to_target_dtype: Dict[str, Dict[str, Optional[Union[torch.dtype, type]]]] = defaultdict(dict)
    cache_for_no_tensor_check: Dict[Node, bool] = dict()

    inputs_seen_counter = 0
    outputs_seen_counter = 0

    # first, populate the dtype map based only on qconfig and qhandler
    # this assumes:
    # graph inputs are fp32 by default, and int8 where overriden
    # other nodes output dtype is specified by the qconfig
    modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        root_node, matched_nodes, pattern, qhandler, qconfig = matches.get(
            node.name, (None, None, None, None, None))
        node_name_to_target_dtype[node.name] = get_target_activation_dtype_for_node(
            node, qconfig, inputs_seen_counter, outputs_seen_counter,
            input_quantized_idxs, output_quantized_idxs, qhandler,
            modules, cache_for_no_tensor_check)
        if node.op == "placeholder":
            inputs_seen_counter += 1
        if node.op == "output":
            outputs_seen_counter += 1

    # Second, for nodes with known input dtypes, propagate them throughout the
    # graph. For example, if there is a call such as
    #   x1 = x0.masked_fill(mask, 1)
    # we propagate the type of mask to be torch.bool
    propagate_dtypes_for_known_nodes(
        model.graph, node_name_to_target_dtype, matches)

    # After this point, the current node and all of its arguments
    # have a dtype assigned. Now, we insert observers for inputs
    # of this node (if needed for this node), and the output of this node
    # (if needed for this node).

    # Since we are mutating the graph as we go, we iterate over the original
    # nodes before observer insertion, instead of model.graph.nodes.
    nodes_before_observation = list(model.graph.nodes)

    # reset inputs/outputs counters
    inputs_seen_counter = 0
    outputs_seen_counter = 0
    results_node = None
    for node in nodes_before_observation:

        if node.op == 'placeholder':
            # if a graph input is in fp32, it does not need observation
            # if a graph input is in int8, we assume the observation happens
            #   outside of the graph, and no additional observation is needed
            pass

        elif node.op in ('call_module', 'call_method', 'call_function', 'output'):
            # check for matches
            root_node, matched_nodes, pattern, qhandler, qconfig = matches.get(
                node.name, (None, None, None, None, None))
            equalization_qconfig = equalization_config_map.get(node.name, None)

            this_node_dtype = node_name_to_target_dtype[node.name]
            output_not_a_tensor = this_node_dtype is None
            # TODO(future PR): consider stopping matching getitem
            is_getitem = node.op == 'call_function' and \
                node.target == operator.getitem

            skip_inserting_observers = (
                (qconfig is None) or
                output_not_a_tensor or
                is_getitem
            ) and (
                not node.op == 'output'
            )

            is_supported_by_backend = is_pattern_dtype_config_supported_by_backend(
                pattern, matched_nodes, node_name_to_target_dtype, backend_config_dict)

            if not skip_inserting_observers and is_supported_by_backend:
                modules = dict(model.named_modules(remove_duplicate=False))
                if node.op != 'output':
                    assert matched_nodes is not None
                    # add matched nodes to the observed node name set
                    for n in matched_nodes:
                        observed_node_names.add(n.name)

                    # This is currently only used for equalization.
                    # Checks if the current node is in a branch in which the two
                    # first layers are both being quantized.
                    #
                    # ex.       conv2
                    #         /
                    #      x -> conv1
                    #
                    # If this is the case, we will not apply equalization to the
                    # initial two layers.
                    is_quantized_branch = False
                    if (
                        len(node.args) > 0 and
                        isinstance(node.args[0], Node) and
                        len(node.args[0].users) > 1
                    ):
                        for user in node.args[0].users:
                            # Checks if there exists another user being quantized
                            is_user_quantized = (
                                qconfig_map.get(user.name, None) is not None or
                                (user.op == 'call_module' and isinstance(modules[str(user.target)], ObserverBase))
                            )
                            if user != node and is_user_quantized:
                                is_quantized_branch = True

                    # this modifies node inplace
                    maybe_insert_input_observers_for_node(
                        node, qconfig, model, modules, graph,
                        node_name_to_target_dtype,
                        qhandler,
                        prepare_custom_config_dict,
                        backend_config_dict)

                    # Insert equalization input observers if needed
                    maybe_insert_input_equalization_observers_for_node(
                        node, equalization_qconfig, model, modules, graph,
                        node_name_to_target_dtype, is_quantized_branch)

                    is_last_node_of_pattern = root_node is node
                    is_general_tensor_value_op = \
                        (qhandler is not None and qhandler.is_general_tensor_value_op())

                    is_general_tensor_shape_op = \
                        (qhandler is not None and qhandler.is_general_tensor_shape_op())

                    is_reuse_input_qconfig_ = is_reuse_input_qconfig(qconfig)

                    if is_last_node_of_pattern:
                        # this returns the new observer node if it was needed
                        maybe_output_obs_node = maybe_insert_output_observer_for_node(
                            node, model, modules, graph, matches,
                            node_name_to_target_dtype, pattern, qhandler, is_qat)
                        if maybe_output_obs_node is not None:
                            # Update users of original node to use the output observer
                            # instead. For example, change
                            #
                            #           next_node
                            #          /
                            #   cur_node -> obs
                            #
                            # to
                            #
                            #                 next_node
                            #                 /
                            #   cur_node -> obs
                            #
                            # We need to save orig users before updating uses because
                            # the list of users will change as we update uses
                            orig_users = list(node.users.keys())
                            for user_node in orig_users:
                                if user_node is maybe_output_obs_node:
                                    continue
                                user_node.replace_input_with(node, maybe_output_obs_node)

                            is_observer_in_same_graph_ = is_observer_in_same_graph(node, modules, node_name_to_target_dtype)

                            # for general tensor value ops, we modify the graph
                            # to make all inputs and outputs use the first input's
                            # observer
                            if (is_general_tensor_value_op and is_observer_in_same_graph_) or \
                                    is_general_tensor_shape_op or is_reuse_input_qconfig_:
                                if not maybe_make_input_output_share_observers(node, model, modules):
                                    remove_output_observer(node, model, modules)

                            if isinstance(qhandler, CustomModuleQuantizeHandler):
                                swap_custom_module_to_observed(node, qconfig, modules, prepare_custom_config_dict)

                else:  # output
                    maybe_insert_observers_before_graph_output(
                        node, output_quantized_idxs,
                        node_name_to_target_dtype, qconfig_map,
                        model, modules, graph)

        #
        # After this point, the current node has input and output observers
        # that it needs for itself inserted.
        #

        # increment the counters, so future inputs and outputs are assigned
        # correct dtypes
        if node.op == 'placeholder':
            inputs_seen_counter += 1
        elif node.op == 'output':
            outputs_seen_counter += 1
            results_node = node

    return results_node

def run_prepare_fx_on_standalone_modules(
    model: torch.nn.Module,
    is_qat: bool,
    modules: Dict[str, torch.nn.Module],
    matches: Any,
    prepare_custom_config_dict: Dict[str, Any],
    backend_config_dict: Optional[Dict[str, Any]],
) -> None:
    """
    Runs prepare_fx on each standalone module. Note: this does
    not modify the graph, it just replaces the unobserved modules with
    their observed versions.
    """
    for (
        node_name,
        (root_node, matched_nodes, pattern, qhandler, qconfig),
    ) in matches.items():
        if qhandler is None:
            continue
        elif not isinstance(qhandler, StandaloneModuleQuantizeHandler):
            continue

        sm_qconfig_dict, sm_prepare_config_dict, sm_backend_config_dict = \
            prepare_get_standalone_module_configs(
                root_node, modules, prepare_custom_config_dict, qconfig, backend_config_dict)

        standalone_module = modules[root_node.target]
        prepare = \
            torch.ao.quantization.quantize_fx._prepare_standalone_module_fx  # type: ignore[attr-defined]
        observed_standalone_module = \
            prepare(
                standalone_module,
                sm_qconfig_dict,
                is_qat,
                sm_prepare_config_dict,
                backend_config_dict=sm_backend_config_dict)
        preserved_attributes = \
            set(sm_prepare_config_dict.get("preserved_attributes", []))
        observed_standalone_module = ObservedStandaloneGraphModule(
            observed_standalone_module, observed_standalone_module.graph,
            preserved_attributes)
        parent_name, name = _parent_name(root_node.target)
        setattr(modules[parent_name], name,
                observed_standalone_module)
        modules[root_node.target] = observed_standalone_module

def save_state(
    observed: GraphModule,
    qconfig_map: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str, type]],
    patterns: Dict[Pattern, QuantizeHandler],
    prepare_custom_config_dict: Dict[str, Any],
    equalization_qconfig_map: Dict[str, Any],
    qconfig_dict: Dict[str, Dict[Any, Any]],
    is_qat: bool,
    observed_node_names: Set[str],
) -> None:
    observed._patterns = patterns  # type: ignore[assignment]
    observed._qconfig_map = qconfig_map  # type: ignore[assignment]
    observed._prepare_custom_config_dict = \
        prepare_custom_config_dict  # type: ignore[assignment]
    observed._node_name_to_scope = node_name_to_scope  # type: ignore[assignment]
    observed._equalization_qconfig_map = equalization_qconfig_map  # type: ignore[assignment]
    observed._qconfig_dict = qconfig_dict  # type: ignore[assignment]
    observed._is_qat = is_qat  # type: ignore[assignment]
    observed._observed_node_names = observed_node_names  # type: ignore[assignment]

def prepare(
        model: GraphModule,
        qconfig_dict: Any,
        is_qat: bool,
        node_name_to_scope: Dict[str, Tuple[str, type]],
        prepare_custom_config_dict: Optional[Dict[str, Any]] = None,
        equalization_qconfig_dict: Optional[Dict[str, Any]] = None,
        backend_config_dict: Optional[Dict[str, Any]] = None,
        is_standalone_module: bool = False) -> ObservedGraphModule:
    """ standalone_module means it a submodule that is not inlined in
    parent module, and will be quantized separately as one unit.

    How the standalone module is observed is specified by `input_quantized_idxs` and
    `output_quantized_idxs` in the prepare_custom_config for the standalone module
    Args:
        node_name_to_scope: mapping from node name to the scope of the module which contains the node.
        The scope is a tuple of fully qualified path of the module and the type of the module
    Returns:
        model(GraphModule): prepared standalone module
        attributes:
            _standalone_module_input_quantized_idxs(List[Int]): a list of
                indexes for the graph input that is expected to be quantized,
                same as input_quantized_idxs configuration provided
                for the standalone module
            _standalone_module_output_quantized_idxs(List[Int]): a list of
                indexs for the graph output that is quantized
                same as input_quantized_idxs configuration provided
                for the standalone module
    """
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    if equalization_qconfig_dict is None:
        equalization_qconfig_dict = {}

    additional_quant_patterns = \
        prepare_custom_config_dict.get("additional_quant_pattern", {})
    # mapping from a tuple of nodes in reverse order to uninitialized
    #   QuantizeHandler subclass. For example,
    # {
    #   # match a single node
    #   (<class 'torch.nn.modules.conv.Conv3d'>:
    #     <class 'torch.ao.quantization.fx.quantize.ConvRelu'>),
    #   # match multiple nodes in reverse order
    #   ((<function relu at 0x7f766a7360d0>, <built-in function add>):
    #     <class 'torch.ao.quantization.fx.quantize.Add'>),
    # }
    patterns: Dict[Pattern, QuantizeHandler] = {}
    if backend_config_dict is None:
        quant_patterns = get_default_quant_patterns()
        patterns = get_combined_dict(
            quant_patterns, additional_quant_patterns)
    else:
        patterns = get_pattern_to_quantize_handlers(backend_config_dict)

        # TODO: make WEIGHT_INDEX_DICT and BIAS_INDEX_DICT an argument to the functions that needs them
        # TODO: refactor this part to return WEIGHT_INDEX_DICT and BIAS_INDEX_DICT
        pattern_to_input_type_to_index = get_pattern_to_input_type_to_index(backend_config_dict)
        for pattern, input_type_to_index in pattern_to_input_type_to_index.items():
            for input_type, index in input_type_to_index.items():
                index_dicts = {
                    "weight": WEIGHT_INDEX_DICT,
                    "bias": BIAS_INDEX_DICT,
                    "input": {}  # not used right now
                }
                assert input_type in index_dicts.keys(), \
                    f"input type must be one of {index_dicts.keys()} but got: {input_type}"
                index_dict = index_dicts[input_type]
                if pattern in index_dict:  # type: ignore[operator]
                    index_dict[pattern].append(index)  # type: ignore[index]
                else:
                    index_dict[pattern] = [index]  # type: ignore[index]

    convert_dict_to_ordered_dict(qconfig_dict)
    convert_dict_to_ordered_dict(equalization_qconfig_dict)
    qconfig_dict = update_qconfig_for_fusion(model, qconfig_dict)
    equalization_qconfig_dict = update_qconfig_for_fusion(model, equalization_qconfig_dict)
    flattened_qconfig_dict = get_flattened_qconfig_dict(qconfig_dict)
    # TODO: support regex as well
    propagate_qconfig_(model, flattened_qconfig_dict)

    if is_qat:
        additional_qat_module_mapping = prepare_custom_config_dict.get(
            "additional_qat_module_mapping", {})
        # this path will be deprecated after we fully migrate the convert path
        # of fbgemm/qnnpack to use the reference path, it will stay
        # here for a few months
        if backend_config_dict is None:
            module_to_qat_module = get_combined_dict(
                get_default_qat_module_mappings(), additional_qat_module_mapping)
        else:
            module_to_qat_module = get_module_to_qat_module(backend_config_dict)
        qat_swap_modules(model, module_to_qat_module)
        qconfig_dict = update_qconfig_for_qat(qconfig_dict, additional_qat_module_mapping)

    # mapping from fully qualified module name to module instance
    # for example,
    # {
    #   '': Model(...),
    #   'linear': Linear(...),
    #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
    # }
    modules = dict(model.named_modules(remove_duplicate=False))

    # fill qconfig_map, a map from node name to qconfig, used in find_matches
    equalization_qconfig_map = generate_qconfig_map(model, modules, model.graph, equalization_qconfig_dict, node_name_to_scope)
    qconfig_map = generate_qconfig_map(model, modules, model.graph, qconfig_dict, node_name_to_scope)

    # match the patterns that will get quantized
    standalone_module_name_configs = prepare_custom_config_dict.get(
        "standalone_module_name", [])
    standalone_module_class_configs = prepare_custom_config_dict.get(
        "standalone_module_class", [])

    standalone_module_names = [config[0] for config in standalone_module_name_configs]
    standalone_module_classes = [config[0] for config in standalone_module_class_configs]
    custom_module_classes = get_custom_module_class_keys(
        prepare_custom_config_dict, "float_to_observed_custom_module_class")
    matches = find_matches(
        model.graph, modules, patterns, qconfig_map, standalone_module_names,
        standalone_module_classes, custom_module_classes)

    input_quantized_idxs: List[int] = prepare_custom_config_dict.get(
        "input_quantized_idxs", [])
    output_quantized_idxs: List[int] = prepare_custom_config_dict.get(
        "output_quantized_idxs", [])

    run_prepare_fx_on_standalone_modules(
        model, is_qat, modules, matches, prepare_custom_config_dict, backend_config_dict)

    # record names for the set of observed node, so that in convert step
    # we know whether we need to convert a floating point module to reference
    # quantized module or not
    observed_node_names: Set[str] = set()

    result_node = insert_observers_for_model(
        model, modules, matches, qconfig_map,
        model.graph, prepare_custom_config_dict,
        equalization_qconfig_map,
        input_quantized_idxs,
        output_quantized_idxs,
        backend_config_dict,
        observed_node_names,
        is_qat)

    save_state(model, qconfig_map, node_name_to_scope, patterns,
               prepare_custom_config_dict, equalization_qconfig_map, qconfig_dict, is_qat, observed_node_names)

    preserved_attributes = set(prepare_custom_config_dict.get("preserved_attributes", []))
    model = ObservedGraphModule(model, model.graph, preserved_attributes)
    if is_standalone_module:
        assert result_node is not None
        assert isinstance(result_node.args[0], Node), \
            "standalone module only supports returning simple value currently"\
            "(not tuple, dict etc.)"
        # these inputs are observed in parent
        # converting List[int] to Tensor since module attribute is
        # Union[Tensor, Module]
        model._standalone_module_input_quantized_idxs = \
            torch.tensor(input_quantized_idxs)
        model._standalone_module_output_quantized_idxs = torch.tensor(output_quantized_idxs)
    return model
