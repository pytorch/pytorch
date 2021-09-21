import inspect
import re
from typing import NamedTuple, Optional, Callable, Dict, List, Tuple, Union, Any, Set

import torch.fx.experimental.fx_acc.acc_utils as acc_utils
import torch
import torch.fx
from torch.fx.node import _get_qualified_name

# Need to keep up-to-date with https://fburl.com/codesearch/7r2hhh53
ALIAS_MAP = {
    "input": ("input", "x", "a", "x1"),
    "dim": ("dim", "axis"),
    "keepdim": ("keepdim", "keepdims"),
    "other": ("other", "x2"),
}

# Type used for arg replacement tuples. The list represents the argument signature of
# some callable. Each item in the list is a tuple, where for each member of a tuple:
# - The first member is union of either:
#   - A tuple of all potential alias kwarg str names of the source signature, or
#   - A tuple of a single str representing the single kwarg name allowed.
# - The second member is the str name of the kwarg to map it to. This is either from the
#   signature of the acc_op, or for custom mapped nodes from the original unnormalized op.
# - The third member is a bool representing whether this arg is optional, i.e. whether it
#   is allowed to not be present in the original input args.
ArgReplacementTuplesType = List[Tuple[Tuple[str, ...], str, bool]]


class NormalizationInfo(NamedTuple):
    """
    Holds normalization info for some FX node, where the FX node will be mapped either
    via new_fn_target and arg_replacement_tuples, or via custom_mapping_fn.

    If via new_fn_target and arg_replacement_tuples:
      - new_fn_target is the target function to replace the original node with
        (generally some function from acc_ops).

      - arg_replacement_tuples describes how to map the original FX node's args/kwargs to
        the new FX node. If set to None, then the kwargs are copied directly from the
        original FX node. Else, this is list of three-member tuples, where each tuple
        represents a mapping from either an arg or kwarg in the original FX node to the
        kwarg it should be mapped to. If for ops registered with `register_acc_op` then
        this is a mapping to the the new FX node for the acc_op. Otherwise it is for some
        op registered with `register_custom_acc_mapper_fn`, in which case this is a
        mapping for the original input node so its args are normalized to kwargs before
        being custom normalized to acc_ops. The third member of the tuple is a bool
        representing whether this argument is optional; if False and the arg is not
        present then an assertion will be thrown. The index of the tuple indicates where
        the original arg is in node.args and the string name indicates which original
        kwarg it is.

    If via custom_mapping_fn, then custom_mapping_fn is some function that takes the
    original FX node as input and returns the FX node that should replace it. This means
    it was registered via `register_custom_acc_mapper_fn`.
    """

    new_fn_target: Callable
    arg_replacement_tuples: Optional[ArgReplacementTuplesType]
    custom_mapping_fn: Optional[Callable]
    kwargs_to_move_to_acc_out_ty: Optional[Optional[List[Tuple[str, str]]]]
    needs_shapes_for_normalization: bool


# Dict from (op, target) to NormalizationInfo for that op.
_normalization_dict: Dict[Tuple[str, Union[str, Callable]], NormalizationInfo] = {}

# Set of all the acc ops.
_acc_ops: Set[Callable] = set()


def _insert_fun(
    op_and_target: Tuple[str, Union[str, Callable]],
    arg_replacement_tuples: List[Tuple],
    new_fn_target: Optional[Callable] = None,
    custom_mapping_fn: Optional[Callable] = None,
    kwargs_to_move_to_acc_out_ty: Optional[Optional[List[Tuple[str, str]]]] = None,
    needs_shapes_for_normalization=False,
    allow_normalize_from_torch_package=False,
):
    if op_and_target[0] == "call_function":
        assert callable(op_and_target[1])
    elif op_and_target[0] == "call_method":
        assert isinstance(op_and_target[1], str)
    elif op_and_target[0] == "call_module":
        assert isinstance(op_and_target[1], type)

    # Finalize arg replacement tuples.
    # 1. Check to see if they have the `is_optional` bool, and if not defaulting it to
    #   False.
    # 2. Some kwargs might have aliases. e.g. "a", "x" and "x1" are aliases of "input".
    #   Here we replace `orig_kwarg` with a tuple of all aliases if it has aliases.
    final_arg_replacement_tuples = []
    for arg_replacement_tuple in arg_replacement_tuples:
        if len(arg_replacement_tuple) == 2:
            orig_kwarg, new_kwarg, is_optional = *arg_replacement_tuple, False
        else:
            assert len(arg_replacement_tuple) == 3
            orig_kwarg, new_kwarg, is_optional = arg_replacement_tuple

        if not isinstance(orig_kwarg, tuple):
            orig_kwarg = (orig_kwarg,)

        # Use set to avoid duplicates.
        orig_kwarg_set = set(orig_kwarg)

        for k in orig_kwarg:
            if k in ALIAS_MAP:
                orig_kwarg_set.update(ALIAS_MAP[k])
        final_arg_replacement_tuples.append(
            (tuple(orig_kwarg_set), new_kwarg, is_optional)
        )

    assert op_and_target not in _normalization_dict.keys()
    norm_info = NormalizationInfo(
        new_fn_target=new_fn_target,  # type: ignore[arg-type]
        arg_replacement_tuples=final_arg_replacement_tuples,
        custom_mapping_fn=custom_mapping_fn,
        kwargs_to_move_to_acc_out_ty=kwargs_to_move_to_acc_out_ty,
        needs_shapes_for_normalization=needs_shapes_for_normalization,
    )
    _normalization_dict[op_and_target] = norm_info

    # If allow_normalize_from_torch_package then add another entry to
    # _normalization_dict where we look for the qualified name of the target with the
    # torch_package module prefix. Note that we leave off any integer at the end of
    # "<torch_package_>" in order to allow for whatever mangling index is used.
    if allow_normalize_from_torch_package:
        torch_package_op_and_target = (
            op_and_target[0],  # type: ignore[]
            f"<torch_package_>.{_get_qualified_name(op_and_target[1])}",  # type: ignore[arg-type]
        )
        _normalization_dict[torch_package_op_and_target] = norm_info


def _get_dup_signature_tuples(fn: Callable) -> List[Tuple[str, str]]:
    """
    Helper that inspects the arg signature of `fn` and returns a list of tuples, where
    each tuple is a pair of duplicated names which is used for arg_replacement_tuples.
    """
    sig_tuples: List[Tuple[str, str]] = []
    for param in inspect.signature(inspect.unwrap(fn)).parameters:
        sig_tuples.append((param, param))
    return sig_tuples


def register_acc_op(acc_op: Callable):
    """
    For a new acc op, add this as decorator to register it.
    """
    _acc_ops.add(acc_op)
    return acc_op


def register_acc_op_mapping(
    op_and_target: Tuple[str, Union[str, Callable]],
    arg_replacement_tuples: Optional[
        List[Union[Tuple[Union[str, Tuple[str, ...]], str], Tuple[Union[str, Tuple[str, ...]], str, bool]]]
    ] = None,
    kwargs_to_move_to_acc_out_ty: Optional[List[Tuple[str, str]]] = None,
):
    """
    Use this decorator to map a non-acc operator to an acc operator.

    Args:
        op_and_target: A tuple that contains op and target of the node that represents the non-acc operator.
        arg_replacement_tuples: Please refer to the comment on above for `ArgReplacementTuplesType`.
        kwargs_to_move_to_acc_out_ty: The kwargs we want to move out from the non-acc op kwargs to acc_out_ty.
    """

    def insert(new_fn_target: Callable):
        # If arg_replacement_tuples is None then assume we use the same signature for
        # the acc_op and the original op.
        if arg_replacement_tuples is None:
            final_arg_replacement_tuples = _get_dup_signature_tuples(new_fn_target)
        else:
            final_arg_replacement_tuples = arg_replacement_tuples  # type: ignore[assignment]

        _insert_fun(
            op_and_target=op_and_target,
            new_fn_target=new_fn_target,
            arg_replacement_tuples=final_arg_replacement_tuples,  # type: ignore[arg-type]
            kwargs_to_move_to_acc_out_ty=kwargs_to_move_to_acc_out_ty,
        )
        return new_fn_target

    return insert


def register_custom_acc_mapper_fn(
    op_and_target: Tuple[str, Union[str, Callable]],
    arg_replacement_tuples: List[Union[Tuple[Union[str, Tuple[str, ...]], str], Tuple[Union[str, Tuple[str, ...]], str, bool]]],
    needs_shapes_for_normalization=False,
    allow_normalize_from_torch_package=False,
):
    def insert(custom_mapping_fn: Callable):
        _insert_fun(
            op_and_target=op_and_target,
            custom_mapping_fn=custom_mapping_fn,
            arg_replacement_tuples=arg_replacement_tuples,  # type: ignore[arg-type]
            needs_shapes_for_normalization=needs_shapes_for_normalization,
            allow_normalize_from_torch_package=allow_normalize_from_torch_package,
        )
        return custom_mapping_fn

    return insert


def move_kwargs_to_acc_out_ty(
    node_or_normalization_info: Union[NormalizationInfo, torch.fx.Node],
    new_kwargs: Dict[str, Any],
):
    """
    Given `node_or_normalization_info` which is either NormalizationInfo for a node, or
    a node to fetch NormalizationInfo for, check if kwargs_to_move_to_acc_out_ty exists
    in the NormalizationInfo, and if so perform the move of kwargs to acc_out_ty.
    """

    if isinstance(node_or_normalization_info, torch.fx.Node):
        node = node_or_normalization_info
        normalization_info = _normalization_dict.get((node.op, node.target))
    else:
        assert isinstance(node_or_normalization_info, NormalizationInfo)
        normalization_info = node_or_normalization_info

    assert normalization_info is not None
    if normalization_info.kwargs_to_move_to_acc_out_ty is None:
        return

    assert acc_utils.is_acc_op_with_kwarg(
        normalization_info.new_fn_target, "acc_out_ty"
    )

    # Build a dict representing the new TensorMetadata to use for acc_out_ty,
    # and then remove the kwarg from the new_kwargs since it's passed in via
    # acc_out_ty instead.
    tmd_dict: Dict[str, Any] = {}
    for (
        orig_kwarg_name,
        tmd_field_name,
    ) in normalization_info.kwargs_to_move_to_acc_out_ty:
        tmd_dict[tmd_field_name] = new_kwargs[orig_kwarg_name]
        del new_kwargs[orig_kwarg_name]
    # Note: allow_partial_spec here because we are only using the tensor metadata tuple
    # here to pass specific values into the function. For example, for quantization we
    # only need to provide dtype/q_scale/q_zero_point, but is_quantized and qscheme are
    # not passed in.
    new_kwargs["acc_out_ty"] = acc_utils.build_raw_tensor_meta(**tmd_dict)


def get_normalized_kwargs(
    node: torch.fx.Node, arg_replacement_tuples: ArgReplacementTuplesType
):
    new_kwargs = {}
    final_arg_is_varg = False
    for i, replacement_tuple in enumerate(arg_replacement_tuples):
        orig_kwargs_names, new_kwarg_name, is_optional = replacement_tuple

        # Check if this is a varg and if so break/process the rest outside the loop.
        if len(orig_kwargs_names) == 1 and orig_kwargs_names[0] == "*":
            assert i == len(arg_replacement_tuples) - 1
            final_arg_is_varg = True
            break

        # If nothing is found in node.kwargs it means the kwarg is in node.arg
        # or it's optional. In this case, we set orig_kwargs_name to None.
        assert isinstance(orig_kwargs_names, tuple)
        orig_kwargs_name = next(
            (key for key in orig_kwargs_names if key in node.kwargs),
            None,
        )

        # If can't find in node.kwargs then it should be in the i index
        # of node.args.
        if orig_kwargs_name is None:
            if i < len(node.args):
                new_kwargs[new_kwarg_name] = node.args[i]
            else:
                # Verify the arg we're trying to normalize was optional.
                assert is_optional
        else:
            new_kwargs[new_kwarg_name] = node.kwargs[orig_kwargs_name]

    # If using var args then process the rest of the args now.
    if final_arg_is_varg:
        var_arg_idx = len(arg_replacement_tuples) - 1
        new_kwarg_name = arg_replacement_tuples[var_arg_idx][1]
        rest_of_args = []
        for i in range(var_arg_idx, len(node.args)):
            rest_of_args.append(node.args[i])
        new_kwargs[new_kwarg_name] = rest_of_args

    return new_kwargs


def normalize(mod: torch.fx.GraphModule, expect_nodes_have_shapes: bool = False):
    assert len(_normalization_dict) > 0
    graph = mod.graph

    # For "call_module" node we return _base_class_origin if it's a
    # RewrittenModule, otherwise, return its type. For other nodes,
    # we return node.target.
    def get_target(mod: torch.fx.GraphModule, node: torch.fx.Node):
        if node.op != "call_module":
            return node.target

        # Find the module that node.target points to
        m = dict(mod.named_modules())[node.target]
        return getattr(m, "_base_class_origin", type(m))

    def normalize_to_acc_op(
        node: torch.fx.Node,
        normalization_info: NormalizationInfo,
        normalized_args: Tuple[Any, ...],
        normalized_kwargs: Dict[str, Any],
    ):
        # If there's a custom mapping function then use it.
        if normalization_info.custom_mapping_fn is not None:
            # For custom mapping, the normalized_kwargs are used for the original op,
            # i.e. *before* custom acc_ops normalization. Do that now.
            node.args = normalized_args
            node.kwargs = normalized_kwargs
            new_node = normalization_info.custom_mapping_fn(node, mod)
            # If a new node is returned then use it to replace the old node. Otherwise
            # the custom mapping function did its own replacement, so return early.
            if new_node is None:
                return
        else:
            # If there's kwargs_to_move_to_acc_out_ty then use it to setup acc_out_ty in
            # normalized_kwargs, and remove the kwarg from normalized_kwargs.
            move_kwargs_to_acc_out_ty(normalization_info, normalized_kwargs)

            # All acc ops are functions. Create a call to the correct acc_ops target using
            # the normalized kwargs provided.
            with graph.inserting_before(node):
                new_node = graph.create_node(
                    "call_function",
                    normalization_info.new_fn_target,
                    args=normalized_args,
                    kwargs=normalized_kwargs,
                    name=node.name,
                )
                new_node.meta = node.meta.copy()

        # Finally replace the original node with the normalized node.
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

    for node in graph.nodes:
        if node.op in {"placeholder", "get_attr", "output"}:
            continue

        normalization_info = _normalization_dict.get((node.op, get_target(mod, node)))

        # Also check if the torch_packaged version of the op was specified to be normalized.
        if normalization_info is None and node.op == "call_function":
            # Strip off the mangle_index suffix here before checking the map.
            target = re.sub(
                r"\A<torch_package_\d+>",
                "<torch_package_>",
                _get_qualified_name(node.target),
            )
            torch_package_op_and_target = (node.op, target)
            normalization_info = _normalization_dict.get(torch_package_op_and_target)

        if normalization_info is None:
            continue

        # Get the normalized kwargs to be used by normalize_to_acc_op below. If
        # normalization_info.arg_replacement_tuples is empty then assume the function
        # signature must be left as is.
        assert normalization_info.arg_replacement_tuples is not None
        if len(normalization_info.arg_replacement_tuples) == 0:
            normalized_args = node.args
            normalized_kwargs = node.kwargs
        else:
            normalized_args = ()
            try:
                normalized_kwargs = get_normalized_kwargs(
                    node, normalization_info.arg_replacement_tuples
                )
            except Exception:
                print(
                    f"Error during kwarg normalization for: {node.format_node()}; "
                    f"arg_replacement_tuples={normalization_info.arg_replacement_tuples}"
                )
                raise

        if (
            normalization_info.needs_shapes_for_normalization
            and not expect_nodes_have_shapes
        ):
            # All nodes needing shapes for normalization should be custom mapped.
            assert normalization_info.custom_mapping_fn is not None
            # For custom mapping, the normalized_kwargs are used for the original op,
            # i.e. *before* custom acc_ops normalization. Do that now so that whoever
            # consumes the graph next (e.g. shape inference) can use kwargs safely.
            node.args = normalized_args
            node.kwargs = normalized_kwargs
            continue

        try:
            normalize_to_acc_op(
                node, normalization_info, normalized_args, normalized_kwargs
            )
        except Exception:
            print(f"Error during normalization for node: {node.format_node()}")
            raise

    # If there are any dead nodes left after normalization, eliminate them now.
    mod.graph.eliminate_dead_code()
