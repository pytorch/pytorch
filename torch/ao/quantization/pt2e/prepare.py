import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.prepare import (
    _get_arg_as_input_act_obs_or_fq,
    _get_output_act_obs_or_fq,
    _get_dtype_and_is_dynamic,
    _insert_obs_or_fq,
    _maybe_insert_output_observer_for_node,
    _save_state,
    _is_activation_post_process_node,
    _get_qspec_for_arg,
)
from torch.fx import (
    GraphModule,
    Node,
)
from torch.fx.node import Argument

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    EdgeOrNode,
    SharedQuantizationSpec,
    QuantizationSpecBase,
)
from torch.ao.quantization import ObserverOrFakeQuantize

def _find_root(edge_or_node: EdgeOrNode, shared_with_map: Dict[EdgeOrNode, EdgeOrNode]) -> EdgeOrNode:
    parent = shared_with_map[edge_or_node]
    if parent == edge_or_node:
        return edge_or_node
    root = _find_root(parent, shared_with_map)
    # path compression
    shared_with_map[edge_or_node] = root
    return root

def _union(a: EdgeOrNode, b: EdgeOrNode, shared_with_map: Dict[EdgeOrNode, EdgeOrNode]):
    """Merge the subtree for `b` with `a`, the order is important here
    """
    root_a = _find_root(a, shared_with_map)
    root_b = _find_root(b, shared_with_map)
    # union the two trees
    print("union:", root_a, root_b)
    shared_with_map[root_b] = root_a
    return

def _update_shared_with(edge_or_node: EdgeOrNode, qspec: QuantizationSpecBase, shared_with_map: Dict[EdgeOrNode, EdgeOrNode]):
    if isinstance(qspec, SharedQuantizationSpec):
        sharing_with = qspec.edge_or_node
        _union(sharing_with, edge_or_node, shared_with_map)

def _find_root_qspec(qspec: QuantizationSpecBase, edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase], shared_with_map: Dict[EdgeOrNode, EdgeOrNode]):
    if isinstance(qspec, SharedQuantizationSpec):
        sharing_with = qspec.edge_or_node
        root = _find_root(sharing_with, shared_with_map)
        qspec = edge_or_node_to_qspec[root]
        return _find_root_qspec(qspec, edge_or_node_to_qspec, shared_with_map)
    return qspec

def _get_edge_or_node_to_group_id(model: torch.fx.GraphModule):
    """Map from edge/node to the group ID, generated from quantization annotations,
    edge/node with the same group ID should use the same observer/fake_quant instance
    """
    print("model.graph:", model.graph)
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase] = {}
    for n in model.graph.nodes:
        if hasattr(n, "meta") and "quantization_annotation" in n.meta:
            qa = n.meta["quantization_annotation"]
            for input_to_n, qspec in qa.input_qspec_map.items():
                input_edge = (input_to_n, n)
                edge_or_node_to_qspec[input_edge] = qspec
            if qa.output_qspec is not None:
                output_node = n
                qspec = qa.output_qspec
                edge_or_node_to_qspec[output_node] = qspec

    # means the observer of key should be shared with observer with value, by default it will
    # be shared with itself
    shared_with_map: Dict[EdgeOrNode, EdgeOrNode] = {k: k for k in edge_or_node_to_qspec.keys()}
    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        if isinstance(edge_or_node, torch.fx.Node):
            output_node = edge_or_node
            _update_shared_with(output_node, qspec, shared_with_map)
        else:
            input_edge = edge_or_node
            arg, n = input_edge
            arg_as_output_root_qspec = None
            if arg in edge_or_node_to_qspec:
                arg_as_output_qspec = edge_or_node_to_qspec[arg]
                arg_as_output_root_qspec = _find_root_qspec(arg_as_output_qspec, edge_or_node_to_qspec, shared_with_map)

            input_edge_root = _find_root(input_edge, shared_with_map)
            input_edge_root_qspec = edge_or_node_to_qspec[input_edge_root]
            input_edge_root_qspec = _find_root_qspec(input_edge_root_qspec, edge_or_node_to_qspec, shared_with_map)
            print(f"{input_edge} root qspec: {input_edge_root_qspec} output qspec: {arg_as_output_root_qspec}")
            if (
                arg_as_output_root_qspec is not None and
                hasattr(arg_as_output_root_qspec, "dtype") and
                hasattr(input_edge_root_qspec, "dtype") and
                arg_as_output_root_qspec.dtype == input_edge_root_qspec.dtype
            ):
                print(f"union input edge: {input_edge} with {arg}")
                # the input arg to the node should reuse the existing output observer for arg
                # since dtype is the same (we may want to extend this to be a more strict check
                # in the future)
                _union(arg, input_edge, shared_with_map)
            _update_shared_with(input_edge, qspec, shared_with_map)

    print("sharing:", shared_with_map)
    # now that we get the sharing relations between all edges and nodes, we can assingn group ids
    cur_group_id = 0
    edge_or_node_to_group_id: Dict[EdgeOrNode, int] = {}
    for edge_or_node, shared_with in shared_with_map.items():
        root = _find_root(edge_or_node, shared_with_map)
        if root not in edge_or_node_to_group_id:
            edge_or_node_to_group_id[root] = cur_group_id
            cur_group_id += 1
        edge_or_node_to_group_id[edge_or_node] = edge_or_node_to_group_id[root]

    return edge_or_node_to_group_id

def _maybe_insert_input_observer_for_arg_or_kwarg(
    node: Union[Node, Any],
    arg: Argument,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    edge_or_node_to_group_id: Dict[EdgeOrNode, int],
    group_id_to_obs_or_fq: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
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
            new_inner_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
                node, inner_arg, qconfig, model, named_modules, edge_or_node_to_group_id, group_id_to_obs_or_fq, is_qat,
            )
            new_arg_to_return.append(new_inner_arg)
        return type(arg)(new_arg_to_return)

    if not isinstance(arg, Node):
        return arg
    assert isinstance(arg, Node)
    # default (no observer)
    new_arg = arg

    quantization_annotation = node.meta.get("quantization_annotation", QuantizationAnnotation())
    arg_as_input_act_obs_or_fq = _get_arg_as_input_act_obs_or_fq(arg, node, named_modules, edge_or_node_to_group_id, group_id_to_obs_or_fq, is_qat)
    arg_as_input_target_dtype, arg_as_input_target_is_dynamic = _get_dtype_and_is_dynamic(arg_as_input_act_obs_or_fq)

    arg_as_output_act_obs_or_fq = _get_output_act_obs_or_fq(arg, named_modules, edge_or_node_to_group_id, group_id_to_obs_or_fq, is_qat)
    arg_as_output_target_dtype, arg_as_output_target_is_dynamic = _get_dtype_and_is_dynamic(arg_as_output_act_obs_or_fq)

    if arg_as_input_target_is_dynamic or arg_as_input_target_dtype not in [torch.float, None]:
        if arg_as_input_target_dtype == arg_as_output_target_dtype and \
           arg_as_input_target_is_dynamic == arg_as_output_target_is_dynamic:
            assert _is_activation_post_process_node(arg, named_modules)
            assert arg_as_input_act_obs_or_fq is not None
            observed_arg = arg.args[0]
            assert isinstance(observed_arg, Node), f"expect observed argument to be a Node, but got: {type(observed_arg)}"
            assert observed_arg in edge_or_node_to_group_id, \
                f"can't find a sharing group for node: {observed_arg}"
            group_id = edge_or_node_to_group_id[observed_arg]
            # reuse the existing obs/fq
            arg_as_input_act_obs_or_fq = group_id_to_obs_or_fq[group_id]
            # we don't need to insert new observer node
            new_arg = arg
        else:
            # skip inserting new observers if there is an observer inserted for the arg before
            # that has the same dtype that we want to insert here
            # alternatively we could have a dedup pass after we insert all observers to deduplicate
            # observers
            # Example:
            # arg -> existing_obs -> conv1
            #    \ -> conv2
            #
            # instead of inserting new observers we will have:
            # arg -> existing_obs -> conv1
            #                   \ -> conv2
            existing_obs_node = None
            for maybe_obs_node in arg.users.keys():
                if maybe_obs_node.op == 'call_module':
                    maybe_obs_mod = named_modules[maybe_obs_node.target]  # type: ignore[index]
                    if (
                        type(maybe_obs_mod) == type(arg_as_input_act_obs_or_fq) and
                        maybe_obs_mod.dtype == arg_as_input_target_dtype
                    ):
                        arg_as_input_act_obs_or_fq = maybe_obs_mod  # type: ignore[assignment]
                        existing_obs_node = maybe_obs_node
                        break

            assert arg_as_input_act_obs_or_fq is not None
            if existing_obs_node is None:
                # When quantizing two layers with different configs we can have
                # conv2d (int8) -> avgpool(uint8)
                # In this case observer insertion for avgpool will come here but the input
                # to avgpool will be output observer of conv2d
                # Now the obs map that we update must correspond to the original input of
                # avgpool and not the output obs of conv2d
                # This is because when referring to the edge, quantizer would refer to
                # original input and not the observed one.
                while _is_activation_post_process_node(arg, named_modules):
                    arg = arg.args[0]  # type: ignore[assignment]
                group_id = edge_or_node_to_group_id[(arg, node)]
                print(f"group id: {group_id} {group_id_to_obs_or_fq}")
                if group_id not in group_id_to_obs_or_fq:
                    group_id_to_obs_or_fq[group_id] = arg_as_input_act_obs_or_fq
                arg_as_input_act_obs_or_fq = group_id_to_obs_or_fq[group_id]
                print(f"group id: {group_id} {group_id_to_obs_or_fq}")
                new_obs_node = _insert_obs_or_fq(
                    arg, arg_as_input_act_obs_or_fq, model, named_modules, model.graph)
                # override this arg to be the observed arg
                new_arg = new_obs_node
            else:
                new_arg = existing_obs_node

    return new_arg

def _maybe_insert_input_observers_for_node(
    node: Node,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    edge_or_node_to_group_id: Dict[EdgeOrNode, int],
    group_id_to_obs_or_fq: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> None:
    """
    If needed, inserts observers to the input args and kwargs of `node`.
    Note: modifies `node` inplace.

    For example, if cur_node needs an observer after prev_node, we change from

      prev_node -> cur_node

    To

      prev_node -> obs -> cur_node

    """
    # Look through every input arg.  If that arg's target dtype does not
    # match the current node's target dtype, insert an observer.
    new_args = []
    for arg in node.args:
        new_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
            node, arg, qconfig, model, named_modules, edge_or_node_to_group_id, group_id_to_obs_or_fq, is_qat,
        )
        new_args.append(new_arg)

    # Clone has memory_format kwarg that persist in exported graph
    # this is just a work around for that.
    assert (
        node.target == torch.ops.aten.clone.default or len(node.kwargs) == 0
    ), " expecting kwargs for aten op IR to be empty"

    # assign the new args to the node, inplace
    node.args = tuple(new_args)

def _maybe_insert_input_and_output_observers_for_node(
    node: Node,
    model: torch.fx.GraphModule,
    edge_or_node_to_group_id: Dict[EdgeOrNode, int],
    group_id_to_obs_or_fq: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
):
    this_node_quantization_annotation = node.meta["quantization_annotation"] if "quantization_annotation" in node.meta else None
    if "val" in node.meta:
        output_is_a_tensor = (
            this_node_quantization_annotation is not None and
            isinstance(node.meta["val"], FakeTensor)
        )
    else:
        output_is_a_tensor = this_node_quantization_annotation is not None

    skip_inserting_input_and_output_observers = (
        this_node_quantization_annotation is None
    )

    if skip_inserting_input_and_output_observers:
        return

    named_modules = dict(model.named_modules(remove_duplicate=False))

    _maybe_insert_input_observers_for_node(
        node,
        None,  # qconfig
        model,
        named_modules,
        edge_or_node_to_group_id,
        group_id_to_obs_or_fq,
        is_qat,
    )

    skip_inserting_output_observers = (
        not output_is_a_tensor
    )

    if skip_inserting_output_observers:
        return

    # this returns the new observer node if it was needed
    maybe_output_obs_node = _maybe_insert_output_observer_for_node(node, model, named_modules, model.graph, edge_or_node_to_group_id, group_id_to_obs_or_fq, is_qat)

    if maybe_output_obs_node is None:
        return
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

def prepare(
    model: GraphModule,
    node_name_to_scope: Dict[str, Tuple[str, type]],
    is_qat: bool,
) -> GraphModule:
    # Since we are mutating the graph as we go, we iterate over the original
    # nodes before observer insertion, instead of model.graph.nodes.
    nodes_before_observation = list(model.graph.nodes)
    # obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize] = {}
    edge_or_node_to_group_id = _get_edge_or_node_to_group_id(model)
    print("edge or node to group:", edge_or_node_to_group_id)
    group_id_to_obs_or_fq: Dict[int, ObserverOrFakeQuantize] = {}

    for node in nodes_before_observation:
        _maybe_insert_input_and_output_observers_for_node(node, model, edge_or_node_to_group_id, group_id_to_obs_or_fq, is_qat)

    model = GraphModule(model, model.graph)

    _save_state(
        model,
        {},  # node_name_to_qconfig
        node_name_to_scope,
        PrepareCustomConfig(),
        {},  # equalization_node_name_to_qconfig
        QConfigMapping(),
        is_qat,
        set()  # observed_node_names
    )
    return model
