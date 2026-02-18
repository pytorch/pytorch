# mypy: allow-untyped-defs
import contextlib
import operator
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, Optional

import sympy

import torch
import torch.fx
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    rebind_unbacked,
    statically_known_true,
    sym_eq,
)
from torch.fx.graph_module import _get_attr
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map
from torch.utils.flop_counter import flop_registry

from .virtualized import V


# Check the pattern: (nn.module, F.function/torch.Tensor.method) matched.
# Works for length 2 patterns with 1 module and 1 function/method.
def matches_module_function_pattern(
    pattern: tuple[type[torch.nn.modules.Module], Callable[..., Any]],
    node: torch.fx.node.Node,
    modules: dict[str, torch.nn.modules.Module],
) -> bool:
    if len(node.args) == 0:
        return False
    if not isinstance(node.args[0], torch.fx.Node) or not isinstance(
        node, torch.fx.Node
    ):
        return False
    # the first node is call_module
    if node.args[0].op != "call_module":
        return False
    if not isinstance(node.args[0].target, str):
        return False
    if node.args[0].target not in modules:
        return False
    if type(modules[node.args[0].target]) is not pattern[0]:
        return False
    # the second node is call_function or call_method
    if node.op != "call_function" and node.op != "call_method":
        return False
    if node.target != pattern[1]:
        return False
    # make sure node.args[0] output is only used by current node.
    if len(node.args[0].users) > 1:
        return False
    return True


class FakeTensorUpdater:
    """
    The main idea here is that it's difficult to maintain accurate fake
    tensors (our primary form of metadata) for each node in our graph as we
    transform it.

    The most reliable way to obtain this information is by rerunning
    faketensor propagation. However, in general, faketensor propagation is
    fairly expensive. So, instead we'd like to only rerun faketensor
    propagation on nodes that have changed.

    In order to detect which nodes have changed, we first hash its node,
    target, and argument lists (which are immutable in FX).

    Then, whenever we call incremental_update, we check which FX nodes have a
    new hash, and recompute the faketensor metadata for that node. Then, we
    continue to recursively compute the faketensors for all users until the
    fake tensors stop changing.
    """

    def __init__(self, graph: torch.fx.Graph) -> None:
        self.processed_hashes = OrderedSet[Any]()
        self.graph = graph

        for node in self.graph.nodes:
            self.processed_hashes.add(self.hash_node(node))

    def hash_node(self, node: torch.fx.Node):
        # todo(chilli): Not a great hash function
        return (node, node.target, id(node.args), id(node.kwargs))

    def incremental_update(self):
        """Update FakeTensors on self.graph. We will try to do the minimum amount of work."""
        existing_storages: defaultdict[Optional[int], int] = defaultdict(int)
        for node in self.graph.nodes:
            existing_storages[get_node_storage(node)] += 1

        def is_intlist_same(new, old):
            return statically_known_true(sym_eq(new, old))

        def is_fake_tensor_same(new, old, *, node):
            if type(new) is not type(old):
                return False
            if isinstance(new, (list, tuple)):
                if len(new) != len(old):
                    return False
                return all(
                    is_fake_tensor_same(new_i, old_i, node=node)
                    for new_i, old_i in zip(new, old)
                )
            if new is None:
                return old is None
            if not isinstance(new, torch.Tensor):
                assert isinstance(new, (torch.SymInt, torch.SymBool, torch.SymFloat)), (
                    f"Unknown type {type(new)} in {self.graph}"
                )
                return (
                    new.node.shape_env._maybe_evaluate_static(
                        sympy.Eq(new.node.expr, old.node.expr)
                    )
                    == sympy.true
                )
            if not is_intlist_same(new.shape, old.shape) or new.layout != old.layout:
                return False
            if new.layout == torch.strided and (
                not is_intlist_same(new.stride(), old.stride())
                or not statically_known_true(
                    new.storage_offset() == old.storage_offset()
                )
            ):
                return False

            if new.device != old.device:
                return False

            if get_storage(new) == get_storage(old):
                return True

            def any_user_may_alias(node):
                if not isinstance(node.meta["val"], torch.Tensor):
                    # analysis too complicated on lists, can support in the future
                    return True
                for user in node.users:
                    if not (
                        isinstance(
                            user.target,
                            (torch._ops.OpOverload, torch._ops.HigherOrderOperator),
                        )
                        or user.target
                        is torch._inductor.fx_passes.reinplace._generalized_scatter
                    ):
                        return True
                    if isinstance(user.target, torch._ops.HigherOrderOperator):
                        # HOPs that survive until inductor are all non-aliasing HOPs.
                        # We will likely never support HOPs that are aliasing.
                        continue
                    # Strategy: do a FakeTensor prop, see if the storage aliases.
                    # If Inductor ever gets tighter invariants on OpOverloads
                    # (that is, we ban things like torch.ops.aten.reshape calls in the graph),
                    # Then this could just be a fast schema lookup.
                    is_valid, args, kwargs = get_fake_args_kwargs(user)
                    if not is_valid:
                        return True
                    with (
                        V.fake_mode,
                        enable_python_dispatcher(),
                        contextlib.ExitStack() as stack,
                    ):
                        # Ignore unbacked symbols (if they exist): we're making
                        # this FakeTensor and then throwing it away.
                        shape_env = V.fake_mode.shape_env
                        if shape_env is not None:
                            stack.enter_context(
                                shape_env.ignore_fresh_unbacked_symbols()
                            )
                        new_fake_tensor = user.target(*args, **kwargs)
                    if not isinstance(new_fake_tensor, torch.Tensor):
                        # analysis too complicated on lists, can support in the future
                        return True
                    if get_storage(new_fake_tensor) == get_storage(node.meta["val"]):
                        return True
                return False

            # This is the case where it returns a completely fresh storage that's used nowhere else.
            # If the FakeTensor's storage is fresh and none of the node's users can alias it, then
            # we don't need to update this node.
            if (
                existing_storages[get_storage(old)] == 1
                and get_storage(new) not in existing_storages
                and not any_user_may_alias(node)
            ):
                return True

            return False

        def should_process_node(node):
            # node.target for nodes returning true from this function
            # are called under fake mode and does not work for inductor
            # lowerings. We check if the node.target is an aten operator
            # or operator.getitem which is used when returning multiple
            # tensors from an op.
            return node.op == "call_function" and (
                isinstance(node.target, torch._ops.OpOverload)
                or node.target is operator.getitem
                or node.target
                is torch._inductor.fx_passes.reinplace._generalized_scatter
            )

        to_process = OrderedSet[int]()
        for node in self.graph.nodes:
            # NB: Be very careful about skipping nodes (via continues) here
            # and ask for a careful review when changing this code. The
            # consequence for incorrect FakeTensor metadata is difficult-to-debug
            # silent incorrectness.
            if (
                self.hash_node(node) in self.processed_hashes
                and id(node) not in to_process
            ):
                continue

            if not should_process_node(node):
                continue

            is_valid, args, kwargs = get_fake_args_kwargs(node)
            if not is_valid:
                continue
            with V.fake_mode, enable_python_dispatcher():
                new_fake_tensor = node.target(*args, **kwargs)

            if "val" in node.meta and is_fake_tensor_same(
                new_fake_tensor, node.meta["val"], node=node
            ):
                continue

            rebind_unbacked(V.fake_mode.shape_env, node, new_fake_tensor)

            node.meta["val"] = new_fake_tensor
            if (shape_env := V.fake_mode.shape_env) and (
                symbol_to_path := compute_unbacked_bindings(shape_env, new_fake_tensor)
            ):
                # Refresh the bindings to the new symbols

                node.meta["unbacked_bindings"] = symbol_to_path

            existing_storages[get_node_storage(node)] += 1

            to_process.update([id(user) for user in node.users])

            self.processed_hashes.add(self.hash_node(node))


def get_storage(t: torch.Tensor) -> int:
    return t.untyped_storage()._cdata


def get_node_storage(node: torch.fx.Node) -> Optional[int]:
    if "val" not in node.meta:
        return None
    if not isinstance(node.meta["val"], torch.Tensor):
        return None
    if not torch._C._has_storage(node.meta["val"]):
        return None
    return get_storage(node.meta["val"])


def get_fake(x):
    if isinstance(x, torch.fx.Node):
        if "val" not in x.meta:
            return x
        return x.meta["val"]
    return x


def get_fake_args_kwargs(x: torch.fx.Node) -> tuple[bool, tuple[Any], dict[str, Any]]:
    """
    First value returns a boolean if any of the input nodes don't have a faketensor.
    """
    args, kwargs = tree_map(get_fake, (x.args, x.kwargs))
    if any(
        isinstance(a, torch.fx.Node) for a in pytree.arg_tree_leaves(*args, **kwargs)
    ):
        return False, args, kwargs
    return True, args, kwargs


def is_node_realized(node: torch.fx.Node) -> bool:
    """Returns true if a node is always realized when lowered to inductor IR.

    NOTE: This may return some false negatives. e.g. it doesn't
    handle buffers realized heuristically during lowering, or
    buffers realized indirectly through view ops.
    """
    from torch._inductor.lowering import fallbacks, needs_realized_inputs

    def is_buffer(node: torch.fx.Node) -> bool:
        if node.op == "call_function" and node.target is operator.getitem:
            # For nodes with multiple outputs, we get the fx graph:
            #     foo = torch.ops.aten.foo(...)
            #     getitem = foo[0]
            #     getitem_1 = foo[1]
            # where we need to check if foo is a fallback kernel
            return is_buffer(node.args[0])  # type: ignore[arg-type]
        return node.op in ("placeholder", "output") or node.target in fallbacks

    if is_buffer(node):
        return True

    def realizes_inputs(node: torch.fx.Node) -> bool:
        return node.op == "output" or node.target in needs_realized_inputs

    if any(realizes_inputs(user) for user in node.users):
        return True

    # Otherwise, assume node isn't realized
    return False


def count_flops_fx(node: torch.fx.Node) -> Optional[int]:
    if not countable_fx(node) or isinstance(node.target, str):
        return None
    with FakeTensorMode(allow_non_fake_inputs=True):
        success, args, kwargs = get_fake_args_kwargs(node)

        if success:
            with torch.utils.flop_counter.FlopCounterMode(
                display=False
            ) as flop_counter_mode:
                node.target(*args, **kwargs)

            counted_flops = flop_counter_mode.get_total_flops()
            return counted_flops
    return None


def countable_fx(node: torch.fx.Node) -> bool:
    """
    Whether or not we can count the flops of an FX node.
    """
    assert isinstance(node, torch.fx.Node)
    if not hasattr(node, "target"):
        return False
    target = node.target
    if not hasattr(target, "overloadpacket"):
        return target in flop_registry
    packet = target.overloadpacket
    return packet in flop_registry


def get_arg_value(
    node: torch.fx.Node, arg_number: int, kwarg_name: Optional[str] = None
) -> Any:
    if len(node.args) > arg_number:
        return node.args[arg_number]
    elif kwarg_name is None:
        return None
    else:
        return node.kwargs.get(kwarg_name)


def filter_nodes(nodes: Iterable[torch.fx.Node], fn: Any) -> list[torch.fx.Node]:
    fns = [fn]
    if isinstance(fn, torch._ops.OpOverloadPacket):
        fns.extend([getattr(fn, overload) for overload in fn.overloads()])

    return [node for node in nodes if node.target in fns]


def extract_target(node: torch.fx.Node) -> torch.fx.node.Target:
    """For call_function and call_method, we directly use the target function;
    For call_module, the target is string, and we treat the module class
     as a function.
    """
    if node.op == "call_module":
        assert isinstance(node.target, str)
        return _get_attr(node.graph.owning_module, node.target).__class__
    return node.target


def _args(n: torch.fx.Node) -> list[torch.fx.node.Argument]:
    args: list[torch.fx.node.Argument] = []
    torch.fx.map_arg((n.args, n.kwargs), args.append)
    return args


def stable_topological_sort(graph: torch.fx.Graph) -> None:
    # Nodes are in exactly one of these three collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.
    ready = OrderedSet[torch.fx.Node]()

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
    cursor = None
    while pending:
        node = pending.pop()
        waiting_for = [x for x in _args(node) if x not in ready]
        if waiting_for:
            # We have unprocessed input nodes. Might as well wait for the last
            # arg so an already sorted list will only recheck this node once.
            waiting[waiting_for[-1]].append(node)
        else:
            ready.add(node)
            if cursor and cursor.next is not node:
                cursor.append(node)
            cursor = node
            # Mark the nodes that have been waiting for this node to finish as
            # ready to check again.
            pending.extend(reversed(waiting.pop(node, ())))

    assert not waiting and len(ready) == len(graph.nodes)


def is_start_of_fx_graph(graph: torch.fx.Graph, node: torch.fx.Node) -> bool:
    # first node in the graph
    return node is next(iter(graph.nodes))


# match: copy_, relu_, _set_grad_enabled, manual_seed, _enter_autocast, etc
# doesn't match: __rshift__, etc
_mutation_op_re = re.compile(r"(?<!_)(_$|_[.]|(\b|_)(set|enter|exit|seed)(\b|_))(?!_)")


def fixme_incorrect_inductor_schema_op(op: torch._ops.OpOverload) -> bool:
    if op.namespace != "inductor":
        return False

    # TODO - fix schema
    # Dont add any more !
    return op in (
        torch.ops.inductor.accumulate_grad_.default,
        torch.ops.inductor.resize_storage_bytes_.default,
    )


def is_mutation_op(node: torch.fx.Node) -> bool:
    if isinstance(
        node.target, torch._ops.OpOverload
    ) and not fixme_incorrect_inductor_schema_op(node.target):
        return node.target._schema.is_mutable
    elif isinstance(
        node.target, torch._higher_order_ops.auto_functionalize.AutoFunctionalized
    ):
        return False
    if node.op == "call_function":
        assert callable(node.target)
        if _mutation_op_re.search(node.target.__name__):
            return True
    elif node.op == "call_method":
        assert isinstance(node.target, str)
        if _mutation_op_re.search(node.target):
            return True
    return node.kwargs.get("out") is not None


def same_mutation_regions(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    assert "mutation_region_id" in a.meta
    assert "mutation_region_id" in b.meta
    return a.meta["mutation_region_id"] == b.meta["mutation_region_id"]


def get_mutation_region_id(graph: torch.fx.Graph, node: torch.fx.Node) -> int:
    n = node
    while "mutation_region_id" not in n.meta and not is_start_of_fx_graph(graph, n):
        n = n.prev
    mutation_region_id = n.meta.get("mutation_region_id", 0)
    while n is not node:
        n = n.next
        if is_mutation_op(n):
            mutation_region_id += 1
        n.meta["mutation_region_id"] = mutation_region_id
    return mutation_region_id


def should_compute_mutation_region_ids(graph: torch.fx.Graph) -> bool:
    return "mutation_region_id" not in next(iter(graph.nodes)).meta


def compute_mutation_region_ids(graph: torch.fx.Graph) -> None:
    mutation_region_id = 0
    for nd in graph.nodes:
        if is_mutation_op(nd):
            mutation_region_id += 1
        nd.meta["mutation_region_id"] = mutation_region_id


def same_meta(node1: torch.fx.Node, node2: torch.fx.Node):
    """True if two nodes have the same metadata"""
    val1 = node1.meta.get("val")
    val2 = node2.meta.get("val")
    return (
        val1 is not None
        and val2 is not None
        and statically_known_true(sym_eq(val1.size(), val2.size()))
        and val1.layout == val2.layout
        and val1.dtype == val2.dtype
        and val1.device == val2.device
        and (
            val1.layout != torch.strided
            or statically_known_true(sym_eq(val1.stride(), val2.stride()))
        )
        # Check conjugate and negative bits - a clone that resolves these is not a no-op
        and val1.is_conj() == val2.is_conj()
        and val1.is_neg() == val2.is_neg()
    )


def apply_pass_to_subgraphs(
    pass_fn: Callable[[torch.fx.Graph], None], graph: torch.fx.Graph
):
    """Recursively apply a pass function to all subgraphs referenced by get_attr nodes."""
    gm = graph.owning_module
    if gm is None:
        return
    subgraph_names: OrderedSet[str] = OrderedSet(
        x.target for x in graph.find_nodes(op="get_attr")
    )
    for child_name, child_mod in gm.named_children():
        if child_name in subgraph_names and isinstance(child_mod, torch.fx.GraphModule):
            pass_fn(child_mod.graph)


def find_next_users(node: torch.fx.Node) -> list[torch.fx.Node]:
    """For a multi-output node, look through getitem users to find actual consumers."""
    next_users = []
    for getitem_node in node.users:
        for getitem_user in getitem_node.users:
            if getitem_user not in next_users:
                next_users.append(getitem_user)
    return next_users
