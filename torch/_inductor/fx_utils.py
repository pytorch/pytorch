# mypy: allow-untyped-defs
import contextlib
import operator
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import sympy

import torch
import torch.fx
from torch._dispatch.python import enable_python_dispatcher
from torch._inductor.fx_passes.control_dependencies import control_deps
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    rebind_unbacked,
    statically_known_true,
    sym_eq,
)
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map
from torch.utils.flop_counter import flop_registry

from .virtualized import V


# Check the pattern: (nn.module, F.function/torch.Tensor.method) matched.
# Works for length 2 patterns with 1 module and 1 function/method.
def matches_module_function_pattern(
    pattern: tuple[type[torch.nn.modules.Module], Callable[..., Any]],
    node: torch.fx.Node,
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


@dataclass(frozen=True)
class _FxNodeHash:
    node: torch.fx.Node
    target: torch.fx.node.Target
    args_id: int
    kwargs_id: int


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

    Since this runs in the context of Inductor, we assume that the input and
    output semantics for the graph (and any subgraphs) are not subject to change
    after class initialization.  Any violations of this assumption may result in
    undefined behavior.
    """

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        self.processed_hashes = OrderedSet[_FxNodeHash]()
        self.gm = gm

        # Import here to avoid circular import issues.
        from torch._inductor.compile_fx import _get_subgraph_names

        self.subgraph_updaters: dict[torch.fx.GraphModule, FakeTensorUpdater] = {
            (subgraph := getattr(self.gm, subgraph_name)): FakeTensorUpdater(subgraph)
            for subgraph_name in _get_subgraph_names(self.gm)
        }

        for node in self.gm.graph.nodes:
            self.processed_hashes.add(self.hash_node(node))

    def hash_node(self, node: torch.fx.Node) -> _FxNodeHash:
        return _FxNodeHash(node, node.target, id(node.args), id(node.kwargs))

    def incremental_update(self) -> int:
        """Update FakeTensors on self.graph. We will try to do the minimum amount of work."""
        existing_storages: defaultdict[Optional[int], int] = defaultdict(int)
        for node in self.gm.graph.nodes:
            existing_storages[get_node_storage(node)] += 1

        def is_intlist_same(new, old):
            return statically_known_true(sym_eq(new, old))

        def is_fake_tensor_same(
            new, old, *, strict: bool = True, node: torch.fx.Node | None = None
        ) -> bool:
            """Validate that two FakeTensors (or iterables thereof) are the same,
            including storage locations if strict mode is enabled.

            strict: disabling this flag will cause this function to only evaluate size,
            layout, stride, and device.  This is used to validate that arguments are
            equivalent enough for updating subgraphs."""

            if type(new) is not type(old):
                return False

            if isinstance(new, (list, tuple)):
                return len(new) == len(old) and all(
                    is_fake_tensor_same(new_i, old_i, strict=strict, node=node)
                    for new_i, old_i in zip(new, old)
                )

            if new is None:
                return old is None

            if not isinstance(new, torch.Tensor):
                assert isinstance(new, (torch.SymInt, torch.SymBool, torch.SymFloat)), (
                    f"Unknown type {type(new)} in {self.gm.graph}"
                )
                return (
                    new.node.shape_env._maybe_evaluate_static(
                        sympy.Eq(new.node.expr, old.node.expr)
                    )
                    == sympy.true
                )

            if new.layout != old.layout or not is_intlist_same(new.shape, old.shape):
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

            if not strict:
                return True

            if get_storage(new) == get_storage(old):
                return True

            def any_user_may_alias(node):
                if not isinstance(node.meta["val"], torch.Tensor):
                    # analysis too complicated on lists, can support in the future
                    return True

                for user in node.users:
                    if not (
                        isinstance(user.target, torch._ops.OperatorBase)
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
                        if (shape_env := V.fake_mode.shape_env) is not None:
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

        def should_process_node(node: torch.fx.Node) -> bool:
            return (
                callable(node.target)
                # control_deps doesn't have an impl for DispatchKey.AutogradCUDA, which
                # causes a test failure in
                # TestComputeCommReorderingBucketing::test_bucketing_split_for_overlap_blocking_deps_inductor.
                # TODO: remove this when resolving https://github.com/pytorch/pytorch/issues/165786
                and node.target is not control_deps
                # node.target will called with FakeTensor arguments, which are not
                # supported by Inductor lowerings. TODO: Investigate how to remove
                # this. See https://github.com/pytorch/pytorch/issues/164920
                and not hasattr(node.target, "_inductor_lowering_function")
            )

        def node_invokes_subgraph(
            node: torch.fx.Node, *args: Any, **kwargs: Any
        ) -> bool:
            return node.op == "call_function" and pytree.tree_any_only(
                torch.fx.GraphModule,
                lambda s: s in self.subgraph_updaters,
                (args, kwargs),
            )

        def extract_subgraphs_and_args(
            node: torch.fx.Node, *args: Any, **kwargs: Any
        ) -> tuple[tuple[torch.fx.GraphModule, ...], tuple[Any, ...] | None]:
            """HOPs that invoke subgraphs take a number of different forms.  This
            function regularizes them, returning a tuple of subgraphs contained in the
            args and a tuple of the args for the subgraphs.  This function assumes all
            subgraphs share a set of common arguments.

            This function assumes that node_invokes_subgraph(node, *args, **kwargs) is
            True.

            If the second return value is None, this function was unable to determine
            what args to pass to the subgraph(s)."""
            if node.target is torch.ops.higher_order.invoke_subgraph:
                return ((args[0],), tuple(args[2:]))
            if node.target is torch.ops.higher_order.cond:
                return (tuple(args[1:3]), tuple(args[3]))
            if node.target in (
                torch.ops.higher_order.invoke_quant_packed,
                torch.ops.higher_order.invoke_quant,
            ):
                return ((args[0],), tuple(args[1:]))
            if node.target is control_deps:
                assert not kwargs, (
                    "Subgraph arguments can be renamed, so we cannot consistently "
                    "handle kwargs at this point in the stack."
                )
                return ((args[1],), tuple(args[2:]))
            if node.target is torch.ops.higher_order.map_impl:
                # args[1][0] represents the first value in the list of values being
                # mapped over.  We assume it's representative of the whole list, since
                # Dynamo should have provided us multiple subgraphs otherwise.
                return ((args[0],), (args[1][0], *args[2:]))
            if node.target in (
                torch.ops.higher_order.while_loop,
                torch.ops.higher_order.while_loop_stack_output,
            ):
                return (tuple(args[:2]), (*args[2], *args[3]))
            if node.target is torch.ops.higher_order.scan:
                # Similarly to map_impl, we assume the first elements of init and xs
                # must be representative of the whole list.
                return ((args[0],), (args[1][0], args[2][0][0]))
            if node.target is torch.ops.higher_order.flex_attention:
                # The flex attention transformation is complicated, and a mapping from
                # args to subgraph inputs is inconsistent at best.  We'll skip for now.
                return tuple(
                    s
                    for s in pytree.tree_flatten(args)
                    if isinstance(s, torch.fx.GraphModule)
                    and s in self.subgraph_updaters
                ), None

            raise RuntimeError(
                f"Please add support for subgraph args to function {node.target}!"
            )

        @dataclass
        class SubgraphUpdating:
            args_updated: bool = False
            outputs_updated: bool = False

        nodes_updated: int = 0
        to_process = OrderedSet[int]()
        subgraph_updatings: dict[torch.fx.GraphModule, SubgraphUpdating] = {}
        for node in self.gm.graph.nodes:
            is_valid, args, kwargs = get_fake_args_kwargs(node, self.gm)
            if not is_valid:
                continue

            # NB: Be very careful about skipping nodes (via continues) here
            # and ask for a careful review when changing this code. The
            # consequence for incorrect FakeTensor metadata is difficult-to-debug
            # silent incorrectness.
            if (
                # Always run updates on nodes that invoke subgraphs
                not (invokes_subgraph := node_invokes_subgraph(node, *args, **kwargs))
                and self.hash_node(node) in self.processed_hashes
                and id(node) not in to_process
            ):
                continue

            if not should_process_node(node):
                continue

            if invokes_subgraph:
                subgraphs, subgraph_args = extract_subgraphs_and_args(
                    node, *args, **kwargs
                )

                for subgraph in subgraphs:
                    update_subgraph = subgraph not in subgraph_updatings
                    if update_subgraph:
                        subgraph_updatings[subgraph] = SubgraphUpdating()

                    # If the arguments being passed into the subgraph differ from the
                    # existing args the first time we see the subgraph, update the
                    # placeholder nodes in the subgraph.  Any updates past that point would
                    # make the graph internally inconsistent.
                    #
                    # NOTE: is_fake_tensor_equivalent deliberately does not check storages,
                    # since every invocation of the subgraph will have different storages.
                    # This means that we may incorrectly add or remove arg aliasing
                    # relationships, but there's no clear way around this if subgraphs can
                    # be multiply invoked.
                    if subgraph_args is not None:
                        for p, a in zip(
                            subgraph.graph.find_nodes(op="placeholder"), subgraph_args
                        ):
                            if not is_fake_tensor_same(
                                a, get_fake(p, subgraph), strict=False
                            ):
                                assert update_subgraph, (
                                    "subgraph args must have consistent values!"
                                )

                                subgraph_updatings[subgraph].args_updated = True
                                p.meta["val"] = a
                                nodes_updated += 1

                    if update_subgraph:
                        _, orig_output_args, _ = get_fake_args_kwargs(
                            subgraph.graph.output_node(), subgraph
                        )
                        nodes_updated += self.subgraph_updaters[
                            subgraph
                        ].incremental_update()
                        _, new_output_args, _ = get_fake_args_kwargs(
                            subgraph.graph.output_node(), subgraph
                        )

                        if not is_fake_tensor_same(
                            new_output_args, orig_output_args, strict=False
                        ):
                            subgraph_updatings[subgraph].outputs_updated = True

                # If the outputs of the subgraphs have not changed (and have been
                # previously cached), we can skip updating.  If the outputs of the
                # subgraph have changed, we'll run FakeTensor propagation for every
                # invocation of the subgraph, so that each node gets unique FakeTensor
                # values which maintain appropriate aliasing relationships.
                if (
                    not any(subgraph_updatings[s].outputs_updated for s in subgraphs)
                    and "val" in node.meta
                ):
                    continue

            with V.fake_mode, enable_python_dispatcher():
                new_fake_tensor = node.target(*args, **kwargs)

            if "val" in node.meta and is_fake_tensor_same(
                new_fake_tensor, node.meta["val"], node=node
            ):
                continue

            rebind_unbacked(V.fake_mode.shape_env, node, new_fake_tensor)

            node.meta["val"] = new_fake_tensor
            nodes_updated += 1

            if (shape_env := V.fake_mode.shape_env) and (
                symbol_to_path := compute_unbacked_bindings(shape_env, new_fake_tensor)
            ):
                # Refresh the bindings to the new symbols

                node.meta["unbacked_bindings"] = symbol_to_path

            existing_storages[get_node_storage(node)] += 1

            to_process.update(id(user) for user in node.users)

            self.processed_hashes.add(self.hash_node(node))

        return nodes_updated


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


def get_fake(x: Any, gm: Optional[torch.fx.GraphModule]) -> Any:
    """Return a fake tensor from the meta values of an input FX node.  If the input node
    is a get_attr node, we attempt to resolve it as a member of gm."""
    if isinstance(x, torch.fx.Node):
        if "val" in x.meta:
            return x.meta["val"]
        if "example_value" in x.meta:
            return x.meta["example_value"]
        if x.op == "get_attr" and isinstance(x.target, str) and hasattr(gm, x.target):
            return getattr(gm, x.target)
    # If there are no example values, return x
    return x


def get_fake_args_kwargs(
    x: torch.fx.Node, gm: Optional[torch.fx.GraphModule] = None
) -> tuple[bool, tuple[Any, ...], dict[str, Any]]:
    """
    First value returns a boolean if any of the input nodes don't have a faketensor and
    weren't resolved from gm.
    """
    args, kwargs = tree_map(partial(get_fake, gm=gm), (x.args, x.kwargs))
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
