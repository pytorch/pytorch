# mypy: allow-untyped-defs
import contextlib
import operator
from collections import defaultdict
from collections.abc import (
    Callable,
    Collection,
    Container,
    Generator,
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Any

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


def _is_fake_tensor_same(
    new: Any,
    old: Any,
    existing_storages: Mapping[int | None, int] | None = None,
    *,
    check_dtype: bool = True,
    check_strides: bool = True,
    check_storage: bool = True,
    node: torch.fx.Node | None = None,
    recursive_ids: OrderedSet[tuple[int, int]] | None = None,
) -> bool:
    """Validate that two FakeTensors or collections thereof are the same."""

    def is_intlist_same(new, old):
        return statically_known_true(sym_eq(new, old))

    if type(new) is not type(old):
        return False

    if not isinstance(new, torch.Tensor):
        if new is None:
            return old is None

        if isinstance(new, Mapping):
            if new.keys() != old.keys():
                return False
            return all(
                _is_fake_tensor_same(
                    new[key],
                    old[key],
                    existing_storages,
                    check_dtype=check_dtype,
                    check_strides=check_strides,
                    check_storage=check_storage,
                    node=node,
                    recursive_ids=recursive_ids,
                )
                for key in new
            )

        if isinstance(new, Collection) and not isinstance(new, (str, bytes)):
            if recursive_ids is None:
                recursive_ids = OrderedSet()

            id_pair = (id(new), id(old))
            visited = id_pair in recursive_ids
            recursive_ids.add(id_pair)

            # If we've visited this exact check before while recursing, all elements in
            # this collection have already been validated (or will be validated in the
            # future) by a call at a different layer of recursion.
            return visited or (
                len(new) == len(old)
                and all(
                    _is_fake_tensor_same(
                        new_i,
                        old_i,
                        existing_storages,
                        check_dtype=check_dtype,
                        check_strides=check_strides,
                        check_storage=check_storage,
                        node=node,
                        recursive_ids=recursive_ids,
                    )
                    for new_i, old_i in zip(new, old)
                )
            )

        if isinstance(new, torch.types.py_sym_types):
            return (
                new.node.shape_env._maybe_evaluate_static(
                    sympy.Eq(new.node.expr, old.node.expr)
                )
                == sympy.true
            )

        # If this is not a collection, a sym-type, or a Tensor, then check Python
        # equality. This is conservative for objects without custom __eq__ methods.
        return new == old

    if new.layout != old.layout or not is_intlist_same(new.shape, old.shape):
        return False

    if new.device != old.device:
        return False

    if check_dtype and new.dtype != old.dtype:
        return False

    if (
        check_strides
        and new.layout == torch.strided
        and not is_intlist_same(new.stride(), old.stride())
    ):
        return False

    if not check_storage:
        return True

    if not statically_known_true(new.storage_offset() == old.storage_offset()):
        return False

    if get_storage(new) == get_storage(old):
        return True

    if node is None or existing_storages is None:
        return False

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
                    stack.enter_context(shape_env.ignore_fresh_unbacked_symbols())
                new_fake_tensor = user.target(*args, **kwargs)

            if not isinstance(new_fake_tensor, torch.Tensor):
                # analysis too complicated on lists, can support in the future
                return True

            if get_storage(new_fake_tensor) == get_storage(node.meta["val"]):
                return True

        return False

    # This is the case where it returns a completely fresh storage that's used nowhere
    # else. If the FakeTensor's storage is fresh and none of the node's users can alias
    # it, then we don't need to update this node.
    if (
        existing_storages.get(get_storage(old), 0) == 1
        and get_storage(new) not in existing_storages
        and not any_user_may_alias(node)
    ):
        return True

    return False


def _extract_subgraphs_and_args(
    node: torch.fx.Node,
    valid_subgraphs: Container[torch.fx.GraphModule],
    *args: Any,
    **kwargs: Any,
) -> Generator[tuple[torch.fx.GraphModule, tuple[Any, ...] | None]]:
    """HOPs that invoke subgraphs take a number of different forms.  This function
    regularizes them, yielding subgraphs from the args and kwargs coupled with
    appropriate args to update those subgraphs.

    If the second yielded value is None, this function was unable to determine what args
    to pass to the subgraph."""
    if node.target is torch.ops.higher_order.associative_scan:
        # Associative scan operates on slices of xs (see: scan), but multiple slices.
        # Use the same slice twice to account for cases where only a single slice is
        # input.
        yield args[0], (*(a[0] for a in args[1]), *(a[0] for a in args[1]), *args[2])
    elif node.target is torch.ops.higher_order.cond:
        subgraph_args = tuple(args[3])
        yield args[1], subgraph_args
        yield args[2], subgraph_args
    elif node.target in (
        torch.ops.higher_order.flex_attention,
        torch.ops.higher_order.flex_attention_backward,
    ):
        # flex_attention subgraphs have a small fixed scalar prefix plus optional
        # captured tensors.  The backward HOP also has a joint graph with one extra
        # gradient scalar before the optional captured tensors.  We assume this format
        # here (adding some defensive assertions), which means that the only inputs we
        # may need to update are the optional additional tensors. See
        # torch._higher_order_ops.flex_attention._math_attention_inner for more details.

        def get_subgraph_args(
            subgraph: torch.fx.GraphModule,
        ) -> tuple[torch.Tensor, ...]:
            return tuple(
                get_fake(n, subgraph)
                for n in subgraph.graph.find_nodes(op="placeholder")
            )

        def is_integer(d: torch.dtype) -> bool:
            return not d.is_floating_point and not d.is_complex

        def new_score_arg(reference: torch.Tensor) -> torch.Tensor:
            return args[0].new_zeros(
                (),
                dtype=args[0].dtype,
                requires_grad=reference.requires_grad,
            )

        if node.target is torch.ops.higher_order.flex_attention:
            score_subgraph_idx = 3
            mask_subgraph = args[4][-1]
            score_mod_other_buffers_idx = 7
            mask_mod_other_buffers_idx = 8
        else:
            score_subgraph_idx = 7
            mask_subgraph = args[9][-1]
            score_mod_other_buffers_idx = 12
            mask_mod_other_buffers_idx = 13

        score_subgraph: torch.fx.GraphModule = args[score_subgraph_idx]
        score_subgraph_args = get_subgraph_args(score_subgraph)
        mask_subgraph_args = get_subgraph_args(mask_subgraph)
        score_args = (new_score_arg(score_subgraph_args[0]), *score_subgraph_args[1:5])

        integer_arg_dtypes = OrderedSet[torch.dtype](
            a.dtype for a in chain(score_args[1:], mask_subgraph_args[:4])
        )
        assert (
            len(integer_arg_dtypes) == 1
            and is_integer(integer_arg_dtypes.pop())
            and all(
                len(a.size()) == 0 for a in chain(score_args, mask_subgraph_args[:4])
            )
        ), "flex_attention subgraph arg format has changed!"

        yield (
            score_subgraph,
            (
                *score_args,
                *args[score_mod_other_buffers_idx],
            ),
        )
        yield (
            mask_subgraph,
            (
                *mask_subgraph_args[:4],
                *args[mask_mod_other_buffers_idx],
            ),
        )

        if node.target is torch.ops.higher_order.flex_attention_backward:
            if isinstance(joint_subgraph := args[8], torch.fx.GraphModule):
                joint_subgraph_args = get_subgraph_args(joint_subgraph)
                joint_args = (
                    new_score_arg(joint_subgraph_args[0]),
                    *joint_subgraph_args[1:5],
                    new_score_arg(joint_subgraph_args[5]),
                )
                yield (
                    joint_subgraph,
                    (
                        *joint_args,
                        *args[score_mod_other_buffers_idx],
                    ),
                )
    elif node.target in (
        torch.ops.higher_order.foreach_map,
        torch.ops.higher_order.invoke_quant_packed,
        torch.ops.higher_order.invoke_quant,
    ):
        yield args[0], tuple(args[1:])
    elif node.target is torch.ops.higher_order.with_effects:
        if args[1] is torch.ops.higher_order.invoke_subgraph:
            yield args[2], tuple(args[4:])
        else:
            raise AssertionError(
                "Please add FakeTensorUpdater subgraph arg support for "
                f"with_effects wrapping {args[1]}!"
            )
    elif node.target is torch.ops.higher_order.invoke_subgraph:
        yield args[0], tuple(args[2:])
    elif node.target is torch.ops.higher_order.run_const_graph:
        yield args[0], tuple(args[1])
    elif node.target is torch.ops.higher_order.map_impl:
        # Map is applied over slices from the first dimension of each value in args[1].
        yield args[0], (*(a[0] for a in args[1]), *args[2])
    elif node.target is torch.ops.higher_order.scan:
        # Scans accept a dim keyword, but the dimensions will be reordered so that at
        # this point we always scan over dim 0.
        yield args[0], (*args[1], *(a[0] for a in args[2]), *args[3])
    elif node.target in (
        torch.ops.higher_order.while_loop,
        torch.ops.higher_order.while_loop_stack_output,
    ):
        subgraph_args = (*args[2], *args[3])
        yield args[0], subgraph_args
        yield args[1], subgraph_args
    elif node.target is control_deps:
        assert not kwargs, (
            "Subgraph arguments can be renamed, so we cannot consistently "
            "handle kwargs at this point in the stack."
        )
        yield args[1], tuple(args[2:])
    else:
        raise AssertionError(
            f"Please add FakeTensorUpdater subgraph arg support for {node.target}!"
        )


@dataclass(frozen=True)
class _FxNodeHash:
    node: torch.fx.Node
    target: torch.fx.node.Target
    arg_hashes: tuple[int, ...]
    kwarg_hashes: tuple[int, ...]


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
    output semantics for the outermost graph are not subject to change after class
    initialization, but we allow dtype and striding changes for subgraphs.  Any other
    changes will result in errors or undefined behavior.
    """

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        self.processed_hashes = OrderedSet[_FxNodeHash]()
        self.gm = gm
        self.subgraph_updaters: dict[torch.fx.GraphModule, FakeTensorUpdater] = {}
        self.refresh_subgraph_updaters()

        for node in self.gm.graph.nodes:
            is_valid, _, _ = get_fake_args_kwargs(node, self.gm)
            if is_valid:
                self.processed_hashes.add(self.hash_node(node))

    def refresh_subgraph_updaters(self) -> None:
        # Import here to avoid circular import issues.
        from torch._inductor.compile_fx import _get_subgraph_names

        for subgraph_name in _get_subgraph_names(self.gm):
            subgraph = getattr(self.gm, subgraph_name)
            if subgraph not in self.subgraph_updaters:
                self.subgraph_updaters[subgraph] = FakeTensorUpdater(subgraph)

    def hash_node(self, node: torch.fx.Node) -> _FxNodeHash:
        def get_hash_or_ids(n_iter: Iterable[Any]) -> tuple[int, ...]:
            """Replace unhashable items from the input with the ids of those items, and
            hash all other items.  This is kludgy, but allows us to attempt to account
            for unhashable classes like torch.fx.Node when hashing args and kwargs for
            other nodes."""

            def get_hash_or_id(o: object) -> int:
                try:
                    return hash(o)
                except TypeError:
                    return id(o)

            return tuple(get_hash_or_id(n) for n in n_iter)

        return _FxNodeHash(
            node,
            node.target,
            get_hash_or_ids(node.args),
            get_hash_or_ids(chain.from_iterable(node.kwargs.items())),
        )

    def incremental_update(self, nodes_to_process: Iterable[torch.fx.Node] = ()) -> int:
        """Update FakeTensors on self.graph. We will try to do the minimum amount of work.

        nodes_to_process: nodes to force update even if their hashes are unchanged.

        Returns the number of nodes updated, including recursive updates on subgraphs."""
        existing_storages: defaultdict[int | None, int] = defaultdict(int)
        for node in self.gm.graph.nodes:
            existing_storages[get_node_storage(node)] += 1

        def should_process_node(node: torch.fx.Node) -> bool:
            if node.op != "call_function":
                return False

            if isinstance(node.target, torch._ops.HigherOrderOperator):
                return True

            return (
                isinstance(node.target, torch._ops.OpOverload)
                or node.target is operator.getitem
                or node.target
                is torch._inductor.fx_passes.reinplace._generalized_scatter
            )

        def node_invokes_subgraph(
            node: torch.fx.Node, *args: Any, **kwargs: Any
        ) -> bool:
            return (
                node.op == "call_function"
                and node.target
                not in (
                    # auto_functionalized doesn't call a subgraph, but the pytree call
                    # below can return subgraphs if the functionalized call itself
                    # invokes a subgraph.
                    torch.ops.higher_order.auto_functionalized,
                    torch.ops.higher_order.auto_functionalized_v2,
                )
                and pytree.tree_any_only(
                    torch.fx.GraphModule,
                    lambda s: s in self.subgraph_updaters,
                    (args, kwargs),
                )
            )

        # Update self.processed_hashes every time.  This allows us to account for
        # situations where a node gets modified, updated, then reverted to its original
        # state.  Without doing this, we wouldn't update downstream nodes, because the
        # reverted node would already be in our cache.
        current_graph_hashes = OrderedSet[_FxNodeHash]()

        nodes_updated: int = 0
        to_process = OrderedSet(id(node) for node in nodes_to_process)
        # Value records whether subgraph outputs have been updated.
        subgraph_updatings: dict[torch.fx.GraphModule, bool] = {}
        subgraph_nodes_to_process: dict[
            torch.fx.GraphModule, OrderedSet[torch.fx.Node]
        ] = {}
        self.refresh_subgraph_updaters()
        for node in self.gm.graph.nodes:
            hash = self.hash_node(node)
            is_valid, args, kwargs = get_fake_args_kwargs(node, self.gm)
            if not is_valid:
                continue
            current_graph_hashes.add(hash)
            node_needs_update = (
                hash not in self.processed_hashes or id(node) in to_process
            )

            # NB: Be very careful about skipping nodes (via continues) here
            # and ask for a careful review when changing this code. The
            # consequence for incorrect FakeTensor metadata is difficult-to-debug
            # silent incorrectness.
            if (
                # Always run updates on nodes that invoke subgraphs
                not (invokes_subgraph := node_invokes_subgraph(node, *args, **kwargs))
                and not node_needs_update
            ):
                continue

            if not should_process_node(node):
                continue

            if invokes_subgraph:
                any_output_updated = False

                for subgraph, subgraph_args in _extract_subgraphs_and_args(
                    node, self.subgraph_updaters, *args, **kwargs
                ):
                    update_subgraph = subgraph not in subgraph_updatings
                    if update_subgraph:
                        subgraph_updatings[subgraph] = False

                    # If the arguments being passed into the subgraph differ from the
                    # existing args the first time we see the subgraph, update the
                    # placeholder nodes in the subgraph.  Any updates past that point
                    # would make subgraph updates inconsistent, so throw errors.
                    if subgraph_args is not None:
                        for p, a in zip(
                            subgraph.graph.find_nodes(op="placeholder"),
                            subgraph_args,
                            strict=True,
                        ):
                            # Do an update iff anything except the storage has changed,
                            # since every invocation of the subgraph will use different
                            # storages.  This implies potentially incorrect arg
                            # aliasing relationships.
                            if not _is_fake_tensor_same(
                                a,
                                p_fake := get_fake(p, subgraph),
                                existing_storages,
                                check_storage=False,
                            ):
                                assert update_subgraph, (
                                    "subgraph args must have consistent values!"
                                )
                                # Check that only dtype or stride has changed.  Other
                                # changes cannot be handled without manual intervention.
                                assert _is_fake_tensor_same(
                                    a,
                                    p_fake,
                                    existing_storages,
                                    check_dtype=False,
                                    check_strides=False,
                                    check_storage=False,
                                ), (
                                    "A subgraph argument other than dtype or striding "
                                    "has been modified; FakeTensorUpdater cannot "
                                    "update this argument!"
                                )

                                p.meta["val"] = a
                                nodes_updated += 1
                                subgraph_nodes_to_process.setdefault(
                                    subgraph, OrderedSet()
                                ).update(p.users)

                    if update_subgraph:
                        _, orig_output_args, _ = get_fake_args_kwargs(
                            subgraph.graph.output_node(), subgraph
                        )
                        nodes_updated += self.subgraph_updaters[
                            subgraph
                        ].incremental_update(
                            subgraph_nodes_to_process.get(subgraph, ())
                        )
                        _, new_output_args, _ = get_fake_args_kwargs(
                            subgraph.graph.output_node(), subgraph
                        )

                        if not _is_fake_tensor_same(
                            new_output_args,
                            orig_output_args,
                            existing_storages,
                        ):
                            subgraph_updatings[subgraph] = True

                    any_output_updated = (
                        any_output_updated or subgraph_updatings[subgraph]
                    )

                # If the outputs of the subgraphs have not changed (and have been
                # previously cached), we can skip updating.  If the outputs of the
                # subgraph have changed, we'll run FakeTensor propagation for every
                # invocation of the subgraph, so that each node gets unique FakeTensor
                # values which maintain appropriate aliasing relationships.
                if (
                    not any_output_updated
                    and not node_needs_update
                    and "val" in node.meta
                ):
                    continue

            with V.fake_mode, enable_python_dispatcher():
                new_fake_tensor = node.target(*args, **kwargs)

            if "val" in node.meta and _is_fake_tensor_same(
                new_fake_tensor,
                node.meta["val"],
                existing_storages,
                node=node,
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

        self.processed_hashes = current_graph_hashes

        return nodes_updated


def get_storage(t: torch.Tensor) -> int:
    return t.untyped_storage()._cdata


def get_node_storage(node: torch.fx.Node) -> int | None:
    if "val" not in node.meta:
        return None
    if not isinstance(node.meta["val"], torch.Tensor):
        return None
    if not torch._C._has_storage(node.meta["val"]):
        return None
    return get_storage(node.meta["val"])


def get_fake(x: Any, gm: torch.fx.GraphModule | None) -> Any:
    """Return a fake tensor from the meta values of an input FX node.  If the input node
    is a get_attr node, we attempt to resolve it as a member of gm."""
    if isinstance(x, torch.fx.Node):
        if "val" in x.meta:
            return x.meta["val"]
        if "example_value" in x.meta:
            return x.meta["example_value"]
        if gm is not None and x.op == "get_attr" and isinstance(x.target, str):
            attr = _get_attr(gm, x.target)
            if pytree.tree_any_only(torch.Tensor, lambda _: True, attr):
                return x
            return attr
        return x
    # If there are no example values, return x
    return x


def get_fake_args_kwargs(
    x: torch.fx.Node, gm: torch.fx.GraphModule | None = None
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


def count_flops_fx(node: torch.fx.Node) -> int | None:
    if not countable_fx(node) or isinstance(node.target, str):
        return None
    with FakeTensorMode(allow_non_fake_inputs=True):
        success, args, kwargs = get_fake_args_kwargs(node)

        if success:
            # flex_attention HOPs have registered formulas, but invoking them
            # here can require tracing-only context, e.g. TransformGetItemToIndex.
            if node.target in (
                torch.ops.higher_order.flex_attention,
                torch.ops.higher_order.flex_attention_backward,
            ):
                flop_formula = flop_registry.get(node.target)
                if flop_formula is not None:
                    return flop_formula(*args, **kwargs, out_val=node.meta.get("val"))

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
