"""
# Inductor Pattern Matcher

The pattern matcher enables search/replace within an FX graph.

The main entrypoint to the pattern matcher is register_replacement(). Given a
search function and a replacement function this will register a replacement with
a pass (such as torch._inductor.fx_passes.joint_graph.patterns).

Internally the pattern matcher represents patterns as a graph (a DAG). Creating
new patterns manually as a graph is cumbersome and error-prone so the standard
way to create patterns (using register_replacement()) is to provide a search
function and a replacement function which is traced and converted into a graph.

Because the search functions are built somewhat generic (they tend to ignore
tensor sizes, for example) register_replacement() allows you to specify an
`extra_check` function which performs additional checks to verify that the
matched pattern fully matches before returning it.

## Precompiled Patterns

New patterns are added using register_replacement(). Patterns added in this way
can have a compile-time overhead because they need to be traced before
use. Patterns can be precompiled and added using gen_register_replacement()
instead. To do this you call gen_register_replacement() instead of
register_replacement(). The arguments are the same except for an additional
unique name which is used as a lookup key.

## Internals

The match DAG is represented by a graph of `PatternExpr` nodes. Each PatternExpr
implements a `_match` method which returns either a `Match` object for a
successful match or a `FailedMatch` object for a failure to match.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import itertools
import logging
import operator
import os
import re
import typing
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator, Mapping, Sequence
from typing import Any, Callable, Optional, Protocol, TypeVar, Union
from typing_extensions import Self, TypeIs

import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.utils import is_view
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph_module import _get_attr
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

Constant = Any
NodeOrConstant = Union[Constant, torch.fx.Node]


@functools.lru_cache(None)
def _warn_complex_not_supported() -> None:
    warnings.warn(
        "Torchinductor does not support code generation for complex operators. Performance may be worse than eager."
    )


# There are some types (CPU) which we accept as input but not as
# output.
def unsupported_input_tensor(
    t: torch.Tensor,
    parent: Optional[torch.fx.Node] = None,
    node: Optional[torch.fx.Node] = None,
) -> bool:
    "Do not support reading or writing to this tensor"
    if t.is_complex():
        # Complex views are supported with IR ComplexView
        if parent and parent.target in (
            torch.ops.aten.view.dtype,
            torch.ops.prims.convert_element_type.default,
        ):
            return False
        _warn_complex_not_supported()
        return True

    if t.is_meta:
        return True

    if t.dtype == torch.float8_e8m0fnu:
        if not node:
            return True

        # allow bitcast, views, memory movement, but not arithmetic
        # TODO: delete once triton adds native support
        return not (
            isinstance(parent.target, torch._ops.OpOverload)  # type: ignore[union-attr]
            and parent.target  # type: ignore[union-attr]
            in (
                aten.view.dtype,
                aten.cat.default,
                aten._scaled_mm.default,
            )
            or (isinstance(node.target, torch._ops.OpOverload) and is_view(node.target))  # type: ignore[arg-type]
        )

    return False


def unsupported_output_tensor(
    t: torch.Tensor,
    parent: Optional[torch.fx.Node] = None,
    node: Optional[torch.fx.Node] = None,
) -> bool:
    "Do not support writing tensor but can read from it"
    if unsupported_input_tensor(t, parent):
        return True
    return t.is_cpu and config.disable_cpp_codegen


def fallback_node_due_to_unsupported_type(
    node: torch.fx.Node, allow_cpu_inputs: bool = True
) -> bool:
    # Custom fallback lowering
    if node.target is aten.view_as_complex.default:
        return False

    # We should be able to remove this special case once `disable_cpp_codegen` is killed.
    if node.target is aten.lift_fresh_copy.default:
        return False

    def check_skip_condition(node, parent, is_output: bool) -> bool:  # type: ignore[no-untyped-def]
        if not isinstance(node, torch.fx.Node):
            return False

        if "val" not in node.meta:
            return False

        for meta in pytree.tree_leaves(node.meta["val"]):
            if not isinstance(meta, torch._subclasses.FakeTensor):
                continue

            if is_output:
                if unsupported_output_tensor(meta, parent, node):
                    return True
            else:
                if unsupported_input_tensor(meta, parent, node):
                    return True

        return False

    # only skip codegen if there is a cpu output, not input
    for arg in pytree.arg_tree_leaves(*node.args, **node.kwargs):
        if check_skip_condition(arg, node, is_output=False):
            return True

    return check_skip_condition(node, node, is_output=True)


class SearchFn(Protocol):
    __name__: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


class ReplaceFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


class TraceFn(Protocol):
    def __call__(
        self, fn: Union[SearchFn, ReplaceFn], *args: Any, **kwargs: Any
    ) -> torch.fx.GraphModule:
        ...


T = TypeVar("T")

# What's a better name for this?
FnsType = Union[torch.fx.node.Target, str]


class Multiple:
    def __init__(self) -> None:
        # Ensure we're really a singleton.
        assert "MULTIPLE" not in globals() or self is MULTIPLE


# Sentinel indicating multiple quantities can be matched
MULTIPLE = Multiple()


def _transfer_meta(
    new_meta: dict[str, Any], old_node: torch.fx.Node, pass_name: str = ""
) -> None:
    from torch.fx.traceback import NodeSource, NodeSourceAction

    # transfer metadata after pattern matching occurs.
    # skip "val" and "tensor_meta" because this info is too specific; it's unlikely
    # to remain accurate after pattern matching has occurred.
    if config.trace.enabled:
        # We handle "from_node" field of the node meta specially to record that the new node comes from the old_node.
        new_from_node = new_meta.get("from_node", []).copy()
        new_from_node.append(NodeSource(old_node, pass_name, NodeSourceAction.REPLACE))
        new_meta.update(
            (k, v)
            for k, v in old_node.meta.items()
            if k in torch.fx.proxy._COPY_META_FIELDS
        )
        new_meta["from_node"] = new_from_node
    else:
        new_meta.update(
            (k, v)
            for k, v in old_node.meta.items()
            if k in torch.fx.proxy._COPY_META_FIELDS
        )


class Match:
    """
    Represents a successfully matched pattern.

    The `Match` object is returned to represent a successfully matched
    pattern. Included in the Match are the pattern that was matched, the graph
    nodes matched, and any args that were used during the matching.

    The args and kwargs are specific to the type of pattern that was matched and
    provide hints about what was matched.
    """

    pattern: PatternExpr
    args: list[Any]
    kwargs: dict[str, Any]
    nodes: list[torch.fx.Node]
    targets: dict[_TargetExpr, torch.fx.node.Target]
    ctx: MatchContext
    replacement_graph: Optional[torch.fx.GraphModule]

    def __init__(
        self,
        ctx: MatchContext,
        pattern: PatternExpr,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.pattern = pattern
        # The input nodes that must be passed in to the result
        self.args = list(args or [])
        self.kwargs = kwargs or {}
        # The nodes matched in this expression
        self.nodes = []
        # Mapping CallFunction to the node.target
        self.targets = {}
        self.ctx = ctx
        self.replacement_graph = None

    @property
    def graph(self) -> torch.fx.Graph:
        return self.ctx.graph

    def extend(self, other: Match) -> None:
        if self.kwargs:
            for key in OrderedSet(self.kwargs.keys()) & OrderedSet(other.kwargs.keys()):
                if self.kwargs[key] != other.kwargs[key]:
                    raise FailedMatch("kwarg mismatch: {}", key)
        self.args.extend(other.args)
        self.nodes.extend(other.nodes)
        self.kwargs.update(other.kwargs)
        self.targets.update(other.targets)

    def bundle(self) -> Match:
        # Wrap args in an extra list
        self.args = [tuple(self.args)] if self.args else []
        return self

    def __repr__(self) -> str:
        return f"Match(..., {self.args}, {self.kwargs})"

    def erase_nodes(self) -> None:
        graph = self.graph
        for n in reversed(self.nodes):
            if not n._erased and not n.users:
                graph.erase_node(n)

    def output_nodes(self) -> list[Optional[torch.fx.Node]]:
        return [
            (self.ctx.pattern_to_node[p] if p is not None else None)
            for p in self.ctx.outputs
        ]

    def output_node(self) -> torch.fx.Node:
        return next(p for p in self.output_nodes() if p)

    def replace_with_graph(
        self, replacement_graph: torch.fx.Graph, args: Sequence[Any]
    ) -> None:
        ReplacementPatternEntry.replace_with_graph(
            self, self.ctx.graph, replacement_graph, args
        )

    def replace_by_example(
        self,
        replacement_fn: ReplaceFn,
        args: Sequence[Any],
        trace_fn: Optional[TraceFn] = None,
        run_functional_passes: bool = True,
    ) -> None:
        """Replace with a graph generated by tracing the replacement_fn.

        Args:
            run_functional_passes (bool). If we should run passes that
                assume functional IR (like DCE, remove_noop_ops), on the
                replacement graph.

        """
        from torch._inductor.virtualized import NullHandler, V

        context = (
            V.fake_mode
            if (not isinstance(V.fake_mode, NullHandler) or (V.fake_mode is None))
            else contextlib.nullcontext()
        )

        def should_propagate_eager_input_vals(nodes: list[torch.fx.Node]) -> bool:
            if len(nodes) != 1:
                return False
            node = nodes[0]
            if "eager_input_vals" not in node.meta:
                return False
            return node.target in OrderedSet(
                [
                    torch.ops.higher_order.triton_kernel_wrapper_functional,
                    torch.ops.higher_order.auto_functionalized,
                    torch.ops.higher_order.auto_functionalized_v2,
                ]
            )

        with context:
            if trace_fn is None:
                trace_fn = functools.partial(
                    fwd_only, run_functional_passes=run_functional_passes
                )

            if should_propagate_eager_input_vals(self.nodes):
                # Our strategy is:
                # 1) trace out the graph with eager_input_vals (which have accurate eager-mode metadata)
                # 2) trace out the graph with vals (which have the accurate Inductor metadata)
                # 3) Propagate the eager_input_vals from the first graph to the second.
                # 4) Use the second graph as the replacement graph.

                # Construct a map of node -> FakeTensor val in eager_input_vals
                node_to_val = {}

                fake_args, fake_kwargs = self.nodes[0].meta["eager_input_vals"]
                fake_kwargs = {**fake_kwargs}
                match_args, match_kwargs = tuple(self.args), self.kwargs

                def record(node: torch.fx.Node, val: Any) -> None:
                    if isinstance(node, torch.fx.Node):
                        node_to_val[node] = val

                torch.utils._pytree.tree_map(
                    record, (match_args, match_kwargs), (fake_args, fake_kwargs)
                )
                # map args to their FakeTensor val in eager_input_vals
                example_vals = torch.fx.map_arg(args, lambda arg: node_to_val[arg])

                # first graph
                graph_with_eager_vals = trace_fn(replacement_fn, example_vals)

                # second graph
                example_vals = torch.fx.map_arg(args, lambda arg: arg.meta["val"])
                replacement = trace_fn(graph_with_eager_vals, example_vals)

                # propagate metadata from first graph to second
                # NB: This assertion might not be true in general, but it is true for
                # the two use cases we have
                # (triton_kernel_wrapper_functional, auto_functionalized)
                assert len(graph_with_eager_vals.graph.nodes) == len(
                    replacement.graph.nodes
                )
                for old_node, new_node in zip(
                    graph_with_eager_vals.graph.nodes, replacement.graph.nodes
                ):
                    if "eager_input_vals" in old_node.meta:
                        new_node.meta["eager_input_vals"] = old_node.meta[
                            "eager_input_vals"
                        ]

            else:
                example_vals = torch.fx.map_arg(args, lambda arg: arg.meta["val"])
                replacement = trace_fn(replacement_fn, example_vals)
            if len(self.nodes) == 1:
                for n in replacement.graph.nodes:
                    _transfer_meta(
                        new_meta=n.meta,
                        old_node=self.nodes[0],
                        pass_name="replace_by_example",
                    )

            ReplacementPatternEntry.replace_with_graph(
                self,
                self.ctx.graph,
                replacement,
                args,
            )


class FailedMatch(RuntimeError):
    """
    Represents a unsuccessful match.

    The `FailedMatch` object is returned to represent a failure to match a
    pattern.
    """

    format_string: str

    def __init__(self, format_string: str, *args: Any, **kwargs: Any) -> None:
        self.format_string = format_string
        # We want to construct error messages lazily instead of eagerly, as
        # constructing them eagerly can significantly worsen compile times.
        if len(format_string) > 200:
            raise RuntimeError(
                f"Format string too long - use lazy construction of strings instead. Format string is\n {format_string}"
            )
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        return self.format_string.format(*self.args, **self.kwargs)

    def __bool__(self) -> bool:
        return False


MatchResult = Union[Match, FailedMatch]


def is_match(m: MatchResult) -> TypeIs[Match]:
    """
    TypeIs cannot act on `self`. Thus this function exists to let mypy
    recognize FailedMatch.__bool__ as a TypeIs.
    """
    return bool(m)


class MatchContext:
    """
    Internal state needed while running PatternExpr._match().
    """

    outputs: list[Optional[PatternExpr]]
    pattern_to_node: dict[PatternExpr, Optional[torch.fx.Node]]
    graph: torch.fx.Graph
    exclusive_node_set: list[NodeOrConstant]

    def __init__(
        self,
        outputs: list[Optional[PatternExpr]],
        pattern_to_node: Optional[dict[PatternExpr, torch.fx.Node]] = None,
        *,
        graph: torch.fx.Graph,
    ) -> None:
        self.outputs = outputs
        self.pattern_to_node = {} if pattern_to_node is None else dict(pattern_to_node)
        self.graph = graph
        self.exclusive_node_set = []

    def match(self, pattern: PatternExpr, node: NodeOrConstant) -> MatchResult:
        """wrapper to check reused nodes in patterns"""
        if pattern in self.pattern_to_node:
            if self.pattern_to_node[pattern] == node:
                return Match(self, pattern)  # already checked this node
            else:
                return FailedMatch("repeated pattern differs")
        m = pattern._match(node, self)
        assert pattern not in self.pattern_to_node
        self.pattern_to_node[pattern] = node if m else None
        return m

    def filter_multi_user_patterns(self) -> dict[PatternExpr, torch.fx.Node]:
        return {
            pattern: node
            for pattern, node in self.pattern_to_node.items()
            if pattern.has_multiple_users() and node is not None
        }


class PatternExpr(ABC):
    """
    Base class for types of patterns.
    """

    @abstractmethod
    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        ...

    def match(self, node: torch.fx.Node) -> MatchResult:
        try:
            return MatchContext([self], graph=node.graph).match(self, node)
        except FailedMatch as e:
            return e

    def has_multiple_users(self) -> bool:
        return False

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[Optional[torch.fx.Node], None, None]:
        if self in ctx.pattern_to_node:
            yield ctx.pattern_to_node[self]

    def pattern_eq(self, other: Any) -> bool:
        """
        Compare two `PatternExpr`s and return true if they are the
        same. Note this is NOT matching a pattern - it is comparing the pattern
        structures (for debugging).
        """
        return isinstance(other, self.__class__)


class _TargetExpr(PatternExpr):
    """
    Base class for filtering match by node.target
    """

    fns: list[FnsType]
    fns_set: OrderedSet[FnsType]

    def __init__(
        self, fns: Union[FnsType, Sequence[FnsType]], users: Union[Multiple, int] = 1
    ) -> None:
        super().__init__()
        fns = [fns] if callable(fns) or isinstance(fns, str) else list(fns)
        for fn in fns:
            if isinstance(fn, torch._ops.OpOverloadPacket):
                fns.extend(getattr(fn, overload) for overload in fn.overloads())

        self.fns = fns
        self.fns_set = OrderedSet(fns)
        self.users = users

    @property
    @abstractmethod
    def op(self) -> str:
        ...

    def fns_repr(self) -> str:
        first_repr = self.fns[0]
        if not isinstance(first_repr, str):
            first_repr = first_repr.__name__

        if len(self.fns) > 1:
            return f"[{first_repr}, ...]"
        elif self.fns[0] is getattr(torch, first_repr, None):
            return f"torch.{first_repr}"
        elif self.fns[0] is getattr(operator, first_repr, None):
            return f"operator.{first_repr}"
        elif isinstance(self.fns[0], torch._ops.OpOverload):
            return str(self.fns[0])
        else:
            return first_repr

    def __repr__(self) -> str:
        if self.users is MULTIPLE:
            comma_users = ", MULTIPLE"
        elif self.users != 1:
            comma_users = f", {self.users})"
        else:
            comma_users = ""
        return f"{self.__class__.__name__}({self.fns_repr()}{comma_users})"

    def has_multiple_users(self) -> bool:
        return isinstance(self.users, Multiple) or self.users > 1

    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[Optional[torch.fx.Node], None, None]:
        raise NotImplementedError

    def _match_fns(self, node: torch.fx.Node) -> bool:
        return (
            isinstance(node, torch.fx.Node)
            and node.op == self.op
            and extract_target(node) in self.fns_set
        )

    def _match_users(self, node: torch.fx.Node, ctx: MatchContext) -> bool:
        return (
            self in ctx.outputs
            or self.users is MULTIPLE
            or len(node.users) == self.users
        )

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and self.op == other.op
            and self.fns == other.fns
            and self.users == other.users
        )


_SimpleSpec = tuple[Any, ...]


class _TargetArgsExpr(_TargetExpr):
    """
    Base class for filtering match by node.{target,args,kwargs}
    """

    def __init__(
        self,
        fns: Union[torch.fx.node.Target, str, Sequence[Any]],
        *args: Any,
        _users: Union[int, Multiple] = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(fns, _users)
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        if any(
            isinstance(x, (dict, list, tuple))
            for x in itertools.chain(args, kwargs.values())
        ):
            self.flatten = self.pytree_flatten
        else:
            self.flatten = self.simple_flatten
        self.flat_args_kwargs = self.flatten(self.args, self.kwargs)

    @staticmethod
    def simple_flatten(
        args: Sequence[Any], kwargs: Mapping[Any, Any]
    ) -> tuple[Sequence[Any], Union[_SimpleSpec, pytree.TreeSpec]]:
        values = (*args, *kwargs.values())
        spec = (len(args), *kwargs.keys())
        return values, spec

    @staticmethod
    def pytree_flatten(
        args: Sequence[Any], kwargs: Mapping[Any, Any]
    ) -> tuple[Sequence[Any], Union[_SimpleSpec, pytree.TreeSpec]]:
        type_mapping: dict[type, type] = {
            immutable_list: tuple,
            list: tuple,
            immutable_dict: dict,
        }

        def convert_type(x: Any) -> Any:
            cls = type(x)
            convert_fn = type_mapping.get(cls)
            if convert_fn is not None:
                return pytree.tree_map(
                    convert_type,
                    convert_fn(x),
                    is_leaf=lambda x: type(x) in type_mapping,
                )
            return x

        normalized_args_tree = pytree.tree_map(
            convert_type,
            (args, kwargs),
            is_leaf=lambda x: type(x) in type_mapping,
        )
        flat, spec = pytree.tree_flatten(normalized_args_tree)
        return flat, spec

    def __repr__(self) -> str:
        args = [
            self.fns_repr(),
            *map(repr, self.args),
            *[f"{k}={v}" for k, v in self.kwargs.items()],
        ]
        if self.users is MULTIPLE:
            args.append("_users=MULTIPLE")
        elif self.users != 1:
            args.append(f"_users={self.users}")
        return f"{self.__class__.__name__}({', '.join(args)})"

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        args = [
            self.fns_repr(),
            *(pp.pretty_print(x) for x in self.args),
            *[f"{k}={pp.pretty_print(v)}" for k, v in self.kwargs.items()],
        ]
        if self.users is MULTIPLE:
            args.append("_users=MULTIPLE")
        elif self.users != 1:
            args.append(f"_users={self.users}")

        joiner_str = ", "
        return f"{self.__class__.__name__}({joiner_str.join(args)})"

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        if not self._match_fns(node) or len(node.args) != len(self.args):
            return FailedMatch("function_mismatch: node={}, pattern={}", node, self)

        if not self._match_users(node, ctx):
            return FailedMatch("multiple_users {}", self)

        _args = node.args
        _kwargs = node.kwargs
        if len(_kwargs) < len(self.kwargs):
            from torch.fx.operator_schemas import normalize_function

            assert callable(node.target)
            normalized_args_and_kwargs = normalize_function(
                node.target, node.args, node.kwargs
            )

            if normalized_args_and_kwargs is None:
                return FailedMatch("function_mismatch: node={}, pattern={}", node, self)
            else:
                _args, _kwargs = normalized_args_and_kwargs
                if len(_args) == len(self.args) and len(_kwargs) >= len(self.kwargs):
                    _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}
                else:
                    return FailedMatch(
                        "function_mismatch: node={}, pattern={}", node, self
                    )
        else:
            _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}

        node_items, node_spec = self.flatten(_args, _kwargs)
        self_items, self_spec = self.flat_args_kwargs
        if node_spec != self_spec:
            return FailedMatch("args_structure {} {}", node_spec, self_spec)
        assert len(node_items) == len(self_items)

        m = Match(ctx, self)
        for i, pattern, child_node in zip(itertools.count(), self_items, node_items):
            if isinstance(pattern, PatternExpr):
                child_match = ctx.match(pattern, child_node)
                if not is_match(child_match):
                    return child_match
                m.extend(child_match)
            elif isinstance(child_node, torch.fx.Node) or child_node != pattern:
                return FailedMatch(
                    "constant_args: {} {!r}!={pattern!r}", node, child_node
                )
        m.nodes.append(node)
        m.targets[self] = node.target
        return m

    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[Optional[torch.fx.Node], None, None]:
        """
        This is used when we are matching a pattern with multiple outputs.
        There is a partial match (stored in ctx) and we want to walk
        this pattern to find a connection to an already-matched node.

        Yields candidate nodes that `self._match` might like.
        """
        if self in ctx.pattern_to_node:
            yield ctx.pattern_to_node[self]
            return

        for pattern in self.flat_args_kwargs[0]:
            if isinstance(pattern, PatternExpr):
                for other_node in pattern.find_anchor_nodes(ctx, searched):
                    if not isinstance(other_node, torch.fx.Node):
                        continue
                    for node in other_node.users:
                        if node not in searched:
                            if self._match_fns(node):
                                yield node
                                searched.add(node)

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and self.flat_args_kwargs[1] == other.flat_args_kwargs[1]
            and all(
                a.pattern_eq(b) if isinstance(a, PatternExpr) else a == b
                for a, b in zip(self.flat_args_kwargs[0], other.flat_args_kwargs[0])
            )
        )


class _TargetExprVarArgs(_TargetExpr):
    """
    Matches a call_function node with any arguments which are passed into the pattern
    """

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        if not self._match_fns(node):
            return FailedMatch("function_mismatch")

        if not self._match_users(node, ctx):
            return FailedMatch("multiple_users")

        m = Match(ctx, self)
        m.nodes.append(node)
        m.targets[self] = node.target
        m.args.extend(node.args)
        m.kwargs.update(node.kwargs)
        return m


class Placeholder(_TargetExprVarArgs):
    op = "placeholder"


class PatternPrettyPrinter:
    """
    Serializes Patterns to executable python.
    XXX: currently only used and tested for fuse attention patterns. May not cover
    all patterns.
    """

    def __init__(self) -> None:
        self.namespace = torch.fx.graph._Namespace()
        self.memoized_objs_names: dict[PatternExpr, str] = {}
        self.memoized_objs_pp: dict[PatternExpr, str] = {}

    @staticmethod
    @functools.lru_cache(None)
    def run(obj: PatternExpr, output_name: str = "output") -> str:
        """
        Serializes obj to python code with obj written out to `output_name`
        """

        pp = PatternPrettyPrinter()
        assert hasattr(obj, "pretty_print")
        out_str = obj.pretty_print(pp=pp)

        output = [
            f"{pp.memoized_objs_names[key]} = {pp.memoized_objs_pp[key]}"
            for key in pp.memoized_objs_names
        ]

        output.append(f"{output_name} = {out_str}")

        return "\n".join(output)

    def pretty_print(self, obj: Any) -> str:
        if isinstance(obj, _TargetArgsExpr):
            if memoized_name := self.memoized_objs_names.get(obj):
                return memoized_name
            else:
                return self.memoize(obj)
        if hasattr(obj, "pretty_print"):
            return obj.pretty_print(self)

        return repr(obj)

    def memoize(self, obj: _TargetArgsExpr) -> str:
        obj_str = obj.pretty_print(self)
        obj_name = obj.fns_repr()
        for prefix in ("aten.", "torch.", "prims."):
            obj_name = obj_name.replace(prefix, "")

        tmp_name = self.namespace.create_name(obj_name, None)
        self.memoized_objs_names[obj] = tmp_name
        self.memoized_objs_pp[obj] = obj_str
        return tmp_name


class _PassDictsType(Protocol):
    def __getitem__(self, k: tuple[str, torch.fx.node.Target]) -> list[PatternEntry]:
        ...


@dataclasses.dataclass
class PatternEntry:
    pattern: PatternExpr
    extra_check: Callable[[Match], bool]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        raise NotImplementedError

    def register(
        self,
        pass_dicts: Union[_PassDictsType, Sequence[_PassDictsType]],
        target: Union[torch.fx.node.Target, None] = None,
        prepend: bool = False,
    ) -> None:
        if target is None:
            assert hasattr(self.pattern, "fns")
            for fn in self.pattern.fns:
                self.register(pass_dicts, fn, prepend=prepend)
        elif isinstance(pass_dicts, (dict, PatternMatcherPass)):
            assert hasattr(self.pattern, "op")
            if prepend:
                pass_dicts[(self.pattern.op, target)].insert(0, self)
            else:
                pass_dicts[(self.pattern.op, target)].append(self)
        else:
            pass_dicts = typing.cast(Sequence[_PassDictsType], pass_dicts)
            for x in pass_dicts:
                self.register(x, target, prepend=prepend)


@dataclasses.dataclass
class GraphPatternEntry(PatternEntry):
    """
    A pattern that runs a function on the FX graph
    """

    handler: Callable[..., Any]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        with graph.inserting_before(node):
            self.handler(match, *match.args, **match.kwargs)


@dataclasses.dataclass
class ReplacementPatternEntry(PatternEntry):
    normalize_args: Callable[..., list[Any]]

    @staticmethod
    def replace_with_graph(
        match: Match,
        graph: torch.fx.Graph,
        replacement_graph: Union[torch.fx.Graph, torch.fx.GraphModule],
        args: Sequence[torch.fx.Node],
    ) -> None:
        class Replacer(torch.fx.Interpreter):
            call_method = None  # type: ignore[assignment]
            call_module = None  # type: ignore[assignment]
            get_attr = None  # type: ignore[assignment]

            def run_node(self, node: torch.fx.Node) -> Any:
                if node.op in ("placeholder", "output"):
                    return super().run_node(node)
                target = node.target
                args, kwargs = self.fetch_args_kwargs_from_env(node)
                if node.op == "call_function":
                    assert callable(target)
                    result = graph.call_function(target, args, kwargs)
                    _transfer_meta(
                        new_meta=result.meta,
                        old_node=node,
                        pass_name="Interpreter_Replacer",
                    )
                    # This function copy-pastes the replacement graph into
                    # the graph. If the replacement graph had any eager_input_vals,
                    # or val/tensor_meta, we propagate those over.
                    if "eager_input_vals" in node.meta:
                        result.meta["eager_input_vals"] = node.meta["eager_input_vals"]
                    if "val" in node.meta and "val" not in result.meta:
                        result.meta["val"] = node.meta["val"]
                        if isinstance(node.meta["val"], torch.Tensor):
                            assert "tensor_meta" in node.meta
                            result.meta["tensor_meta"] = node.meta["tensor_meta"]
                    return result
                if node.op == "get_attr":
                    # If the replacement graph contains a HOP, the subgraphs of the HOP are "get_attr" nodes.
                    # We need to fetch the subgraph of the HOP then register the subgraph to the replaced graph's root.
                    from torch._higher_order_ops.utils import (
                        unique_graph_name_with_root,
                    )

                    sub_gm = super().get_attr(target, args, kwargs)
                    if not isinstance(sub_gm, torch.fx.GraphModule):
                        raise NotImplementedError(
                            f"NYI: replacement_graph.{target} is not a graph module. Got {sub_gm}."
                        )

                    assert graph.owning_module is not None
                    _, graph_name = unique_graph_name_with_root(
                        graph.owning_module, str(target)
                    )
                    graph.owning_module.register_module(graph_name, sub_gm)
                    return graph.get_attr(graph_name)

                raise NotImplementedError(f"unhandled {node}")

        output_nodes = match.output_nodes()

        if len(output_nodes) == 1:
            last_node = output_nodes[0]
        else:
            assert output_nodes[0]
            nodes = list(output_nodes[0].graph.nodes)
            indices = [
                (nodes.index(n), n)
                for n in output_nodes
                if isinstance(n, torch.fx.Node)
            ]
            last_node = min(indices, key=operator.itemgetter(0))[1]

        def percolate_tags(
            node: torch.fx.Node,
            tag_name: str,
            tag_value: str,
            input_stops: OrderedSet[torch.fx.Node],
        ) -> None:
            queue = [node]
            visited = OrderedSet[torch.fx.Node]()

            while queue:
                arg = queue.pop()
                if (
                    arg not in visited
                    and arg not in input_stops
                    and hasattr(arg, "meta")
                ):
                    visited.add(arg)
                    arg.meta[tag_name] = tag_value
                    queue.extend(arg.all_input_nodes)

        with graph.inserting_before(last_node):
            assert isinstance(replacement_graph, torch.fx.GraphModule)
            replacement = Replacer(replacement_graph).run(*args)
            if isinstance(replacement, torch.fx.Node):
                replacement = [replacement]

            def maybe_getitem(node: torch.fx.Node) -> Any:
                if node.op != "call_function":
                    return None
                if node.target != operator.getitem:
                    return None
                assert len(node.args) == 2
                return node.args[1]

            def replace(
                old: Union[torch.fx.Node, None],
                new: Union[torch.fx.Node, Sequence[torch.fx.Node], None],
            ) -> None:
                if old is None:
                    assert new is None
                    return
                assert isinstance(old, torch.fx.Node)
                if new is None:
                    old.replace_all_uses_with(None)  # type: ignore[arg-type]
                    graph.erase_node(old)
                    return
                if isinstance(new, torch.fx.Node):
                    if "val" not in new.meta:
                        new.meta.update(old.meta)

                    # Preserve the recompute tags in the replacement graph. We
                    # look at the recompute tags of the original output node to
                    # propagate the tag from the output all the way to the input
                    # args (named as args in the replace_with_graph).
                    # Note that this is best effort. Since patterns are from
                    # many to many, there is no easy way to correctly map the
                    # recomputable tags. It is possible in some scenarios that we
                    # incorrectly tag some nodes as recomputables.
                    for tag_name in ["recompute", "ac_graph_id"]:
                        if tag_name in old.meta:
                            percolate_tags(
                                new, tag_name, old.meta[tag_name], OrderedSet(args)
                            )

                    old.replace_all_uses_with(new)
                    graph.erase_node(old)
                    return

                # `new` is not a node: it's a list of nodes.
                #
                # This happens when we want to replace a node that has a single
                # packed return with multiple unpacked returns. We need to do
                # some graph surgery here.
                #
                # Example:
                #   def original_graph(x):
                #      a = op(x)
                #      b = a[0]
                #      c = a[1]
                #      ...
                #
                # Assume that we want to replace op(x) with the graph
                #   def new_op(x):
                #      w = x + 1
                #      z = x + 2
                #      return (w, z)
                #
                # We need to replace `op` with the contents of `new_op`,
                # and then rewrite a[0] to be w and a[1] to be z, as so:
                #   def new_graph(x):
                #     w = x + 1
                #     z = x + 2
                #     b = w
                #     c = z
                #     ...
                old_uses = list(old.users.keys())
                for user in old_uses:
                    idx = maybe_getitem(user)
                    if idx is None:
                        raise AssertionError("can't handle")
                    replace(user, new[idx])
                graph.erase_node(old)

            if len(output_nodes) == len(replacement):
                for old, new in zip(output_nodes, replacement):
                    replace(old, new)
            else:
                assert len(output_nodes) == 1
                replace(output_nodes[0], replacement)

        match.erase_nodes()

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        assert match.replacement_graph is not None
        self.replace_with_graph(
            match,
            graph,
            match.replacement_graph,
            self.normalize_args(*match.args, **match.kwargs),
        )


def _return_true(match: Match) -> bool:
    return True


def register_graph_pattern(
    pattern: PatternExpr,
    extra_check: Callable[[Match], bool] = _return_true,
    *,
    pass_dict: _PassDictsType,
    prepend: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Register a pattern that runs a function on the FX graph, allowing
    custom transformation code.
    """

    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
        assert callable(handler)
        GraphPatternEntry(
            pattern=pattern, extra_check=extra_check, handler=handler
        ).register(pass_dict, prepend=prepend)
        return handler

    return decorator


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


class PatternMatcherPass:
    def __init__(
        self,
        pass_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.patterns: defaultdict[
            tuple[str, torch.fx.node.Target], list[PatternEntry]
        ] = defaultdict(list)
        self.pass_name = pass_name

        # For a particular generated pattern repr, store all of the str representations
        # of the graph used to generate them. Because we ignore certain patterns
        # in searching, but not in matching, use the graph to distinguish if two equivalent
        # searches are actually different.
        self.seen_patterns: dict[str, list[Optional[str]]] = defaultdict(list)

    def __getitem__(self, item: tuple[str, torch.fx.node.Target]) -> list[PatternEntry]:
        return self.patterns[item]

    def apply(self, gm: Union[torch.fx.GraphModule, torch.fx.Graph]) -> int:
        if not self.patterns:
            return 0
        if isinstance(gm, torch.fx.GraphModule):
            graph = gm.graph
        elif isinstance(gm, torch.fx.Graph):
            graph = gm
            gm = graph.owning_module
        else:
            raise RuntimeError(
                f"The input to PatternMatcherPass must be a GraphModule or a Graph, but got {type(gm)}"
            )
        if should_compute_mutation_region_ids(graph):
            compute_mutation_region_ids(graph)
        get_mutation_region_id_partial = functools.partial(
            get_mutation_region_id, graph
        )
        count = 0
        nodes = []
        has_call_module = False
        for op, target in list(self.patterns.keys()):
            if op == "call_module":
                has_call_module = True
            elif op == "placeholder" and target == "*":
                input_nodes = [node for node in graph.nodes if node.op == "placeholder"]
                for input_node in input_nodes:
                    new_target = input_node.target
                    nodes.append(graph.find_nodes(op=op, target=new_target, sort=False))
                    self.patterns[(op, new_target)] = copy.deepcopy(
                        self.patterns[op, "*"]
                    )
                    entries = self.patterns[(op, new_target)]
                    for entry in entries:
                        users = entry.pattern.users  # type: ignore[attr-defined]
                        entry.pattern = Placeholder(new_target, users=users)
                self.patterns.pop((op, "*"))
            else:
                nodes.append(graph.find_nodes(op=op, target=target, sort=False))
        if has_call_module:
            nodes.append(graph.find_nodes(op="call_module", sort=False))
        pass_name = self.pass_name if self.pass_name is not None else "pattern_matcher"
        assert isinstance(gm, torch.fx.GraphModule)
        with GraphTransformObserver(gm, pass_name):
            for node in sorted(itertools.chain.from_iterable(nodes), reverse=True):
                target = extract_target(node)
                if node.op == "call_module":
                    if (node.op, target) not in self.patterns:
                        continue

                # conservatively not applying pattern for cpu input,
                # since some of the patterns induce codegen and split nodes.
                # Note: we will only skip cpu compute if disable_cpp_codegen=True
                if fallback_node_due_to_unsupported_type(node, allow_cpu_inputs=False):
                    continue

                for entry in self.patterns[(node.op, target)]:
                    if node._erased:
                        break
                    m = entry.pattern.match(node)
                    # pattern match crosses mutation barrier - discard
                    if (
                        is_match(m)
                        and len(
                            OrderedSet(map(get_mutation_region_id_partial, m.nodes))
                        )
                        != 1
                    ):
                        continue
                    if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG") == node.name:
                        log.warning("%s%s %s %s", node, node.args, m, entry.pattern)
                    if is_match(m) and entry.extra_check(m):
                        count += 1
                        entry.apply(m, graph, node)
                        counters["inductor"]["pattern_matcher_count"] += 1
                        counters["inductor"]["pattern_matcher_nodes"] += len(m.nodes)
        return count

    def clear(self) -> None:
        self.patterns.clear()


@torch.no_grad()
def fwd_only(
    fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    run_functional_passes: bool = True,
    get_decomp_fn: Optional[Callable[..., Any]] = None,
) -> torch.fx.GraphModule:
    """Build a normalized inference graph, for use with fx_to_pattern"""
    # TODO - look into using aot autograd, asserting no mutating ops here
    with enable_python_dispatcher():
        decompositions = (
            get_decomp_fn() if get_decomp_fn is not None else select_decomp_table()
        )
        gm = make_fx(fn, decompositions, tracing_mode="real")(*args)

    from torch._inductor.fx_passes.post_grad import remove_noop_ops

    if run_functional_passes:
        remove_noop_ops(gm.graph)
        gm.graph.eliminate_dead_code()

    gm.recompile()
    return gm


def extract_target(node: torch.fx.Node) -> torch.fx.node.Target:
    """For call_function and call_method, we directly use the target function;
    For call_module, the target is string, and we treat the module class
     as a function.
    """
    if node.op == "call_module":
        assert isinstance(node.target, str)
        return _get_attr(node.graph.owning_module, node.target).__class__
    return node.target
