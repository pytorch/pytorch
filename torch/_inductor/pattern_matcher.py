# mypy: allow-untyped-decorators
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
import dataclasses
import functools
import importlib
import inspect
import itertools
import logging
import operator
import os
import re
import textwrap
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import Self, TypeIs

import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import guard_size_oblivious
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.passes.graph_transform_observer import GraphTransformObserver

from .._functorch import config as functorch_config
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensor, FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

Constant = Any
NodeOrConstant = Union[Constant, torch.fx.Node]


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


def _transfer_meta(new_meta: Dict[str, Any], old_meta: Dict[str, Any]) -> None:
    # transfer metadata after pattern matching occurs.
    # skip "val" and "tensor_meta" because this info is too specific; it's unlikely
    # to remain accurate after pattern matching has occurred.
    new_meta.update(
        (k, v) for k, v in old_meta.items() if k in torch.fx.proxy._COPY_META_FIELDS
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
    args: List[Any]
    kwargs: Dict[str, Any]
    nodes: List[torch.fx.Node]
    targets: Dict[_TargetExpr, torch.fx.node.Target]
    ctx: MatchContext
    replacement_graph: Optional[torch.fx.GraphModule]

    def __init__(
        self,
        ctx: MatchContext,
        pattern: PatternExpr,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
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
            for key in set(self.kwargs.keys()) & set(other.kwargs.keys()):
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

    def output_nodes(self) -> List[Optional[torch.fx.Node]]:
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

        with context:
            if trace_fn is None:
                trace_fn = functools.partial(
                    fwd_only, run_functional_passes=run_functional_passes
                )
            replacement = trace_fn(
                replacement_fn, torch.fx.map_arg(args, lambda arg: arg.meta["val"])  # type: ignore[arg-type]
            )
            if len(self.nodes) == 1:
                for n in replacement.graph.nodes:
                    _transfer_meta(new_meta=n.meta, old_meta=self.nodes[0].meta)

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

    outputs: List[Optional[PatternExpr]]
    pattern_to_node: Dict[PatternExpr, Optional[torch.fx.Node]]
    graph: torch.fx.Graph
    exclusive_node_set: List[NodeOrConstant]

    def __init__(
        self,
        outputs: List[Optional[PatternExpr]],
        pattern_to_node: Optional[Dict[PatternExpr, torch.fx.Node]] = None,
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

    def filter_multi_user_patterns(self) -> Dict[PatternExpr, torch.fx.Node]:
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
        self, ctx: MatchContext, searched: Set[torch.fx.Node]
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


class Arg(PatternExpr):
    """
    Capture an arg which will become an input to the handler.  Args are
    passed in depth first order.
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        return Match(ctx, self, args=[node])  # matches anything


class Ignored(PatternExpr):
    """
    Match an arg, but don't pass it to handler
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        return Match(ctx, self)  # matches anything

    def __repr__(self) -> str:
        return "*"

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        return "Ignored()"


class KeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return f"KeywordArg({self.name!r})"

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        return Match(ctx, self, kwargs={self.name: node})  # matches anything

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return super().pattern_eq(other) and self.name == other.name


class ExclusiveKeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return f"ExclusiveKeywordArg({self.name!r})"

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        if node in ctx.exclusive_node_set:
            return FailedMatch("exclusive arg appears twice")

        ctx.exclusive_node_set.append(node)
        return Match(ctx, self, kwargs={self.name: node})  # matches anything

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return super().pattern_eq(other) and self.name == other.name


class _TargetExpr(PatternExpr):
    """
    Base class for filtering match by node.target
    """

    fns: List[FnsType]
    fns_set: Set[FnsType]

    def __init__(
        self, fns: Union[FnsType, Sequence[FnsType]], users: Union[Multiple, int] = 1
    ) -> None:
        super().__init__()
        fns = [fns] if callable(fns) or isinstance(fns, str) else list(fns)
        for fn in fns:
            if isinstance(fn, torch._ops.OpOverloadPacket):
                fns.extend(getattr(fn, overload) for overload in fn.overloads())

        self.fns = fns
        self.fns_set = set(fns)
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
        self, ctx: MatchContext, searched: Set[torch.fx.Node]
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


_SimpleSpec = Tuple[Any, ...]


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
    ) -> Tuple[Sequence[Any], Union[_SimpleSpec, pytree.TreeSpec]]:
        values = (*args, *kwargs.values())
        spec = (len(args), *kwargs.keys())
        return values, spec

    @staticmethod
    def pytree_flatten(
        args: Sequence[Any], kwargs: Mapping[Any, Any]
    ) -> Tuple[Sequence[Any], Union[_SimpleSpec, pytree.TreeSpec]]:
        type_mapping = {immutable_list: tuple, list: tuple, immutable_dict: dict}

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

            normalized_args_and_kwargs = normalize_function(
                node.target, node.args, node.kwargs  # type: ignore[arg-type]
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
        self, ctx: MatchContext, searched: Set[torch.fx.Node]
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


class CallFunction(_TargetArgsExpr):
    """
    Matches a call_function node in the FX graphs: `fns[i](*args, **kwargs)`
    """

    op = "call_function"


class CallMethod(_TargetArgsExpr):
    """
    Matches a call_method node in the FX graphs: `fns[i].method(*args, **kwargs)`
    """

    op = "call_method"


class CallModule(_TargetArgsExpr):
    """
    Matches a call_module node in the FX graphs: `module(*args, **kwargs)`
    """

    op = "call_module"


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


class CallFunctionVarArgs(_TargetExprVarArgs):
    op = "call_function"


class CallMethodVarArgs(_TargetExprVarArgs):
    op = "call_method"


class CallModuleVarArgs(_TargetExprVarArgs):
    op = "call_module"


class ListOf(PatternExpr):
    """
    Matches a repeated pattern
    """

    def __init__(self, pattern: PatternExpr, partial: bool = False) -> None:
        super().__init__()
        assert isinstance(pattern, PatternExpr)
        self.pattern = pattern
        self.partial = partial

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pattern})"

    def _match(self, node: List[torch.fx.Node], ctx: MatchContext) -> MatchResult:  # type: ignore[override]
        if not isinstance(node, (list, tuple)) or len(node) == 0:
            return FailedMatch("non_list")
        m = Match(ctx, self)
        # Propagating patterns with multiple users will ensure we don't revisit
        # the same nodes
        pattern_to_node = ctx.filter_multi_user_patterns()
        matched = False
        for i, child_node in enumerate(node):
            child_ctx = MatchContext(
                ctx.outputs, pattern_to_node, graph=child_node.graph
            )
            child_match = child_ctx.match(self.pattern, child_node)
            pattern_to_node = child_ctx.filter_multi_user_patterns()
            if not is_match(child_match):
                if not self.partial:
                    return FailedMatch("list[{}]: {}", i, child_match)
                continue
            matched = True
            m.extend(child_match.bundle())
        if not matched:
            return FailedMatch("list: no_match")
        return m.bundle()

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and self.pattern.pattern_eq(other.pattern)
            and self.partial == other.partial
        )


class MultiOutputPattern(PatternExpr):
    outputs: List[Optional[PatternExpr]]

    def __init__(self, outputs: Sequence[Optional[PatternExpr]]) -> None:
        super().__init__()
        assert isinstance(outputs[0], _TargetExpr)
        assert all(x is None or isinstance(x, PatternExpr) for x in outputs), outputs
        self.outputs = list(outputs)
        self.op = outputs[0].op

    @property
    def fns(self) -> Union[Callable[..., Any], str, Sequence[Any]]:
        # This cast is checked above in __init__()
        output = typing.cast(_TargetExpr, self.outputs[0])
        return output.fns

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.outputs})"

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        args = [pp.pretty_print(x) for x in self.outputs]
        joiner_str = f",\n{'  '}"
        str_out = f"{self.__class__.__name__}([{joiner_str.join(args)}"
        str_out = f"{str_out}\n])"
        return str_out

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        output = typing.cast(_TargetExpr, self.outputs[0])
        m = ctx.match(output, node)
        if not is_match(m):
            return m

        for pattern in self.outputs[1:]:
            if pattern is None:
                continue
            child_match = self._match_from_anchors(pattern, ctx)
            if not is_match(child_match):
                return child_match
            m.extend(child_match)

        return m

    def _match_from_anchors(
        self, pattern: PatternExpr, ctx: MatchContext
    ) -> MatchResult:
        prior = dict(ctx.pattern_to_node)
        m: MatchResult = FailedMatch("no anchor found")
        for node in pattern.find_anchor_nodes(ctx, set()):
            m = ctx.match(pattern, node)
            if is_match(m):
                return m
            # revert any partial matches
            ctx.pattern_to_node = dict(prior)
        return m

    def match(self, node: torch.fx.Node) -> MatchResult:
        try:
            return MatchContext(self.outputs, graph=node.graph).match(self, node)
        except FailedMatch as e:
            return e

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and len(self.outputs) == len(other.outputs)
            and all(
                a.pattern_eq(b) if isinstance(a, PatternExpr) else a == b
                for a, b in zip(self.outputs, other.outputs)
            )
        )


class RepeatedExpr(PatternExpr):
    """
    Checks for a repeated pattern. Useful for repeated operations after a node such as `split` or `unbind`
    """

    def __init__(self, inner_pattern: _TargetExpr) -> None:
        super().__init__()
        self.inner_pattern = inner_pattern
        self.op = inner_pattern.op

    @property
    def fns(self) -> Sequence[FnsType]:
        return self.inner_pattern.fns

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        m = ctx.match(self.inner_pattern, node)
        if not is_match(m):
            return m
        ctx.pattern_to_node.pop(
            self.inner_pattern,
        )
        # Check all anchor nodes match the pattern
        for anchor_node in self.inner_pattern.find_anchor_nodes(ctx, set()):
            anchor_m = MatchContext([self], graph=node.graph).match(
                self.inner_pattern, anchor_node
            )
            if not is_match(anchor_m):
                return anchor_m
            m.extend(anchor_m)
        return m

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return super().pattern_eq(other) and self.inner_pattern.pattern_eq(
            other.inner_pattern
        )


class PatternPrettyPrinter:
    """
    Serializes Patterns to executable python.
    XXX: currently only used and tested for fuse attention patterns. May not cover
    all patterns.
    """

    def __init__(self) -> None:
        self.namespace = torch.fx.graph._Namespace()
        self.memoized_objs_names: Dict[PatternExpr, str] = {}
        self.memoized_objs_pp: Dict[PatternExpr, str] = {}

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
    def __getitem__(self, k: Tuple[str, torch.fx.node.Target]) -> List[PatternEntry]:
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
class LoweringPatternEntry(PatternEntry):
    handler: Callable[..., Any]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        handler = functools.wraps(self.handler)(functools.partial(self.handler, match))
        with graph.inserting_before(node):
            replacement = graph.call_function(handler, tuple(match.args), match.kwargs)
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
        assert match.nodes[-1] is node
        match.erase_nodes()


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
    normalize_args: Callable[..., List[Any]]

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
                if node.op == "call_function":
                    target = node.target
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    result = graph.call_function(target, args, kwargs)  # type: ignore[arg-type]
                    _transfer_meta(new_meta=result.meta, old_meta=node.meta)
                    if "val" in node.meta and "val" not in result.meta:
                        result.meta["val"] = node.meta["val"]
                        if isinstance(node.meta["val"], torch.Tensor):
                            assert "tensor_meta" in node.meta
                            result.meta["tensor_meta"] = node.meta["tensor_meta"]
                    return result
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
            input_stops: Set[torch.fx.Node],
        ) -> None:
            queue = [node]
            visited = set()

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
            replacement = Replacer(replacement_graph).run(*args)  # type: ignore[arg-type]
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
                            percolate_tags(new, tag_name, old.meta[tag_name], set(args))

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
                    replace(user, new[idx])  # type: ignore[index]
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


def log_trace_failure(search_fn: Callable[..., Any], e: RuntimeError) -> None:
    log.info(
        "Replacement pattern %s failed to apply due to shape mismatch: %s",
        search_fn.__name__,
        e,
    )


def register_replacement(
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    pass_dicts: Union[_PassDictsType, Sequence[_PassDictsType]],
    extra_check: Callable[[Match], bool] = _return_true,
    scalar_workaround: Union[Dict[str, Union[float, int]], None] = None,
    exclusive_arg_names: Sequence[str] = (),
    search_fn_pattern: Union[PatternExpr, None] = None,
) -> bool:
    """
    Create a replacement rule based on example functions that get traced
    to create patterns.  This supports both training and inference when
    run on a joint forward+backward graph.

    Args:
        search_fn: traced to give original pattern
        replace_fn: traced to give replacement graph
        example_inputs: example inputs for initial trace
        trace_fn: fwd_only or joint_fwd_bwd
        pass_dict: dict of passes to register to
        extra_check: additional check to run on match(using real shapes)
    """
    argnames_static = [*inspect.signature(search_fn).parameters.keys()]

    def check_fn(match: Match) -> bool:
        """
        Often shapes get burned into the pattern, so our initial match ran with
        `ignore_types=(int, ...)`.

        Recheck the match with the correct shapes.
        """
        argnames = list(argnames_static)
        for name in argnames:
            if name not in match.kwargs:
                raise RuntimeError(
                    f"Not all inputs to pattern found in match.kwargs. Perhaps one "
                    f"of the inputs is unused? argnames={argnames}, match.kwargs={match.kwargs}"
                )

        args = list(
            torch.fx.map_arg(  # type: ignore[arg-type]
                [match.kwargs[name] for name in argnames], lambda n: n.meta["val"]
            )
        )
        sym_args: List[torch.SymInt] = []
        with torch._dynamo.utils.detect_fake_mode(args):
            for i, grad in enumerate(requires_grad):
                if isinstance(args[i], torch.Tensor):
                    if grad and is_integer_dtype(args[i].dtype):
                        return False

                    args[i] = torch.empty_strided(
                        args[i].size(),
                        args[i].stride(),
                        dtype=args[i].dtype,
                        device=args[i].device,
                        requires_grad=grad,
                    )
                    for v in itertools.chain(args[i].shape, args[i].stride()):
                        if isinstance(v, torch.SymInt) and all(
                            guard_size_oblivious(v != a) for a in sym_args
                        ):
                            sym_args.append(v)

            # If we were given a pre-traced pattern then use that instead of
            # retracing. Note that this means the pattern has to be independent
            # of its args.
            specific_pattern = search_fn_pattern

            if not specific_pattern:
                if sym_args:
                    # AOT Autograd and make fx will dedupe symbolic shape size
                    # accesses of sym ints that appear as inputs
                    # We don't want the sym_size uses to interfere with pattern matching
                    # so we provide them as inputs.
                    # Later, when we actually do the replacement, the symbolic shape
                    # sizes will get re-traced and added to the graph.

                    def search_fn_new(*args_new: Any) -> Any:
                        return search_fn(*args_new[len(args_new) - len(args) :])

                    try:
                        specific_graph = trace_fn(search_fn_new, sym_args + args)
                    except RuntimeError as e:
                        log_trace_failure(search_fn, e)
                        return False

                    # correct argnames in the graph
                    sym_arg_names = []
                    for i, placeholder in zip(
                        range(len(sym_args) + len(args)),
                        specific_graph.graph.nodes,
                    ):
                        if i < len(sym_args):
                            sym_arg_names.append(placeholder.target)
                            continue

                        with specific_graph.graph.inserting_after(placeholder):
                            new_node = specific_graph.graph.placeholder(
                                argnames[i - len(sym_args)]
                            )
                            new_node.target = new_node.name
                            placeholder.replace_all_uses_with(new_node)
                            specific_graph.graph.erase_node(placeholder)

                    argnames = sym_arg_names + argnames
                else:
                    try:
                        specific_graph = trace_fn(search_fn, args)
                    except RuntimeError as e:
                        log_trace_failure(search_fn, e)
                        return False

                specific_pattern = fx_to_pattern(
                    specific_graph,
                    argnames=argnames,
                    exclusive_arg_names=exclusive_arg_names,
                    scalar_workaround=scalar_workaround,
                )

            node = match.output_nodes()[0]
            assert node is not None
            specific_pattern_match = specific_pattern.match(node)

            if is_match(specific_pattern_match) and extra_check(specific_pattern_match):
                # trace the pattern using the shapes from the user program
                match.replacement_graph = trace_fn(replace_fn, args)
                if len(match.nodes) == 1:
                    for n in match.replacement_graph.graph.nodes:
                        _transfer_meta(
                            new_meta=n.meta,
                            old_meta=match.nodes[0].meta,
                        )
                return True
            return False

    def normalize_args(**kwargs: Any) -> List[Any]:
        args = [kwargs.pop(name) for name in argnames_static]
        for i in range(1, len(kwargs) + 1):
            if f"tangents_{i}" not in kwargs:
                break
            args.append(kwargs.pop(f"tangents_{i}"))
        assert not kwargs, f"leftover kwargs: {kwargs!r}"
        return args

    if trace_fn is joint_fwd_bwd:
        # If inference mode is enabled during compilation, assume that we don't
        # want to match on any training graph patterns
        if torch.is_inference_mode_enabled():
            return False

    # TODO: Revisit the functionalize_rng_ops for lowmem dropout
    with functorch_config.patch(functionalize_rng_ops=False):
        requires_grad: List[bool] = [
            isinstance(x, torch.Tensor) and x.requires_grad for x in example_inputs
        ]
        if search_fn_pattern is None:
            pattern = gen_pattern(
                search_fn,
                example_inputs,
                trace_fn,
                scalar_workaround,
                exclusive_arg_names,
            )
        else:
            pattern = search_fn_pattern

        pattern_repr = PatternPrettyPrinter.run(pattern)
        assert pattern_repr not in _seen_patterns
        _seen_patterns.add(pattern_repr)
        pattern = ReplacementPatternEntry(
            pattern=pattern,
            extra_check=check_fn,
            normalize_args=normalize_args,
        )
        pattern.register(pass_dicts)
        return pattern.pattern


_serialized_patterns: Set[str] = set()


def _serialize_pattern(
    unique_name: str,
    search_fn: SearchFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    scalar_workaround: Union[Dict[str, Union[float, int]], None],
) -> PatternExpr:
    def get_file_template() -> str:
        auto_generated_msg = textwrap.dedent(
            """\
            # This is an auto-generated file. Please do not modify it by hand.
            # To re-generate, run:
            # cd ~/pytorch && python torchgen/fuse/gen_patterns.py
            """
        )

        file_template = textwrap.dedent(
            """\
            

            # noqa: F401, E501
            {msg}
            import torch
            import torch._inductor

            aten = torch.ops.aten
            prims = torch.ops.prims

            """
        ).format(msg=auto_generated_msg)

        pattern_matcher_imports = []
        for name in dir(torch._inductor.pattern_matcher):
            attr = getattr(torch._inductor.pattern_matcher, name)
            if isinstance(attr, type) and issubclass(attr, (PatternExpr, _TargetExpr)):
                pattern_matcher_imports.append(name)

        formatted_imports = ",\n   ".join(pattern_matcher_imports)
        formatted_imports = f"from torch._inductor.pattern_matcher import (\n   {formatted_imports},\n)\n"
        return f"{file_template}{formatted_imports}"

    if not SERIALIZED_PATTERN_PATH.is_dir():
        raise RuntimeError(
            f"Could not find serialized patterns directory at {SERIALIZED_PATTERN_PATH}"
        )

    pattern_name = search_fn.__name__

    from torch._functorch import config as functorch_config

    with functorch_config.patch(functionalize_rng_ops=False):
        pattern = gen_pattern(search_fn, example_inputs, trace_fn, scalar_workaround)

    serialized_pattern = PatternPrettyPrinter.run(pattern, output_name=unique_name)
    if pattern_name not in _serialized_patterns:
        write_mode = "w"
        _serialized_patterns.add(pattern_name)
    else:
        write_mode = "a"

    file_template = get_file_template()

    with open(SERIALIZED_PATTERN_PATH / f"{pattern_name}.py", write_mode) as f:
        if write_mode == "w":
            f.write(file_template)
        else:
            f.write("\n\n")
        f.write(serialized_pattern)
        f.write("\n")

    return pattern


SERIALIZED_PATTERN_PATH = Path(__file__).parent / "fx_passes" / "serialized_patterns"

# This is the set of serialized patterns that we've registered.  Used by
# test_serialized_patterns_up_to_date() to ensure the patterns are up
# to date.
_known_precompiled_patterns: List[
    Tuple[
        Any,
        Iterable[Any],
        Callable[[Callable[..., Any], Iterable[Any]], torch.fx.GraphModule],
        Any,
        PatternExpr,
    ]
] = []


def gen_register_replacement(
    unique_name: str,
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    pass_dicts: Union[_PassDictsType, Sequence[_PassDictsType]],
    extra_check: Callable[[Match], bool] = _return_true,
    scalar_workaround: Union[Dict[str, Union[float, int]], None] = None,
    exclusive_arg_names: Sequence[str] = (),
    skip_duplicates: bool = False,
) -> None:
    # Make sure the example_inputs is materialized.
    example_inputs = tuple(example_inputs)

    if "PYTORCH_GEN_PATTERNS" in os.environ:
        pat = _serialize_pattern(
            unique_name, search_fn, example_inputs, trace_fn, scalar_workaround
        )
    else:
        pattern_name = search_fn.__name__
        m = importlib.import_module(
            f"torch._inductor.fx_passes.serialized_patterns.{pattern_name}"
        )
        if not m or not hasattr(m, unique_name):
            log.warning(
                "Precompiled pattern %r not found. Run torchgen/fuse/gen_patterns.py.",
                unique_name,
            )
        pat = getattr(m, unique_name)

    for arg in pytree.tree_iter(example_inputs):
        if isinstance(arg, FakeTensor) and arg.constant is not None:
            # This can be a problem - small fake tensors (e.g. `tensor(2)`) will
            # hold onto their original constant value - and by stashing it here
            # will cause a memory leak if the constant value is on GPU.
            # Since this is just an optimization we can clear it out.
            arg.constant = None

    if PatternPrettyPrinter.run(pat) in _seen_patterns and skip_duplicates:
        return
    _known_precompiled_patterns.append(
        (search_fn, example_inputs, trace_fn, scalar_workaround, pat)
    )
    register_replacement(
        search_fn,
        replace_fn,
        example_inputs,
        trace_fn,
        pass_dicts,
        extra_check,
        scalar_workaround,
        exclusive_arg_names,
        search_fn_pattern=pat,
    )


@functorch_config.patch(functionalize_rng_ops=False)
def gen_pattern(
    search_fn: SearchFn,
    example_inputs: Sequence[Any],
    trace_fn: TraceFn,
    scalar_workaround: Union[Dict[str, Union[float, int]], None] = None,
    exclusive_arg_names: Sequence[str] = (),
) -> PatternExpr:
    argnames = [*inspect.signature(search_fn).parameters.keys()]

    if scalar_workaround is None:
        scalar_workaround = {}
    flat_inputs = []
    input_idx = 0  # Positional arguments index

    for argname in argnames:
        if argname in scalar_workaround:
            flat_inputs.append(scalar_workaround[argname])
        else:
            flat_inputs.append(example_inputs[input_idx])
            input_idx += 1

    search_gm = trace_fn(search_fn, flat_inputs)
    return fx_to_pattern(
        search_gm,
        ignore_types=(int, float, list, torch.device, torch.dtype),
        argnames=argnames,
        scalar_workaround=scalar_workaround,
        exclusive_arg_names=exclusive_arg_names,
    )


def register_lowering_pattern(
    pattern: PatternExpr,
    extra_check: Callable[[Match], bool] = _return_true,
    *,
    pass_dict: _PassDictsType,
    prepend: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Register an aten to inductor IR replacement pattern.  The decorated
    function is saved and then called a lowering time allowing direct
    pattern to inductor IR conversion.
    """

    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
        assert callable(handler)
        LoweringPatternEntry(
            pattern=pattern, extra_check=extra_check, handler=handler
        ).register(pass_dict, prepend=prepend)
        handler._inductor_lowering_function = True  # type: ignore[attr-defined]
        return handler

    return decorator


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
        if _mutation_op_re.search(node.target.__name__):  # type: ignore[union-attr]
            return True
    elif node.op == "call_method":
        if _mutation_op_re.search(node.target):  # type: ignore[union-attr, arg-type]
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


def should_compute_mutation_region_ids(graph: torch.fx.GraphModule) -> bool:
    return "mutation_region_id" not in next(iter(graph.nodes)).meta  # type: ignore[arg-type]


def compute_mutation_region_ids(graph: torch.fx.GraphModule) -> None:
    mutation_region_id = 0
    for nd in graph.nodes:  # type: ignore[union-attr]
        if is_mutation_op(nd):
            mutation_region_id += 1
        nd.meta["mutation_region_id"] = mutation_region_id


class PatternMatcherPass:
    def __init__(
        self,
        pass_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.patterns: DefaultDict[
            Tuple[str, torch.fx.node.Target], List[PatternEntry]
        ] = defaultdict(list)
        self.pass_name = pass_name

    def __getitem__(self, item: Tuple[str, torch.fx.node.Target]) -> List[PatternEntry]:
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
        if should_compute_mutation_region_ids(graph):  # type: ignore[arg-type]
            compute_mutation_region_ids(graph)  # type: ignore[arg-type]
        get_mutation_region_id_partial = functools.partial(
            get_mutation_region_id, graph
        )
        count = 0
        nodes = []
        has_call_module = False
        for op, target in self.patterns:
            if op == "call_module":
                has_call_module = True
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
                        and len(set(map(get_mutation_region_id_partial, m.nodes))) != 1  # type: ignore[possibly-undefined]
                    ):
                        continue
                    if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG") == node.name:
                        log.warning("%s%s %s %s", node, node.args, m, entry.pattern)
                    if is_match(m) and entry.extra_check(m):
                        count += 1
                        entry.apply(m, graph, node)  # type: ignore[arg-type]
                        counters["inductor"]["pattern_matcher_count"] += 1
                        counters["inductor"]["pattern_matcher_nodes"] += len(m.nodes)
        return count

    def clear(self) -> None:
        self.patterns.clear()


def _not_implemented(*args: Any, **kwargs: Any) -> NoReturn:
    raise NotImplementedError


def fx_to_pattern(
    gm: Union[torch.fx.GraphModule, torch.fx.Graph],
    ignore_types: Sequence[Type[Any]] = (),
    argnames: Sequence[str] = (),
    scalar_workaround: Union[Dict[str, Union[float, int]], None] = None,
    exclusive_arg_names: Sequence[str] = (),
) -> PatternExpr:
    """
    Convert an FX graph into a PatternExpr.  This is useful for simple
    patterns that can only match single functions and fixed-length lists.
    """
    # scalar_workaround is a hack to capture dropout_p
    # see https://github.com/pytorch/pytorch/issues/97894
    scalar_workaround = scalar_workaround or {}
    inv_scalar_workaround = {v: k for k, v in scalar_workaround.items()}
    assert len(inv_scalar_workaround) == len(scalar_workaround)

    def process_arg(
        x: T, ignore_types_override: Optional[Sequence[Type[Any]]] = None
    ) -> Union[T, KeywordArg, Ignored]:
        current_ignore_types = (
            ignore_types_override if ignore_types_override is not None else ignore_types
        )
        if isinstance(x, (float, int)) and x in inv_scalar_workaround:
            return KeywordArg(inv_scalar_workaround[x])
        if type(x) in current_ignore_types:
            return Ignored()
        if isinstance(x, list) and all(isinstance(y, Ignored) for y in x) and x:
            return Ignored()
        return x

    argnum = itertools.count()

    class Converter(torch.fx.Interpreter):
        call_method = _not_implemented
        call_module = _not_implemented
        get_attr = _not_implemented

        def placeholder(
            self, target: str, args: Sequence[Any], kwargs: Mapping[str, Any]  # type: ignore[override]
        ) -> Union[ExclusiveKeywordArg, KeywordArg]:
            n = next(argnum)
            if n < len(argnames):
                name = argnames[n]
            elif argnames:
                assert target.startswith("tangent")
                name = target
            else:
                target = re.sub(r"_\d+$", "", target)  # de-mangle arg name
                name = target
            if name in exclusive_arg_names:
                return ExclusiveKeywordArg(name)
            else:
                return KeywordArg(name)

        def call_function(
            self, target: str, args: Sequence[Any], kwargs: Mapping[str, Any]  # type: ignore[override]
        ) -> PatternExpr:
            process_arg_fn = process_arg
            # Indexing is critical for matching getitem nodes, so we can't ignore int args here
            if target == operator.getitem:

                def process_arg_fn_impl(
                    x: T,
                    ignore_types_override: Optional[Sequence[Type[Any]]] = tuple(
                        t for t in ignore_types if t is not int
                    ),
                ) -> Union[T, KeywordArg, Ignored]:
                    return process_arg(x, ignore_types_override)

                process_arg_fn = process_arg_fn_impl

            args, kwargs = pytree.tree_map(process_arg_fn, (args, kwargs))
            if list in ignore_types:
                # Handle a burned in tensor size which are now [Ignored(), Ignored(), ...]
                args = [process_arg_fn(a) for a in args]
                kwargs = {k: process_arg_fn(a) for k, a in kwargs.items()}
            return CallFunction(target, *args, **kwargs)

        def run_node(self, n: torch.fx.Node) -> Any:
            rv = super().run_node(n)
            if n.op == "output" and isinstance(rv, tuple):
                assert len(rv) == len(n.args[0])  # type: ignore[arg-type]
                for r, arg in zip(rv, n.args[0]):  # type: ignore[arg-type]
                    r.users = len(arg.users)
            else:
                rv.users = len(n.users)
            return rv

    pattern = Converter(gm).run()  # type: ignore[arg-type]
    if not isinstance(pattern, PatternExpr):
        return MultiOutputPattern(pytree.tree_leaves(pattern))
    return pattern


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

    from .fx_passes.post_grad import remove_noop_ops

    if run_functional_passes:
        remove_noop_ops(gm.graph)
        gm.graph.eliminate_dead_code()

    gm.recompile()
    return gm


@torch.enable_grad()
def joint_fwd_bwd(fn: Callable[..., Any], args: Sequence[Any]) -> torch.fx.GraphModule:
    """Build a normalized training graph, for use with fx_to_pattern"""
    gm: Optional[torch.fx.GraphModule] = None

    def record_joint_graph(
        joint_graph: torch.fx.GraphModule, inputs: Sequence[Any], **kwargs: Any
    ) -> Tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        nonlocal gm
        assert not gm
        gm = clone_graph(joint_graph)
        return default_partition(joint_graph, inputs, **kwargs)

    with torch._guards.tracing(None):
        aot_function(
            fn,
            lambda g, i: make_boxed_func(g),
            partition_fn=record_joint_graph,
            decompositions=select_decomp_table(),
            keep_inference_input_mutations=True,
            enable_log=False,
        )(*args)
    assert gm

    from .fx_passes.post_grad import remove_noop_ops

    remove_noop_ops(gm.graph)

    from .fx_passes.joint_graph import pointless_view

    matcher_pass = PatternMatcherPass()

    pattern = CallFunction(
        torch.ops.aten.view.default, KeywordArg("arg"), KeywordArg("size")
    )
    GraphPatternEntry(
        pattern=pattern, handler=pointless_view, extra_check=_return_true
    ).register(matcher_pass.patterns)
    matcher_pass.apply(gm.graph)  # type: ignore[arg-type]

    # remove in/out specs
    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def _args(n: torch.fx.Node) -> List[torch.fx.node.Argument]:
    args: List[torch.fx.node.Argument] = []
    torch.fx.map_arg((n.args, n.kwargs), args.append)
    return args


def stable_topological_sort(graph: torch.fx.Graph) -> None:
    # Nodes are in exactly one of these three collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.
    ready = set()

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


def init_once_fakemode(fn: Callable[..., Any]) -> Callable[[], Any]:
    """Wrapper around lazy init functions in fx_passes/"""

    @functools.lru_cache(None)
    @functools.wraps(fn)
    def lazy_init() -> Any:
        counters_ref = counters["inductor"].copy()

        with torch._guards.tracing(None), unset_fake_temporarily(), FakeTensorMode():
            result = fn()

        # clear view matches encountered during tracing
        counters["inductor"] = counters_ref

        return result

    return lazy_init


def config_flag(name: str) -> Callable[[Match], Any]:
    """Function for extra_check to put pass behind a flag"""

    def flag_check(match: Match) -> Any:
        return getattr(config, name)

    return flag_check


def clone_graph(input_graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
    class CopyGraph(Transformer):
        def run_node(self, old_node: torch.fx.Node) -> torch.fx.Node:
            new_node = super().run_node(old_node)
            if isinstance(new_node, torch.fx.Proxy):
                new_node.node.meta.update(old_node.meta)
                new_node.node.name = self.new_graph._graph_namespace.create_name(
                    old_node.name, None
                )
            return new_node

    return CopyGraph(input_graph).transform()


_seen_patterns: Set[str] = set()


def get_arg_value(
    node: torch.fx.Node, arg_number: int, kwarg_name: Optional[str] = None
) -> Any:
    return (
        node.args[arg_number]
        if len(node.args) > arg_number
        else node.kwargs.get(kwarg_name)  # type: ignore[arg-type]
    )


def filter_nodes(nodes: Iterable[torch.fx.Node], fn: Any) -> List[torch.fx.Node]:
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
        return getattr(node.graph.owning_module, node.target).__class__  # type: ignore[arg-type]
    return node.target
