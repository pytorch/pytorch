from __future__ import annotations

import dataclasses
import functools
import inspect
import itertools
import logging
import operator
import os
import re
from collections import defaultdict
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    NoReturn,
    Optional,
    Set,
    Union,
)

from typing_extensions import TypeGuard

import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.experimental.symbolic_shapes import guard_size_oblivious
from torch.fx.immutable_collections import immutable_dict, immutable_list

from .._functorch import config as functorch_config
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

Constant = Any
NodeOrConstant = Union[Constant, torch.fx.Node]


class Multiple:
    pass


# Sentinel indicating multiple quantities can be matched
MULTIPLE = Multiple()


class Match:
    """
    Represents a successfully matched pattern.
    """

    def __init__(self, pattern: PatternExpr, args=None, kwargs=None):
        super().__init__()
        self.pattern = pattern
        # The input nodes that must be passed in to the result
        self.args = args or []
        self.kwargs = kwargs or {}
        # The nodes matched in this expression
        self.nodes: List[torch.fx.Node] = []
        # Mapping CallFunction to the node.target
        self.targets: Dict[_TargetExpr, torch.fx.node.Target] = {}
        self.ctx: Optional[MatchContext] = None
        self.replacement_graph: Optional[torch.fx.Graph] = None

    @property
    def graph(self) -> torch.fx.Graph:
        assert self.ctx
        return self.ctx.graph

    def extend(self, other: Match):
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

    def __repr__(self):
        return f"Match(..., {self.args}, {self.kwargs})"

    def erase_nodes(self, graph: torch.fx.Graph):
        for n in reversed(self.nodes):
            if not n._erased:
                graph.erase_node(n)

    def output_nodes(self) -> List[Optional[torch.fx.Node]]:
        assert self.ctx
        return [
            (self.ctx.pattern_to_node[p] if p is not None else None)
            for p in self.ctx.outputs
        ]

    def output_node(self) -> torch.fx.Node:
        return next(p for p in self.output_nodes() if p)

    def replace_with_graph(self, replacement_graph, args):
        assert self.ctx
        ReplacementPatternEntry.replace_with_graph(
            self, self.ctx.graph, replacement_graph, args
        )

    def replace_by_example(self, replacement_fn, args, trace_fn=None, run_dce=True):
        assert self.ctx
        if trace_fn is None:
            trace_fn = functools.partial(fwd_only, run_dce=run_dce)
        replacement = trace_fn(
            replacement_fn, torch.fx.map_arg(args, lambda arg: arg.meta["val"])
        )
        ReplacementPatternEntry.replace_with_graph(
            self,
            self.ctx.graph,
            replacement,
            args,
        )


class FailedMatch(RuntimeError):
    def __init__(self, format_string, *args, **kwargs):
        self.format_string = format_string
        # We want to construct error messages lazily instead of eagerly, as
        # constructing them eagerly can significantly worsen compile times.
        if len(format_string) > 200:
            raise RuntimeError(
                f"Format string too long - use lazy construction of strings instead. Format string is\n {format_string}"
            )
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.format_string.format(*self.args, **self.kwargs)

    def __bool__(self):
        return False


def is_match(m: Union[Match, FailedMatch]) -> TypeGuard[Match]:
    """
    TypeGuards cannot act on `self`. Thus this function exists to let mypy
    recognize FailedMatch.__bool__ as a TypeGuard.
    """
    return bool(m)


class MatchContext:
    """
    State needed while running PatternExpr._match().
    """

    def __init__(
        self,
        outputs: List[Optional[PatternExpr]],
        pattern_to_node: Optional[Dict[PatternExpr, Node]] = None,
        *,
        graph: torch.fx.Graph,
    ):
        self.outputs = outputs
        self.pattern_to_node = {} if pattern_to_node is None else pattern_to_node
        self.graph = graph
        self.exclusive_node_set: List[NodeOrConstant] = []

    def match(self, pattern, node):
        """wrapper to check reused nodes in patterns"""
        if pattern in self.pattern_to_node:
            if self.pattern_to_node[pattern] == node:
                return Match(pattern)  # already checked this node
            else:
                return FailedMatch("repeated pattern differs")
        m = pattern._match(node, self)
        assert pattern not in self.pattern_to_node
        self.pattern_to_node[pattern] = node if m else None
        m.ctx = self
        return m

    def filter_multi_user_patterns(self):
        return {
            pattern: node
            for pattern, node in self.pattern_to_node.items()
            if pattern.has_multiple_users() and node is not None
        }


class PatternExpr:
    """
    Base class for types of patterns
    """

    def _match(
        self, node: torch.fx.Node, ctx: MatchContext
    ) -> Union[Match, FailedMatch]:
        raise NotImplementedError()

    def match(self, node: torch.fx.Node) -> Union[Match, FailedMatch]:
        try:
            return MatchContext([self], graph=node.graph).match(self, node)
        except FailedMatch as e:
            return e

    def has_multiple_users(self) -> bool:
        return False

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def find_anchor_nodes(self, ctx: MatchContext, searched):
        if self in ctx.pattern_to_node:
            yield ctx.pattern_to_node[self]


class Arg(PatternExpr):
    """
    Capture an arg which will become an input to the handler.  Args are
    passed in depth first order.
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext):
        return Match(self, args=[node])  # matches anything


class Ignored(PatternExpr):
    """
    Match an arg, but don't pass it to handler
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext):
        return Match(self)  # matches anything

    def __repr__(self):
        return "*"

    def pretty_print(self, pp: PatternPrettyPrinter):
        return "Ignored()"


class KeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"KeywordArg({self.name!r})"

    def _match(self, node: NodeOrConstant, ctx: MatchContext):
        return Match(self, kwargs={self.name: node})  # matches anything


class ExclusiveKeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"ExclusiveKeywordArg({self.name!r})"

    def _match(self, node: NodeOrConstant, ctx: MatchContext):
        if node in ctx.exclusive_node_set:
            return FailedMatch("exclusive arg appears twice")

        ctx.exclusive_node_set.append(node)
        return Match(self, kwargs={self.name: node})  # matches anything


class _TargetExpr(PatternExpr):
    """
    Base class for filtering match by node.target
    """

    op: Optional[str] = None

    def __init__(self, fns, users=1):
        if not self.op:
            raise NotImplementedError("Shouldn't directly use _BaseNodeMatch")
        super().__init__()
        fns = [fns] if callable(fns) or isinstance(fns, str) else list(fns)
        for fn in list(fns):
            if isinstance(fn, torch._ops.OpOverloadPacket):
                fns.extend([getattr(fn, overload) for overload in fn.overloads()])

        self.fns: List[Union[Callable[..., Any], str]] = fns
        self.fns_set: Set[Union[Callable[..., Any], str]] = set(fns)
        self.users: Union[int, Multiple] = users

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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.fns_repr()})"

    def has_multiple_users(self) -> bool:
        return isinstance(self.users, Multiple) or self.users > 1

    def find_anchor_nodes(self, ctx: MatchContext, searched):
        raise NotImplementedError()

    def _match_fns(self, node: torch.fx.Node):
        return (
            isinstance(node, torch.fx.Node)
            and node.op == self.op
            and extract_target(node) in self.fns_set
        )

    def _match_users(self, node: torch.fx.Node, ctx: MatchContext):
        return (
            self in ctx.outputs
            or self.users is MULTIPLE
            or len(node.users) == self.users
        )


class _TargetArgsExpr(_TargetExpr):
    """
    Base class for filtering match by node.{target,args,kwargs}
    """

    def __init__(self, fns, *args, _users=1, **kwargs):
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
    def simple_flatten(args, kwargs: Dict[Any, Any]):
        return (*args, *kwargs.values()), (len(args), *kwargs.keys())

    @staticmethod
    def pytree_flatten(args, kwargs: Dict[Any, Any]):
        def norm_spec(s: pytree.TreeSpec):
            if s.type is None:
                return s
            mapping = {immutable_list: list, tuple: list, immutable_dict: dict}
            return pytree.TreeSpec(
                mapping.get(s.type, s.type),
                s.context,
                list(map(norm_spec, s.children_specs)),
            )

        flat, spec = pytree.tree_flatten([args, kwargs])
        spec = norm_spec(spec)
        return flat, spec

    def __repr__(self):
        args = [
            self.fns_repr(),
            *map(repr, self.args),
            *[f"{k}={v}" for k, v in self.kwargs.items()],
        ]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def pretty_print(self, pp: PatternPrettyPrinter):
        args = [
            self.fns_repr(),
            *(pp.pretty_print(x) for x in self.args),
            *[f"{k}={pp.pretty_print(v)}" for k, v in self.kwargs.items()],
        ]
        if isinstance(self.users, Multiple):
            args.append("_users=MULTIPLE")
        elif self.users > 1:
            args.append(f"_users={self.users}")

        joiner_str = ", "
        return f"{self.__class__.__name__}({joiner_str.join(args)})"

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        if not self._match_fns(node) or len(node.args) != len(self.args):
            return FailedMatch("function_mismatch: node={}, pattern={}", node, self)

        if not self._match_users(node, ctx):
            return FailedMatch("multiple_users {}", self)

        _args = node.args
        _kwargs = node.kwargs
        if len(_kwargs) < len(self.kwargs):
            from torch.fx.operator_schemas import normalize_function

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

        m = Match(self)
        for i, pattern, child_node in zip(itertools.count(), self_items, node_items):
            if isinstance(pattern, PatternExpr):
                child_match = ctx.match(pattern, child_node)
                if not child_match:
                    return child_match
                m.extend(child_match)
            elif isinstance(child_node, torch.fx.Node) or child_node != pattern:
                return FailedMatch(
                    "constant_args: {} {!r}!={pattern!r}", node, child_node
                )
        m.nodes.append(node)
        m.targets[self] = node.target
        return m

    def find_anchor_nodes(self, ctx: MatchContext, searched):
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

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        if not self._match_fns(node):
            return FailedMatch("function_mismatch")

        if not self._match_users(node, ctx):
            return FailedMatch("multiple_users")

        m = Match(self)
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

    def __init__(self, pattern: PatternExpr, partial=False):
        super().__init__()
        assert isinstance(pattern, PatternExpr)
        self.pattern = pattern
        self.partial = partial

    def __repr__(self):
        return f"{self.__class__.__name__}({self.pattern})"

    def _match(self, node: List[torch.fx.Node], ctx: MatchContext):  # type: ignore[override]
        if not isinstance(node, (list, tuple)) or len(node) == 0:
            return FailedMatch("non_list")
        m = Match(self)
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
            if not child_match:
                if not self.partial:
                    return FailedMatch("list[{}]: {}", i, child_match)
                continue
            matched = True
            m.extend(child_match.bundle())
        if not matched:
            return FailedMatch("list: no_match")
        return m.bundle()


class MultiOutputPattern(PatternExpr):
    def __init__(self, outputs):
        super().__init__()
        assert all(isinstance(x, (PatternExpr, type(None))) for x in outputs), outputs
        self.outputs: List[Optional[PatternExpr]] = outputs

    @property
    def fns(self):
        assert self.outputs[0] and hasattr(self.outputs[0], "fns")
        return self.outputs[0].fns

    def __repr__(self):
        return f"{self.__class__.__name__}({self.outputs})"

    def pretty_print(self, pp: PatternPrettyPrinter):
        args = [pp.pretty_print(x) for x in self.outputs]
        joiner_str = f",\n{'  '}"
        str_out = f"{self.__class__.__name__}([{joiner_str.join(args)}"
        str_out = f"{str_out}\n])"
        return str_out

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        m = ctx.match(self.outputs[0], node)
        if not m:
            return m

        for pattern in self.outputs[1:]:
            if pattern is None:
                continue
            child_match = self._match_from_anchors(pattern, ctx)
            if not child_match:
                return child_match
            m.extend(child_match)

        return m

    def _match_from_anchors(self, pattern, ctx):
        prior = dict(ctx.pattern_to_node)
        m = FailedMatch("no anchor found")
        for node in pattern.find_anchor_nodes(ctx, set()):
            m = ctx.match(pattern, node)
            if m:
                return m
            # revert any partial matches
            ctx.pattern_to_node = dict(prior)
        return m

    def match(self, node: torch.fx.Node) -> Union[Match, FailedMatch]:
        try:
            return MatchContext(self.outputs, graph=node.graph).match(self, node)
        except FailedMatch as e:
            return e


class RepeatedExpr(PatternExpr):
    """
    Checks for a repeated pattern. Useful for repeated operations after a node such as `split` or `unbind`
    """

    def __init__(self, inner_pattern: PatternExpr):
        super().__init__()
        assert hasattr(inner_pattern, "fns")
        self.inner_pattern = inner_pattern

    @property
    def fns(self):
        return self.inner_pattern.fns

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        m = ctx.match(self.inner_pattern, node)
        if not m:
            return m
        ctx.pattern_to_node.pop(
            self.inner_pattern,
        )
        # Check all anchor nodes match the pattern
        for anchor_node in self.inner_pattern.find_anchor_nodes(ctx, set()):
            anchor_m = MatchContext([self], graph=node.graph).match(
                self.inner_pattern, anchor_node
            )
            if not anchor_m:
                return anchor_m
            m.extend(anchor_m)
        return m


class PatternPrettyPrinter:
    """
    Serializes Patterns to executable python.
    XXX: currently only used and tested for fuse attention patterns. May not cover
    all patterns.
    """

    def __init__(self):
        self.namespace = torch.fx.graph._Namespace()
        self.memoized_objs_names: Dict[PatternExpr, str] = {}
        self.memoized_objs_pp: Dict[PatternExpr, str] = {}

    @staticmethod
    def run(obj: PatternExpr, output_name="output"):
        """
        Serializes obj to python code with obj written out to `output_name`
        """

        pp = PatternPrettyPrinter()
        assert hasattr(obj, "pretty_print")
        out_str = obj.pretty_print(pp=pp)

        output = []
        for key in pp.memoized_objs_names:
            output.append(f"{pp.memoized_objs_names[key]} = {pp.memoized_objs_pp[key]}")

        output.append(f"{output_name} = {out_str}")

        return "\n".join(output)

    def pretty_print(self, obj):
        if isinstance(obj, _TargetArgsExpr):
            if memoized_name := self.memoized_objs_names.get(obj):
                return memoized_name
            else:
                return self.memoize(obj)
        if hasattr(obj, "pretty_print"):
            return obj.pretty_print(self)

        return repr(obj)

    def memoize(self, obj):
        obj_str = obj.pretty_print(self)
        obj_name = obj.fns_repr()
        for prefix in ("aten.", "torch.", "prims."):
            obj_name = obj_name.replace(prefix, "")

        tmp_name = self.namespace.create_name(obj_name, None)
        self.memoized_objs_names[obj] = tmp_name
        self.memoized_objs_pp[obj] = obj_str
        return tmp_name


@dataclasses.dataclass
class PatternEntry:
    pattern: PatternExpr
    extra_check: Callable[[Match], bool]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        raise NotImplementedError()

    def register(self, pass_dicts, target=None, prepend=False):
        if target is None:
            assert hasattr(self.pattern, "fns")
            for fn in self.pattern.fns:
                self.register(pass_dicts, fn, prepend=prepend)
        elif isinstance(pass_dicts, (dict, PatternMatcherPass)):
            if prepend:
                pass_dicts[target].insert(0, self)
            else:
                pass_dicts[target].append(self)
        else:
            for x in pass_dicts:
                self.register(x, target, prepend=prepend)


@dataclasses.dataclass
class LoweringPatternEntry(PatternEntry):
    handler: Callable[..., Any]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        handler = functools.wraps(self.handler)(functools.partial(self.handler, match))
        with graph.inserting_before(node):
            replacement = graph.call_function(handler, tuple(match.args), match.kwargs)
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
        assert match.nodes[-1] is node
        match.erase_nodes(graph)


@dataclasses.dataclass
class GraphPatternEntry(PatternEntry):
    """
    A pattern that runs a function on the FX graph
    """

    handler: Callable[..., Any]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        with graph.inserting_before(node):
            self.handler(match, *match.args, **match.kwargs)


@dataclasses.dataclass
class ReplacementPatternEntry(PatternEntry):
    normalize_args: Callable[..., List[Any]]

    @staticmethod
    def replace_with_graph(
        match: Match,
        graph: torch.fx.Graph,
        replacement_graph: torch.fx.Graph,
        args: List[Any],
    ):
        output_nodes = match.output_nodes()
        first_node = output_nodes[0]

        class Replacer(torch.fx.Interpreter):
            call_method = None  # type: ignore[assignment]
            call_module = None  # type: ignore[assignment]
            get_attr = None  # type: ignore[assignment]

            def run_node(self, node) -> Any:
                if node.op in ("placeholder", "output"):
                    return super().run_node(node)
                if node.op == "call_function":
                    target = node.target
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    result = graph.call_function(target, args, kwargs)
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
            last_node = min(indices, key=lambda tup: tup[0])[1]

        def percolate_tags(node, recompute_tag, input_stops):
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
                    arg.meta["recompute"] = recompute_tag
                    queue.extend(arg.all_input_nodes)

        with graph.inserting_before(last_node):
            replacement = Replacer(replacement_graph).run(*args)
            if isinstance(replacement, torch.fx.Node):
                replacement = [replacement]

            def maybe_getitem(node):
                if node.op != "call_function":
                    return None
                if node.target != operator.getitem:
                    return None
                assert len(node.args) == 2
                return node.args[1]

            def replace(old, new):
                if old is None:
                    assert new is None
                    return
                assert isinstance(old, torch.fx.Node)
                if new is None:
                    old.replace_all_uses_with(None)
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
                    if "recompute" in old.meta:
                        percolate_tags(new, old.meta["recompute"], args)

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

        match.erase_nodes(graph)

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        self.replace_with_graph(
            match,
            graph,
            match.replacement_graph,  # type: ignore[arg-type]
            self.normalize_args(*match.args, **match.kwargs),
        )


def _return_true(match):
    return True


def log_trace_failure(search_fn, e):
    log.info(
        "Replacement pattern %s failed to apply due to shape mismatch: %s",
        search_fn.__name__,
        e,
    )


def register_replacement(
    search_fn,
    replace_fn,
    example_inputs: Iterable[Any],
    trace_fn: Callable[[Callable[..., Any], Iterable[Any]], torch.fx.GraphModule],
    pass_dicts,
    extra_check=_return_true,
    scalar_workaround=(),
    exclusive_arg_names=(),
    search_fn_pattern=None,
):
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

    def check_fn(match: Match):
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
            torch.fx.map_arg(
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

            if sym_args:
                # AOT Autograd and make fx will dedupe symbolic shape size
                # accesses of sym ints that appear as inputs
                # We don't want the sym_size uses to interfere with pattern matching
                # so we provide them as inputs.
                # Later, when we actually do the replacement, the symbolic shape
                # sizes will get re-traced and added to the graph.

                def search_fn_new(*args_new):
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
            specific_pattern_match = specific_pattern.match(match.output_nodes()[0])  # type: ignore[arg-type]
            if specific_pattern_match and extra_check(specific_pattern_match):
                # trace the pattern using the shapes from the user program
                match.replacement_graph = trace_fn(replace_fn, args)  # type: ignore[assignment]
                return True
            return False

    def normalize_args(**kwargs):
        args = []
        for name in argnames_static:
            args.append(kwargs.pop(name))
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


@functorch_config.patch(functionalize_rng_ops=False)
def gen_pattern(
    search_fn, example_inputs, trace_fn, scalar_workaround=(), exclusive_arg_names=()
) -> PatternExpr:
    argnames = [*inspect.signature(search_fn).parameters.keys()]

    if scalar_workaround == ():
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
    pattern: PatternExpr, extra_check=_return_true, *, pass_dict, prepend=False
):
    """
    Register an aten to inductor IR replacement pattern.  The decorated
    function is saved and then called a lowering time allowing direct
    pattern to inductor IR conversion.
    """

    def decorator(handler):
        assert callable(handler)
        LoweringPatternEntry(
            pattern=pattern, extra_check=extra_check, handler=handler
        ).register(pass_dict, prepend=prepend)
        handler._inductor_lowering_function = True
        return handler

    return decorator


def register_graph_pattern(
    pattern: PatternExpr, extra_check=_return_true, *, pass_dict, prepend=False
):
    """
    Register a pattern that runs a function on the FX graph, allowing
    custom transformation code.
    """

    def decorator(handler):
        assert callable(handler)
        GraphPatternEntry(
            pattern=pattern, extra_check=extra_check, handler=handler
        ).register(pass_dict, prepend=prepend)
        return handler

    return decorator


def is_start_of_fx_graph(graph: torch.fx.Graph, node: torch.fx.Node) -> bool:
    # first node in the graph
    return node is next(iter(graph.nodes))


# match: copy_, relu_, _set_grad_enabled, manual_seed, enter_functional_autocast, etc
_mutation_op_re = re.compile(r"_$|_[.]|(\b|_)(set|enter|exit|seed)(\b|_)")


def is_mutation_op(node: torch.fx.Node) -> bool:
    if node.op == "call_function":
        if _mutation_op_re.search(node.target.__name__):  # type: ignore[union-attr]
            return True
    elif node.op == "call_method":
        if _mutation_op_re.search(node.target):  # type: ignore[union-attr, arg-type]
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


def should_compute_mutation_region_ids(graph: torch.fx.GraphModule) -> bool:
    return "mutation_region_id" not in next(iter(graph.nodes)).meta


def compute_mutation_region_ids(graph: torch.fx.GraphModule):
    mutation_region_id = 0
    for nd in graph.nodes:
        if is_mutation_op(nd):
            mutation_region_id += 1
        nd.meta["mutation_region_id"] = mutation_region_id


class PatternMatcherPass:
    def __init__(
        self, prevent_match_across_mutations=False, pass_name: Optional[str] = None
    ):
        super().__init__()
        self.patterns: DefaultDict[
            torch.fx.node.Target, List[PatternEntry]
        ] = defaultdict(list)
        self.prevent_match_across_mutations = prevent_match_across_mutations
        self.pass_name = pass_name

    def __getitem__(self, item: torch.fx.node.Target) -> List[PatternEntry]:
        return self.patterns[item]

    def apply(self, graph: torch.fx.GraphModule) -> int:
        if not self.patterns:
            return 0
        if isinstance(graph, torch.fx.GraphModule):
            graph = graph.graph
        if self.prevent_match_across_mutations:
            if should_compute_mutation_region_ids(graph):
                compute_mutation_region_ids(graph)
            get_mutation_region_id_partial = functools.partial(
                get_mutation_region_id, graph
            )
        count = 0
        for node in reversed(graph.nodes):
            target = extract_target(node)
            if (
                node.op in ["call_function", "call_method", "call_module"]
                and target in self.patterns
            ):
                # conservatively not applying pattern for cpu input,
                # since some of the patterns induce codegen and split nodes.
                # Note: we will only skip cpu compute if disable_cpp_codegen=True
                if fallback_node_due_to_unsupported_type(node, allow_cpu_inputs=False):
                    continue

                for entry in self.patterns[target]:
                    if node._erased:
                        break
                    m = entry.pattern.match(node)
                    # pattern match crosses mutation barrier - discard
                    if (
                        self.prevent_match_across_mutations
                        and is_match(m)
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

    def clear(self):
        self.patterns.clear()


def _not_implemented(*args, **kwargs) -> NoReturn:
    raise NotImplementedError()


def fx_to_pattern(
    gm,
    ignore_types=(),
    argnames=(),
    scalar_workaround=(),
    exclusive_arg_names=(),
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

    def process_arg(x):
        if isinstance(x, (float, int)) and x in inv_scalar_workaround:
            return KeywordArg(inv_scalar_workaround[x])
        if type(x) in ignore_types:
            return Ignored()
        if isinstance(x, list) and all(isinstance(y, Ignored) for y in x) and x:
            return Ignored()
        return x

    argnum = itertools.count()

    class Converter(torch.fx.Interpreter):
        call_method = _not_implemented
        call_module = _not_implemented
        get_attr = _not_implemented

        def placeholder(self, target, args, kwargs):
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

        def call_function(self, target, args, kwargs):
            args, kwargs = pytree.tree_map(process_arg, (args, kwargs))
            if list in ignore_types:
                # Handle a burned in tensor size which are now [Ignored(), Ignored(), ...]
                args = [process_arg(a) for a in args]
                kwargs = {k: process_arg(a) for k, a in kwargs.items()}
            return CallFunction(target, *args, **kwargs)

        def run_node(self, n):
            rv = super().run_node(n)
            if n.op == "output" and isinstance(rv, tuple):
                assert len(rv) == len(n.args[0])
                for r, arg in zip(rv, n.args[0]):
                    r.users = len(arg.users)
            else:
                rv.users = len(n.users)
            return rv

    pattern = Converter(gm).run()
    if not isinstance(pattern, PatternExpr):
        return MultiOutputPattern(pytree.tree_leaves(pattern))
    return pattern


@torch.no_grad()
def fwd_only(fn, args, *, run_dce=True) -> torch.fx.GraphModule:
    """Build a normalized inference graph, for use with fx_to_pattern"""
    # TODO - look into using aot autograd, asserting no mutating ops here
    with enable_python_dispatcher():
        mode = (
            "real" if not torch._inductor.utils.any_is_symbolic(*args) else "symbolic"
        )
        gm = make_fx(fn, select_decomp_table(), tracing_mode=mode)(*args)
    if run_dce:
        gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


@torch.enable_grad()
def joint_fwd_bwd(fn, args) -> torch.fx.GraphModule:
    """Build a normalized training graph, for use with fx_to_pattern"""
    gm: Optional[torch.fx.GraphModule] = None

    def record_joint_graph(joint_graph, inputs, **kwargs):
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
    args: List[torch.fx.node.Argument] = list()
    torch.fx.map_arg((n.args, n.kwargs), args.append)
    return args


def stable_topological_sort(graph: torch.fx.Graph):
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


def init_once_fakemode(fn: Callable[..., Any]):
    """Wrapper around lazy init functions in fx_passes/"""

    @functools.lru_cache(None)
    @functools.wraps(fn)
    def lazy_init():
        counters_ref = counters["inductor"].copy()

        with torch._guards.tracing(
            None
        ), maybe_disable_fake_tensor_mode(), FakeTensorMode():
            result = fn()

        # clear view matches encountered during tracing
        counters["inductor"] = counters_ref

        return result

    return lazy_init


def config_flag(name):
    """Function for extra_check to put pass behind a flag"""

    def flag_check(match):
        return getattr(config, name)

    return flag_check


def clone_graph(input_graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
    class CopyGraph(Transformer):
        def run_node(self, old_node):
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
):
    return (
        node.args[arg_number]
        if len(node.args) > arg_number
        else node.kwargs.get(kwarg_name)  # type: ignore[arg-type]
    )


def filter_nodes(nodes: Iterable[torch.fx.Node], fn) -> List[torch.fx.Node]:
    fns = [fn]
    if isinstance(fn, torch._ops.OpOverloadPacket):
        fns.extend([getattr(fn, overload) for overload in fn.overloads()])

    return [node for node in nodes if node.target in fns]


def extract_target(node: Node):
    """For call_function and call_method, we directly use the target function;
    For call_module, the target is string, and we treat the module class
     as a function.
    """
    if node.op == "call_module":
        return getattr(node.graph.owning_module, node.target).__class__  # type: ignore[arg-type]
    return node.target
