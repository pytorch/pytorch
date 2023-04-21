import dataclasses
import functools
import inspect
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import patch

import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.immutable_collections import immutable_dict, immutable_list
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from ..fx import Transformer
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type

log = logging.getLogger(__name__)
aten = torch.ops.aten

Constant = Any
NodeOrConstant = Union[Constant, torch.fx.Node]

# Sentinel indicating multiple quantities can be matched
MULTIPLE = object()


class Match:
    """
    Represents a successfully matched pattern.
    """

    def __init__(self, pattern, args=None, kwargs=None):
        super().__init__()
        self.pattern = pattern
        # The input nodes that must be passed in to the result
        self.args = args or []
        self.kwargs = kwargs or {}
        # The nodes matched in this expression
        self.nodes = []
        # Mapping CallFunction to the node.target
        self.targets = {}
        self.ctx: MatchContext = None

    def extend(self, other):
        if self.kwargs:
            for key in set(self.kwargs.keys()) & set(other.kwargs.keys()):
                if self.kwargs[key] != other.kwargs[key]:
                    raise FailedMatch(f"kwarg mismatch: {key}")
        self.args.extend(other.args)
        self.nodes.extend(other.nodes)
        self.kwargs.update(other.kwargs)
        self.targets.update(other.targets)

    def bundle(self):
        # Wrap args in an extra list
        self.args = [tuple(self.args)] if self.args else []
        return self

    def __repr__(self):
        return f"Match(..., {self.args}, {self.kwargs})"

    def erase_nodes(self, graph: torch.fx.Graph):
        for n in reversed(self.nodes):
            graph.erase_node(n)

    def output_nodes(self):
        return [
            (self.ctx.pattern_to_node[p] if p is not None else None)
            for p in self.ctx.outputs
        ]


class FailedMatch(RuntimeError):
    def __bool__(self):
        return False


class MatchContext:
    """
    State needed while running PatternExpr._match().
    """

    def __init__(
        self,
        outputs: List["PatternExpr"],
        pattern_to_node: Optional[Dict["PatternExpr", Node]] = None,
    ):
        self.outputs = outputs
        self.pattern_to_node = pattern_to_node
        if self.pattern_to_node is None:
            self.pattern_to_node = {}

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
            if pattern.has_multiple_users()
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
            return MatchContext([self]).match(self, node)
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


class KeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"KeywordArg({self.name!r})"

    def _match(self, node: NodeOrConstant, ctx: MatchContext):
        return Match(self, kwargs={self.name: node})  # matches anything


class CallFunction(PatternExpr):
    """
    Matches a call_function node in the FX graphs: `fns[i](*args, **kwargs)`
    """

    def __init__(self, fns, *args, _users=1, **kwargs):
        super().__init__()
        fns = [fns] if callable(fns) else list(fns)
        for fn in list(fns):
            if isinstance(fn, torch._ops.OpOverloadPacket):
                fns.extend([getattr(fn, overload) for overload in fn.overloads()])

        self.fns = fns
        self.fns_set = set(fns)
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self.users = _users
        if any(
            isinstance(x, (dict, list, tuple))
            for x in itertools.chain(args, kwargs.values())
        ):
            self.flatten = self.pytree_flatten
        else:
            self.flatten = self.simple_flatten
        self.flat_args_kwargs = self.flatten(self.args, self.kwargs)

    @staticmethod
    def simple_flatten(args, kwargs):
        return (*args, *kwargs.values()), (len(args), *kwargs.keys())

    @staticmethod
    def pytree_flatten(args, kwargs):
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
            f"[{self.fns[0].__name__}, ...]"
            if len(self.fns) > 1
            else self.fns[0].__name__,
            *map(repr, self.args),
            *[f"{k}={v}" for k, v in self.kwargs.items()],
        ]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        if (
            not isinstance(node, torch.fx.Node)
            or node.op != "call_function"
            or node.target not in self.fns_set
            or len(node.args) != len(self.args)
            or len(node.kwargs) != len(self.kwargs)
        ):
            return FailedMatch("function_mismatch")

        if (
            self not in ctx.outputs
            and self.users is not MULTIPLE
            and len(node.users) != self.users
        ):
            return FailedMatch("multiple_users")

        node_items, node_spec = self.flatten(node.args, node.kwargs)
        self_items, self_spec = self.flat_args_kwargs
        if node_spec != self_spec:
            return FailedMatch(f"args_structure {node_spec} {self_spec}")
        assert len(node_items) == len(self_items)

        m = Match(self)
        for i, pattern, child_node in zip(itertools.count(), self_items, node_items):
            if isinstance(pattern, PatternExpr):
                child_match = ctx.match(pattern, child_node)
                if not child_match:
                    return child_match
                m.extend(child_match)
            elif isinstance(child_node, torch.fx.Node) or child_node != pattern:
                return FailedMatch(f"constant_args: {node} {child_node!r}!={pattern!r}")
        m.nodes.append(node)
        m.targets[self] = node.target
        return m

    def has_multiple_users(self) -> bool:
        return self.users is MULTIPLE or self.users > 1

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
                    for node in other_node.users:
                        if (
                            node not in searched
                            and node.op == "call_function"
                            and node.target in self.fns_set
                        ):
                            yield node
                            searched.add(node)


class ListOf(PatternExpr):
    """
    Matches a repeated pattern
    """

    def __init__(self, pattern):
        super().__init__()
        assert isinstance(pattern, PatternExpr)
        self.pattern = pattern

    def __repr__(self):
        return f"{self.__class__.__name__}({self.pattern})"

    def _match(self, node: List[torch.fx.Node], ctx: MatchContext):
        if not isinstance(node, (list, tuple)) or len(node) == 0:
            return FailedMatch("non_list")
        m = Match(self)
        # Propogating patterns with multiple users will ensure we don't revisit
        # the same nodes
        pattern_to_node = ctx.filter_multi_user_patterns()
        for i, child_node in enumerate(node):
            child_ctx = MatchContext(ctx.outputs, pattern_to_node)
            child_match = child_ctx.match(self.pattern, child_node)
            if not child_match:
                return FailedMatch(f"list[{i}]: {child_match}")
            pattern_to_node = child_ctx.filter_multi_user_patterns()
            m.extend(child_match.bundle())
        return m.bundle()


class MultiOutputPattern(PatternExpr):
    def __init__(self, outputs):
        super().__init__()
        assert all(isinstance(x, (PatternExpr, type(None))) for x in outputs), outputs
        self.outputs = outputs

    @property
    def fns(self):
        return self.outputs[0].fns

    def __repr__(self):
        return f"{self.__class__.__name__}({self.outputs})"

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
            return MatchContext(self.outputs).match(self, node)
        except FailedMatch as e:
            return e


@dataclasses.dataclass
class PatternEntry:
    pattern: PatternExpr
    extra_check: Callable[[Match], bool]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        raise NotImplementedError()

    def register(self, pass_dicts, target=None):
        if target is None:
            for fn in self.pattern.fns:
                self.register(pass_dicts, fn)
        elif isinstance(pass_dicts, (dict, PatternMatcherPass)):
            pass_dicts[target].append(self)
        else:
            for x in pass_dicts:
                self.register(x, target)


@dataclasses.dataclass
class LoweringPatternEntry(PatternEntry):
    handler: Any

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        handler = functools.wraps(self.handler)(functools.partial(self.handler, match))
        with graph.inserting_before(node):
            replacement = graph.call_function(handler, tuple(match.args), match.kwargs)
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
        assert match.nodes[-1] is node
        match.erase_nodes(graph)


@dataclasses.dataclass
class ReplacementPatternEntry(PatternEntry):
    normalize_args: Callable

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        class Replacer(torch.fx.Interpreter):
            call_method = None
            call_module = None
            get_attr = None

            def run_node(self, node) -> Any:
                if node.op in ("placeholder", "output"):
                    return super().run_node(node)
                if node.op == "call_function":
                    target = node.target
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    result = graph.call_function(target, args, kwargs)
                    if "val" in node.meta and "val" not in result.meta:
                        result.meta["val"] = node.meta["val"]
                    return result
                raise NotImplementedError(f"unhandled {node}")

        with graph.inserting_before(node):
            replacement = Replacer(match.replacement_graph).run(
                *self.normalize_args(*match.args, **match.kwargs)
            )
            if isinstance(replacement, torch.fx.Node):
                replacement = [replacement]
            output_nodes = match.output_nodes()
            assert len(replacement) == len(output_nodes)
            for old, new in zip(output_nodes, replacement):
                if old is None:
                    assert new is None
                elif new is None:
                    old.replace_all_uses_with(None)
                else:
                    if "val" not in new.meta:
                        new.meta.update(old.meta)
                    old.replace_all_uses_with(new)

        match.erase_nodes(graph)


def _return_true(match):
    return True


def register_replacement(
    search_fn,
    replace_fn,
    example_inputs,
    trace_fn,
    pass_dict,
    extra_check=_return_true,
    scalar_workaround=(),
):
    """
    Create a replacement rule based on example functions that get traced
    to create patterns.  This supports both training and inference when
    run on a joint foward+backward graph.

    Args:
        search_fn: traced to give original pattern
        replace_fn: traced to give replacement graph
        example_inputs: example inputs for initial trace
        trace_fn: inference_graph or training_graph
        pass_dict: dict of passes to register to
        extra_check: additional check to run on match(using real shapes)
    """

    def check_fn(match: Match):
        """
        Often shapes get burned into the pattern, so our initial match ran with
        `ignore_types=(int, ...)`.

        Recheck the match with the correct shapes.
        """
        args = list(
            torch.fx.map_arg(
                [match.kwargs[name] for name in argnames], lambda n: n.meta["val"]
            )
        )
        for i, grad in enumerate(requires_grad):
            if isinstance(args[i], torch.Tensor):
                args[i] = torch.empty_strided(
                    args[i].size(),
                    args[i].stride(),
                    dtype=args[i].dtype,
                    device=args[i].device,
                    requires_grad=grad,
                )
        specific_graph = trace_fn(search_fn, args)
        specific_pattern = fx_to_pattern(specific_graph, argnames=argnames)
        specific_pattern_match = specific_pattern.match(match.output_nodes()[0])
        if specific_pattern_match and extra_check(specific_pattern_match):
            # trace the pattern using the shapes form the user program
            match.replacement_graph = trace_fn(replace_fn, args)
            return True
        return False

    def normalize_args(**kwargs):
        args = []
        for name in argnames:
            args.append(kwargs.pop(name))
        for i in range(1, len(kwargs) + 1):
            args.append(kwargs.pop(f"tangents_{i}"))
        assert not kwargs, f"leftover kwargs: {kwargs!r}"
        return args

    # TODO: Revisit the functionalize_rng_ops for lowmem dropout
    with patch("functorch.compile.config.functionalize_rng_ops", False):
        argnames = [*inspect.signature(search_fn).parameters.keys()]
        requires_grad = [
            isinstance(x, torch.Tensor) and x.requires_grad for x in example_inputs
        ]
        search_gm = trace_fn(search_fn, example_inputs)
        pattern = fx_to_pattern(
            search_gm,
            ignore_types=(int, float, torch.device, torch.dtype),
            argnames=argnames,
            scalar_workaround=scalar_workaround,
        )
        assert repr(pattern) not in _seen_patterns
        _seen_patterns.add(repr(pattern))
        pattern = ReplacementPatternEntry(
            pattern=pattern,
            extra_check=check_fn,
            normalize_args=normalize_args,
        )
        pattern.register(pass_dict)


def register_lowering_pattern(pattern, extra_check=_return_true, *, pass_dict):
    """
    Register an aten to inductor IR replacement pattern
    """

    def decorator(handler):
        assert callable(handler)
        for target in pattern.fns:
            LoweringPatternEntry(
                pattern=pattern, extra_check=extra_check, handler=handler
            ).register(pass_dict, target)
        handler._inductor_lowering_function = True
        return handler

    assert isinstance(pattern, CallFunction)
    return decorator


class PatternMatcherPass:
    def __init__(self):
        super().__init__()
        self.patterns = defaultdict(list)

    def __getitem__(self, item):
        return self.patterns[item]

    def apply(self, graph):
        if not self.patterns:
            return 0
        count = 0
        for node in reversed(graph.nodes):
            if node.op == "call_function" and node.target in self.patterns:
                # conservatively not applying pattern for cpu input,
                # since some of the patterns induce codegen and split nodes.
                # Note: we will only skip cpu compute if disable_cpp_codegen=True
                if fallback_node_due_to_unsupported_type(node, allow_cpu_inputs=False):
                    continue

                for entry in self.patterns[node.target]:
                    if node._erased:
                        break
                    m = entry.pattern.match(node)
                    if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG") == node.name:
                        log.warning("%s%s %s %s", node, node.args, m, entry.pattern)
                    if m and entry.extra_check(m):
                        count += 1
                        entry.apply(m, graph, node)
                        counters["inductor"]["pattern_matcher_count"] += 1
                        counters["inductor"]["pattern_matcher_nodes"] += len(m.nodes)
        return count


def _not_implemented(*args, **kwargs):
    raise NotImplementedError()


def fx_to_pattern(gm, ignore_types=(), argnames=(), scalar_workaround=()):
    """
    Convert an FX graph into a PatternExpr.  This is useful for simple
    patterns that can only match single functions and fixed length lists.
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
        return x

    argnum = itertools.count()

    class Converter(torch.fx.Interpreter):
        call_method = _not_implemented
        call_module = _not_implemented
        get_attr = _not_implemented

        def placeholder(self, target, args, kwargs):
            n = next(argnum)
            if n < len(argnames):
                return KeywordArg(argnames[n])
            if argnames:
                assert target.startswith("tangent")
                return KeywordArg(target)
            else:
                target = re.sub(r"_\d+$", "", target)  # de-mangle arg name
                return KeywordArg(target)

        def call_function(self, target, args, kwargs):
            args, kwargs = pytree.tree_map(process_arg, (args, kwargs))
            return CallFunction(target, *args, **kwargs)

        def run_node(self, n):
            rv = super().run_node(n)
            rv.users = len(n.users)
            return rv

    pattern = Converter(gm).run()
    if not isinstance(pattern, PatternExpr):
        return MultiOutputPattern(pytree.tree_flatten(pattern)[0])
    return pattern


@torch.no_grad()
def inference_graph(fn, args):
    """Build a normalized inference graph, for use with fx_to_pattern"""
    gm = make_fx(fn, select_decomp_table())(*args)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


@torch.enable_grad()
def training_graph(fn, args):
    """Build a normalized training graph, for use with fx_to_pattern"""
    gm = None

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
            enable_log=False,
        )(*args)

    # remove in/out specs
    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def clone_graph(input_graph):
    class CopyGraph(Transformer):
        def run_node(self, old_node):
            new_node = super().run_node(old_node)
            if isinstance(new_node, torch.fx.Proxy):
                new_node.node.meta.update(old_node.meta)
            return new_node

    return CopyGraph(input_graph).transform()


_seen_patterns = set()


def get_arg_value(node, arg_number, kwarg_name=None):
    return (
        node.args[arg_number]
        if len(node.args) > arg_number
        else node.kwargs.get(kwarg_name)
    )


def filter_nodes(nodes, fn):
    fns = [fn]
    if isinstance(fn, torch._ops.OpOverloadPacket):
        fns.extend([getattr(fn, overload) for overload in fn.overloads()])

    return [node for node in nodes if node.target in fns]
