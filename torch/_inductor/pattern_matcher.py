import dataclasses
import functools
import inspect
import itertools
import logging
import operator
import os
from collections import defaultdict
from typing import Any, Callable, List, Union

import torch
import torch._inductor as inductor
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch.fx.immutable_collections import immutable_dict, immutable_list

from . import config, ir
from .lowering import lowerings as L
from .virtualized import V

log = logging.getLogger(__name__)
aten = torch.ops.aten

Constant = Any
NodeOrConstant = Union[Constant, torch.fx.Node]


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
        self.args = [tuple(self.args)]
        return self

    def __repr__(self):
        return f"Match(..., {self.args}, {self.kwargs})"

    def erase_nodes(self, graph: torch.fx.Graph):
        for n in reversed(self.nodes):
            graph.erase_node(n)


class FailedMatch(RuntimeError):
    def __bool__(self):
        return False


class MatchContext:
    """
    State needed while running PatternExpr._match().
    """

    def __init__(self, outputs: List["PatternExpr"]):
        self.outputs = outputs
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
        return m


class PatternExpr:
    """
    Base class for types of patterns
    """

    def _match(self, node: torch.fx.Node, outputs) -> Union[Match, FailedMatch]:
        raise NotImplementedError()

    def match(self, node: torch.fx.Node) -> Union[Match, FailedMatch]:
        try:
            return MatchContext([self]).match(self, node)
        except FailedMatch as e:
            return e

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Arg(PatternExpr):
    """
    Capture an arg which will become an input to the handler.  Args are
    passed in depth first order.
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext):
        return Match(self, args=[node])  # matches anything


class KeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def _match(self, node: NodeOrConstant, ctx: MatchContext):
        return Match(self, kwargs={self.name: node})  # matches anything


class CallFunction(PatternExpr):
    """
    Matches a call_function node in the FX graps: `fns[i](*args, **kwargs)`
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
            f"[{self.fns[0].__name__}, ...]",
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

        if self not in ctx.outputs and len(node.users) != self.users:
            return FailedMatch("multiple_users")

        node_items, node_spec = self.flatten(node.args, node.kwargs)
        self_items, self_spec = self.flat_args_kwargs
        if node_spec != self_spec:
            return FailedMatch(f"args_stucture {node_spec} {self_spec}")
        assert len(node_items) == len(self_items)

        m = Match(self)
        for i, pattern, child_node in zip(itertools.count(), self_items, node_items):
            if isinstance(pattern, PatternExpr):
                child_match = ctx.match(pattern, child_node)
                if not child_match:
                    return FailedMatch(f"arg[{i}]: {child_match}")
                m.extend(child_match)
            elif isinstance(child_node, torch.fx.Node) or child_node != pattern:
                return FailedMatch("constant_args")
        m.nodes.append(node)
        m.targets[self] = node.target
        return m


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
        for i, child_node in enumerate(node):
            child_match = MatchContext(ctx.outputs).match(self.pattern, child_node)
            if not child_match:
                return FailedMatch(f"list[{i}]: {child_match}")
            m.extend(child_match.bundle())
        return m.bundle()


pass_patterns = [
    defaultdict(list),
    defaultdict(list),
    defaultdict(list),
]


@dataclasses.dataclass
class PatternEntry:
    pattern: PatternExpr
    extra_check: Callable[[Match], bool]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        raise NotImplementedError()

    def register(self, pass_number, target):
        if isinstance(pass_number, int):
            pass_patterns[pass_number][target].append(self)
        else:
            for x in pass_number:
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
    replacement_graph: torch.fx.GraphModule
    signature: inspect.Signature
    propagate: bool = False

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        class Replacer(torch.fx.Interpreter):
            call_method = None
            call_module = None
            get_attr = None

            def call_function(self, target, args, kwargs):
                result = graph.call_function(target, args, kwargs)
                if propagate and V.fake_mode:
                    fargs, fkwargs = torch.fx.map_arg(
                        (args, kwargs), lambda n: n.meta["val"]
                    )
                    with V.fake_mode:
                        result.meta["val"] = target(*fargs, **fkwargs)
                return result

        propagate = self.propagate
        norm_args = self.signature.bind(*match.args, **match.kwargs)
        with graph.inserting_before(node):
            replacement = Replacer(self.replacement_graph).run(
                *norm_args.arguments.values()
            )
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
        assert match.nodes[-1] is node
        match.erase_nodes(graph)


def _return_true(match):
    return True


def register_replacement_pattern(pattern, extra_check=_return_true, pass_number=1):
    """
    Register an aten to aten replacement pattern
    """

    def decorator(handler):
        signature = inspect.signature(handler)
        replacement_graph = torch.fx.symbolic_trace(handler)
        for target in pattern.fns:
            ReplacementPatternEntry(
                pattern=pattern,
                extra_check=extra_check,
                replacement_graph=replacement_graph,
                signature=signature,
            ).register(pass_number, target)
        return handler

    assert isinstance(pattern, CallFunction)
    return decorator


def register_lowering_pattern(pattern, extra_check=_return_true, pass_number=1):
    """
    Register an aten to inductor IR replacement pattern
    """

    def decorator(handler):
        assert callable(handler)
        for target in pattern.fns:
            LoweringPatternEntry(
                pattern=pattern, extra_check=extra_check, handler=handler
            ).register(pass_number, target)
        handler._inductor_lowering_function = True
        return handler

    assert isinstance(pattern, CallFunction)
    return decorator


register_pattern = register_lowering_pattern


def replace_matched_patterns(graph: torch.fx.Graph):
    # the actual replacement work
    for patterns in pass_patterns:
        if not patterns:
            continue
        for node in reversed(graph.nodes):
            if node.op == "call_function" and node.target in patterns:
                for entry in patterns[node.target]:
                    if node._erased:
                        break
                    m = entry.pattern.match(node)
                    if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG") == node.name:
                        log.warning(f"{node}{node.args} {m} {entry.pattern}")
                    if m and entry.extra_check(m):
                        entry.apply(m, graph, node)
                        counters["inductor"]["pattern_matcher_count"] += 1
                        counters["inductor"]["pattern_matcher_nodes"] += len(m.nodes)


def reorder_for_locality(graph: torch.fx.Graph):
    def visit(other_node):
        if (
            other_node.op == "call_function"
            and other_node.target != operator.getitem
            and all((n in seen_nodes) for n in other_node.users)
        ):
            # move node's producers right before it
            node.prepend(other_node)

    seen_nodes = set()
    for node in reversed(graph.nodes):
        seen_nodes.add(node)
        torch.fx.map_arg((node.args, node.kwargs), visit)


def fx_passes(gm: torch.fx.GraphModule):
    if config.dce:
        # has some issues with mutation in inference mode
        gm.graph.eliminate_dead_code()

    if config.reordering:
        # has some issues with mutation in inference mode
        reorder_for_locality(gm.graph)

    if config.pattern_matcher:
        replace_matched_patterns(gm.graph)

    gm.graph.lint()


################################################################################
# Actual patterns below this point.
# Priority of patterns is:
#   - later output nodes first
#   - order patterns are defined in
################################################################################


@register_lowering_pattern(
    CallFunction(
        aten.add,
        CallFunction(aten.mm, Arg(), Arg()),
        CallFunction(aten.mm, Arg(), Arg()),
    )
)
def mm_plus_mm(match: Match, mat1, mat2, mat3, mat4):
    return inductor.kernel.mm_plus_mm.tuned_mm_plus_mm(mat1, mat2, mat3, mat4)


@register_lowering_pattern(
    CallFunction(aten.cat, ListOf(CallFunction(aten.mm, Arg(), Arg())), Arg()),
)
def cat_mm(match, inputs, dim):
    def shape_of(a, b):
        m, _ = a.get_size()
        _, n = b.get_size()
        return [m, n]

    return cat_tuned_op(match, inputs, dim, op=L[aten.mm], shape_of=shape_of)


@register_lowering_pattern(
    CallFunction(
        aten.cat, ListOf(CallFunction(aten.addmm, Arg(), Arg(), Arg())), Arg()
    ),
)
def cat_addmm(match, inputs, dim):
    def shape_of(bias, a, b):
        m, _ = a.get_size()
        _, n = b.get_size()
        return [m, n]

    return cat_tuned_op(match, inputs, dim, op=L[aten.addmm], shape_of=shape_of)


def cat_tuned_op(match, inputs, dim, *, op, shape_of):
    """
    Memory planning to remove cat.  We can't use the stock memory
    planner since autotuning matmauls needs to know the output layout.
    """
    # TODO(jansel): rewrite this as a bmm?
    if dim < 0:
        dim += len(shape_of(*inputs[0]))
    assert dim in (0, 1)
    notdim = 1 - dim

    new_size = None
    offsets_start = []
    offsets_end = []

    # compute output sizes
    for i in range(len(inputs)):
        shape = shape_of(*inputs[i])
        if new_size is None:
            new_size = shape
        else:
            new_size[notdim] = V.graph.sizevars.guard_equals(
                shape[notdim], new_size[notdim]
            )
            new_size[dim] += shape[dim]
        offsets_start.append(new_size[dim] - shape[dim])
        offsets_end.append(new_size[dim])

    dtype = functools.reduce(
        torch.promote_types, [x.get_dtype() for x in itertools.chain(*inputs)]
    )
    device = inputs[0][0].get_device()
    kernel = ir.ConcatKernel(
        name=None,
        layout=ir.FixedLayout(device, dtype, new_size),
        inputs=[],
    )
    kernel_tensor = ir.TensorBox.create(kernel)

    for i in range(len(inputs)):
        dst = ir.SliceView.create(kernel_tensor, dim, offsets_start[i], offsets_end[i])
        src = op(*inputs[i], layout=dst.get_layout()).data.data
        assert isinstance(src, (ir.ExternKernelOut, ir.TemplateBuffer))
        src.layout = ir.AliasedLayout(dst)
        kernel.inputs.append(src)

    kernel.name = V.graph.register_buffer(kernel)
    kernel.inputs = ir.ConcatKernel.unwrap_storage(kernel.inputs)
    return kernel_tensor


_cat_1 = CallFunction(aten.cat, Arg(), 1, _users=2)


@register_lowering_pattern(
    CallFunction(
        aten.cat,
        [
            _cat_1,
            CallFunction(
                aten.slice,
                CallFunction(aten.slice, _cat_1, 0, 0, 9223372036854775807),
                1,
                0,
                KeywordArg("size"),
            ),
        ],
        1,
    )
)
def cat_slice_cat(match, cat_input, size, dim=1):
    """
    This is an example of a more complex pattern where cat_1 is used
    multiple times inside the pattern.  We fold 2 calls to cat into one.

    Matches:
        cat_1: f32[1024, 4077] = torch.ops.aten.cat.default([add_26, primals_217], 1)
        slice_1: f32[1024, 4077] = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
        slice_2: f32[1024, 19] = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)
        cat_2: f32[1024, 4096] = torch.ops.aten.cat.default([cat_1, slice_2], 1)


    Rewrite to:
        slice_2 = torch.ops.aten.slice.Tensor(add_26, 1, 0, 19)
        cat_2 = torch.ops.aten.cat.default([add_26, primals_217, slice2], 1)
    """
    first, *rest = cat_input
    if V.graph.sizevars.maybe_guard_leq(size, first.get_size()[dim]):
        # fold 2 cats into 1 cat
        return L[aten.cat](
            [
                first,
                *rest,
                L[aten.slice](first, dim, 0, size),
            ],
            dim,
        )
    else:
        # don't expect to hit this case, just fall back
        tmp = L[aten.cat](cat_input, dim)
        return L[aten.cat](
            [
                tmp,
                L[aten.slice](tmp, dim, 0, size),
            ],
            dim,
        )


@register_replacement_pattern(
    CallFunction(
        aten.add,
        CallFunction(aten.mm, Arg(), Arg()),
        KeywordArg("added"),
    ),
    pass_number=2,
)
@register_replacement_pattern(
    CallFunction(
        aten.add,
        KeywordArg("added"),
        CallFunction(aten.mm, Arg(), Arg()),
    ),
    pass_number=2,
)
def addmm(mat1, mat2, added):
    return aten.addmm(added, mat1, mat2)


# This slows things down:
"""
@register_replacement_pattern(
    CallFunction(
        aten.add,
        CallFunction(aten.bmm, Arg(), Arg()),
        KeywordArg("added"),
    ),
    pass_number=3
)
@register_replacement_pattern(
    CallFunction(
        aten.add,
        KeywordArg("added"),
        CallFunction(aten.bmm, Arg(), Arg()),
    ),
    pass_number=3
)
def baddbmm(mat1, mat2, added):
    return aten.baddbmm(added, mat1, mat2)
"""
