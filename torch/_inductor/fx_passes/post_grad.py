import functools
import itertools
import logging
import operator

import torch
import torch._inductor as inductor
from .. import config, ir, pattern_matcher

from ..lowering import lowerings as L
from ..pattern_matcher import (
    _return_true,
    Arg,
    CallFunction,
    filter_nodes,
    get_arg_value,
    Ignored,
    init_once_fakemode,
    KeywordArg,
    ListOf,
    Match,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)
from ..virtualized import V


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

# First pass_patterns[0] are applied, then [1], then [2]
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]


def post_grad_passes(gm: torch.fx.GraphModule, locality_reorder: bool):
    """
    Passes that run on after grad.  This is called once on the forwards
    graph and once on the backwards graph.

    The IR here has been normalized and functionalized.
    """
    if config.dce:
        # has some issues with mutation in inference mode
        gm.graph.eliminate_dead_code()

    if locality_reorder:
        reorder_for_locality(gm.graph)

    if config.pattern_matcher:
        lazy_init()

        for patterns in pass_patterns:
            patterns.apply(gm.graph)

    stable_topological_sort(gm.graph)
    gm.recompile()
    gm.graph.lint()


@init_once_fakemode
def lazy_init():
    if torch._C._has_mkldnn:
        from .mkldnn_fusion import _mkldnn_fusion_init

        _mkldnn_fusion_init()

    from .quantization import register_quantization_lowerings

    register_quantization_lowerings()


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

    # only reorder nodes before the first copy_ in the graph.
    # copy_ will appear at the end of functionalized graphs when there is mutation on inputs,
    # and this reordering doesnt work well with mutation
    first_copy = next(
        (
            node
            for node in graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.copy_.default
        ),
        None,
    )
    past_mutating_epilogue = True if first_copy is None else False

    for node in reversed(graph.nodes):
        seen_nodes.add(node)
        if not past_mutating_epilogue:
            past_mutating_epilogue = node is first_copy
            continue

        torch.fx.map_arg((node.args, node.kwargs), visit)


def register_lowering_pattern(pattern, extra_check=_return_true, pass_number=1):
    """
    Register an aten to inductor IR replacement pattern
    """
    return pattern_matcher.register_lowering_pattern(
        pattern, extra_check, pass_dict=pass_patterns[pass_number]
    )


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


@register_graph_pattern(
    CallFunction(
        aten.cumsum.default,
        CallFunction(
            torch.ops.aten.full.default,
            [Arg(), Arg()],
            1,
            dtype=KeywordArg("dtype"),
            layout=Ignored(),
            device=KeywordArg("device"),
            pin_memory=False,
            _users=MULTIPLE,
        ),
        1,
        _users=MULTIPLE,
    ),
    pass_dict=pass_patterns[1],
)
def pointless_cumsum_replacement(match: Match, size0, size1, device, dtype):
    """Based on a pattern in OPTForCausalLM"""

    def repl(size0, size1):
        return torch.arange(1, size1 + 1, device=device, dtype=dtype).expand(
            size0, size1
        )

    # only replace the output node, not all nodes
    match.nodes = [match.output_node()]
    with V.fake_mode:
        match.replace_by_example(repl, [size0, size1])


def shape_of_mm(a, b):
    m, _ = a.get_size()
    _, n = b.get_size()
    return [m, n]


@register_lowering_pattern(
    CallFunction(aten.cat, ListOf(CallFunction(aten.mm, Arg(), Arg())), Arg()),
)
def cat_mm(match, inputs, dim):
    return cat_tuned_op(match, inputs, dim, op=L[aten.mm], shape_of=shape_of_mm)


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
    Memory planning to remove cat. We can't use the stock memory
    planner since autotuning matmuls needs to know the output layout.
    """
    if len(inputs) == 1:
        return op(*inputs[0])

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
    # Optimization is optional, because we can just not fold the cat
    # size should be within first.get_size()[dim] such that the optimization is valid.
    # For negative `end`, we currently fallback to not optimizing.
    if size >= 0 and V.graph.sizevars.statically_known_leq(size, first.get_size()[dim]):
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


@register_lowering_pattern(
    CallFunction(
        aten.add,
        CallFunction(aten.mm, Arg(), Arg()),
        KeywordArg("inp"),
    ),
    pass_number=2,
)
@register_lowering_pattern(
    CallFunction(
        aten.add,
        KeywordArg("inp"),
        CallFunction(aten.mm, Arg(), Arg()),
    ),
    pass_number=2,
)
def addmm(match, mat1, mat2, inp):
    if isinstance(inp, ir.TensorBox):
        inp_shape = inp.get_size()
        matched = len(inp_shape) <= 2
        mm_shape = shape_of_mm(mat1, mat2)
        for i, m in zip(inp_shape, mm_shape):
            matched &= i == 1 or i == m
    else:  # inp is a Number
        matched = False
    if matched:
        return L[aten.addmm](inp, mat1, mat2)
    else:
        return L[aten.add](inp, L[aten.mm](mat1, mat2))


def is_valid_splitwithsizes_cat(match):
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    get_item_nodes = filter_nodes(match.nodes, operator.getitem)
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    split_node, cat_node = split_nodes[0], cat_nodes[0]
    # The dim of split and cat should match for passthrough
    if get_arg_value(split_node, 2, "dim") != get_arg_value(cat_node, 1, "dim"):
        return False
    get_item_args = {
        get_arg_value(get_item_node, 1) for get_item_node in get_item_nodes
    }
    assert None not in get_item_args
    split_sizes = get_arg_value(split_node, 1, "split_sizes")
    # All parts of split should be included in the cat
    if get_item_args != set(range(len(split_sizes))):
        return False
    # The order of get_item_args should same with cat_node used.
    # For example, if the split_node like split_with_sizes(input, [2, 2, 3], 1),
    # the cat node should be like cat([get_item(0), get_item(1), get_item(2)], 1).
    cat_items_args_order = [
        get_arg_value(item_node, 1) for item_node in get_arg_value(cat_node, 0)
    ]
    if cat_items_args_order != list(range(len(split_sizes))):
        return False

    return True


@register_lowering_pattern(
    CallFunction(
        aten.cat,
        ListOf(
            CallFunction(
                operator.getitem,
                CallFunction(
                    aten.split_with_sizes,
                    KeywordArg("input_"),
                    Ignored(),
                    Ignored(),
                    _users=MULTIPLE,
                ),
                Ignored(),
            ),
        ),
        Ignored(),
    ),
    pass_number=2,
    extra_check=is_valid_splitwithsizes_cat,
)
def splitwithsizes_cat_replace(match, input_):
    return input_


def view_to_reshape(gm):
    """
    Replace view ops in the GraphModule to reshape ops.
    """
    for nd in gm.graph.nodes:
        if nd.target == torch.ops.aten.view.default:
            nd.target = torch.ops.aten.reshape.default
