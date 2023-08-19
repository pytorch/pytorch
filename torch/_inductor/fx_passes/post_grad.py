import functools
import itertools
import logging
import operator
from collections import defaultdict, namedtuple
from typing import List, Optional, Union

from sympy import Expr

import torch
import torch._inductor as inductor
from torch.utils._pytree import tree_map

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
    remove_noop_ops,
    stable_topological_sort,
)
from ..virtualized import V
from .group_batch_fusion import group_batch_fusion_post_grad_passes


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

# First pass_patterns[0] are applied, then [1], then [2]
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]
# patterns applied only in inference
inference_patterns = PatternMatcherPass()

def hash_node(node):
    return (node, node.target, node.args, node.kwargs)

def create_hash(graph):
    old_hash = set()
    for node in graph.nodes:
        old_hash.add(hash_node(node))
    return old_hash

def get_storage(t):
    return t.untyped_storage()._cdata

def get_node_storage(node):
    if not isinstance(node.meta['val'], torch.Tensor):
        return None
    if not torch._C._has_storage(node.meta['val']):
        return None
    return get_storage(node.meta["val"])

def incremental_fake_tensor_update(graph: torch.fx.Graph, old_hash):
    """
    Not fully featured yet. Mainly added to ensure that reinplacing pass is
    sound.

    Todo things:
    1. We probably want to persist this across more passes.
    """
    processed = set()
    existing_tensors = defaultdict(int)
    for node in graph.nodes:
        existing_tensors[get_node_storage(node)] += 1

    def is_fake_tensor_same(new, old):
        if (
            new.shape != old.shape
            or new.stride() != old.stride()
        ):
            return False
        if get_storage(new) == get_storage(old):
            return True
        if existing_tensors[get_storage(old)] == 1 and get_storage(new) not in existing_tensors:
            return True
        return False

    for node in graph.nodes:
        if hash_node(node) in old_hash:
            continue

        def get_fake_tensor(x):
            if isinstance(x, torch.fx.Node):
                return x.meta['val']
            return x

        if node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload):
            processing = [node]
            while len(processing) > 0:
                updating_node = processing.pop()
                if updating_node in processed:
                    continue
                if node.op != 'call_function' or not isinstance(node.target, torch._ops.OpOverload):
                    continue

                args, kwargs = tree_map(get_fake_tensor, (updating_node.args, updating_node.kwargs))
                new_fake_tensor = updating_node.target(*args, **kwargs)
                if is_fake_tensor_same(new_fake_tensor, updating_node.meta['val']):
                    continue
                updating_node.meta['val'] = new_fake_tensor
                processed.add(updating_node)
                for user in updating_node.users:
                    processing.append(user)

            old_hash.add(hash_node(updating_node))


def post_grad_passes(gm: torch.fx.GraphModule, is_inference: bool):
    """
    Passes that run on after grad.  This is called once on the forwards
    graph and once on the backwards graph.

    The IR here has been normalized and functionalized.
    """
    if config.dce:
        # has some issues with mutation in inference mode
        gm.graph.eliminate_dead_code()

    if is_inference and config.reordering:
        reorder_for_locality(gm.graph)

    # old_hash = create_hash(gm.graph)
    if config.pattern_matcher:
        lazy_init()

        group_batch_fusion_post_grad_passes(gm.graph)
        remove_noop_ops(gm.graph)

        for patterns in pass_patterns:
            patterns.apply(gm.graph)
        if is_inference:
            inference_patterns.apply(gm.graph)

    stable_topological_sort(gm.graph)

    # Keep this last, since it introduces
    # mutation.
    # incremental_fake_tensor_update(gm.graph, old_hash)
    reinplace_scatters(gm.graph)
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
    return inductor.kernel.mm_plus_mm.tuned_mm_plus_mm(mat1, mat2, mat3, mat4)  # type: ignore[attr-defined]


def cuda_and_enabled_mixed_mm(match):
    return (config.use_mixed_mm or config.force_mixed_mm) and getattr(
        match.kwargs["mat1"].meta.get("val"), "is_cuda", False
    )


def cuda_and_enabled_mixed_mm_and_not_int8(match):
    return (
        cuda_and_enabled_mixed_mm(match)
        and getattr(match.kwargs["mat1"].meta.get("val"), "is_cuda", False)
        and getattr(match.kwargs["mat2"].meta.get("val"), "dtype", torch.int8)
        != torch.int8
    )  # bitshift numerics in triton and pytorch don't match for torch.int8


"""
    this is intended to be used to unpack a [K,N] int4 tensor from a [K/2, N] uint4x2 tensor
    (where the int4 and uint4x2 are represented with int8 and uint8 respectively)
    where every other row of the int4 is packed with the row above it as:
    uint4x2[k,n] = (8+int4[2*k,n])+(8+int4[2*k+1,n])<<4

    unpack formulas:
    int4[2*k,n]=(uint4x2[k,n] & 0xF) - 8
    int4[2*k+1,n]=(uint4x2[k,n] >> 4) - 8

    thus matching on unpack formula:
    torch.mm(mat1, torch.cat((mat2 & 0xF, mat2>>4),1).reshape(mat2_mm_shape).to(mat2_dtype).sub(8))

    note: although the unpack formula in pytorch and the triton kernel is designed for a uint8 mat2, the behavior
    of the kernel matches the pytorch formula for all dtypes except torch.int8
    where the bitwise numerics in triton do not match those in pytorch.
"""


@register_lowering_pattern(
    CallFunction(
        aten.mm.default,
        KeywordArg("mat1"),
        CallFunction(
            aten.sub.Tensor,
            CallFunction(
                prims.convert_element_type.default,
                CallFunction(
                    aten.reshape.default,
                    CallFunction(
                        aten.cat.default,
                        ListOf(
                            CallFunction(
                                aten.bitwise_and.Scalar,
                                KeywordArg("mat2"),
                                0xF,
                            ),
                            CallFunction(
                                aten.__rshift__.Scalar,
                                KeywordArg("mat2"),
                                4,
                            ),
                        ),
                        1,
                    ),
                    KeywordArg("mat2_mm_shape"),
                ),
                KeywordArg("mat2_dtype"),
            ),
            8,
        ),
    ),
    extra_check=cuda_and_enabled_mixed_mm_and_not_int8,
)
def uint4x2_mixed_mm(match: Match, mat1, mat2, mat2_mm_shape, mat2_dtype):
    return inductor.kernel.unpack_mixed_mm.tuned_uint4x2_mixed_mm(
        mat1, mat2, mat2_mm_shape, mat2_dtype
    )


"""
    torch.mm(mat1, mat2.to(mat2_dtype))
"""


@register_lowering_pattern(
    CallFunction(
        aten.mm,
        KeywordArg("mat1"),
        CallFunction(
            prims.convert_element_type.default,
            KeywordArg("mat2"),
            KeywordArg("mat2_dtype"),
        ),
    ),
    extra_check=cuda_and_enabled_mixed_mm,
)
def mixed_mm(match: Match, mat1, mat2, mat2_dtype):
    return inductor.kernel.mm.tuned_mixed_mm(mat1, mat2, mat2_dtype)


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

    new_size: Optional[Union[List[Expr], List[int]]] = None
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

    assert new_size is not None
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
                _cat_1,
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


def reinplace_scatters(graph):
    """
    Reinplaces scatter operations in easy cases where the node being mutated
    is only used by the scatter (users == 1), and the node being mutated
    shares storage with no other nodes.

    Also handles input mutations when there is a corresponding copy node.
    """

    copy_nodes = {}
    mutated_inputs = set()
    storage_to_nodes = defaultdict(list)
    for node in reversed(graph.nodes):
        if isinstance(node.meta["val"], torch.Tensor):
            storage_to_nodes[get_node_storage(node)].append(node)
        if node.target == aten.copy_.default:
            copy_nodes[(node.args[0], node.args[1])] = node
            mutated_inputs.add(node.args[0])
        elif node.op == "output":
            pass

    InplaceableOp = namedtuple("InplaceableOp", ["inplace_op", "mutated_arg"])

    inplaceable_ops = {
        aten.index_put.default: InplaceableOp(aten.index_put_.default, 0),
    }
    for node in graph.nodes:
        if inplaceable_op := inplaceable_ops.get(node.target, False):
            mutated_arg = node.args[inplaceable_op.mutated_arg]
            if get_node_storage(mutated_arg) is None:
                continue
            shared_view_nodes = storage_to_nodes[get_node_storage(mutated_arg)]
            if mutated_arg.op == "placeholder":
                if not (copy_node := copy_nodes.get((mutated_arg, node), False)):
                    continue

                if (
                    len(shared_view_nodes) > 2
                ):  # Arg aliases another node other than copy_
                    continue

                # Check for any uses other than current node and copy_ epilogue
                if len(mutated_arg.users) > 2:
                    continue

                graph.erase_node(copy_node)
                node.target = inplaceable_op.inplace_op
            else:
                # NB: This condition could be relaxed if none of the aliases
                # are used after this mutation op. But that's trickier.
                if len(shared_view_nodes) > 1:  # Arg aliases another node
                    continue
                if len(mutated_arg.users) > 1:  # Arg used somewhere else
                    continue
                node.target = inplaceable_op.inplace_op


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


def is_pointwise_use(use):
    if not use.op == "call_function":
        return False

    if not (
        isinstance(use.target, torch._ops.OpOverload) or use.target is operator.getitem
    ):
        return False

    if use.target is operator.getitem or use.target.is_view:
        return all(is_pointwise_use(u) for u in use.users)

    return torch.Tag.pointwise in use.target.tags


@register_graph_pattern(
    CallFunction(aten.addmm, Arg(), Arg(), Arg()),
    pass_dict=pass_patterns[2],
)
def unfuse_bias_add_to_pointwise(match: Match, inp, mat1, mat2):
    if not inp.meta["val"].is_cuda:
        return

    output = match.output_node()
    if not all(is_pointwise_use(use) for use in output.users):
        return

    def repl(inp, x1, x2):
        return x1 @ x2 + inp

    with V.fake_mode:
        match.replace_by_example(repl, [inp, mat1, mat2])
