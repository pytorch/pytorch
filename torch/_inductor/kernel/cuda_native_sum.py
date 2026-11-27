# mypy: allow-untyped-defs
import math

import sympy

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch._tensor_iterator import reduce_op
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, Max, Min, Where
from torch.utils._sympy.value_ranges import bound_sympy

from .. import ir, lowering as L
from ..select_algorithm import SymbolicGridFn, TritonTemplate
from ..utils import ceildiv
from ..virtualized import V
from .mm_common import load_kernel_template


@SymbolicGridFn
def cuda_native_sum_grid(*args, min):
    meta = args[-1]
    return (min(meta["M"], 65535), 1 if meta["FINAL"] else meta["MAX_CTAS"], 1)


cuda_native_sum_template = TritonTemplate(
    name="cuda_native_sum",
    grid=cuda_native_sum_grid,
    source=load_kernel_template("triton_cuda_native_sum"),
)


@SymbolicGridFn
def cuda_dynamic_sum_grid(leaves, partials, meta, *, min):
    return (1 if meta["FINAL"] else min(leaves * partials, 65535), 1, 1)


cuda_dynamic_sum_template = TritonTemplate(
    name="cuda_dynamic_sum",
    grid=cuda_dynamic_sum_grid,
    source=load_kernel_template("triton_cuda_dynamic_sum"),
)


def cuda_dynamic_sum(x, keepdims, dtype):
    device = x.get_device()
    if device is None:
        raise AssertionError(f"Expected non-None values, got {device=}")
    x.realize()
    if V.graph.is_negative_buffer(x.get_name()):
        x = L.clone(x)
    output_dtype = dtype or x.get_dtype()
    if output_dtype != x.get_dtype() and not (
        x.get_dtype() in (torch.float16, torch.bfloat16)
        and output_dtype == torch.float32
    ):
        x = L.to_dtype(x, output_dtype)
        x.realize()
    (n,) = x.get_size()
    (stride,) = x.get_stride()
    stride = V.graph.sizevars.guard_int(stride)
    itemsize = torch.empty((), dtype=x.get_dtype()).element_size()
    iv = 8 if itemsize == 2 else 4
    props = torch.cuda.get_device_properties(device)
    target = props.multi_processor_count * (
        props.max_threads_per_multi_processor // 512
    )
    partials = Max(target, ceildiv(n, 512 * 256))
    leaf_limit = min((1 << 31) - 1, ((1 << 31) - 2) // max(stride * itemsize, 1) + 1)
    leaves = Where(sympy.Le(n, leaf_limit), 1, 2 * ceildiv(n, leaf_limit))
    branches = []
    for vi in (False, True) if stride == 1 else (False,):
        for bw in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            lower = max(128, bw * iv) if vi else bw
            upper = (
                (bw * 2 * iv if bw < 512 else None)
                if vi
                else (
                    min(128, bw * 2) if stride == 1 else (bw * 2 if bw < 512 else None)
                )
            )
            if upper is None or lower < upper:
                occupancy = props.multi_processor_count * (
                    props.max_threads_per_multi_processor // bw
                )
                branches.append((lower, upper, bw, vi, occupancy))
    acc_dtype = torch.float64 if x.get_dtype() == torch.float64 else torch.float32
    options = dict(
        STRIDE=stride,
        LEAF_LIMIT=leaf_limit,
        IV=iv,
        BRANCHES=branches,
        ACC_TYPE="tl.float64" if acc_dtype == torch.float64 else "tl.float32",
    )
    first = cuda_dynamic_sum_template.generate(
        input_nodes=(x,),
        layout=ir.FixedLayout(device, acc_dtype, [leaves, partials]),
        call_sizes=[leaves, partials],
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=False,
        **options,
    )
    partial = first.output_node()
    V.graph.no_fuse_buffer_names.add(partial.get_name())
    final = cuda_dynamic_sum_template.generate(
        input_nodes=(x, partial),
        layout=ir.FixedLayout(device, output_dtype, [1]),
        call_sizes=[leaves, partials],
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=True,
        **options,
    )
    return L.view(final.output_node(), [1] if keepdims else [])


@SymbolicGridFn
def cuda_symbolic_sum_grid(m, leaves, capacity, partials, meta, *, min):
    return (min(m if meta["FINAL"] else leaves * capacity * partials, 65535), 1, 1)


cuda_symbolic_sum_template = TritonTemplate(
    name="cuda_symbolic_sum",
    grid=cuda_symbolic_sum_grid,
    source=load_kernel_template("triton_cuda_symbolic_sum"),
)


def cuda_symbolic_sum(x, dims, keepdims, dtype):
    r"""Generate native CUDA reduction geometry from symbolic tensor metadata."""
    device = x.get_device()
    if device is None:
        raise AssertionError(f"Expected non-None values, got {device=}")
    output_dtype = dtype or x.get_dtype()
    node = V.graph.current_node
    source = (
        node.args[0].meta.get("val")
        if node is not None and isinstance(node.args[0], torch.fx.Node)
        else None
    )
    if source is not None:
        x = L.constrain_to_fake_tensor(x, source)
    x.realize()
    if V.graph.is_negative_buffer(x.get_name()):
        x = L.clone(x)
        if source is not None:
            source = source.clone()
            x = L.constrain_to_fake_tensor(x, source)
    if output_dtype != x.get_dtype() and not (
        x.get_dtype() in (torch.float16, torch.bfloat16)
        and output_dtype == torch.float32
    ):
        x = L.to_dtype(x, output_dtype)
        if source is not None:
            source = source.to(output_dtype)
            x = L.constrain_to_fake_tensor(x, source)
    x.realize()
    size, strides = list(x.get_size()), list(x.get_stride())
    output_size = [1 if d in dims else value for d, value in enumerate(size)]
    order = tuple(d for d in reversed(range(len(size))) if d in dims) + tuple(
        d for d in reversed(range(len(size))) if d not in dims
    )
    rdims = len(dims)
    n = math.prod(size[d] for d in order[:rdims])
    m = math.prod(size[d] for d in order[rdims:])
    itemsize = torch.empty((), dtype=x.get_dtype()).element_size()
    iv = 8 if itemsize == 2 else 4
    props = torch.cuda.get_device_properties(device)
    output_itemsize = torch.empty((), dtype=output_dtype).element_size()
    limit = (1 << 31) - 1
    output_strides = [
        0
        if d in dims
        else math.prod(size[j] for j in range(d + 1, len(size)) if j not in dims)
        for d in range(len(size))
    ]
    fits = sympy.And(
        sympy.Le(n * m, limit),
        sympy.Le(
            1
            + sum(
                (length - 1) * stride * itemsize
                for length, stride in zip(size, strides)
            ),
            limit,
        ),
        sympy.Le(
            1
            + sum(
                (length - 1) * stride * output_itemsize
                for length, stride in zip(size, output_strides)
            ),
            limit,
        ),
    )
    widths = [
        Max(
            1,
            Min(
                length,
                FloorDiv(
                    limit - 1,
                    len(size)
                    * Max(abs(istride) * itemsize, ostride * output_itemsize, 1),
                )
                + 1,
            ),
        )
        for length, istride, ostride in zip(size, strides, output_strides)
    ]
    byte_leaves = math.prod(
        Where(sympy.Le(length, width), 1, Min(length, 2 * ceildiv(length, width)))
        for length, width in zip(size, widths)
    )
    leaf_numel = math.prod(widths)
    leaves = Where(
        fits,
        1,
        byte_leaves
        * Where(sympy.Le(leaf_numel, limit), 1, 2 * ceildiv(leaf_numel, limit)),
    )
    single_leaf = V.graph.sizevars.statically_known_true(fits)
    if single_leaf:
        leaves = 1
    capacity = Max(1, Min(m, limit // output_itemsize + 1))
    partials = (
        Max(
            props.multi_processor_count
            * (props.max_threads_per_multi_processor // 512),
            ceildiv(n, 512 * 256),
        )
        if m == 1
        else Where(
            sympy.Lt(n, 256),
            1,
            Max(
                props.multi_processor_count
                * (props.max_threads_per_multi_processor // 256),
                ceildiv(n, 256),
            ),
        )
    )

    def leaf_extent(dimensions):
        return Min(
            limit,
            math.prod(
                Min(
                    size[d], FloorDiv(limit - 1, Max(abs(strides[d]) * itemsize, 1)) + 1
                )
                for d in dimensions
            ),
        )

    leaf_n = leaf_extent(dims)
    leaf_m = leaf_extent(order[rdims:])
    partials = Where(
        sympy.Lt(leaf_n, 256), 1, Min(partials, Max(1, ceildiv(leaf_n, 16)))
    )
    capacity = Where(sympy.Eq(partials, 1), 1, capacity)

    def samples(value):
        upper = bound_sympy(value, V.graph.sizevars.shape_env.var_to_range).upper
        return tuple(1 << power for power in range(max(int(upper), 1).bit_length()))

    fixed_outer = (
        len(size) == 2
        and tuple(dims) == (0,)
        and isinstance(m, (int, sympy.Integer))
        and m > 1
        and strides == [m, 1]
        and 1 + (m - 1) * output_itemsize <= limit
    )
    # A contiguous outer reduction only splits its reduced dimension here.
    output_samples = (int(m),) if fixed_outer else samples(leaf_m)
    fast_plans = (False,) if fixed_outer else (False, True)
    cases = OrderedSet()
    for nv in samples(leaf_n):
        for mv in output_samples:
            if nv == 0 or mv == 0:
                continue
            for fast_plan in fast_plans:
                for stride in (1, 2) if fast_plan else (1,):
                    dummy_ins = (stride, stride * max(nv, 2)) if fast_plan else (mv, 1)
                    for plan in _plans(
                        (nv, mv), dummy_ins, (0, 1), 1, nv, mv, itemsize, props
                    ):
                        cases.add((plan[1], plan[2], plan[3], plan[5]))
    acc_dtype = torch.float64 if x.get_dtype() == torch.float64 else torch.float32
    options = dict(
        RANK=len(size),
        WIDTH=1 << (len(size) - 1).bit_length(),
        ITEMSIZE=itemsize,
        OUTPUT_ITEMSIZE=output_itemsize,
        RDIMS=rdims,
        ORDER=order,
        IV=iv,
        CASES=tuple(sorted(cases)),
        SMS=props.multi_processor_count,
        THREADS_PER_SM=props.max_threads_per_multi_processor,
        ACC_TYPE="tl.float64" if acc_dtype == torch.float64 else "tl.float32",
        SINGLE_CTA=V.graph.sizevars.statically_known_equals(partials, 1),
        SINGLE_LEAF=single_leaf,
    )
    first = cuda_symbolic_sum_template.generate(
        input_nodes=(x,),
        layout=ir.FixedLayout(device, acc_dtype, [leaves, capacity, partials]),
        call_sizes=[m, leaves, capacity, partials],
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=False,
        **options,
    )
    partial = first.output_node()
    V.graph.no_fuse_buffer_names.add(partial.get_name())
    final = cuda_symbolic_sum_template.generate(
        input_nodes=(x, partial),
        layout=ir.FixedLayout(device, output_dtype, [m]),
        call_sizes=[m, leaves, capacity, partials],
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=True,
        **options,
    )
    result_size = (
        output_size
        if keepdims
        else [value for d, value in enumerate(size) if d not in dims]
    )
    return L.view(final.output_node(), result_size)


def _coalesce_dimensions(shape, ins, outs):
    # Meta TensorIterator skips coalescing, which CUDA's launch geometry uses.
    sizes, input_strides, output_strides = [], [], []
    for size, input_stride, output_stride in zip(shape, ins, outs):
        if sizes and (
            sizes[-1] == 1
            or size == 1
            or (
                sizes[-1] * input_strides[-1] == input_stride
                and sizes[-1] * output_strides[-1] == output_stride
            )
        ):
            if sizes[-1] == 1:
                input_strides[-1], output_strides[-1] = input_stride, output_stride
            sizes[-1] *= size
        else:
            sizes.append(size)
            input_strides.append(input_stride)
            output_strides.append(output_stride)
    return tuple(sizes), tuple(input_strides), tuple(output_strides)


def _plans(
    shape, ins, outs, rdims, n, m, itemsize, props, input_vec_size=None, max_threads=512
):
    fast = not shape or rdims == len(shape) or ins[0] < ins[rdims]
    d0, d1 = (n, m) if fast else (m, n)
    stride = ins[0 if fast else rdims] if shape else 1
    iv = input_vec_size or (8 if itemsize == 2 else 4)
    vi = stride == 1 and fast and d0 >= 128 and rdims == 1
    max_ov = 1
    if stride == 1 and not fast:
        max_ov = 4
        factors = [shape[rdims]] + [s for d, s in enumerate(ins) if d != rdims]
        for factor in factors:
            while factor % max_ov:
                max_ov //= 2
    plans = []
    for ov in (4, 2, 1):
        if ov > max_ov:
            continue
        dim0 = d0 // (iv if vi else ov)
        maximum = max_threads // ov
        d0p = min(1 << (max(dim0, 1).bit_length() - 1), maximum)
        d1p = min(1 << (max(d1, 1).bit_length() - 1), maximum)
        bw = min(d0p, 32)
        bh = min(d1p, maximum // bw)
        bw = min(d0p, maximum // bh)
        step = bw if fast else 1
        output_step = 1 if fast else bw
        split_y = ceildiv(n, step) >= min(bh * 16, 256)
        if split_y:
            step *= bh
        else:
            output_step *= bh
        target = props.multi_processor_count * (
            props.max_threads_per_multi_processor // (bw * bh)
        )
        grid = ceildiv(m // ov, output_step)
        ctas = 1
        vpt = ceildiv(n, step)
        if split_y and vpt >= 256 and grid <= target:
            ctas = min(max(ceildiv(target, grid), ceildiv(vpt, 256)), ceildiv(vpt, 16))
        lanes = 1 << (step - 1).bit_length()
        plans.append((ov, bw, bh, vi, iv, fast, split_y, ctas, lanes))
    return tuple(plans)


cuda_split_sum_template = TritonTemplate(
    name="cuda_split_sum",
    grid=cuda_native_sum_grid,
    source=load_kernel_template("triton_cuda_split_sum"),
)


def _split_32bit_iterators(shape, ins, outs, itemsize, output_itemsize):
    limit = (1 << 31) - 1
    pending = [(shape, (0,) * len(shape), False)]
    leaves = []
    while pending:
        size, starts, accumulate = pending.pop()
        if math.prod(size) <= limit and all(
            1 + sum((n - 1) * stride * width for n, stride in zip(size, strides))
            <= limit
            for strides, width in ((ins, itemsize), (outs, output_itemsize))
        ):
            leaves.append((size, starts, accumulate))
            continue
        dim, extent = -1, -1
        for d in reversed(range(len(size))):
            for strides, width in ((outs, output_itemsize), (ins, itemsize)):
                candidate = (size[d] - 1) * abs(strides[d]) * width
                if candidate > extent:
                    dim, extent = d, candidate
        first = size[dim] // 2
        left, right, offset = list(size), list(size), list(starts)
        left[dim], right[dim] = first, size[dim] - first
        offset[dim] += first
        pending.append((tuple(right), tuple(offset), accumulate or outs[dim] == 0))
        pending.append((tuple(left), starts, accumulate))
    return leaves


def _cuda_split_sum(
    x, shape, ins, outs, rdims, leaves, output_dtype, result_size, props
):
    partial_dtype = torch.float64 if x.get_dtype() == torch.float64 else torch.float32
    partials, configurations = [], []
    for size, starts, accumulate in leaves:
        n, m = math.prod(size[:rdims]), math.prod(size[rdims:])
        plans = _plans(
            size,
            ins,
            outs,
            rdims,
            n,
            m,
            torch.empty((), dtype=x.get_dtype()).element_size(),
            props,
        )
        max_ctas = max(plan[7] for plan in plans)
        offset = sum(start * stride for start, stride in zip(starts, ins))
        first = cuda_native_sum_template.generate(
            input_nodes=(x,),
            layout=ir.FixedLayout(x.get_device(), partial_dtype, [m, max_ctas]),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            SHAPE=size,
            ISTRIDES=ins,
            OSTRIDES=outs,
            RDIMS=rdims,
            N=n,
            M=m,
            PLANS=plans,
            MAX_CTAS=max_ctas,
            INPUT_OFFSET=offset,
            ITEMSIZE=torch.empty((), dtype=x.get_dtype()).element_size(),
            ACC_TYPE="tl.float64" if partial_dtype == torch.float64 else "tl.float32",
            FINAL=False,
        )
        partials.append(first.output_node())
        configurations.append((size, starts, accumulate, n, m, plans, max_ctas, offset))
    m = math.prod(shape[rdims:])
    final = cuda_split_sum_template.generate(
        input_nodes=(x, *partials),
        layout=ir.FixedLayout(x.get_device(), output_dtype, [m]),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        SHAPE=shape,
        ISTRIDES=ins,
        OSTRIDES=outs,
        RDIMS=rdims,
        LEAVES=configurations,
        PARTIAL_NAMES=tuple("P" + str(i) for i in range(len(partials))),
        M=m,
        FINAL=True,
        MAX_CTAS=1,
        ITEMSIZE=torch.empty((), dtype=x.get_dtype()).element_size(),
        ACC_TYPE="tl.float64" if partial_dtype == torch.float64 else "tl.float32",
    )
    return L.view(final.output_node(), result_size)


def cuda_native_sum(x, axis, keepdims, dtype):
    r"""Generate CUDA sum kernels with eager's accumulation order."""
    dims = [axis] if isinstance(axis, int) else list(axis or range(len(x.get_size())))
    dims = (
        [(dim + len(x.get_size())) % len(x.get_size()) for dim in dims]
        if x.get_size()
        else []
    )
    output_dtype = dtype or x.get_dtype()
    if free_unbacked_symbols(x.get_size()):
        if len(x.get_size()) == 1 and not free_unbacked_symbols(x.get_stride()):
            return cuda_dynamic_sum(x, keepdims, dtype)
        return cuda_symbolic_sum(x, dims, keepdims, dtype)
    node = V.graph.current_node
    if node is not None and isinstance(node.args[0], torch.fx.Node):
        x = L.constrain_to_fake_tensor(x, node.args[0].meta.get("val"))
    x.realize()
    negative = V.graph.is_negative_buffer(x.get_name())
    if free_unbacked_symbols(x.get_stride()):
        return cuda_symbolic_sum(x, dims, keepdims, dtype)
    size = [V.graph.sizevars.guard_int(s) for s in x.get_size()]
    stride = [V.graph.sizevars.guard_int(s) for s in x.get_stride()]
    with unset_fake_temporarily():
        source = torch.empty_strided(size, stride, device="meta", dtype=x.get_dtype())
        if negative:
            source = source.clone()
            x = L.constrain_to_fake_tensor(L.clone(x), source)
        if output_dtype != x.get_dtype() and not (
            x.get_dtype() in (torch.float16, torch.bfloat16)
            and output_dtype == torch.float32
        ):
            source = source.to(output_dtype)
            x = L.constrain_to_fake_tensor(L.to_dtype(x, output_dtype), source)
        output_size = [1 if d in dims else s for d, s in enumerate(size)]
        output = torch.empty(output_size, device="meta", dtype=output_dtype)
        iterator = reduce_op(output, source)
    x.realize()
    shape = tuple(iterator.shape)
    ins = tuple(s // source.element_size() for s in iterator.strides(1))
    outs = tuple(s // output.element_size() for s in iterator.strides(0))
    shape, ins, outs = _coalesce_dimensions(shape, ins, outs)
    rdims = next((d for d, stride in enumerate(outs) if stride), len(outs))
    m = math.prod(output_size)
    result_size = (
        output_size if keepdims else [s for d, s in enumerate(size) if d not in dims]
    )
    if m == 0 or source.numel() == 0:
        return L.full(result_size, 0, device=x.get_device(), dtype=output_dtype)
    n = source.numel() // m
    props = torch.cuda.get_device_properties(x.get_device())
    leaves = _split_32bit_iterators(
        shape, ins, outs, source.element_size(), output.element_size()
    )
    if len(leaves) > 1:
        return _cuda_split_sum(
            x, shape, ins, outs, rdims, leaves, output_dtype, result_size, props
        )
    plans = _plans(shape, ins, outs, rdims, n, m, source.element_size(), props)
    max_ctas = max(plan[7] for plan in plans)
    partial_dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
    options = dict(
        SHAPE=shape,
        INPUT_OFFSET=0,
        ISTRIDES=ins,
        OSTRIDES=outs,
        RDIMS=rdims,
        N=n,
        M=m,
        PLANS=plans,
        MAX_CTAS=max_ctas,
        ITEMSIZE=source.element_size(),
        ACC_TYPE="tl.float64" if source.dtype == torch.float64 else "tl.float32",
    )
    first = cuda_native_sum_template.generate(
        input_nodes=(x,),
        layout=ir.FixedLayout(x.get_device(), partial_dtype, [m, max_ctas]),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=False,
        **options,
    )
    partial = first.output_node()
    final = cuda_native_sum_template.generate(
        input_nodes=(x, partial),
        layout=ir.FixedLayout(x.get_device(), output_dtype, [m]),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=True,
        **options,
    )
    return L.view(final.output_node(), result_size)
