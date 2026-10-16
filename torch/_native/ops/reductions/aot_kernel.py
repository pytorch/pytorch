"""CuTeDSL builders for native-AOT reduction callables."""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Int32, Int64

from . import _aot_data as data, kernel_general as general, tile, traits


_DTYPES = {
    "bool": cutlass.Boolean,
    "float16": cutlass.Float16,
    "bfloat16": cutlass.BFloat16,
    "float32": cutlass.Float32,
    "int64": cutlass.Int64,
}
_DTYPE_TAG = {
    "bool": "bool",
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "int64": "i64",
}
_TRAITS = {
    "sum": lambda: traits.SumOps(acc=Float32),
    "mean": lambda: traits.MeanOps(acc=Float32),
    "amax": lambda: traits.AMaxOps(acc=Float32),
    "amin": lambda: traits.AMinOps(acc=Float32),
    "prod": lambda: traits.ProdOps(acc=Float32),
    "argmax": lambda: traits.ArgMaxOps(acc=Float32, idx=Int32),
    "argmin": lambda: traits.ArgMinOps(acc=Float32, idx=Int32),
    "maxdim": lambda: traits.MaxDimOps(acc=Float32, idx=Int32),
    "mindim": lambda: traits.MinDimOps(acc=Float32, idx=Int32),
    "varmean": lambda: traits.VarMeanOps(acc=Float32),
    "var0": lambda: traits.WelfordOps(correction=0, acc=Float32),
    "var1": lambda: traits.WelfordOps(correction=1, acc=Float32),
    "std0": lambda: traits.WelfordOps(correction=0, take_sqrt=True, acc=Float32),
    "std1": lambda: traits.WelfordOps(correction=1, take_sqrt=True, acc=Float32),
    "varmean0": lambda: traits.VarMeanOps(correction=0, acc=Float32),
    "varmean1": lambda: traits.VarMeanOps(correction=1, acc=Float32),
    "stdmean0": lambda: traits.VarMeanOps(correction=0, take_sqrt=True, acc=Float32),
    "stdmean1": lambda: traits.VarMeanOps(correction=1, take_sqrt=True, acc=Float32),
    "vnorm1": lambda: traits.NormOps(1.0, acc=Float32),
    "vnorm2": lambda: traits.NormOps(2.0, acc=Float32),
    "vnorminf": lambda: traits.AbsMaxOps(acc=Float32),
    "vnormninf": lambda: traits.AbsMinOps(acc=Float32),
    "all": lambda: traits.AllOps(acc=Float32),
    "any": lambda: traits.AnyOps(acc=Float32),
    "cnz": lambda: traits.CountNonzeroOps(acc=Float32),
    "aminmax": lambda: traits.AMinMaxOps(acc=Float32),
}


@cute.jit
def _itree_load_vec(trait, mX, row, base, n, vec: cutlass.Constexpr):
    ident = trait.init()
    vals = []
    for i in cutlass.range_constexpr(vec):
        col = base + Int32(i)
        valid = col < n
        safe_col = col if valid else Int32(0)
        leaf = trait.leaf(mX[row, safe_col], col)
        vals.append(tuple(leaf[f] if valid else ident[f] for f in range(trait.nfields)))
    return tile._reduce_vec(vals, vec, trait.combine)


@cute.jit
def _itree_warp_fold(
    trait,
    mX,
    row,
    base,
    n,
    active_loads,
    vec: cutlass.Constexpr,
    loads: cutlass.Constexpr,
):
    ident = trait.init()
    tree = []
    wle = const_expr(32 * vec)
    for load in cutlass.range_constexpr(loads):
        value = _itree_load_vec(
            trait,
            mX,
            row,
            base + Int32(load * wle),
            n,
            vec,
        )
        live = Int32(load) < active_loads
        value = tuple(value[f] if live else ident[f] for f in range(trait.nfields))
        value = traits.warp_reduce(trait, value, 32, ascending=True)
        tile._streaming_push(tree, value, load, 8, trait.combine)
    return tree[0]


@cute.jit
def _itree_merge_warps(trait, value, buffers, lane, warp_id, warps):
    if lane == Int32(0):
        for f in cutlass.range_constexpr(trait.nfields):
            buffers[f][warp_id] = value[f]
    cute.arch.barrier()
    ident = trait.init()
    live = lane < warps
    got = tuple(buffers[f][lane if live else Int32(0)] for f in range(trait.nfields))
    value = tuple(got[f] if live else ident[f] for f in range(trait.nfields))
    return traits.warp_reduce(trait, value, 32, ascending=True)


@cute.jit
def _itree_store(trait, mOuts, row, value, n):
    result = trait.project(value, trait.acc(n))
    if const_expr(trait.nouts == 1):
        mOuts[0][row] = mOuts[0].element_type(result)
    else:
        for k in cutlass.range_constexpr(trait.nouts):
            mOuts[k][row] = mOuts[k].element_type(result[k])


class _InnerTreeMultirow:
    def __init__(self, trait, vec, loads):
        self.trait = trait
        self.vec = vec
        self.loads = loads

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mOuts: list, n: Int32, rows: Int32):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        row = bx * Int32(128) + tx
        live = row < rows
        safe_row = row if live else Int32(0)
        trait = self.trait
        tree = []
        for load in cutlass.range_constexpr(self.loads):
            value = _itree_load_vec(
                trait,
                mX,
                safe_row,
                Int32(load * self.vec),
                n,
                const_expr(self.vec),
            )
            tile._streaming_push(tree, value, load, 6, trait.combine)
        if live:
            _itree_store(trait, mOuts, row, tree[0], n)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mOuts: list,
        n: Int32,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(mX, mOuts, n, rows).launch(
            grid=[cute.ceil_div(rows, 128), 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )


class _InnerTreeLooped:
    def __init__(self, trait, vec, loads):
        self.trait = trait
        self.vec = vec
        self.loads = loads

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mOuts: list,
        n: Int32,
        warps: Int32,
        batches: Int32,
    ):
        lane, warp_id, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        trait = self.trait
        smem = cutlass.utils.SmemAllocator()
        buffers = [
            smem.allocate_tensor(dt, cute.make_layout(16), byte_alignment=8)
            for dt in trait.fdtypes
        ]
        ident = trait.init()
        final = ident
        wle = const_expr(32 * self.vec)
        warp_chunk = const_expr(self.loads * wle)
        batch_elements = Int32(warp_chunk) * warps
        for batch in cutlass.range_constexpr(3):
            batch_live = Int32(batch) < batches
            remaining = Int32(0)
            if batch_live:
                remaining = n - Int32(batch) * batch_elements
                if remaining > batch_elements:
                    remaining = batch_elements
            warp_offset = warp_id * Int32(warp_chunk)
            if warp_offset > remaining:
                warp_offset = remaining
            elements = remaining - warp_offset
            if elements > Int32(warp_chunk):
                elements = Int32(warp_chunk)
            active_loads = (elements + Int32(wle - 1)) // Int32(wle)
            base = Int32(batch) * batch_elements + warp_offset + lane * Int32(self.vec)
            active_loads = active_loads if batch_live & (warp_id < warps) else Int32(0)
            value = _itree_warp_fold(
                trait,
                mX,
                row,
                base,
                n,
                active_loads,
                const_expr(self.vec),
                const_expr(self.loads),
            )
            value = _itree_merge_warps(trait, value, buffers, lane, warp_id, warps)
            combined = trait.combine(final, value)
            final = tuple(
                combined[f] if batch_live else final[f] for f in range(trait.nfields)
            )
            if const_expr(batch < 2):
                cute.arch.barrier()
        if lane == Int32(0):
            if warp_id == Int32(0):
                _itree_store(trait, mOuts, row, final, n)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mOuts: list,
        n: Int32,
        rows: Int32,
        warps: Int32,
        batches: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(mX, mOuts, n, warps, batches).launch(
            grid=[rows, 1, 1],
            block=[32, 16, 1],
            stream=stream,
        )


class _InnerTreeSplit:
    def __init__(self, trait, vec, loads):
        self.trait = trait
        self.vec = vec
        self.loads = loads

    @cute.kernel
    def partial(
        self,
        mX: cute.Tensor,
        mParts: list,
        n: Int32,
        batches: Int32,
    ):
        lane, warp_id, _ = cute.arch.thread_idx()
        pair, _, _ = cute.arch.block_idx()
        row = pair // batches
        batch = pair % batches
        trait = self.trait
        smem = cutlass.utils.SmemAllocator()
        buffers = [
            smem.allocate_tensor(dt, cute.make_layout(16), byte_alignment=8)
            for dt in trait.fdtypes
        ]
        wle = const_expr(32 * self.vec)
        warp_chunk = const_expr(self.loads * wle)
        batch_elements = const_expr(warp_chunk * 16)
        remaining = n - batch * Int32(batch_elements)
        if remaining > Int32(batch_elements):
            remaining = Int32(batch_elements)
        warp_offset = warp_id * Int32(warp_chunk)
        if warp_offset > remaining:
            warp_offset = remaining
        elements = remaining - warp_offset
        if elements > Int32(warp_chunk):
            elements = Int32(warp_chunk)
        active_loads = (elements + Int32(wle - 1)) // Int32(wle)
        base = batch * Int32(batch_elements) + warp_offset + lane * Int32(self.vec)
        value = _itree_warp_fold(
            trait,
            mX,
            row,
            base,
            n,
            active_loads,
            const_expr(self.vec),
            const_expr(self.loads),
        )
        value = _itree_merge_warps(trait, value, buffers, lane, warp_id, Int32(16))
        if lane == Int32(0):
            if warp_id == Int32(0):
                for f in cutlass.range_constexpr(trait.nfields):
                    mParts[f][pair] = trait.fdtypes[f](value[f])

    @cute.kernel
    def accumulate(
        self,
        mParts: list,
        mOuts: list,
        rows: Int32,
        batches: Int32,
        n: Int32,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        row = bx * Int32(128) + tx
        live = row < rows
        safe_row = row if live else Int32(0)
        trait = self.trait
        combine = trait.combine
        base = safe_row * batches
        value = tuple(trait.fdtypes[f](mParts[f][base]) for f in range(trait.nfields))
        for batch in cutlass.range(1, batches):
            value = combine(
                value,
                tuple(
                    trait.fdtypes[f](mParts[f][base + batch])
                    for f in range(trait.nfields)
                ),
            )
        if live:
            _itree_store(trait, mOuts, row, value, n)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mParts: list,
        mOuts: list,
        rows: Int32,
        batches: Int32,
        n: Int32,
        stream: cuda.CUstream,
    ):
        self.partial(mX, mParts, n, batches).launch(
            grid=[rows * batches, 1, 1],
            block=[32, 16, 1],
            stream=stream,
        )
        self.accumulate(mParts, mOuts, rows, batches, n).launch(
            grid=[cute.ceil_div(rows, 128), 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )


class _RowAdapter:
    def __init__(self, op):
        self.op = op

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mOuts: list,
        nchunks: Int32,
        nwaves: Int32,
        n: Int32,
        stream: cuda.CUstream,
    ):
        self.op(
            [mX],
            mOuts,
            nchunks,
            nwaves,
            n,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            stream,
        )


class _XctaAdapter:
    def __init__(self, stage1, stage2):
        self.stage1 = stage1
        self.stage2 = stage2

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mParts: list,
        mOuts: list,
        rows: Int32,
        partials: Int32,
        outputs: Int32,
        n: Int64,
        stream: cuda.CUstream,
    ):
        stage1 = self.stage1
        nchunks = Int32(data.XCTA_SUBROW // stage1.vec)
        nwaves = Int32(-(-(data.XCTA_SUBROW // stage1.vec) // stage1.threads_per_row))
        stage1.kernel(
            [mX],
            mParts,
            None,
            nchunks,
            nwaves,
            n,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ).launch(
            grid=[cute.ceil_div(rows, const_expr(stage1.rows_per_block)), 1, 1],
            block=[const_expr(stage1.threads_per_block), 1, 1],
            stream=stream,
        )

        stage2 = self.stage2
        stage2.tile.kernel(
            mParts,
            mOuts,
            None,
            partials,
            None,
            n,
            None,
            None,
            [cute.FastDivmodDivisorV2(partials)],
            [Int64(1)],
            [cute.FastDivmodDivisorV2(outputs)],
            [Int64(partials)],
            Int64(0),
            Int64(partials),
        ).launch(
            grid=[outputs, 1, 1],
            block=[const_expr(stage2.block), 1, 1],
            stream=stream,
        )


def _build_row(spec):
    dtype = _DTYPES[spec["dtype"]]
    trait = _TRAITS[spec["trait"]]()
    out_dtypes = _itree_out_dtypes(spec, trait)
    threads_per_row, threads_per_block = data.ROW_CLASSES[spec["row_class"]]
    vec = 128 // dtype.width
    op = tile.TileReduce(
        trait,
        dtype,
        "row",
        vec,
        threads_per_row=threads_per_row,
        threads_per_block=threads_per_block,
        nouts=trait.nouts,
        final=True,
        unroll=4,
        use_tma=False,
        order="linear",
    )
    m = cute.sym_int()
    n = cute.sym_int(divisibility=vec)
    x_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (m, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fakes = [
        cute.runtime.make_fake_tensor(
            out_dtype,
            (m,),
            stride=(1,),
            assumed_align=16,
        )
        for out_dtype in out_dtypes
    ]
    in_tag = _DTYPE_TAG[spec["dtype"]]
    out_names = (
        tuple(spec["out_dtypes"]) if "out_dtypes" in spec else (spec["out_dtype"],)
    )
    out_tag = "_".join(_DTYPE_TAG[name] for name in out_names)
    return {
        "prefix": (
            f"reduction_row_{spec['trait']}_{in_tag}_{out_tag}_{spec['row_class']}"
        ),
        "fn": _RowAdapter(op),
        "fake_args": [
            x_fake,
            out_fakes,
            Int32(vec),
            Int32(1),
            Int32(vec),
            cute.runtime.make_fake_stream(),
        ],
        "tensor_args": [
            {
                "name": "mX",
                "dynamic_sizes": [0, 1],
                "dynamic_strides": [0],
                "read_only": True,
            },
            _itree_tensor_list("mOuts", "mOut", trait.nouts),
        ],
        "scalar_args": [
            {"name": "nchunks", "ctype": "int32_t"},
            {"name": "nwaves", "ctype": "int32_t"},
            {"name": "n", "ctype": "int32_t"},
        ],
    }


def _build_xcta(spec):
    dtype = _DTYPES[spec["dtype"]]
    trait = _TRAITS[spec["trait"]]()
    out_dtypes = _itree_out_dtypes(spec, trait)
    subrow = data.XCTA_SUBROW
    threads = 256
    stage1 = tile.TileReduce(
        trait,
        dtype,
        "row",
        subrow,
        threads_per_row=threads,
        threads_per_block=threads,
        nouts=trait.nfields,
        final=False,
        unroll=4,
        use_tma=False,
        order="linear",
    )
    stage2 = general.ReduceBlock(
        trait,
        count=2,
        num_o=1,
        red_pairs=[(2, 1)],
        kept_pairs=[(1, 2)],
        from_partials=True,
        project_n=2 * subrow,
        nouts=trait.nouts,
        final=True,
        block=threads,
    )
    rows = cute.sym_int()
    outputs = cute.sym_int()
    x_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (rows, subrow),
        stride_order=(1, 0),
        assumed_align=16,
    )
    part_fakes = [
        cute.runtime.make_fake_tensor(
            field_dtype,
            (rows,),
            stride=(1,),
            assumed_align=16,
        )
        for field_dtype in trait.fdtypes
    ]
    out_fakes = [
        cute.runtime.make_fake_tensor(
            out_dtype,
            (outputs,),
            stride=(1,),
            assumed_align=16,
        )
        for out_dtype in out_dtypes
    ]
    in_tag = _DTYPE_TAG[spec["dtype"]]
    out_names = (
        tuple(spec["out_dtypes"]) if "out_dtypes" in spec else (spec["out_dtype"],)
    )
    out_tag = "_".join(_DTYPE_TAG[name] for name in out_names)
    return {
        "prefix": f"reduction_xcta_{spec['trait']}_{in_tag}_{out_tag}",
        "fn": _XctaAdapter(stage1, stage2),
        "fake_args": [
            x_fake,
            part_fakes,
            out_fakes,
            Int32(2),
            Int32(2),
            Int32(1),
            Int64(2 * subrow),
            cute.runtime.make_fake_stream(),
        ],
        "tensor_args": [
            {"name": "mX", "dynamic_sizes": [0], "read_only": True},
            _itree_tensor_list("mParts", "mPart", trait.nfields),
            _itree_tensor_list("mOuts", "mOut", trait.nouts),
        ],
        "scalar_args": [
            {"name": "rows", "ctype": "int32_t"},
            {"name": "partials", "ctype": "int32_t"},
            {"name": "outputs", "ctype": "int32_t"},
            {"name": "n", "ctype": "int64_t"},
        ],
    }


def _itree_out_dtypes(spec, trait):
    names = tuple(spec["out_dtypes"]) if "out_dtypes" in spec else (spec["out_dtype"],)
    if len(names) != trait.nouts:
        raise ValueError(
            f"{spec['trait']} projects {trait.nouts} outputs, got out_dtypes={names}"
        )
    return tuple(_DTYPES[name] for name in names)


def _itree_tensors(spec, trait):
    dtype = _DTYPES[spec["dtype"]]
    rows = cute.sym_int()
    n = cute.sym_int()
    x_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (rows, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fakes = [
        cute.runtime.make_fake_tensor(
            out_dtype,
            (rows,),
            stride=(1,),
            assumed_align=16,
        )
        for out_dtype in _itree_out_dtypes(spec, trait)
    ]
    return dtype, x_fake, out_fakes


def _itree_prefix(spec):
    out_names = (
        tuple(spec["out_dtypes"]) if "out_dtypes" in spec else (spec["out_dtype"],)
    )
    out_tag = "_".join(_DTYPE_TAG[name] for name in out_names)
    return (
        f"reduction_{spec['path']}_{spec['trait']}_"
        f"{_DTYPE_TAG[spec['dtype']]}_{out_tag}_l{spec['loads']}"
    )


def _itree_tensor_list(name, item_prefix, count):
    return {
        "name": name,
        "items": [
            {"name": f"{item_prefix}{i}", "dynamic_sizes": [0]} for i in range(count)
        ],
    }


def _build_itree_multirow(spec):
    trait = _TRAITS[spec["trait"]]()
    dtype, x_fake, out_fakes = _itree_tensors(spec, trait)
    nouts = len(out_fakes)
    return {
        "prefix": _itree_prefix(spec),
        "fn": _InnerTreeMultirow(
            trait,
            128 // dtype.width,
            spec["loads"],
        ),
        "fake_args": [
            x_fake,
            out_fakes,
            Int32(128 // dtype.width),
            Int32(1),
            cute.runtime.make_fake_stream(),
        ],
        "tensor_args": [
            {
                "name": "mX",
                "dynamic_sizes": [0, 1],
                "dynamic_strides": [0],
                "read_only": True,
            },
            _itree_tensor_list("mOuts", "mOut", nouts),
        ],
        "scalar_args": [
            {"name": "n", "ctype": "int32_t"},
            {"name": "rows", "ctype": "int32_t"},
        ],
    }


def _build_itree_looped(spec):
    trait = _TRAITS[spec["trait"]]()
    dtype, x_fake, out_fakes = _itree_tensors(spec, trait)
    nouts = len(out_fakes)
    return {
        "prefix": _itree_prefix(spec),
        "fn": _InnerTreeLooped(
            trait,
            128 // dtype.width,
            spec["loads"],
        ),
        "fake_args": [
            x_fake,
            out_fakes,
            Int32(1024),
            Int32(1),
            Int32(4),
            Int32(1),
            cute.runtime.make_fake_stream(),
        ],
        "tensor_args": [
            {
                "name": "mX",
                "dynamic_sizes": [0, 1],
                "dynamic_strides": [0],
                "read_only": True,
            },
            _itree_tensor_list("mOuts", "mOut", nouts),
        ],
        "scalar_args": [
            {"name": "n", "ctype": "int32_t"},
            {"name": "rows", "ctype": "int32_t"},
            {"name": "warps", "ctype": "int32_t"},
            {"name": "batches", "ctype": "int32_t"},
        ],
    }


def _build_itree_split(spec):
    trait = _TRAITS[spec["trait"]]()
    dtype, x_fake, out_fakes = _itree_tensors(spec, trait)
    part_fakes = [
        cute.runtime.make_fake_tensor(
            field_dtype,
            (cute.sym_int(),),
            stride=(1,),
            assumed_align=16,
        )
        for field_dtype in trait.fdtypes
    ]
    nouts = len(out_fakes)
    return {
        "prefix": _itree_prefix(spec),
        "fn": _InnerTreeSplit(
            trait,
            128 // dtype.width,
            spec["loads"],
        ),
        "fake_args": [
            x_fake,
            part_fakes,
            out_fakes,
            Int32(1),
            Int32(4),
            Int32(32768),
            cute.runtime.make_fake_stream(),
        ],
        "tensor_args": [
            {
                "name": "mX",
                "dynamic_sizes": [0, 1],
                "dynamic_strides": [0],
                "read_only": True,
            },
            _itree_tensor_list("mParts", "mPart", trait.nfields),
            _itree_tensor_list("mOuts", "mOut", nouts),
        ],
        "scalar_args": [
            {"name": "rows", "ctype": "int32_t"},
            {"name": "batches", "ctype": "int32_t"},
            {"name": "n", "ctype": "int32_t"},
        ],
    }


_BUILDERS = {
    "row": _build_row,
    "xcta": _build_xcta,
    "itree_multirow": _build_itree_multirow,
    "itree_looped": _build_itree_looped,
    "itree_split": _build_itree_split,
}


def build(spec: dict) -> dict:
    """Build one manifest point through its kernel-path adapter."""
    try:
        builder = _BUILDERS[spec["path"]]
    except KeyError as error:
        raise ValueError(f"unknown reduction AOT path: {spec.get('path')!r}") from error
    return builder(spec)
