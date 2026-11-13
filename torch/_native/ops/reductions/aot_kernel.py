"""CuTeDSL builders for native-AOT reduction callables."""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Int32, Int64

from . import _aot_data as data, kernel_general as general, tile, traits


_DTYPES = {
    "float16": cutlass.Float16,
    "bfloat16": cutlass.BFloat16,
    "float32": cutlass.Float32,
}
_DTYPE_TAG = {
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
}
_TRAITS = {
    "sum": lambda: traits.SumOps(acc=Float32),
    "mean": lambda: traits.MeanOps(acc=Float32),
    "amax": lambda: traits.AMaxOps(acc=Float32),
    "amin": lambda: traits.AMinOps(acc=Float32),
    "prod": lambda: traits.ProdOps(acc=Float32),
}


class _RowAdapter:
    def __init__(self, op):
        self.op = op

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        nchunks: Int32,
        nwaves: Int32,
        n: Int32,
        stream: cuda.CUstream,
    ):
        self.op(
            [mX],
            [mOut],
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
    out_dtype = _DTYPES[spec["out_dtype"]]
    trait = _TRAITS[spec["trait"]]()
    threads_per_row, threads_per_block = data.ROW_CLASSES[spec["row_class"]]
    vec = 128 // dtype.width
    op = tile.TileReduce(
        trait,
        dtype,
        "row",
        vec,
        threads_per_row=threads_per_row,
        threads_per_block=threads_per_block,
        nouts=1,
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
    out_fake = cute.runtime.make_fake_tensor(
        out_dtype,
        (m,),
        stride=(1,),
        assumed_align=16,
    )
    in_tag = _DTYPE_TAG[spec["dtype"]]
    out_tag = _DTYPE_TAG[spec["out_dtype"]]
    return {
        "prefix": (
            f"reduction_row_{spec['trait']}_{in_tag}_{out_tag}_{spec['row_class']}"
        ),
        "fn": _RowAdapter(op),
        "fake_args": [
            x_fake,
            out_fake,
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
            {"name": "mOut", "dynamic_sizes": [0]},
        ],
        "scalar_args": [
            {"name": "nchunks", "ctype": "int32_t"},
            {"name": "nwaves", "ctype": "int32_t"},
            {"name": "n", "ctype": "int32_t"},
        ],
    }


def _tensor_list(name, item_prefix, count):
    return {
        "name": name,
        "items": [
            {"name": f"{item_prefix}{i}", "dynamic_sizes": [0]} for i in range(count)
        ],
    }


def _build_xcta(spec):
    dtype = _DTYPES[spec["dtype"]]
    trait = _TRAITS[spec["trait"]]()
    out_dtype = _DTYPES[spec["out_dtype"]]
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
        nouts=1,
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
    out_fake = cute.runtime.make_fake_tensor(
        out_dtype,
        (outputs,),
        stride=(1,),
        assumed_align=16,
    )
    in_tag = _DTYPE_TAG[spec["dtype"]]
    out_tag = _DTYPE_TAG[spec["out_dtype"]]
    return {
        "prefix": f"reduction_xcta_{spec['trait']}_{in_tag}_{out_tag}",
        "fn": _XctaAdapter(stage1, stage2),
        "fake_args": [
            x_fake,
            part_fakes,
            [out_fake],
            Int32(2),
            Int32(2),
            Int32(1),
            Int64(2 * subrow),
            cute.runtime.make_fake_stream(),
        ],
        "tensor_args": [
            {"name": "mX", "dynamic_sizes": [0], "read_only": True},
            _tensor_list("mParts", "mPart", trait.nfields),
            _tensor_list("mOuts", "mOut", 1),
        ],
        "scalar_args": [
            {"name": "rows", "ctype": "int32_t"},
            {"name": "partials", "ctype": "int32_t"},
            {"name": "outputs", "ctype": "int32_t"},
            {"name": "n", "ctype": "int64_t"},
        ],
    }


_BUILDERS = {
    "row": _build_row,
    "xcta": _build_xcta,
}


def build(spec: dict) -> dict:
    """Build one manifest point through its kernel-path adapter."""
    try:
        builder = _BUILDERS[spec["path"]]
    except KeyError as error:
        raise ValueError(f"unknown reduction AOT path: {spec.get('path')!r}") from error
    return builder(spec)
