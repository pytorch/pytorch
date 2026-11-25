# mypy: allow-untyped-defs
import re
from typing import Any

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

from .. import ir, lowering as L
from ..select_algorithm import SymbolicGridFn, TritonTemplate
from ..virtualized import V
from .mm_common import load_kernel_template


@SymbolicGridFn
def cuda_native_mv_grid(*args):
    meta = args[-1]
    return (
        (meta["M"] + 15) // 16 if meta["KIND"] in (4, 5) else meta["M"],
        1 if meta["FINAL"] else meta["CTAS"],
        1,
    )


@SymbolicGridFn
def cuda_native_mv_aligned_grid(*args):
    meta = args[-1]
    return ((meta["M"] + 15) // 16, 1 if meta["FINAL"] else meta["MAX_CTAS"], 1)


cuda_native_mv_aligned_template = TritonTemplate(
    name="cuda_native_mv_aligned",
    grid=cuda_native_mv_aligned_grid,
    source=load_kernel_template("triton_cuda_native_mv_aligned"),
)


cuda_native_mv_template = TritonTemplate(
    name="cuda_native_mv",
    grid=cuda_native_mv_grid,
    source=load_kernel_template("triton_cuda_native_mv"),
)


cuda_native_complex_mv_template = TritonTemplate(
    name="cuda_native_complex_mv",
    grid=cuda_native_mv_grid,
    source=load_kernel_template("triton_cuda_native_complex_mv"),
)


cuda_native_mv_mma_template = TritonTemplate(
    name="cuda_native_mv_mma",
    grid=cuda_native_mv_grid,
    source=load_kernel_template("triton_cuda_native_mv_mma"),
)


def _copy_complex(x, preserve_format):
    real, imag = L.complex_components(x)
    parts = L.cat((L.unsqueeze(real, -1), L.unsqueeze(imag, -1)), -1)
    if preserve_format:
        sizes = [V.graph.sizevars.guard_int(s) for s in x.get_size()]
        strides = [V.graph.sizevars.guard_int(s) for s in x.get_stride()]
        cloned = torch.empty_strided(sizes, strides, device="meta").clone()
        parts = ir.ExternKernel.require_exact_strides(
            parts, [2 * s for s in cloned.stride()] + [1]
        )
    else:
        parts = ir.ExternKernel.require_contiguous(parts)
    return ir.TensorBox.create(
        ir.ComplexView.create(torch.ops.aten.view_as_complex.default, parts)
    )


def cuda_native_mv(x, vector):
    r"""Generate GEMV kernels using native CUDA launch geometry and arithmetic."""
    node = V.graph.current_node
    if node is not None:
        for index, buffer in enumerate((x, vector)):
            argument = node.args[index]
            if isinstance(argument, torch.fx.Node):
                constrained = L.constrain_to_fake_tensor(
                    buffer, argument.meta.get("val")
                )
                if index == 0:
                    x = constrained
                else:
                    vector = constrained
    device = x.get_device()
    assert device is not None
    complex_dtype = x.get_dtype() if x.get_dtype().is_complex else None
    x.realize()
    vector.realize()
    if V.graph.is_negative_buffer(x.get_name()) or V.graph.is_conjugate_buffer(
        x.get_name()
    ):
        x = _copy_complex(x, True) if complex_dtype is not None else L.clone(x)
    if V.graph.is_negative_buffer(vector.get_name()) or V.graph.is_conjugate_buffer(
        vector.get_name()
    ):
        vector = (
            _copy_complex(vector, True)
            if complex_dtype is not None
            else L.clone(vector)
        )
    m, k = [V.graph.sizevars.guard_int(s) for s in x.get_size()]
    if m == 0 or k == 0:
        if complex_dtype is not None:
            parts = L.full(
                [m, 2],
                0,
                device=x.get_device(),
                dtype=torch.float64
                if complex_dtype == torch.complex128
                else torch.float32,
            )
            return ir.TensorBox.create(
                ir.ComplexView.create(torch.ops.aten.view_as_complex.default, parts)
            )
        return L.full([m], 0, device=x.get_device(), dtype=x.get_dtype())
    s0, s1 = [V.graph.sizevars.guard_int(s) for s in x.get_stride()]
    if s0 == 1 and s1 >= max(1, m):
        transposed, lda = False, s1
    elif s1 == 1 and s0 >= max(1, k):
        transposed, lda = True, s0
    else:
        if complex_dtype is not None:
            x = _copy_complex(x, False)
        else:
            x = ir.ExternKernel.require_contiguous(x)
        s0, s1 = k, 1
        transposed, lda = True, k
    vs = V.graph.sizevars.guard_int(vector.get_stride()[0])
    if vs == 0:
        vector = (
            _copy_complex(vector, False)
            if complex_dtype is not None
            else ir.ExternKernel.require_contiguous(vector)
        )
        vs = 1
    vs = max(1, vs)
    V.graph.preserve_input_alignment(
        (x.get_name(), vector.get_name()),
        "strict CUDA mv requires original input pointer alignment",
    )
    element_size = torch.empty((), dtype=x.get_dtype()).element_size()
    alignments = [
        (15, 0),
        *((2 * value - 1, value) for value in (8, 4, 2) if value >= element_size),
    ]
    plans = {}
    with unset_fake_temporarily(), torch.cuda.device(x.get_device()):
        for matrix_mask, matrix_value in alignments:
            for vector_mask, vector_value in alignments:
                plan = torch._C._cuda_getGemvLaunchPlan(
                    x.get_dtype(),
                    k if transposed else m,
                    m if transposed else k,
                    lda,
                    vs,
                    transposed,
                    matrix_value,
                    vector_value,
                )
                options = _native_options(plan, x.get_dtype(), m, k, s0, s1, vs)
                key = tuple(options.items())
                if key not in plans:
                    plans[key] = (options, [])
                plans[key][1].append(
                    (matrix_mask, matrix_value, vector_mask, vector_value)
                )
    if len(plans) > 1:
        return _aligned_mv(x, vector, tuple(plans.values()))
    options = next(iter(plans.values()))[0]
    ctas, kind = options["CTAS"], options["KIND"]
    acc_dtype = (
        torch.float64
        if x.get_dtype() in (torch.float64, torch.complex128)
        else torch.float32
    )
    if kind in (4, 5):
        first = cuda_native_mv_mma_template.generate(
            input_nodes=(x, vector),
            layout=ir.FixedLayout(
                device,
                acc_dtype if kind == 5 else x.get_dtype(),
                [m, ctas] if kind == 5 else [m],
            ),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            FINAL=kind == 4,
            **options,
        ).output_node()
        if kind == 4:
            return first
        final_options = options.copy()
        final_options["KIND"] = 3
        return cuda_native_mv_template.generate(
            input_nodes=(x, vector, first),
            layout=ir.FixedLayout(device, x.get_dtype(), [m]),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            FINAL=True,
            **final_options,
        ).output_node()
    if complex_dtype is not None:
        x = ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_real.default, x)
        )
        vector = ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_real.default, vector)
        )
        if kind in (2, 3):
            partial = cuda_native_complex_mv_template.generate(
                input_nodes=(x, vector),
                layout=ir.FixedLayout(device, acc_dtype, [m, ctas, 2]),
                num_stages=1,
                num_warps=4,
                triton_meta={"enable_fp_fusion": False},
                FINAL=False,
                **options,
            ).output_node()
            inputs = (x, vector, partial)
        else:
            inputs = (x, vector)
        parts = cuda_native_complex_mv_template.generate(
            input_nodes=inputs,
            layout=ir.FixedLayout(device, acc_dtype, [m, 2]),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            FINAL=True,
            **options,
        ).output_node()
        return ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_complex.default, parts)
        )
    if kind in (2, 3):
        partial = cuda_native_mv_template.generate(
            input_nodes=(x, vector),
            layout=ir.FixedLayout(device, acc_dtype, [m, ctas]),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            FINAL=False,
            **options,
        ).output_node()
        inputs = (x, vector, partial)
    else:
        inputs = (x, vector)
    return cuda_native_mv_template.generate(
        input_nodes=inputs,
        layout=ir.FixedLayout(device, x.get_dtype(), [m]),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=True,
        **options,
    ).output_node()


def _aligned_mv(x, vector, plans):
    options = plans[0][0]
    device = x.get_device()
    assert device is not None
    complex_dtype = x.get_dtype() if x.get_dtype().is_complex else None
    acc_dtype = (
        torch.float64
        if x.get_dtype() in (torch.float64, torch.complex128)
        else torch.float32
    )
    max_ctas = max(p[0]["CTAS"] for p in plans)
    keys = (
        "KIND",
        "CTAS",
        "THREADS",
        "ACTIVE_THREADS",
        "INNER_CHUNK",
        "INNER_STRIDE",
        "SWAP",
        "PRODUCT",
        "PRE_SCALE",
    )
    metadata = dict(
        M=options["M"],
        K=options["K"],
        S0=options["S0"],
        S1=options["S1"],
        VS=options["VS"],
        MAX_CTAS=max_ctas,
        ACC_TYPE=options["ACC_TYPE"],
        COMPLEX=complex_dtype is not None,
        PLANS=tuple(
            (tuple(conditions), *(p[key] for key in keys), p.get("K_SLICE", 0))
            for p, conditions in plans
        ),
    )
    if complex_dtype is not None:
        x = ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_real.default, x)
        )
        vector = ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_real.default, vector)
        )
    partial_shape = [options["M"], max_ctas] + (
        [2] if complex_dtype is not None else []
    )
    partial = cuda_native_mv_aligned_template.generate(
        input_nodes=(x, vector),
        layout=ir.FixedLayout(device, acc_dtype, partial_shape),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=False,
        **metadata,
    ).output_node()
    V.graph.no_fuse_buffer_names.add(partial.get_name())
    output_shape = [options["M"]] + ([2] if complex_dtype is not None else [])
    output = cuda_native_mv_aligned_template.generate(
        input_nodes=(x, vector, partial),
        layout=ir.FixedLayout(
            device,
            acc_dtype if complex_dtype is not None else x.get_dtype(),
            output_shape,
        ),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=True,
        **metadata,
    ).output_node()
    if complex_dtype is not None:
        return ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_complex.default, output)
        )
    return output


def _native_options(plan, dtype, m, k, s0, s1, vs) -> dict[str, Any]:
    name, grid, block, _, partition = next(p for p in plan if "reduce" not in p[0])
    ctas, threads, kind = 1, 1, 0
    tile_k = 1
    if "dot_kernel" in name:
        kind, ctas, threads = 2, grid[0], block[0]
    elif name.startswith("nvjet_"):
        kind = 4
        tiles = re.search(r"nvjet_[^_]+_(\d+)x(\d+)_(\d+)x(\d+)_", name)
        if tiles is None:
            raise NotImplementedError(f"Strict CUDA mv tensor-core geometry for {name}")
        tile_m = max(int(tiles[1]), int(tiles[2]))
        tile_k = int(tiles[3])
        if any("splitKreduce" in p[0] for p in plan):
            kind = 5
            ctas = grid[0] * grid[1] * grid[2] // ((m + tile_m - 1) // tile_m)
    elif any("splitKreduce" in p[0] for p in plan):
        kind, ctas, threads = 3, grid[2], block[1]
    elif "gemvx" in name:
        flags = [int(v) for v in re.findall(r"Lb([01])E", name)]
        threads = block[0] if flags[2] else block[1]

    elif "gemv2N_kernel" in name:
        threads = 1
    elif "gemvNSP_kernel" in name:
        threads = block[1]
        partition = (4, 1, 0)
    elif k == 1:
        threads = 1
    else:
        raise NotImplementedError(f"Strict CUDA mv reduction order for {name}")
    acc_dtype = (
        torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
    )
    options = dict(
        M=m,
        K=k,
        S0=s0,
        S1=s1,
        VS=vs,
        CTAS=ctas,
        THREADS=1 << (threads - 1).bit_length(),
        ACTIVE_THREADS=threads,
        INNER_CHUNK=partition[0],
        INNER_STRIDE=partition[1],
        KIND=kind,
        ACC_TYPE="tl.float64" if acc_dtype == torch.float64 else "tl.float32",
    )
    if kind in (4, 5):
        options.update(K_SLICE=((k + ctas - 1) // ctas + tile_k - 1) // tile_k * tile_k)
    options.update(
        SWAP=not bool(partition[2]) if kind == 2 else bool(partition[2]),
        PRODUCT="gemmk1_kernel" in name,
        PRE_SCALE="gemv2N_kernel" in name,
    )
    return options
