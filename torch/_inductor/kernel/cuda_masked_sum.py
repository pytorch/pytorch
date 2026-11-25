# mypy: allow-untyped-defs
import math

import torch

from .. import ir, lowering as L
from ..select_algorithm import SymbolicGridFn, TritonTemplate
from ..virtualized import V
from .cuda_native_sum import _plans
from .mm_common import load_kernel_template


@SymbolicGridFn
def cuda_masked_sum_grid(*args):
    meta = args[-1]
    if meta["PHASE"] in (0, 1):
        return (meta["BLOCKS"], 2 if meta["COMPLEX"] and meta["PHASE"] == 1 else 1, 1)
    return (1 if meta["FINAL"] else meta["MAX_CTAS"], 2 if meta["COMPLEX"] else 1, 1)


cuda_masked_sum_template = TritonTemplate(
    name="cuda_masked_sum",
    grid=cuda_masked_sum_grid,
    source=load_kernel_template("triton_cuda_masked_sum"),
)


def cuda_masked_sum(x, mask):
    r"""Compact selected values on the device and sum in eager's reduction order."""
    device = x.get_device()
    assert device is not None
    complex_dtype = x.get_dtype() if x.get_dtype().is_complex else None
    x, mask = L.broadcast_tensors(x, mask)
    x.realize()
    if V.graph.is_negative_buffer(x.get_name()):
        x = L.clone(x)
    n = math.prod(V.graph.sizevars.guard_int(s) for s in x.get_size())
    itemsize = torch.empty((), dtype=x.get_dtype()).element_size()
    if complex_dtype is not None:
        real, imag = L.complex_components(x)
        x = L.cat((L.unsqueeze(real, -1), L.unsqueeze(imag, -1)), -1)
    x = ir.ExternKernel.require_contiguous(x)
    mask = ir.ExternKernel.require_contiguous(mask)
    if n == 0:
        if complex_dtype is not None:
            parts = L.full([2], 0, device=x.get_device(), dtype=x.get_dtype())
            return ir.TensorBox.create(
                ir.ComplexView.create(torch.ops.aten.view_as_complex.default, parts)
            )
        return L.full([], 0, device=x.get_device(), dtype=x.get_dtype())
    blocks = (n + 1023) // 1024
    props = torch.cuda.get_device_properties(x.get_device())
    iv = 8 if itemsize == 2 else 4
    max_threads = 256 if complex_dtype == torch.complex128 else 512
    branches = []
    for vi in (False, True):
        for bw in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            if bw > max_threads:
                continue
            lower = max(128, bw * iv) if vi else bw
            upper = (
                (bw * 2 * iv if bw < max_threads else n + 1) if vi else min(128, bw * 2)
            )
            if lower < upper and lower <= n:
                target = props.multi_processor_count * (
                    props.max_threads_per_multi_processor // bw
                )
                branches.append((lower, upper, bw, vi, target))
    max_ctas = _plans(
        (n,), (1,), (0,), 1, n, 1, itemsize, props, max_threads=max_threads
    )[0][7]
    acc_dtype = torch.float64 if x.get_dtype() == torch.float64 else torch.float32
    options = dict(
        N=n,
        BLOCKS=blocks,
        BLOCK=1024,
        MAX_CTAS=max_ctas,
        IV=iv,
        BRANCHES=branches,
        ACC_TYPE="tl.float64" if acc_dtype == torch.float64 else "tl.float32",
        COMPLEX=complex_dtype is not None,
    )

    def generate(phase, inputs, dtype, size, final=False):
        return cuda_masked_sum_template.generate(
            input_nodes=inputs,
            layout=ir.FixedLayout(device, dtype, size),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            PHASE={"counts": 0, "compact": 1, "sum": 2}[phase],
            FINAL=final,
            **options,
        ).output_node()

    counts = generate("counts", (mask,), torch.int64, [blocks])
    scan_kwargs = L._make_scan_inner(counts, axis=0, dtype=torch.int64)
    (prefix,) = ir.Scan.create(
        **scan_kwargs,
        combine_fn=lambda a, b: (L.ops.add(a[0], b[0]),),
        can_fallback_to_aten=False,
    )
    assert prefix is not None
    prefix.realize()
    compact = generate(
        "compact",
        (x, mask, prefix),
        x.get_dtype(),
        [2, n] if complex_dtype is not None else [n],
    )
    # Compaction's write mask cannot cover a sibling pointwise epilogue.
    V.graph.no_fuse_buffer_names.add(compact.get_name())
    partial = generate(
        "sum",
        (compact, prefix),
        acc_dtype,
        [2, max_ctas] if complex_dtype is not None else [max_ctas],
    )
    V.graph.no_fuse_buffer_names.add(partial.get_name())
    result = generate(
        "sum",
        (compact, prefix, partial),
        x.get_dtype(),
        [2, 1] if complex_dtype is not None else [1],
        final=True,
    )
    if complex_dtype is not None:
        parts = L.view(result, [2])
        return ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_complex.default, parts)
        )
    return L.view(result, [])
