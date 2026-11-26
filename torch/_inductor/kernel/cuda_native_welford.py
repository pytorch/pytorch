# mypy: allow-untyped-defs
import math

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch._tensor_iterator import reduce_op

from .. import ir, lowering as L
from ..select_algorithm import SymbolicGridFn, TritonTemplate
from ..virtualized import V
from .cuda_native_sum import _coalesce_dimensions, _plans
from .mm_common import load_kernel_template


@SymbolicGridFn
def cuda_native_welford_grid(*args):
    meta = args[-1]
    return (meta["M"], 1 if meta["FINAL"] else meta["MAX_CTAS"], 1)


cuda_native_welford_template = TritonTemplate(
    name="cuda_native_welford",
    grid=cuda_native_welford_grid,
    source=load_kernel_template("triton_cuda_native_welford"),
)


def cuda_native_bn_welford(x, value):
    x = L.constrain_to_fake_tensor(x, value)
    x.realize()
    size = [V.graph.sizevars.guard_int(s) for s in x.get_size()]
    stride = [V.graph.sizevars.guard_int(s) for s in x.get_stride()]
    dtype = torch.float64 if x.get_dtype() == torch.float64 else torch.float32
    with unset_fake_temporarily():
        source = torch.empty_strided(size, stride, device="meta", dtype=x.get_dtype())
        if x.get_dtype() == torch.bfloat16:
            # The two-output reduction retains mixed Half input, but casts BFloat16.
            source = source.to(torch.float32)
            x = L.constrain_to_fake_tensor(L.to_dtype(x, torch.float32), source)
        output_size = [1] * len(size)
        output_size[1] = size[1]
        mean = torch.empty(output_size, device="meta", dtype=dtype)
        variance = torch.empty_like(mean)
        iterator = reduce_op(variance, source, out2=mean)
    x.realize()
    shape = tuple(iterator.shape)
    ins = tuple(s // source.element_size() for s in iterator.strides(2))
    outs = tuple(s // mean.element_size() for s in iterator.strides(0))
    shape, ins, outs = _coalesce_dimensions(shape, ins, outs)
    rdims = next((d for d, stride in enumerate(outs) if stride), len(outs))
    m = size[1]
    n = math.prod(size) // m
    plans = _plans(
        shape,
        ins,
        outs,
        rdims,
        n,
        m,
        source.element_size(),
        torch.cuda.get_device_properties(x.get_device()),
        input_vec_size=2,
    )
    max_ctas = max(plan[7] for plan in plans)
    options = dict(
        SHAPE=shape,
        INS=ins,
        OUTS=outs,
        RDIMS=rdims,
        N=n,
        M=m,
        PLANS=plans,
        MAX_CTAS=max_ctas,
        ITEMSIZE=source.element_size(),
        ACC_TYPE="tl.float64" if dtype == torch.float64 else "tl.float32",
    )
    first = cuda_native_welford_template.generate(
        input_nodes=(x,),
        layout=ir.FixedLayout(x.get_device(), dtype, [3, m, max_ctas]),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=False,
        **options,
    )
    final = cuda_native_welford_template.generate(
        input_nodes=(x, first.output_node()),
        layout=ir.FixedLayout(x.get_device(), dtype, [2, m]),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=True,
        **options,
    )
    output = final.output_node()
    return L.clone(L.select(output, 0, 0)), L.clone(L.select(output, 0, 1))
