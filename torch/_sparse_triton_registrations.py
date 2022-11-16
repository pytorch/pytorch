import torch
from torch._inductor.utils import has_triton

if has_triton():
    from torch.sparse.triton_ops import bsr_dense_mm

    lib = torch.library.Library("aten", "IMPL")
    lib.impl(
        "aten::_triton_bsr_dense_mm",
        lambda *args, **kwargs: bsr_dense_mm(*args, skip_checks=True, **kwargs),
        "SparseCsrCUDA",
    )
