import torch
import triton
from torch import Tensor
import triton.language as tl



@triton.jit
def kernel(
        o_ptr,
        # Meta-parameters.
        B0: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr):

        qk = tl.zeros((BLOCK_M, B0), dtype=tl.float32)
        v = tl.zeros((B0, D), dtype=tl.float32)

        # This must be here to trigger the 2D stuff. If I remove this, the assert goes away. 
        row_max = tl.max(qk, axis=-1)
        qk -= row_max[:, None]

        tl.static_print(qk.shape, "qk shape")
        tl.static_print(v.shape, "v shape")


        # o = tl.dot(s, v) # does not work because BLOCK_M < 16.
        o = qk[:, :, None] * v[None, :, :]
        tl.static_print(o.shape, "o shape")
        o = tl.sum(o, axis=-2)
        tl.static_print(o.shape, "o shape")



        idx_m = tl.arange(0, BLOCK_M)[:, None]
        idx_d = tl.arange(0, D)[None, :]
        xindex = idx_d + (D*idx_m)
        tl.store(o_ptr + (xindex), o, None)


o_ptr = torch.zeros(2, 64, device='cuda', dtype=torch.float32)
grid = lambda meta: (1, 1, 1)
kernel[grid](o_ptr, B0=128, D=64, BLOCK_M=2)
print(o_ptr)
