
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.triton.unique_kernel_names = True






from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()


    @torch.no_grad()
    def forward(self, l__self___attn_qkv_w_int_repr_t, clamp, x_vals_int8):
        x_scales = clamp.div(127);  clamp = None
        y = torch.ops.higher_order.out_dtype(torch.ops.aten.mm.default, torch.int32, x_vals_int8, l__self___attn_qkv_w_int_repr_t);  x_vals_int8 = l__self___attn_qkv_w_int_repr_t = None
        y_1 = y * x_scales;  y = x_scales = None
        return (y_1,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('b5194b7aeb0844ec42bba009d7d688ebb956e9c8', 4915200, device=device(type='cuda', index=0), dtype_hint=torch.int8)
    reader.tensor(buf0, (1280, 3840), dtype=torch.int8, is_leaf=True)  # l__self___attn_qkv_w_int_repr_t
    buf1 = reader.storage('44c0eea6271f93ab44d195c4e0e683a500a4e7b9', 313600, device=device(type='cuda', index=0))
    reader.tensor(buf1, (78400, 1), is_leaf=True)  # clamp
    buf2 = reader.storage('df3ab49b8a9c96aaf6301f429b4f37ba83ea21f8', 100352000, device=device(type='cuda', index=0), dtype_hint=torch.int8)
    reader.tensor(buf2, (78400, 1280), dtype=torch.int8, is_leaf=True)  # x_vals_int8
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='run',
        save_dir='/home/cdhernandez/local/ao_benchmarks/checkpoints', autocast=False, backend='inductor')
