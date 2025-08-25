import torch
import torch.nn as nn
from torch.utils.flop_counter import _FlopCounterMode, FlopCounterMode, register_flop_formula, register_flop_formula_for_triton_kernel
from torch._ops import OpOverload
D = 1024


@torch._library.triton.triton_op("mylib::op", mutates_args=())
def op() -> None:
    print("op IN HERE")
    kernel1()
    kernel2()
    pass


@torch._library.triton.triton_op("mylib::kernel1", mutates_args=())
def kernel1() -> None:
    print("kernel1 IN HERE")
    pass

@torch._library.triton.triton_op("mylib::kernel2", mutates_args=())
def kernel2() -> None:
    print("kernel2 IN HERE")
    pass

def op_decompose(*args, **kwargs):
    print("op_decompose IN HERE")
    # Call the decomposed operations
    torch.ops.mylib.kernel1()
    torch.ops.mylib.kernel2()
    # Return None since the original op returns None
    return None


# torch.library.register_torch_dispatch("mylib::op", _FlopCounterMode, op_decompose)

# This of course works since we manually write in the decompose
# torch.ops.mylib.op.default.decompose = op_decompose

@register_flop_formula_for_triton_kernel(torch.ops.mylib.kernel1)
def compute_flops_kernel1(*args, **kwargs) -> int:
   return 1

@register_flop_formula_for_triton_kernel(torch.ops.mylib.kernel2)
def compute_flops_kernel2(*args, **kwargs) -> int:
   return 2

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dtype):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim, dtype=dtype),
                nn.LayerNorm(hidden_dim, dtype=dtype),
                nn.Linear(hidden_dim, output_dim, dtype=dtype),
                nn.LayerNorm(output_dim, dtype=dtype),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

mod = SimpleMLP(input_dim=D, hidden_dim=D, output_dim=D, dtype=torch.float).to("cuda")
x = torch.randn(D, device="cuda", dtype=torch.float)

with FlopCounterMode() as m:
#    mod(x)
#    op_impl()
    torch.ops.mylib.op()
    # kernel1()
    # kernel2()

print(m)
