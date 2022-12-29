import torch
from typing import List
import torch._inductor.config as config
config.debug=True

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    gm.graph.print_tabular()
    return gm.forward

def nt_relu(nt):
    result = nt.relu()
    return result

def nt_tanh_relu(nt):
    result = nt.tanh().relu()
    return result

nt = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(5, 3)], requires_grad=False)
# st = torch.arange(6, dtype=torch.float).reshape(2, 3).to_sparse()
# opt_nt_relu = torch.compile(nt_relu)
opt_nt_tanh_relu = torch.compile(nt_tanh_relu)
# opt_nt_relu = torch.compile(nt_relu, backend=custom_backend)
result = opt_nt_tanh_relu(nt)

print(result)
