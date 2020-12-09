import torch

torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._debug_set_fusion_group_inlining(False)
from typing import List

# import  pdb; pdb.set_trace()

x = torch.tensor([2, 4])


@torch.jit.script
def foo(x: List[int]):
    return torch.tensor(x)

x = [1, 2, 3, 4]
# x = torch.rand([2, 2], dtype=torch.double)
foo(x)
foo(x)
print(foo(x), torch.tensor(x))
print(torch.jit.last_executed_optimized_graph())
