import torch

from typing import List, Optional

@torch.jit.script
def test():
    # type: () -> List[Optional[int]]
    return torch.jit.annotate(List[Optional[int]], [])


print(test.graph)
