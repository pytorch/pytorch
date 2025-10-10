# LOCAL_RANK=0 python test_dispatch.py

import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

class PrintDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"PrintDispatchMode op: {func} {kwargs=}")
        return func(*args, **(kwargs or {}))


def main():
    bsz = 4
    seq_len = 4096
    dim = 2048
    inputs = torch.randn(bsz, seq_len, dim, device="cuda")
    weight = torch.randn(3072, dim, device="cuda")
    with PrintDispatchMode():
        F.linear(inputs, weight)
    return

if __name__ == "__main__":
    main()
