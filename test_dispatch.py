import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

class PrintDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        import fbvscode
        fbvscode.set_trace()
        print(f"PrintDispatchMode op: {func.__name__} {kwargs=}")
        return func(*args, **(kwargs or {}))


def main():
    bsz = 4
    seqlen = 4096
    dim = 2048
    out_features = 3072
    input = torch.randn(bsz, seqlen, dim, device="cuda")
    weight = torch.randn(out_features, dim, device="cuda")
    with PrintDispatchMode():
        F.linear(input, weight, None)
    return

if __name__ == "__main__":
    main()
