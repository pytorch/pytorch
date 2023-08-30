import itertools
import time

import torch
from torch._inductor.utils import do_bench
from torch.utils.flop_counter import FlopCounterMode


def get_flops_achieved(f):
    """
    Measures and prints the Software FLOPs achieved by a given function.
    """
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        f()
    total_flops = sum(flop_counter.get_flop_counts()["Global"].values())
    ms_per_iter = do_bench(f)
    iters_per_second = 1e3 / ms_per_iter
    print(f"{iters_per_second * total_flops / 1e12} TF/s")


def evaluate_model_flops(model, inp):
    """
    Evaluates the Software FLOPs of a given model with a specific input.

    Example usage:
    --------------
    from torchvision.models import resnet18
    model = resnet18().cuda().half()
    inp = torch.randn(128, 3, 224, 224, device='cuda', dtype=torch.half)
    evaluate_model_flops(model, inp)

    # If you have a compiled model, you can also use:
    # compiled_model = torch.compile(model)
    # evaluate_model_flops(compiled_model, inp)
    """
    get_flops_achieved(lambda: model(inp).sum())


def memory_bandwidth(model, inp, num_samples=1):
    """
    Memory bandwidth is most useful in regimes with small batch sizes
    The formula is bw = model_size / (batch_size * time) for a specific number of samples
    In a more real world use case this would also include tokenization encoding and decoding
    But keeping it agnostic for now so this can work for all modalities

    Example usage:
    --------------
    from torchvision.models import resnet18
    model = resnet18().cuda().half()
    inp = torch.randn(128, 3, 224, 224, device='cuda', dtype=torch.half)
    memory_bandwidth(model, inp)
    """
    model.eval()
    model_size = sum(
        [
            p.numel() * p.data.element_size()
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )

    # Assume batch size is the first dimension
    batch_size = inp.size(0)

    total_time = 0.0

    device = next(model.parameters()).device

    for i in range(num_samples):
        if device == torch.device("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(inp)

        if device == torch.device("cuda"):
            torch.cuda.synchronize()

        t = time.perf_counter() - t0
        total_time += t

    avg_time = total_time / num_samples
    bandwidth = (model_size / batch_size) / avg_time / 1e9
    print(
        f"Average memory bandwidth per image over {num_samples} samples: {bandwidth:.02f} GB/s"
    )
    return bandwidth
