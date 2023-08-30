import torch
from torch.utils.flop_counter import FlopCounterMode
from torch._inductor.utils import do_bench

def get_flops_achieved(f):
    """
    Measures and prints the Software FLOPs achieved by a given function.
    """
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        f()
    total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
    ms_per_iter = do_bench(f)
    iters_per_second = 1e3/ms_per_iter
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
