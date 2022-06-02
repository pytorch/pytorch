import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from objprint import op

def compute_self_time(profiler : torch.autograd.profiler.profile):
    call_tree = profiler.kineto_results.experimental_event_tree()
    events = profiler.kineto_results.events()
    op(call_tree)
    for e in events:
        print(e)


if __name__ == '__main__':
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    compute_self_time(prof.profiler)
