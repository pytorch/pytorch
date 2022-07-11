import torch
import torchvision.models as models

import torch.autograd.profiler as profiler

for with_cuda in [False, True]:
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)
    sort_key = "self_cpu_memory_usage"
    if with_cuda and torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        sort_key = "self_cuda_memory_usage"
        print("Profiling CUDA Resnet model")
    else:
        print("Profiling CPU Resnet model")

    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        with profiler.record_function("root"):
            model(inputs)

    print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=-1))
