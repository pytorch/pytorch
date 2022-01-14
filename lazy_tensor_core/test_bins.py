
import torch

import copy
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import torch.optim as optim
import lazy_tensor_core.core.lazy_model as ltm
import random
import collections

lazy_tensor_core._LAZYC._ltc_init_ts_backend()


def generate_tensors(num, choices, bs, features, dev):
    return [torch.rand(bs, random.choice(choices), features).to(device = dev) for _ in range(num)]


for _ in range(10):

    input_tensors = generate_tensors(15, [128, 64, 256], 8, 32, 'lazy')

    bins = collections.defaultdict(list)

    for t in input_tensors:
        bins[t.size()[1]].append(t)


    for k, v in bins.items():
        print(f"looking at {k}")
        for t in v:
            print(f"\t{t.size()[1]}")

    r = torch.stack(bins[64], 1).sum([1,2]) + torch.stack(bins[128], 1).sum([1,2]) + torch.stack(bins[256], 1).sum([1,2])
    ltm.mark_step()
    print(r.sum().item())