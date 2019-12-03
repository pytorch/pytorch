import torch
import json
from . import unary, binary
from collections import defaultdict

results = defaultdict(list)

def warm_up_cuda():
    a = torch.randn(100 * 1024 * 1024)
    for _ in range(10):
        _ = a + a

def run(more):
    warm_up_cuda()
    for title, result in unary.run(more):
        results[title].append(result)
    for title, result in binary.run(more):
        results[title].append(result)

def dump(filename):
    with open(filename, 'w') as f:
        json.dump(results, f)
