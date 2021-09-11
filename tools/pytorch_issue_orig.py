import torch
import time
from itertools import product

inp_sizes = [
    ((14, 768), (768, 768), (768,)),
    ((64, 1024), (4096, 1024), (4096,)),
]


for device, sizes in product(["cpu", "cuda"], inp_sizes):
    inp = torch.rand(sizes[0]).to(device)
    weights = [torch.rand(sizes[1]).to(device) for _ in range(3)]
    biases = [torch.rand(sizes[2]).to(device) for _ in range(3)]

    warmup = 5
    outputs = [0 for _ in range(3)]

    def test():
        for i in range(3):
            outputs[i] = torch._C._nn.linear(inp, weights[i], biases[i])

    NITER = 40
    NWARMUP = 10

    def benchmark(func):
        torch.cuda.synchronize()
        for _ in range(NWARMUP):
            func()
            torch.cuda.synchronize()
        s = time.time()
        for _ in range(NITER):
            func()
            torch.cuda.synchronize()
        e = time.time()
        return (e - s) / NITER

    non_fused = benchmark(test)
    weights = torch.cat(weights)
    biases = torch.cat(biases)
    outputs2 = []

    def fused():
        out = torch._C._nn.linear(inp, weights, biases)
        outputs2 = torch.chunk(out, 3)

    fused = benchmark(fused)
    for elem1, elem2 in zip(outputs, outputs2):
        torch.testing.assert_allclose(elem1, elem2)

    outputs3 = []

    def fused_transposed():
        out = (weights @ inp.T).T + biases
        outputs3 = torch.chunk(out, 3)

    print(
        "sizes",
        sizes,
        "device",
        device,
        "fused speedup",
        (non_fused - fused) / non_fused,
    )

    fused2 = benchmark(fused_transposed)
    for elem1, elem2 in zip(outputs2, outputs3):
        torch.testing.assert_allclose(elem1, elem2)

    print(
        "sizes",
        sizes,
        "device",
        device,
        "fused transposed speedup",
        (non_fused - fused2) / non_fused,
    )
