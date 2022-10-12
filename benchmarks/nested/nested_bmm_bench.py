import argparse

import torch


def bench(nt_a, nt_b, niter):
    # Warmup
    nt_c = nt_a.bmm(nt_b)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for iter in range(niter):
        nt_c = nt_a.bmm(nt_b)
    end_event.record()
    torch.cuda.synchronize()
    runtime = (start_event.elapsed_time(end_event) * 1.0e-3) / niter
    return runtime


def sweep_n(ntensor, niter, dtype):
    print("n, dtype, ntensor, gflop, runtime, tflop/s")
    for n in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        nt_a = torch.nested_tensor(
            [torch.randn(n, n).to(dtype).cuda() for t in range(ntensor)]
        )
        nt_b = torch.nested_tensor(
            [torch.randn(n, n).to(dtype).cuda() for t in range(ntensor)]
        )
        runtime = bench(nt_a, nt_b, niter)
        tflop = n * n * n * ntensor * 2 / 1e12
        print(n, dtype, ntensor, tflop, runtime, tflop / runtime)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nested Tensor BMM Benchmark")
    parser.add_argument("--niter", default="10", type=int)
    parser.add_argument("--ntensor", default="20", type=int)

    args = parser.parse_args()
    niter = args.niter
    ntensor = args.ntensor

    sweep_n(ntensor, niter, torch.float32)
    sweep_n(ntensor, niter, torch.float16)
