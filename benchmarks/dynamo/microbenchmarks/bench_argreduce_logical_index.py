import argparse
import json

import torch
from torch._inductor.runtime.benchmarking import benchmarker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--shape", type=int, nargs="+", default=(64, 128, 256))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and Triton")

    def fn(x):
        return torch.argmax(x + 1)

    x = torch.randn(args.shape, device="cuda")
    compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
    compiled_fn(x)
    latency_ms, _, _ = benchmarker.benchmark_gpu(lambda: compiled_fn(x))
    print(
        json.dumps(
            {
                "label": args.label,
                "device": torch.cuda.get_device_name(),
                "shape": args.shape,
                "latency_ms": latency_ms,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
