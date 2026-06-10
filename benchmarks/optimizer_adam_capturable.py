#!/usr/bin/env python3

import argparse
import statistics
import time
from unittest import mock

import torch
import torch.optim.adam as adam_module


def make_optimizer(optim_cls, args):
    params = [
        torch.randn(
            args.numel,
            device=args.device,
            dtype=args.dtype,
            requires_grad=True,
        )
        for _ in range(args.nparams)
    ]
    for param in params:
        param.grad = torch.randn_like(param)
    optimizer = optim_cls(
        params,
        lr=args.lr,
        capturable=True,
        foreach=False,
        amsgrad=args.amsgrad,
    )
    return params, optimizer


def check_single_tensor_path(optimizer):
    def wrong_path(*args, **kwargs):
        raise RuntimeError("expected single-tensor Adam path")

    with (
        mock.patch.object(
            adam_module, "_single_tensor_adam", wraps=adam_module._single_tensor_adam
        ) as wrapped,
        mock.patch.object(adam_module, "_multi_tensor_adam", wrong_path),
        mock.patch.object(adam_module, "_fused_adam", wrong_path),
    ):
        optimizer.step()
    if wrapped.call_count != 1:
        raise RuntimeError(
            f"expected _single_tensor_adam once, got {wrapped.call_count}"
        )


def time_optimizer(optimizer, args):
    for _ in range(args.warmup):
        optimizer.step()

    if args.device == "cuda":
        torch.cuda.synchronize()

    measurements = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        for _ in range(args.steps):
            optimizer.step()
        if args.device == "cuda":
            torch.cuda.synchronize()
        measurements.append((time.perf_counter() - start) * 1000 / args.steps)
    return measurements


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark capturable single-tensor Adam/AdamW."
    )
    parser.add_argument("--optimizer", choices=("Adam", "AdamW"), default="AdamW")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", type=lambda name: getattr(torch, name), default=torch.float16
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--nparams", type=int, default=32)
    parser.add_argument("--numel", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=9)
    args = parser.parse_args()

    optim_cls = getattr(torch.optim, args.optimizer)
    _, optimizer = make_optimizer(optim_cls, args)
    check_single_tensor_path(optimizer)
    measurements = time_optimizer(optimizer, args)
    print(
        f"{args.optimizer} dtype={args.dtype} device={args.device} "
        f"capturable=True foreach=False amsgrad={args.amsgrad}"
    )
    print("single_tensor_calls=1")
    print(
        "ms/step "
        f"median={statistics.median(measurements):.6f} "
        f"mean={statistics.mean(measurements):.6f} "
        f"min={min(measurements):.6f} "
        f"max={max(measurements):.6f}"
    )


if __name__ == "__main__":
    main()
