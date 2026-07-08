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
        foreach=args.foreach,
        amsgrad=args.amsgrad,
    )
    return params, optimizer


def check_optimizer_path(optimizer, foreach):
    def wrong_path(*args, **kwargs):
        expected = "multi-tensor" if foreach else "single-tensor"
        raise RuntimeError(f"expected {expected} Adam path")

    if foreach:
        path_name = "multi_tensor"
        wrapped_path = "_multi_tensor_adam"
        other_path = "_single_tensor_adam"
    else:
        path_name = "single_tensor"
        wrapped_path = "_single_tensor_adam"
        other_path = "_multi_tensor_adam"

    with mock.patch.object(
        adam_module, wrapped_path, wraps=getattr(adam_module, wrapped_path)
    ) as wrapped:
        with (
            mock.patch.object(adam_module, other_path, wrong_path),
            mock.patch.object(adam_module, "_fused_adam", wrong_path),
        ):
            optimizer.step()
    if wrapped.call_count != 1:
        raise RuntimeError(f"expected {wrapped_path} once, got {wrapped.call_count}")
    return path_name


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
        description="Benchmark capturable Adam/AdamW single-tensor or foreach paths."
    )
    parser.add_argument("--optimizer", choices=("Adam", "AdamW"), default="AdamW")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", type=lambda name: getattr(torch, name), default=torch.float16
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--foreach", action="store_true")
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--nparams", type=int, default=32)
    parser.add_argument("--numel", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=9)
    args = parser.parse_args()

    optim_cls = getattr(torch.optim, args.optimizer)
    _, optimizer = make_optimizer(optim_cls, args)
    path_name = check_optimizer_path(optimizer, args.foreach)
    measurements = time_optimizer(optimizer, args)
    print(
        f"{args.optimizer} dtype={args.dtype} device={args.device} "
        f"capturable=True foreach={args.foreach} amsgrad={args.amsgrad}"
    )
    print(f"optimizer_path={path_name}")
    print(f"{path_name}_calls=1")
    print(
        "ms/step "
        f"median={statistics.median(measurements):.6f} "
        f"mean={statistics.mean(measurements):.6f} "
        f"min={min(measurements):.6f} "
        f"max={max(measurements):.6f}"
    )


if __name__ == "__main__":
    main()
