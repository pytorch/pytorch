from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch._inductor.runtime.benchmarking import benchmarker


def benchmark_kernel_in_milliseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    return benchmarker.benchmark_gpu(lambda: func(*args, **kwargs))


@dataclass
class Performance:
    # Benchmark setting usually the shape of the input tensor
    setting: str

    # Latency in milliseconds
    latency: float

    # Number of  memory access in bytes
    memory_bytes: float

    # Number of flops computation
    flops: float

    # Memory bandwidth in GB/s
    memory_bandwidth: float = 0.0

    # Compute intensity in FLOPs/byte
    compute_intensity: float = 0.0

    # Computationi per second
    tflops_per_seconds: float = 0.0

    def __post_init__(self):
        self.memory_bandwidth = self.memory_bytes / (self.latency / 1000) / 1e9
        self.compute_intensity = self.flops / self.memory_bytes
        self.flops_per_seconds = self.flops / (self.latency / 1000) / 1e12

    def __str__(self):
        return f"setting: {self.setting}, latency: {self.latency} ms, memory bandwidth: {self.memory_bandwidth} GB/s"


class BenchmarkKernel:
    def __init__(self):
        self.name = self.__class__.__name__
        self.available_backends = []

    def get_memory_bytes(self, args, kwargs) -> int:
        # Get the necessary memory access in bytes for the kernelßß
        raise NotImplementedError

    def get_flops(self, args, kwargs) -> int:
        # Get the number of flops for the kernel
        raise NotImplementedError

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # Get a list of input shapes to benchmark the kernel
        raise NotImplementedError

    def eager(self, args, kwargs) -> Any:
        raise NotImplementedError

    def compiled(self, args, kwargs) -> Any:
        raise NotImplementedError

    def helion(self, args, kwargs) -> Any:
        raise NotImplementedError

    def quack(self, args, kwargs) -> Any:
        raise NotImplementedError

    def liger(self, args, kwargs) -> Any:
        raise NotImplementedError

    def triton(self, args, kwargs) -> Any:
        raise NotImplementedError

    def benchmark(self):
        raise NotImplementedError

    def clone_inputs(self, args, kwargs) -> Any:
        args_ref = [
            arg.clone().detach().requires_grad_(arg.requires_grad) for arg in args
        ]

        kwargs_ref = (
            {
                k: (
                    v.clone().detach().requires_grad_(v.requires_grad)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in kwargs.items()
            }
            if kwargs
            else kwargs
        )

        return args_ref, kwargs_ref

    def check_accuracy(self, args, kwargs) -> None:
        res = {}
        for backend in self.available_backends:
            args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
            res[backend] = getattr(self, backend)(args_ref, kwargs_ref)()
        gold = res["eager"]
        for backend in self.available_backends:
            if backend == "eager":
                continue
            try:
                torch.testing.assert_close(res[backend], gold)
                print(
                    f"Accuracy check \033[92m✓ succeed\033[0m for {backend} backend on {self.name} kernel"
                )
            except Exception as e:
                print(
                    f"Accuracy check \033[91m✗ failed\033[0m for {backend} backend on {self.name} kernel. Error {e}"
                )

    def benchmark_single_shape(
        self, args, kwargs=None, should_check_accuracy=True, setting: str = ""
    ):
        for backend in self.available_backends:
            args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
            try:
                avg_time = benchmark_kernel_in_milliseconds(
                    getattr(self, backend)(args_ref, kwargs_ref)
                )
            except Exception as e:
                print(
                    f"Failed to run {backend} backend on {self.name} kernel for {setting} due to {e}"
                )
                self.available_backends.remove(backend)
                continue
            mem_bytes = self.get_memory_bytes(args_ref, kwargs_ref)
            flops = self.get_flops(args_ref, kwargs_ref)
            perf = Performance(setting, avg_time, mem_bytes, flops)
            print(f"{self.name} kernel on {backend} backend. {perf}")

        if should_check_accuracy:
            self.check_accuracy(args, kwargs)
