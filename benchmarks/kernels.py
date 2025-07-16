from typing import Any, Callable

import torch

import cutlass
import cutlass.torch as cutlass_torch

torch._dynamo.config.automatic_dynamic_shapes = False
# Needed since changing args to function causes recompiles
torch._dynamo.config.recompile_limit = 1000

# TODO1: visualization
# TODO2: roof line analysis


from torch._inductor.runtime.benchmarking import benchmarker


def benchmark_kernel_in_milliseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    return benchmarker.benchmark_gpu(lambda: func(*args, **kwargs))


class BenchmarkKernel:
    def __init__(self):
        self.name = self.__class__.__name__
        self.available_backends = []

    def get_memory_bytes(self, args, kwargs) -> int:
        raise NotImplementedError

    def eager(self, args, kwargs) -> Any:
        raise NotImplementedError

    def compiled(self, args, kwargs) -> Any:
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
        args_ref = [arg.clone().detach().requires_grad_(arg.requires_grad) for arg in args]
        
        kwargs_ref = {k: (v.clone().detach().requires_grad_(v.requires_grad) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()} if kwargs else kwargs

        return args_ref, kwargs_ref

    def check_accuracy(self, args, kwargs) -> bool:
        res = {}
        for backend in self.available_backends:
            args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
            res[backend] = getattr(self, backend)(args_ref, kwargs_ref)
        
        gold = res["eager"]
        for backend in self.available_backends:
            if backend == "eager":
                continue
            try:
                torch.testing.assert_close(res[backend], gold, atol=1e-2, rtol=1e-2) # TODO: decide the atol and rtol
                print(f"Accuracy check passed for {backend} backend on {self.name} kernel")
            except:
                print(f"Accuracy check failed for {backend} backend on {self.name} kernel")

    def benchmark_single_shape(self, args, kwargs=None, should_check_accuracy=True):
        for backend in self.available_backends:
            args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
            avg_time = benchmark_kernel_in_milliseconds(getattr(self, backend)(args, kwargs))
            mem_bytes = self.get_memory_bytes(args_ref, kwargs_ref)
            print(f"{self.name} kernel on {backend} backend, latency: {avg_time} ms, mem throughput: {mem_bytes / (avg_time / 1000) / 1e9} GB/s")

        if should_check_accuracy:
            self.check_accuracy(args, kwargs)


class CrossEntropy(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack"]

    def get_memory_bytes(self, args, kwargs) -> int:
        x, target = args
        M, N = x.shape
        dtype = x.dtype
        return (M * N + M + M) * dtype.itemsize

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target = args
        return lambda: torch.nn.functional.cross_entropy(x, target, reduction="none")

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target = args
        compiled_cross_entropy = torch.compile(lambda x, target: torch.nn.functional.cross_entropy(x, target, reduction='none'), mode="max-autotune-no-cudagraphs")
        fn = lambda: compiled_cross_entropy(x, target)
        return fn

    def quack(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target = args
        from quack.cross_entropy import _cross_entropy
        return lambda: _cross_entropy(x, target)

    def benchmark(self):
        candidate_shapes = ((32768,256), (32768,512),(32768,1024),(32768,2048),(32768,4096),(32768,8192),(32768,16384),(32768,32768),(32768,65536),(16384,131072), (8192,262144))
        for (M, N) in candidate_shapes:
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype)
            target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
            self.benchmark_single_shape((x, target))


CrossEntropy().benchmark()
