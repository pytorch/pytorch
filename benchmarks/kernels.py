from dataclasses import dataclass
from typing import Any, Callable
import cutlass
import cutlass.torch as cutlass_torch

import torch
from torch._inductor.runtime.benchmarking import benchmarker
import torch.nn.functional as F

torch._dynamo.config.automatic_dynamic_shapes = False
# Needed since changing args to function causes recompiles
torch._dynamo.config.recompile_limit = 1000

torch._inductor.config.force_disable_caches = True

# TODO1: visualization
# TODO2: roof line analysis
# TODO: Record peak memory


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
        args_ref = [arg.clone().detach().requires_grad_(arg.requires_grad) for arg in args]
        
        kwargs_ref = {k: (v.clone().detach().requires_grad_(v.requires_grad) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()} if kwargs else kwargs

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
                print(f"Accuracy check \033[92m✓ succeed\033[0m for {backend} backend on {self.name} kernel")
            except:
                print(f"Accuracy check \033[91m✗ failed\033[0m for {backend} backend on {self.name} kernel")

    def benchmark_single_shape(self, args, kwargs=None, should_check_accuracy=True, setting: str = ""):
        for backend in self.available_backends:
            args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
            try:
                avg_time = benchmark_kernel_in_milliseconds(getattr(self, backend)(args, kwargs))
            except:
                print(f"Failed to run {backend} backend on {self.name} kernel for {setting}")
                self.available_backends.remove(backend)
                continue
            mem_bytes = self.get_memory_bytes(args_ref, kwargs_ref)
            flops = self.get_flops(args_ref, kwargs_ref)
            perf = Performance(setting, avg_time, mem_bytes, flops)
            print(f"{self.name} kernel on {backend} backend. {perf}")

        if should_check_accuracy:
            self.check_accuracy(args, kwargs)

    # def roofline_analysis(self, backend: str, perfs: list[Performance]) -> None:
    #     for perf in perfs:
    #         compute_intensity = perf.flops / perf.memory_bytes




    #     for backend, backend_perf in perf.items():
    #         for shape, shape_perf in backend_perf.items():
    #             mem_bw = shape_perf["mem_bw"]
    #             flops = shape_perf["flops"]
    #             latency = shape_perf["latency"]
    #             print(f"{self.name} kernel on {backend} backend, shape: {shape}, latency: {latency} ms, mem throughput: {mem_bw} GB/s, flops: {flops} FLOPs")


class CrossEntropyForward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        return ((32768,256), (32768,512), (32768,1024), (32768,2048), (32768,4096), (32768,8192), (32768,16384), (32768,32768), (32768,65536), (16384,131072), (8192,262144))

    def get_memory_bytes(self, args, kwargs) -> int:
        # Read x (M*N elements) + read target (M elements) + write loss (M elements)
        x, target = args
        M, N = x.shape
        dtype = x.dtype
        return (M * N + M + M) * dtype.itemsize

    def get_flops(self, args, kwargs) -> int:
        # TODO
        return 0

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target = args
        return lambda: F.cross_entropy(x, target, reduction="none")

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target = args
        # Need `lambda` otherwise torch.compile will not trace the function.
        # More discussion: https://github.com/pytorch/pytorch/issues/158455
        compiled_cross_entropy = torch.compile(lambda x, target: F.cross_entropy(x, target, reduction='none'), mode="max-autotune-no-cudagraphs")
        fn = lambda: compiled_cross_entropy(x, target)
        return fn

    def quack(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target = args
        from quack.cross_entropy import _cross_entropy
        return lambda: _cross_entropy(x, target)

    def liger(self, args, kwargs=None) -> Any:
        assert kwargs is None
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

        x, target = args
        cross_entropy = LigerCrossEntropyLoss(reduction='none')
        fn = lambda: cross_entropy(x, target)
        return fn

    def benchmark(self):
        for (M, N) in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype)
            target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
            self.benchmark_single_shape((x, target), setting=f"shape: [{M}, {N}]")


class CrossEntropyBackward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        return ((32768,256), (32768,512), (32768,1024), (32768,2048), (32768,4096), (32768,8192), (32768,16384), (32768,32768), (32768,65536), (16384,131072), (8192,262144))

    def get_memory_bytes(self, args, kwargs) -> int:
        # Read x (M*N elements) + read target (M elements) + read dloss (M elements) + write grad(M*N elements)
        x, target, dloss = args
        # Memory ba
        M, N = x.shape
        return 2*M*N * x.dtype.itemsize + M * target.dtype.itemsize + M * dloss.dtype.itemsize

    def get_flops(self, args, kwargs) -> int:
        # TODO
        return 0

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target, dloss = args
        loss = F.cross_entropy(x, target, reduction='none')
        fn = lambda: torch.autograd.grad(loss, x, grad_outputs=dloss, retain_graph=True)
        return fn

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target, dloss = args

        compiled_cross_entropy = torch.compile(lambda x, target: F.cross_entropy(x, target, reduction='none'), mode="max-autotune-no-cudagraphs")
        loss = compiled_cross_entropy(x, target)
        fn = lambda: torch.autograd.grad(loss, x, grad_outputs=dloss, retain_graph=True)
        return fn

    def quack(self, args, kwargs=None) -> Any:
        from quack.cross_entropy import cross_entropy

        assert kwargs is None
        x, target, dloss = args
        loss = cross_entropy(x, target)
        fn = lambda: torch.autograd.grad(loss, x, grad_outputs=dloss, retain_graph=True)
        return fn

    def liger(self, args, kwargs=None) -> Any:
        assert kwargs is None
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

        x, target, dloss = args
        cross_entropy = LigerCrossEntropyLoss(reduction='none')
        loss = cross_entropy(x, target)
        fn = lambda: torch.autograd.grad(loss, x, grad_outputs=dloss, retain_graph=True)
        return fn

    def benchmark(self):
        for (M, N) in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype, requires_grad=True)
            target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
            dloss = torch.randn(M, device="cuda", dtype=torch.float32)
            self.benchmark_single_shape((x, target, dloss), setting=f"shape: [{M}, {N}]")


class SoftmaxForward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        return ((32768,256), (32768,512), (32768,1024), (32768,2048), (32768,4096), (32768,8192), (32768,16384), (32768,32768), (32768,65536), (16384,131072), (8192,262144))

    def get_memory_bytes(self, args, kwargs) -> int:
        x, = args
        M, N = x.shape
        return 2 * M * N * x.dtype.itemsize

    def get_flops(self, args, kwargs) -> int:
        return 0

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, = args
        return lambda: F.softmax(x, dim=-1)

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, = args
        compiled_softmax = torch.compile(lambda x: F.softmax(x, dim=-1), mode="max-autotune-no-cudagraphs")
        fn = lambda: compiled_softmax(x)
        return fn

    def quack(self, args, kwargs=None) -> Any:
        from quack.softmax import softmax

        assert kwargs is None
        x, = args
        fn = lambda: softmax(x)
        return fn

    def liger(self, args, kwargs=None) -> Any:
        from liger_kernel.transformers.softmax import LigerSoftmax
        assert kwargs is None
        x, = args
        softmax = LigerSoftmax().to("cuda")
        fn = lambda: softmax(x)
        return fn

    def benchmark(self):
        for (M, N) in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype)
            self.benchmark_single_shape((x,), setting=f"shape: [{M}, {N}]")


class SoftmaxBackward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        return ((32768,256), (32768,512), (32768,1024), (32768,2048), (32768,4096), (32768,8192), (32768,16384), (32768,32768), (32768,65536), (16384,131072), (8192,262144))
    
    def get_memory_bytes(self, args, kwargs) -> int:
        # Memory: read dy and y, write ax backward
        x, dy = args
        M, N = x.shape
        return 3 * M * N * x.dtype.itemsize

    def get_flops(self, args, kwargs) -> int:
        return 0

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, dy = args
        y = F.softmax(x, dim=-1)
        fn = lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)
        return fn

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, dy = args
        compiled_softmax = torch.compile(lambda x: F.softmax(x, dim=-1), mode="max-autotune-no-cudagraphs")
        y = compiled_softmax(x)
        fn = lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)
        return fn

    def quack(self, args, kwargs=None) -> Any:
        from quack.softmax import softmax

        assert kwargs is None
        x, dy = args

        y = softmax(x)
        fn = lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)
        return fn

    def liger(self, args, kwargs=None) -> Any:
        from liger_kernel.transformers.softmax import LigerSoftmax
        assert kwargs is None
        x, dy = args
        softmax = LigerSoftmax().to("cuda")
        y = softmax(x)
        fn = lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)
        return fn

    def benchmark(self):
        for (M, N) in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype, requires_grad=True)
            dy = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            self.benchmark_single_shape((x, dy), setting=f"shape: [{M}, {N}]")


class RMSNormForward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        return ((32768,256), (32768,512), (32768,1024), (32768,2048), (32768,4096), (32768,8192), (32768,16384), (32768,32768), (32768,65536), (16384,131072), (8192,262144))

    def get_memory_bytes(self, args, kwargs) -> int:
        x, w = args
        M, N = x.shape
        return 2 * M * N * x.dtype.itemsize + N * w.dtype.itemsize

    def get_flops(self, args, kwargs) -> int:
        return 0

    def rms_norm_ref(self, x, w):
        x_f32 = x.float()
        return (x_f32 *torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6) * w).to(x.dtype)

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w = args
        fn = lambda: self.rms_norm_ref(x, w)
        return fn

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w = args
        compiled_rms_norm = torch.compile(self.rms_norm_ref, mode="max-autotune-no-cudagraphs")
        fn = lambda: compiled_rms_norm(x, w)
        return fn

    def quack(self, args, kwargs=None) -> Any:
        # Note: only supper weight with float32 dtype
        from quack.rmsnorm import _rmsnorm_fwd

        x, w = args
        fn = lambda: _rmsnorm_fwd(x, w, eps=1e-6)
        return fn

    def liger(self, args, kwargs) -> Any:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm

        x, w = args
        M, N = x.shape
        liger_rmsnorm = LigerRMSNorm(hidden_size=N, eps=1e-6).cuda()
        liger_rmsnorm.weight.data.copy_(w)
        fn = lambda: liger_rmsnorm(x)
        return fn

    def benchmark(self):
        for (M, N) in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            w = torch.randn(N, device="cuda", dtype=torch.float32)
            self.benchmark_single_shape((x, w), setting=f"shape: [{M}, {N}]")


class RMSNormBackward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # TODO: OOM for (32768, 65536) on h100 
        return ((32768,256), (32768,512), (32768,1024), (32768,2048), (32768,4096), (32768,8192), (32768,16384), (32768,32768))

    def get_memory_bytes(self, args, kwargs) -> int:
        x, w, dy = args
        # x, dy: [M, N], w: [N]
        M, N = x.shape
        # Read x, w, dy, write dx, dw
        return 3*M*N*x.dtype.itemsize + 2*N*w.dtype.itemsize

    def get_flops(self, args, kwargs) -> int:
        return 0

    def rms_norm_ref(self, x, w):
        x_f32 = x.float()
        return (x_f32 *torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6) * w).to(x.dtype)

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w, dy = args
        y = self.rms_norm_ref(x, w)
        fn = lambda: torch.autograd.grad(y, [x, w], grad_outputs=dy, retain_graph=True)
        return fn

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w, dy = args
        y = torch.compile(self.rms_norm_ref, mode="max-autotune-no-cudagraphs")(x, w)
        fn = lambda: torch.autograd.grad(y, [x, w], grad_outputs=dy, retain_graph=True)
        return fn

    def quack(self, args, kwargs=None) -> Any:
        from quack.rmsnorm import _rmsnorm_backward

        x, w, dy, = args
        M, N = x.shape
        rstd = torch.randn(M, device="cuda", dtype=torch.float32)
        fn = lambda: _rmsnorm_backward(x, w, dy, rstd)
        return fn

    def liger(self, args, kwargs=None) -> Any:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm

        x, w, dy = args
        M, N = x.shape
        liger_rmsnorm = LigerRMSNorm(hidden_size=N, eps=1e-6).cuda()
        liger_rmsnorm.weight.data.copy_(w)
        y = liger_rmsnorm(x)
        fn = lambda: torch.autograd.grad(y, [x,liger_rmsnorm.weight], grad_outputs=dy, retain_graph=True)
        return fn

    def benchmark(self):
        for (M, N) in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = torch.randn(M, N, device="cuda", dtype=torch_dtype, requires_grad=True)
            w = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
            dy = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            self.benchmark_single_shape((x, w, dy), setting=f"shape: [{M}, {N}]")


class LayerNormForward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        return ((32768,256), (32768,512), (32768,1024), (32768,2048), (32768,4096), (32768,8192), (32768,16384), (32768,32768), (32768,65536), (16384,131072), (8192,262144))

    def get_memory_bytes(self, args, kwargs) -> int:
        x, w = args
        M, N = x.shape
        # Read x ([M, N]), w ([N]), write y ([M, N])
        return 2 * M * N * x.dtype.itemsize + N * w.dtype.itemsize  

    def get_flops(self, args, kwargs) -> int:
        return 0

    def layernorm_ref(self, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        x_f32 = x.float()
        return F.layer_norm(x_f32, w.shape, w, None, eps).to(x.dtype)

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w = args
        fn = lambda: self.layernorm_ref(x, w)
        return fn

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w = args
        compiled_layernorm = torch.compile(self.layernorm_ref, mode="max-autotune-no-cudagraphs")
        fn = lambda: compiled_layernorm(x, w, eps=1e-6)
        return fn

    def quack(self, args, kwargs) -> Any:
        from quack.layernorm import layernorm

        x, w = args
        fn = lambda: layernorm(x, w, eps=1e-6)
        return fn

    def benchmark(self):
        for (M, N) in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            w = torch.randn(N, device="cuda", dtype=torch.float32)
            self.benchmark_single_shape((x, w), setting=f"shape: [{M}, {N}]")


CrossEntropyForward().benchmark()
CrossEntropyBackward().benchmark()
SoftmaxForward().benchmark()
SoftmaxBackward().benchmark()
RMSNormForward().benchmark()
RMSNormBackward().benchmark()
LayerNormForward().benchmark()
