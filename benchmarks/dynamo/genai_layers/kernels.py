from typing import Any

import cutlass
import cutlass.torch as cutlass_torch
from utils import BenchmarkKernel

import torch
import torch.nn.functional as F


class CrossEntropyForward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
            (16384, 131072),
            (8192, 262144),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        # Read x (M*N elements) + read target (M elements) + write loss (M elements)
        x, target = args
        M, N = x.shape
        dtype = x.dtype
        return (M * N + M + M) * dtype.itemsize

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target = args
        return lambda: F.cross_entropy(x, target, reduction="none")

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target = args

        # Mark batch size as dynamic for realistic workload
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(target, 0)

        # Need `lambda` otherwise torch.compile will not trace the function.
        # More discussion: https://github.com/pytorch/pytorch/issues/158455
        compiled_cross_entropy = torch.compile(
            lambda x, target: F.cross_entropy(x, target, reduction="none"),
            mode="max-autotune-no-cudagraphs",
        )
        return lambda: compiled_cross_entropy(x, target)

    def quack(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target = args
        from quack.cross_entropy import _cross_entropy

        return lambda: _cross_entropy(x, target)

    def liger(self, args, kwargs=None) -> Any:
        assert kwargs is None
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

        x, target = args
        cross_entropy = LigerCrossEntropyLoss(reduction="none")
        return lambda: cross_entropy(x, target)

    def benchmark(self):
        for M, N in self.get_shapes():
            print(f"\n Tensor dimensions: [{M}, {N}]")
            # quack requires cutlass dtype
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype)
            target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
            self.benchmark_single_shape((x, target), setting=f"shape: [{M}, {N}]")

    def check_accuracy(self, args, kwargs) -> None:
        res = {}
        for backend in self.available_backends:
            args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
            res[backend] = getattr(self, backend)(args_ref, kwargs_ref)()
        gold = res["eager"]
        for backend in self.available_backends:
            if backend == "eager":
                continue
            if backend == "quack":
                # quack's cross_entropy only returns float32 loss output.
                # Need to convert it to the same dtype as gold for comparison.
                res[backend] = res[backend].to(gold.dtype)
            try:
                torch.testing.assert_close(res[backend], gold)
                print(
                    f"Accuracy check \033[92m✓ succeed\033[0m for {backend} backend on {self.name} kernel"
                )
            except Exception as e:
                print(
                    f"Accuracy check \033[91m✗ failed\033[0m for {backend} backend on {self.name} kernel. Error {e}"
                )


class CrossEntropyBackward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
            (16384, 131072),
            (8192, 262144),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        # Read x (M*N elements) + read target (M elements) + read dloss (M elements) + write grad(M*N elements)
        x, target, dloss = args
        # Memory ba
        M, N = x.shape
        return (
            2 * M * N * x.dtype.itemsize
            + M * target.dtype.itemsize
            + M * dloss.dtype.itemsize
        )

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target, dloss = args
        loss = F.cross_entropy(x, target, reduction="none")
        return lambda: torch.autograd.grad(
            loss, x, grad_outputs=dloss, retain_graph=True
        )

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, target, dloss = args

        compiled_cross_entropy = torch.compile(
            lambda x, target: F.cross_entropy(x, target, reduction="none"),
            mode="max-autotune-no-cudagraphs",
        )
        loss = compiled_cross_entropy(x, target)
        return lambda: torch.autograd.grad(
            loss, x, grad_outputs=dloss, retain_graph=True
        )

    def quack(self, args, kwargs=None) -> Any:
        from quack.cross_entropy import cross_entropy

        assert kwargs is None
        x, target, dloss = args
        loss = cross_entropy(x, target)
        return lambda: torch.autograd.grad(
            loss, x, grad_outputs=dloss, retain_graph=True
        )

    def liger(self, args, kwargs=None) -> Any:
        assert kwargs is None
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

        x, target, dloss = args
        cross_entropy = LigerCrossEntropyLoss(reduction="none")
        loss = cross_entropy(x, target)
        return lambda: torch.autograd.grad(
            loss, x, grad_outputs=dloss, retain_graph=True
        )

    def benchmark(self):
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(
                M, N, device="cuda", dtype=torch_dtype, requires_grad=True
            )
            target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
            dloss = torch.randn(M, device="cuda", dtype=torch.float32)
            self.benchmark_single_shape(
                (x, target, dloss), setting=f"shape: [{M}, {N}]"
            )


class SoftmaxForward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
            (16384, 131072),
            (8192, 262144),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        (x,) = args
        M, N = x.shape
        return 2 * M * N * x.dtype.itemsize

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        (x,) = args
        return lambda: F.softmax(x, dim=-1)

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        (x,) = args

        # Mark batch size as dynamic for realistic workload
        torch._dynamo.mark_dynamic(x, 0)

        compiled_softmax = torch.compile(
            lambda x: F.softmax(x, dim=-1), mode="max-autotune-no-cudagraphs"
        )
        return lambda: compiled_softmax(x)

    def quack(self, args, kwargs=None) -> Any:
        from quack.softmax import softmax

        assert kwargs is None
        (x,) = args
        return lambda: softmax(x)

    def liger(self, args, kwargs=None) -> Any:
        from liger_kernel.transformers.softmax import LigerSoftmax

        assert kwargs is None
        (x,) = args
        softmax = LigerSoftmax().to("cuda")
        return lambda: softmax(x)

    def benchmark(self):
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype)
            self.benchmark_single_shape((x,), setting=f"shape: [{M}, {N}]")


class SoftmaxBackward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
            (16384, 131072),
            (8192, 262144),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        # Memory: read dy and y, write ax backward
        x, dy = args
        M, N = x.shape
        return 3 * M * N * x.dtype.itemsize

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, dy = args
        y = F.softmax(x, dim=-1)
        return lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, dy = args
        compiled_softmax = torch.compile(
            lambda x: F.softmax(x, dim=-1), mode="max-autotune-no-cudagraphs"
        )
        y = compiled_softmax(x)
        return lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)

    def quack(self, args, kwargs=None) -> Any:
        from quack.softmax import softmax

        assert kwargs is None
        x, dy = args

        y = softmax(x)
        return lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)

    def liger(self, args, kwargs=None) -> Any:
        from liger_kernel.transformers.softmax import LigerSoftmax

        assert kwargs is None
        x, dy = args
        softmax = LigerSoftmax().to("cuda")
        y = softmax(x)
        return lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)

    def benchmark(self):
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(
                M, N, device="cuda", dtype=torch_dtype, requires_grad=True
            )
            dy = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            self.benchmark_single_shape((x, dy), setting=f"shape: [{M}, {N}]")


class RMSNormForward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
            (16384, 131072),
            (8192, 262144),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        x, w = args
        M, N = x.shape
        return 2 * M * N * x.dtype.itemsize + N * w.dtype.itemsize

    def rms_norm_ref(self, x, w):
        x_f32 = x.float()
        return (
            x_f32
            * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
            * w
        ).to(x.dtype)

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w = args
        return lambda: self.rms_norm_ref(x, w)

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w = args

        # Mark batch size as dynamic for realistic workload
        torch._dynamo.mark_dynamic(x, 0)

        compiled_rms_norm = torch.compile(
            self.rms_norm_ref, mode="max-autotune-no-cudagraphs"
        )
        return lambda: compiled_rms_norm(x, w)

    def quack(self, args, kwargs=None) -> Any:
        # Note: only supper weight with float32 dtype
        from quack.rmsnorm import _rmsnorm_fwd

        x, w = args
        return lambda: _rmsnorm_fwd(x, w, eps=1e-6)

    def liger(self, args, kwargs) -> Any:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm

        x, w = args
        M, N = x.shape
        liger_rmsnorm = LigerRMSNorm(hidden_size=N, eps=1e-6).cuda()
        liger_rmsnorm.weight.data.copy_(w)
        return lambda: liger_rmsnorm(x)

    def benchmark(self):
        for M, N in self.get_shapes():
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
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        x, w, dy = args
        # x, dy: [M, N], w: [N]
        M, N = x.shape
        # Read x, w, dy, write dx, dw
        return 3 * M * N * x.dtype.itemsize + 2 * N * w.dtype.itemsize

    def rms_norm_ref(self, x, w):
        x_f32 = x.float()
        return (
            x_f32
            * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
            * w
        ).to(x.dtype)

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w, dy = args
        y = self.rms_norm_ref(x, w)
        return lambda: torch.autograd.grad(
            y, [x, w], grad_outputs=dy, retain_graph=True
        )

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w, dy = args
        y = torch.compile(self.rms_norm_ref, mode="max-autotune-no-cudagraphs")(x, w)
        return lambda: torch.autograd.grad(
            y, [x, w], grad_outputs=dy, retain_graph=True
        )

    def quack(self, args, kwargs=None) -> Any:
        from quack.rmsnorm import _rmsnorm_backward

        (
            x,
            w,
            dy,
        ) = args
        M, N = x.shape
        rstd = torch.randn(M, device="cuda", dtype=torch.float32)
        return lambda: _rmsnorm_backward(x, w, dy, rstd)

    def liger(self, args, kwargs=None) -> Any:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm

        x, w, dy = args
        M, N = x.shape
        liger_rmsnorm = LigerRMSNorm(hidden_size=N, eps=1e-6).cuda()
        liger_rmsnorm.weight.data.copy_(w)
        y = liger_rmsnorm(x)
        return lambda: torch.autograd.grad(
            y, [x, liger_rmsnorm.weight], grad_outputs=dy, retain_graph=True
        )

    def benchmark(self):
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = torch.randn(M, N, device="cuda", dtype=torch_dtype, requires_grad=True)
            w = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
            dy = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            self.benchmark_single_shape((x, w, dy), setting=f"shape: [{M}, {N}]")


class LayerNormForward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # OOM for (16384, 131072) on h100
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        x, w = args
        M, N = x.shape
        # Read x ([M, N]), w ([N]), write y ([M, N])
        return 2 * M * N * x.dtype.itemsize + N * w.dtype.itemsize

    def layernorm_ref(self, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        x_f32 = x.float()
        return F.layer_norm(x_f32, w.shape, w, None, eps).to(x.dtype)

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w = args
        return lambda: self.layernorm_ref(x, w)

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w = args

        # Mark batch size as dynamic for realistic workload
        torch._dynamo.mark_dynamic(x, 0)

        compiled_layernorm = torch.compile(
            self.layernorm_ref, mode="max-autotune-no-cudagraphs"
        )
        return lambda: compiled_layernorm(x, w, eps=1e-6)

    def quack(self, args, kwargs) -> Any:
        # Note: quack layernorm does not support bias
        from quack.layernorm import layernorm

        x, w = args
        return lambda: layernorm(x, w, eps=1e-6)

    def liger(self, args, kwargs) -> Any:
        from liger_kernel.transformers.layer_norm import LigerLayerNorm

        x, w = args
        M, N = x.shape
        liger_layernorm = LigerLayerNorm(hidden_size=N, eps=1e-6).cuda()
        liger_layernorm.weight.data.copy_(w)
        liger_layernorm.bias.data.copy_(
            torch.zeros(N, device="cuda", dtype=torch.float32)
        )
        return lambda: liger_layernorm(x)

    def benchmark(self):
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            w = torch.randn(N, device="cuda", dtype=torch.float32)
            self.benchmark_single_shape((x, w), setting=f"shape: [{M}, {N}]")


class LayerNormBackward(BenchmarkKernel):
    def __init__(self):
        super().__init__()
        self.available_backends = ["eager", "compiled", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # OOM for (16384, 131072), (8192, 262144)
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        x, w, dy = args
        M, N = x.shape
        # Read x ([M, N]), w ([N]), dy ([M, N]), write dx ([M, N]), dw ([N])
        return (
            2 * M * N * x.dtype.itemsize
            + 2 * N * w.dtype.itemsize
            + M * N * dy.dtype.itemsize
        )

    def layernorm_ref(self, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        x_f32 = x.float()
        return F.layer_norm(x_f32, w.shape, w, None, eps).to(x.dtype)

    def eager(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w, dy = args
        y = self.layernorm_ref(x, w)
        return lambda: torch.autograd.grad(
            y, [x, w], grad_outputs=dy, retain_graph=True
        )

    def compiled(self, args, kwargs=None) -> Any:
        assert kwargs is None
        x, w, dy = args
        compiled_layernorm = torch.compile(
            self.layernorm_ref, mode="max-autotune-no-cudagraphs"
        )
        y = compiled_layernorm(x, w)
        return lambda: torch.autograd.grad(
            y, [x, w], grad_outputs=dy, retain_graph=True
        )

    def liger(self, args, kwargs) -> Any:
        from liger_kernel.transformers.layer_norm import LigerLayerNorm

        x, w, dy = args
        M, N = x.shape
        liger_layernorm = LigerLayerNorm(hidden_size=N, eps=1e-6).cuda()
        liger_layernorm.weight.data.copy_(w)
        liger_layernorm.bias.data.copy_(
            torch.zeros(N, device="cuda", dtype=torch.float32)
        )
        y = liger_layernorm(x)
        return lambda: torch.autograd.grad(
            y, [x, liger_layernorm.weight], grad_outputs=dy, retain_graph=True
        )

    def benchmark(self):
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = torch.randn(M, N, device="cuda", dtype=torch_dtype, requires_grad=True)
            w = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
            dy = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            self.benchmark_single_shape((x, w, dy), setting=f"shape: [{M}, {N}]")
