import torch
from functorch.compile import memory_efficient_fusion
import benchmark_helper


device = "cuda"
dtype = torch.float16

# LightSeq pattern 1
class DropoutResBias:
    @staticmethod
    def fn(input, bias, residual):
        a = torch.add(input, bias)
        b = torch.nn.functional.dropout(a, p=0.7, training=True)
        c = b + residual
        return c

    @staticmethod
    def args():
        batch_size, seq_len, hidden_size = 32, 196, 1024
        input = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            requires_grad=True,
            device=device,
            dtype=dtype,
        )
        bias = torch.randn(hidden_size, requires_grad=True, device=device, dtype=dtype)
        residual = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            requires_grad=False,
            device=device,
            dtype=dtype,
        )
        args = (input, bias, residual)
        return args


class DropoutResBiasScalar:
    @staticmethod
    def fn(input, bias, residual, p: float):
        a = torch.add(input, bias)
        b = torch.nn.functional.dropout(a, p, training=True)
        c = b + residual
        return c

    @staticmethod
    def args():
        batch_size, seq_len, hidden_size = 32, 196, 1024
        input = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            requires_grad=True,
            device=device,
            dtype=dtype,
        )
        bias = torch.randn(hidden_size, requires_grad=True, device=device, dtype=dtype)
        residual = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            requires_grad=False,
            device=device,
            dtype=dtype,
        )
        args = (input, bias, residual, 0.7)
        return args



# LightSeq pattern 2
class BiasReluDropout:
    @staticmethod
    def fn(input, bias):
        a = torch.add(input, bias)
        b = torch.nn.functional.relu(a)
        c = torch.nn.functional.dropout(b, p=0.6, training=True)
        return c

    @staticmethod
    def args():
        batch_size = 32
        seq_len = 196
        intermediate_size = 4096
        input = torch.randn(
            batch_size,
            seq_len,
            intermediate_size,
            requires_grad=True,
            device=device,
            dtype=dtype,
        )
        bias = torch.randn(
            intermediate_size, requires_grad=True, device=device, dtype=dtype
        )
        args = (input, bias)
        return args


class BiasDropoutResLayerNorm:
    @staticmethod
    def fn(input, bias, residual):
        hidden_size = 1024
        a = torch.add(input, bias)
        b = torch.nn.functional.dropout(a, p=0.7, training=True)
        c = b + residual
        d = torch.nn.functional.layer_norm(c, normalized_shape=(hidden_size,))
        return d

    @staticmethod
    def args():
        batch_size = 32
        seq_len = 196
        hidden_size = 1024

        input = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            requires_grad=True,
            device=device,
            dtype=dtype,
        )
        bias = torch.randn(hidden_size, requires_grad=True, device=device, dtype=dtype)
        residual = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            requires_grad=False,
            device=device,
            dtype=dtype,
        )
        args = (input, bias, residual)
        return args


class LayerNormSigmoid:
    @staticmethod
    def fn(inp):
        hidden_size = 512
        a = torch.nn.functional.layer_norm(inp, normalized_shape=(hidden_size,))
        b = torch.sigmoid(a)
        return b

    @staticmethod
    def args():
        batch_size = 8192
        hidden_size = 512
        inp = torch.randn(
            batch_size, hidden_size, requires_grad=True, device=device, dtype=dtype
        )
        args = (inp,)
        return args


for cl in [DropoutResBias, BiasReluDropout, DropoutResBiasScalar, BiasDropoutResLayerNorm, LayerNormSigmoid]:
    # Clear the compile cache

    # Get the function and inputs
    obj = cl()
    fn = obj.fn
    args = obj.args()

    # Find the static args
    static_argnums = []
    for idx, arg in enumerate(args):
        if not isinstance(arg, torch.Tensor):
            static_argnums.append(idx)

    # Get the optimized function
    opt_fn = memory_efficient_fusion(fn, static_argnums)

    # Profile cuda kernels
    benchmark_helper.profile_cuda_kernels(fn, args, "Eager")
    with torch.jit.fuser("fuser2"):
        benchmark_helper.profile_cuda_kernels(opt_fn, args, "AOTAutograd")

    # Time it with Torch Timer
    benchmark_helper.time_with_torch_timer(fn, args, "Eager")
    with torch.jit.fuser("fuser2"):
        benchmark_helper.time_with_torch_timer(opt_fn, args, "AOTAutograd")

    # Time it with manual Timer
    benchmark_helper.time_with_manual_timer(fn, args, "Eager")
    with torch.jit.fuser("fuser2"):
        benchmark_helper.time_with_manual_timer(opt_fn, args, "AOTAutograd")
