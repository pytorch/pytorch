import operator_benchmark as op_bench

import torch


"""Microbenchmarks for grouped_mm on MoE expert GEMMs.

Shapes follow the Primus-Turbo grouped GEMM benchmark
(benchmark/ops/training/bench_grouped_gemm_torch.py): for each MoE config, the GateUP
(N = 2 * moe_intermediate_size, K = hidden_size) and Down (N = hidden_size,
K = moe_intermediate_size) projections with B = n_routed_experts / EP local experts and
M tokens routed to every expert. Each M uses one EP, rotating through (32, 16, 8) as M
grows: that still reaches every ck_tile config these shapes select on gfx950, and every
test fits in 40 GB.

The forward test is x (B * M, K) @ w (B, N, K)^T with jagged M. The gradient test
times backward through both inputs (dX and dW).

The long tag runs bf16 and fp16 everywhere, and fp32 only on ROCm.

On ROCm, the CK-Tile numbers need ROCM_ALLOW_GROUP_GEMM_CK_TILE=1; without it grouped_mm
runs hipBLASLt.
"""

# (n_routed_experts, moe_intermediate_size, hidden_size), in Primus-Turbo's order
MOE_CONFIGS = [
    (256, 2048, 7168),
    (384, 3072, 7168),
    (128, 3072, 6144),
    (160, 1536, 5120),
    (64, 1408, 2048),
    (8, 16384, 8192),
    (32, 2880, 2880),
    (128, 2048, 2048),
    (128, 4096, 4096),
    (8, 14336, 4096),
    (8, 16384, 6144),
    (384, 2048, 7168),
    (224, 1920, 8192),
]
EP_SIZES = (32, 16, 8)
TOKENS_PER_EXPERT = (512, 1024, 2048, 4096, 8192)


def moe_shapes(configs=MOE_CONFIGS):
    shapes = []
    for experts, intermediate, hidden in configs:
        projs = {"GateUP": (2 * intermediate, hidden), "Down": (hidden, intermediate)}
        eps = [ep for ep in EP_SIZES if experts % ep == 0]
        for i, M in enumerate(TOKENS_PER_EXPERT):
            for proj, (N, K) in projs.items():
                shapes.append([proj, experts // eps[i % len(eps)], M, N, K])
    return shapes


grouped_mm_configs_short = op_bench.config_list(
    attr_names=["case", "B", "M", "N", "K"],
    attrs=[s for s in moe_shapes(MOE_CONFIGS[:1]) if s[2] in (512, 4096)],
    cross_product_configs={
        "device": ["cuda"],
        "dtype": [torch.bfloat16],
    },
    tags=["short"],
)

# fp32 only on ROCm, where ck_tile has an fp32 grouped GEMM. On CUDA, fp32 grouped_mm is
# the per-group mm fallback at the non-tensor-core rate, which would push the A100 job
# past its timeout.
DTYPES = [torch.bfloat16, torch.float16]
if torch.version.hip:
    DTYPES.append(torch.float32)

grouped_mm_configs_long = op_bench.config_list(
    attr_names=["case", "B", "M", "N", "K"],
    attrs=moe_shapes(),
    cross_product_configs={"device": ["cuda"], "dtype": DTYPES},
    tags=["long"],
)


class DenseGrad(torch.autograd.Function):
    """Identity whose backward densifies the gradient: the runner backprops out.mean(), whose
    stride-0 expanded gradient grouped_mm's backward rejects. The runner repeats the same
    backward through one graph, so the copy is made once and reused, keeping it out of the
    timed backward."""

    @staticmethod
    def forward(ctx, out):
        ctx.dense_grad = None
        return out.view_as(out)

    @staticmethod
    def backward(ctx, grad):
        if ctx.dense_grad is None:
            ctx.dense_grad = grad.contiguous()
        return ctx.dense_grad


class GroupedMMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, case, B, M, N, K, device, dtype):
        # Both inputs need grad so the gradient test times dX and dW together, like Primus.
        opts = {"device": device, "dtype": dtype, "requires_grad": True}
        self.inputs = {
            "x": torch.randn(B * M, K, **opts),
            "w": torch.randn(B, N, K, **opts),
            "offs": torch.arange(1, B + 1, device=device, dtype=torch.int32) * M,
        }
        # The gradient test runs backward hundreds of times. Drop each .grad once it lands so
        # a backward never holds new and accumulated gradients at once (OOM on 40 GB GPUs).
        for t in (self.inputs["x"], self.inputs["w"]):
            t.register_post_accumulate_grad_hook(lambda t: setattr(t, "grad", None))
        self.set_module_name("grouped_mm")

    def forward(self, x, w, offs):
        out = torch.nn.functional.grouped_mm(x, w.transpose(-2, -1), offs=offs)
        return DenseGrad.apply(out)

    def get_memory_traffic_bytes(self):
        """read x (B*M, K) and w (B, N, K), write out (B*M, N)"""
        x, w = self.inputs["x"], self.inputs["w"]
        B, N, K = w.shape
        total_m = x.shape[0]
        return (total_m * K + B * N * K + total_m * N) * x.element_size()


grouped_mm_configs = grouped_mm_configs_short + grouped_mm_configs_long
op_bench.generate_pt_test(grouped_mm_configs, GroupedMMBenchmark)
op_bench.generate_pt_gradient_test(grouped_mm_configs, GroupedMMBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
