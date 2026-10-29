import operator_benchmark as op_bench

import torch


"""Microbenchmarks for grouped_mm on MoE expert GEMMs.

Shapes follow the Primus-Turbo grouped GEMM benchmark
(benchmark/ops/training/bench_grouped_gemm_torch.py): for each MoE config, the GateUP
(N = 2 * moe_intermediate_size, K = hidden_size) and Down (N = hidden_size,
K = moe_intermediate_size) projections with B = n_routed_experts / EP local experts
for EP in (32, 16, 8), and M tokens routed to every expert.

The forward test is x (B * M, K) @ w (B, N, K)^T with jagged M. The gradient test
times backward through both inputs (dX and dW). Each test first checks the output, dX
and dW against per-group matmuls with Primus-Turbo's tolerances.
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
TOKENS_PER_EXPERT = (512, 1024, 2048, 4096, 8192, 16384)


def moe_shapes(configs=MOE_CONFIGS, tokens_per_expert=TOKENS_PER_EXPERT):
    shapes = []
    for experts, intermediate, hidden in configs:
        projs = {"GateUP": (2 * intermediate, hidden), "Down": (hidden, intermediate)}
        for ep in EP_SIZES:
            if experts % ep != 0:
                continue
            for M in tokens_per_expert:
                for proj, (N, K) in projs.items():
                    shapes.append([proj, experts // ep, M, N, K])
    return shapes


grouped_mm_configs_short = op_bench.config_list(
    attr_names=["case", "B", "M", "N", "K"],
    attrs=[s for s in moe_shapes(MOE_CONFIGS[:1], (512, 4096)) if s[1] == 8],
    cross_product_configs={
        "device": ["cuda"],
        "dtype": [torch.bfloat16],
    },
    tags=["short"],
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

grouped_mm_configs_long = op_bench.config_list(
    attr_names=["case", "B", "M", "N", "K"],
    attrs=moe_shapes(),
    cross_product_configs={"device": ["cuda"], "dtype": list(DTYPES.values())},
    tags=["long"],
)

# The long shapes again, one tag per dtype (e.g. --tag-filter long_bf16): the test
# name filter is an exact match, so tags are the only way to run a single dtype.
grouped_mm_configs_long_per_dtype = [
    config
    for name, dtype in DTYPES.items()
    for config in op_bench.config_list(
        attr_names=["case", "B", "M", "N", "K"],
        attrs=moe_shapes(),
        cross_product_configs={"device": ["cuda"], "dtype": [dtype]},
        tags=[f"long_{name}"],
    )
]


# Primus-Turbo's check_allclose rtol and atol
TOLERANCES = {torch.bfloat16: 1e-2, torch.float16: 1e-2, torch.float32: 1e-4}


def check_correctness(x, w, offs):
    """Primus-Turbo's check: output, dX and dW against per-group matmuls."""
    x_ref, w_ref = (t.detach().requires_grad_() for t in (x, w))
    x, w = (t.detach().requires_grad_() for t in (x, w))
    with torch.enable_grad():
        out = torch.nn.functional.grouped_mm(x, w.transpose(-2, -1), offs=offs)
        ends = offs.tolist()
        groups = zip([0] + ends[:-1], ends)
        ref = torch.cat([x_ref[s:e] @ w_ref[i].t() for i, (s, e) in enumerate(groups)])
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        ref.backward(grad_out)
    tol = TOLERANCES[x.dtype]
    pairs = {"fwd": (out, ref), "dX": (x.grad, x_ref.grad), "dW": (w.grad, w_ref.grad)}
    ok = {k: torch.allclose(a, b, rtol=tol, atol=tol) for k, (a, b) in pairs.items()}
    status = "PASS" if all(ok.values()) else "FAIL"
    details = ", ".join(f"{k}={v}" for k, v in ok.items())
    shapes = f"x {tuple(x.shape)} w {tuple(w.shape)} {x.dtype}"
    print(f"# Correctness Check: {status} ({details}) {shapes}")


class GroupedMMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, case, B, M, N, K, device, dtype):
        # Both inputs need grad so the gradient test times dX and dW together, like Primus.
        opts = {"device": device, "dtype": dtype, "requires_grad": True}
        self.inputs = {
            "x": torch.randn(B * M, K, **opts),
            "w": torch.randn(B, N, K, **opts),
            "offs": torch.arange(1, B + 1, device=device, dtype=torch.int32) * M,
        }
        self.checked = False
        self.set_module_name("grouped_mm")

    def forward(self, x, w, offs):
        # Once per test, on the first call, which is in the untimed warmup.
        if not self.checked:
            self.checked = True
            check_correctness(x, w, offs)
        return torch.nn.functional.grouped_mm(x, w.transpose(-2, -1), offs=offs)

    def get_memory_traffic_bytes(self):
        """read x (B*M, K) and w (B, N, K), write out (B*M, N)"""
        x, w = self.inputs["x"], self.inputs["w"]
        B, N, K = w.shape
        total_m = x.shape[0]
        return (total_m * K + B * N * K + total_m * N) * x.element_size()


grouped_mm_configs = (
    grouped_mm_configs_short
    + grouped_mm_configs_long
    + grouped_mm_configs_long_per_dtype
)
op_bench.generate_pt_test(grouped_mm_configs, GroupedMMBenchmark)
op_bench.generate_pt_gradient_test(grouped_mm_configs, GroupedMMBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
