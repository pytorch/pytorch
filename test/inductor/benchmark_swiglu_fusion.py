"""
Benchmark script for SwiGLU FFN epilogue fusion with Triton GEMM templates.

Compares configurations at MRS-relevant shapes:
1. Baseline: cuBLAS (ATEN backend)
2. Triton GEMM only (no epilogue fusion)
3. Triton GEMM + epilogue fusion
4. Trunk autoWS TRITON-only: persistent TMA + upstream add_warp_specialize (Blackwell only)
5. Trunk autoWS ATEN+TRITON: persistent TMA + cuBLAS competes (Blackwell only)
6. Meta autoWS TRITON-only: persistent TMA + Meta WS passes (Blackwell only)
7. Meta autoWS ATEN+TRITON: mixed backends + Meta WS (Blackwell only)

Reports latency, peak memory, and runtime kernel launch count for each
configuration. Blackwell WS configs automatically set Inductor config flags
(triton.use_meta_ws, triton.use_meta_partition) - no env vars required.

Usage:
    # Default (MRS shape [5120, 4096]):
    buck2 run @fbcode//mode/opt -c fbcode.enable_gpu_sections=true \
        fbcode//caffe2/test/inductor:run_benchmark_swiglu_fusion

    # RLLayer SwiGLU FFN shape [5120, 256] -> [5120, 512]:
    buck2 run @fbcode//mode/opt -c fbcode.enable_gpu_sections=true \
        fbcode//caffe2/test/inductor:run_benchmark_swiglu_fusion \
        -- --shape-preset rllayer-ffn

    # On Blackwell (autoWS set via Inductor config, no env vars needed):
    buck2 run @fbcode//mode/opt \
        -c fbcode.enable_gpu_sections=true \
        -c fbcode.platform010_cuda_version=12.8 \
        -c fbcode.nvcc_arch=b200a \
        fbcode//caffe2/test/inductor:run_benchmark_swiglu_fusion \
        -- --shape-preset rllayer-ffn --verbose
"""

import argparse
import logging
import math
import os
import time

import torch
import torch._inductor.config as inductor_config


log: logging.Logger = logging.getLogger(__name__)


def _print_diagnostics() -> dict[str, bool]:
    """Print GPU capabilities and Triton template readiness diagnostics."""
    print("\n" + "=" * 72)
    print("DIAGNOSTICS")
    print("=" * 72)

    # GPU info
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        name = torch.cuda.get_device_name(dev)
        major, minor = torch.cuda.get_device_capability(dev)
        arch = major * 10 + minor
        print(f"GPU:                    {name}")
        print(f"Compute Capability:     {major}.{minor} (arch={arch})")
    else:
        print("GPU:                    CUDA not available!")
        return {
            "is_blackwell": False,
            "has_tma_device": False,
            "has_tensor_desc": False,
        }

    # Blackwell arch check
    try:
        from torch._inductor.codegen.cuda.cuda_env import is_datacenter_blackwell_arch

        is_blackwell = is_datacenter_blackwell_arch()
    except ImportError:
        is_blackwell = False
    print(f"is_datacenter_blackwell_arch(): {is_blackwell}")

    # TMA support checks — can_use_tma() gates on has_triton_tma_device()
    try:
        from torch.utils._triton import has_triton_tma_device

        has_tma_device = has_triton_tma_device()
    except (ImportError, AttributeError):
        has_tma_device = False
    print(f"has_triton_tma_device():        {has_tma_device}")

    # Check what TMA device APIs are actually importable
    tma_old_api = False
    try:
        from triton.language.extra.cuda import (  # noqa: F401
            experimental_device_tensormap_create2d,
        )

        tma_old_api = True
    except ImportError:
        pass
    print(
        f"  triton old TMA API (experimental_device_tensormap_create2d): {tma_old_api}"
    )

    tma_new_api = False
    try:
        from triton.language import make_tensor_descriptor  # noqa: F401

        tma_new_api = True
    except ImportError:
        pass
    print(f"  triton new TMA API (tl.make_tensor_descriptor): {tma_new_api}")

    try:
        from torch.utils._triton import has_triton_tensor_descriptor_host_tma

        has_tensor_desc = has_triton_tensor_descriptor_host_tma()
    except (ImportError, AttributeError):
        has_tensor_desc = False
    print(f"has_triton_tensor_descriptor_host_tma(): {has_tensor_desc}")

    # Check the combined Blackwell check
    try:
        from torch.utils._triton import has_datacenter_blackwell_tma_device

        has_bw_tma = has_datacenter_blackwell_tma_device()
    except (ImportError, AttributeError):
        has_bw_tma = False
    print(f"has_datacenter_blackwell_tma_device(): {has_bw_tma}")

    # Config state
    print(
        f"config.triton.enable_persistent_tma_matmul: "
        f"{inductor_config.triton.enable_persistent_tma_matmul}"
    )

    # Environment variables
    env_vars = [
        "TRITON_USE_META_WS",
        "TRITON_USE_META_PARTITION",
        "ENABLE_PERSISTENT_TMA_MATMUL",
        "TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC",
        "TORCH_LOGS",
    ]
    print("\nEnvironment variables:")
    for var in env_vars:
        val = os.environ.get(var, "<not set>")
        print(f"  {var}={val}")

    # Triton version
    try:
        import triton

        print(f"\nTriton version: {triton.__version__}")
    except (ImportError, AttributeError):
        print("\nTriton version: <unavailable>")

    # Template availability summary
    print("\nTemplate readiness:")
    print("  Basic Triton GEMM:          always available")
    tma_ready = has_tma_device and inductor_config.triton.enable_persistent_tma_matmul
    print(f"  Persistent TMA:             {'READY' if tma_ready else 'NOT READY'}")
    if not tma_ready:
        if not has_tma_device:
            print("    -> Missing: has_triton_tma_device()=False")
            print(
                "       Need triton.language.make_tensor_descriptor or legacy TMA API"
            )
        if not inductor_config.triton.enable_persistent_tma_matmul:
            print("    -> Missing: enable_persistent_tma_matmul=True")
    bw_ready = tma_ready and has_tensor_desc and is_blackwell
    print(f"  Blackwell WS + TMA:         {'READY' if bw_ready else 'NOT READY'}")
    if not bw_ready and is_blackwell:
        if not has_tensor_desc:
            print("    -> Missing: triton.tools.tensor_descriptor.TensorDescriptor")
        if not has_tma_device:
            print("    -> Missing: has_triton_tma_device()=False (blocks can_use_tma)")
        if not inductor_config.triton.enable_persistent_tma_matmul:
            print("    -> Missing: enable_persistent_tma_matmul=True")
    print("=" * 72 + "\n")

    return {
        "is_blackwell": is_blackwell,
        "has_tma_device": has_tma_device,
        "has_tensor_desc": has_tensor_desc,
    }


class SwiGLUFFN(torch.nn.Module):
    """SwiGLU FFN: SiLU(x @ W_gate) * (x @ W_up) @ W_down"""

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.w_gate = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w_up = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w_down = torch.nn.Linear(hidden_dim, input_dim, bias=bias)
        self.silu = torch.nn.SiLU()
        self._init_weights()

    def _init_weights(self) -> None:
        for name in ("w_gate", "w_up", "w_down"):
            layer = getattr(self, name)
            torch.nn.init.normal_(
                layer.weight, mean=0.0, std=1.0 / math.sqrt(layer.in_features)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        return self.w_down(hidden)


class SwiGLUFFNWithRMSNormResidual(torch.nn.Module):
    """Full MRS pattern: RMSNorm -> SwiGLU FFN -> residual add."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = torch.nn.RMSNorm(input_dim)
        self.ffn = SwiGLUFFN(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


def _do_bench(fn, warmup=1000, rep=2000):
    """Simple benchmark: run fn with warmup then measure rep iterations."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / rep * 1000  # ms


def _count_runtime_kernels(fn, verbose=False):
    """Count actual CUDA kernel launches using profiler."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        fn()
        torch.cuda.synchronize()

    kernel_count = 0
    kernel_names = []
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.count > 0:
            kernel_count += evt.count
            kernel_names.append((evt.key, evt.count))

    if verbose:
        print("    CUDA kernel trace:")
        if kernel_names:
            for name, count in sorted(kernel_names, key=lambda x: -x[1]):
                tag = ""
                if "triton" in name.lower():
                    tag = " [TRITON]"
                elif "gemm" in name.lower() or "cutlass" in name.lower():
                    tag = " [cuBLAS/CUTLASS]"
                print(f"      {name}: {count}x{tag}")
        else:
            print("      (no CUDA kernels detected)")
    return kernel_count


# Shape presets derived from RLLayer model config
# (batch_size=512, num_targets=10, embedding_dim=256, ffn_hidden_dim=512)
SHAPE_PRESETS: dict[str, tuple[int, int, int]] = {
    # (batch_dim, input_dim, hidden_dim)
    "mrs": (5120, 4096, 8192),  # Original D97188817 MRS region shape
    "rllayer-ffn": (5120, 256, 512),  # Cross-attn SwiGLU FFN
    "rllayer-mlp": (5120, 2048, 1024),  # Sparse module final_mlp mid-layer
    "rllayer-attn": (81920, 256, 512),  # Cross-attn larger batch (5120*16)
}


def _benchmark_config(
    model_cls,
    input_shape,
    hidden_dim,
    dtype,
    config_dict,
    config_name,
    include_backward,
    verbose,
):
    """Compile model with given config and benchmark it."""
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    input_dim = input_shape[1]

    model = model_cls(input_dim, hidden_dim).to(device="cuda", dtype=dtype)
    x = torch.randn(*input_shape, device="cuda", dtype=dtype)

    if verbose:
        print(f"    Config dict: {config_dict}")

    with inductor_config.patch(config_dict):
        if verbose:
            print(
                f"    enable_persistent_tma_matmul="
                f"{inductor_config.triton.enable_persistent_tma_matmul}"
            )
        compiled_model = torch.compile(model, fullgraph=True)

        if include_backward:
            x_grad = x.detach().clone().requires_grad_(True)

            def fn():
                out = compiled_model(x_grad)
                out.sum().backward(retain_graph=True)
        else:

            def fn():
                compiled_model(x)

        # Warmup (includes compilation)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()

    # Count actual runtime kernel launches
    kernel_count = _count_runtime_kernels(fn, verbose=verbose)

    # Measure peak memory
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Benchmark latency
    latency_ms = _do_bench(fn)

    return {
        "config": config_name,
        "kernel_launches": kernel_count,
        "latency_ms": latency_ms,
        "peak_memory_mb": peak_mem_mb,
    }


# ---- Config definitions ----
# Configs are run in order. Blackwell-specific configs are added dynamically
# when a Blackwell GPU is detected (see main()).
# All Triton configs use autotune_in_subproc=True to avoid compilation hangs
# (Triton template compilation can get stuck in scheduling passes).

CONFIGS = {
    "cuBLAS (ATEN)": {
        "max_autotune": False,
        "epilogue_fusion": False,
    },
    "Triton GEMM (no fusion)": {
        "max_autotune": True,
        "max_autotune_gemm_backends": "TRITON",
        "epilogue_fusion": False,
        "autotune_in_subproc": True,
    },
    "Triton GEMM + epilogue fusion": {
        "max_autotune": True,
        "max_autotune_gemm_backends": "TRITON",
        "epilogue_fusion": True,
        "autotune_in_subproc": True,
    },
}

BLACKWELL_CONFIGS = {
    "Trunk autoWS (TRITON-only)": {
        "max_autotune": True,
        "max_autotune_gemm_backends": "TRITON",
        "epilogue_fusion": True,
        "triton.enable_persistent_tma_matmul": True,
        "autotune_in_subproc": True,
    },
    "Trunk autoWS (ATEN+TRITON)": {
        "max_autotune": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
        "epilogue_fusion": True,
        "epilogue_fusion_first": True,
        "coordinate_descent_tuning": True,
        "triton.enable_persistent_tma_matmul": True,
        "autotune_in_subproc": True,
    },
    "Meta autoWS (TRITON-only)": {
        "max_autotune": True,
        "max_autotune_gemm_backends": "TRITON",
        "epilogue_fusion": True,
        "triton.enable_persistent_tma_matmul": True,
        "triton.use_meta_ws": True,
        "triton.use_meta_partition": True,
        "autotune_in_subproc": True,
    },
    "Meta autoWS (ATEN+TRITON)": {
        "max_autotune": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
        "epilogue_fusion": True,
        "epilogue_fusion_first": True,
        "coordinate_descent_tuning": True,
        "triton.enable_persistent_tma_matmul": True,
        "triton.use_meta_ws": True,
        "triton.use_meta_partition": True,
        "autotune_in_subproc": True,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark SwiGLU FFN epilogue fusion")
    parser.add_argument(
        "--shape-preset",
        type=str,
        default=None,
        choices=list(SHAPE_PRESETS.keys()),
        help=(
            "Use a predefined shape. Overrides --batch-dim/--input-dim/--hidden-dim. "
            "Choices: mrs=[5120,4096,8192], rllayer-ffn=[5120,256,512], "
            "rllayer-mlp=[5120,2048,1024], rllayer-attn=[81920,256,512]"
        ),
    )
    parser.add_argument(
        "--batch-dim",
        type=int,
        default=5120,
        help="Batch dimension (default: 5120, matching MRS Region [11/0])",
    )
    parser.add_argument(
        "--input-dim", type=int, default=4096, help="Input dimension (default: 4096)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension (default: input_dim * 2)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type (default: bf16)",
    )
    parser.add_argument(
        "--include-backward",
        action="store_true",
        help="Include backward pass in benchmark",
    )
    parser.add_argument(
        "--with-rmsnorm",
        action="store_true",
        help="Include RMSNorm + residual (full MRS pattern)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed diagnostics, kernel traces, and template selection info",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only configs whose name contains this substring (e.g. --only Blackwell)",
    )
    args = parser.parse_args()

    # Enable TORCH_LOGS for template selection debugging when verbose
    if args.verbose and not os.environ.get("TORCH_LOGS"):
        os.environ["TORCH_LOGS"] = "autotuning"
        logging.basicConfig(level=logging.DEBUG)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Resolve shape
    if args.shape_preset:
        batch_dim, input_dim, hidden_dim = SHAPE_PRESETS[args.shape_preset]
        print(f"[INFO] Using shape preset '{args.shape_preset}'\n")
    else:
        batch_dim = args.batch_dim
        input_dim = args.input_dim
        hidden_dim = args.hidden_dim if args.hidden_dim else input_dim * 2

    input_shape = (batch_dim, input_dim)
    model_cls = SwiGLUFFNWithRMSNormResidual if args.with_rmsnorm else SwiGLUFFN

    # Run diagnostics
    diag = _print_diagnostics()

    model_name = model_cls.__name__
    print(f"{'=' * 72}")
    print("SwiGLU FFN Epilogue Fusion Benchmark")
    print(f"{'=' * 72}")
    print(f"Model:      {model_name}")
    print(
        f"Shape:      [{batch_dim}, {input_dim}] -> hidden [{batch_dim}, {hidden_dim}]"
    )
    gemm_shapes = (
        f"  GEMMs:      [{batch_dim},{input_dim}]@[{input_dim},{hidden_dim}] (gate+up), "
        f"[{batch_dim},{hidden_dim}]@[{hidden_dim},{input_dim}] (down)"
    )
    print(gemm_shapes)
    print(f"Dtype:      {args.dtype}")
    print(f"Backward:   {args.include_backward}")
    print(f"{'=' * 72}\n")

    # Build config list: always include base configs, add Blackwell if detected
    configs = dict(CONFIGS)
    if diag["is_blackwell"] and diag["has_tensor_desc"]:
        configs.update(BLACKWELL_CONFIGS)
        print("[INFO] Blackwell detected — including Blackwell WS configs\n")
    elif diag["is_blackwell"]:
        print(
            "[WARN] Blackwell GPU detected but tensor descriptor API unavailable.\n"
            "       Blackwell WS configs will be SKIPPED.\n"
            "       Check Triton version (need >= 3.3.2+fb).\n"
        )
    else:
        print("[INFO] Non-Blackwell GPU — Blackwell WS configs skipped\n")

    # Filter configs if --only is specified
    if args.only:
        filtered = {k: v for k, v in configs.items() if args.only.lower() in k.lower()}
        if not filtered:
            print(f"[ERROR] No configs match --only='{args.only}'")
            print(f"Available configs: {list(configs.keys())}")
            return
        configs = filtered
        print(f"[INFO] Running only: {list(configs.keys())}\n")

    results = []
    for config_name, config_dict in configs.items():
        print(f"Benchmarking: {config_name}...")
        try:
            result = _benchmark_config(
                model_cls=model_cls,
                input_shape=input_shape,
                hidden_dim=hidden_dim,
                dtype=dtype,
                config_dict=config_dict,
                config_name=config_name,
                include_backward=args.include_backward,
                verbose=args.verbose,
            )
            results.append(result)
            print(
                f"  Kernel launches: {result['kernel_launches']:>3}  "
                f"Latency: {result['latency_ms']:>8.3f} ms  "
                f"Peak Mem: {result['peak_memory_mb']:>8.1f} MB"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            results.append(
                {
                    "config": config_name,
                    "kernel_launches": -1,
                    "latency_ms": float("inf"),
                    "peak_memory_mb": 0.0,
                }
            )

    # Summary table
    print(f"\n{'=' * 72}")
    print(
        f"{'Configuration':<35} {'Launches':>8} {'Latency(ms)':>12}"
        f" {'Peak Mem(MB)':>13}"
    )
    print(f"{'-' * 72}")
    for r in results:
        launches = (
            f"{r['kernel_launches']:>8}" if r["kernel_launches"] >= 0 else "  FAILED"
        )
        latency = (
            f"{r['latency_ms']:>12.3f}"
            if r["latency_ms"] != float("inf")
            else "         N/A"
        )
        print(f"{r['config']:<35} {launches} {latency} {r['peak_memory_mb']:>13.1f}")
    print(f"{'=' * 72}")

    # Compute speedups relative to cuBLAS baseline
    valid_results = [r for r in results if r["kernel_launches"] >= 0]
    if len(valid_results) >= 2:
        baseline = valid_results[0]
        print(f"\nComparison vs {baseline['config']}:")
        for r in valid_results[1:]:
            speedup = baseline["latency_ms"] / r["latency_ms"]
            kernel_diff = r["kernel_launches"] - baseline["kernel_launches"]
            mem_saving = baseline["peak_memory_mb"] - r["peak_memory_mb"]
            print(
                f"  {r['config']}: "
                f"{speedup:.2f}x latency, "
                f"{kernel_diff:+d} kernel launches, "
                f"{mem_saving:+.1f} MB memory"
            )


if __name__ == "__main__":
    main()
