#!/usr/bin/env python3
"""Train all MNIST models on Vulkan/CUDA/CPU x fp32/fp16/bf16, plot convergence.

Usage:
  # Run Vulkan + CPU training (+ load CUDA results if available):
  python plot_convergence.py

  # Plot from previously saved results:
  python plot_convergence.py --plot-only

Each model/backend/dtype combo runs in a separate subprocess to isolate
Vulkan cleanup segfaults from the main process.

CUDA training must be run separately via run_convergence.sh (uninstalls torch_vulkan first).
"""

import gc
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Configuration ────────────────────────────────────────────────

MODELS = {
    "MLP (all activations)": {
        "cls": "MLPAllActivations",
        "kwargs": {},
        "train_kwargs": {"epochs": 5, "lr": 0.1, "batch_size": 256},
        "half_kwargs": {"epochs": 5, "lr": 0.1, "batch_size": 256},
        "max_samples": 2000,
    },
    "CNN": {
        "cls": "CNNBasic",
        "kwargs": {},
        "train_kwargs": {"epochs": 10, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "half_kwargs": {"epochs": 10, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "max_samples": 2000,
    },
    "ResNet": {
        "cls": "ResNetMini",
        "kwargs": {},
        "train_kwargs": {"epochs": 10, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "half_kwargs": {"epochs": 10, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "max_samples": 2000,
    },
    "Transformer": {
        "cls": "TransformerClassifier",
        "kwargs": {},
        "train_kwargs": {"epochs": 8, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "half_kwargs": {"epochs": 8, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "max_samples": 2000,
    },
    "LLM (RMSNorm+SwiGLU)": {
        "cls": "LLMStyleClassifier",
        "kwargs": {"d_model": 32, "nhead": 4, "num_layers": 1},
        "train_kwargs": {"epochs": 5, "lr": 0.05, "batch_size": 256},
        "half_kwargs": {"epochs": 5, "lr": 0.05, "batch_size": 256},
        "max_samples": 1000,
        "vulkan_only": True,
    },
    "AdvancedOps": {
        "cls": "AdvancedOpsModel",
        "kwargs": {},
        "train_kwargs": {"epochs": 10, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "half_kwargs": {"epochs": 10, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "max_samples": 2000,
    },
}

DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}

# ── CUDA Results Loading ─────────────────────────────────────────

CUDA_RESULTS_PATH = Path(__file__).parent / "cuda_results.json"

def load_cuda_results():
    """Load pre-computed CUDA results from cuda_results.json."""
    if CUDA_RESULTS_PATH.exists():
        with open(CUDA_RESULTS_PATH) as f:
            results = json.load(f)
        print(f"Loaded CUDA results from {CUDA_RESULTS_PATH}")
        for model_name, dtypes in results.items():
            for dtype_name, data in dtypes.items():
                if "error" in data:
                    print(f"  [CUDA] {model_name} {dtype_name}: ERROR - {data['error'][:80]}")
                else:
                    losses = data["losses"]
                    accs = data["accs"]
                    marker = "OK" if data["converged"] else "FAIL"
                    print(f"  [CUDA] {model_name} {dtype_name}: loss {losses[0]:.4f} -> {losses[-1]:.4f}  "
                          f"acc {accs[0]*100:.1f}% -> {accs[-1]*100:.1f}%  [{marker}]")
        return results
    else:
        print(f"No CUDA results found at {CUDA_RESULTS_PATH}")
        print("Run: bash tests/run_convergence.sh   (handles uninstall/reinstall)")
        return {}


# ── Subprocess Training ──────────────────────────────────────────

WORKER_SCRIPT = Path(__file__).parent / "run_convergence_worker.py"

def run_training_subprocess(model_name, model_cfg, backend, dtype_str, max_retries=2):
    """Run training in a subprocess to isolate Vulkan cleanup segfaults."""
    for attempt in range(max_retries):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        config = {
            "model_name": model_name,
            "model_cfg": model_cfg,
            "backend": backend,
            "dtype_str": dtype_str,
            "output_path": output_path,
        }

        try:
            result = subprocess.run(
                [sys.executable, str(WORKER_SCRIPT), json.dumps(config)],
                capture_output=True, text=True, timeout=300,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
            )

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                with open(output_path) as f:
                    data = json.load(f)
                # NaN check — rare SwiftShader flake, retry
                if "losses" in data and any(x != x for x in data["losses"]):
                    if attempt < max_retries - 1:
                        continue  # retry
                    return {"error": "persistent NaN in losses after retries"}
                return data

            # Segfault but no result file — retry
            if (result.returncode == -11 or result.returncode == 139) and attempt < max_retries - 1:
                continue
            if result.returncode == -11 or result.returncode == 139:
                return {"error": "segfault (Vulkan cleanup)"}

            return {"error": f"exit code {result.returncode}: {result.stderr[:200]}"}

        except subprocess.TimeoutExpired:
            return {"error": "timeout (300s)"}
        except Exception as e:
            return {"error": str(e)[:200]}
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    return {"error": "all retries failed"}


# ── Main ──────────────────────────────────────────────────────────

def main():
    plot_only = "--plot-only" in sys.argv

    results_path = Path(__file__).parent / "convergence_results.json"

    if plot_only:
        if not results_path.exists():
            print(f"No results found at {results_path}")
            return
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded results from {results_path}")
        plot_results(all_results)
        return

    has_cuda = torch.cuda.is_available()
    print(f"CUDA: {has_cuda}" + (f" ({torch.cuda.get_device_name(0)})" if has_cuda else ""))

    # Phase 1: Load pre-computed CUDA results
    cuda_results = {}
    if has_cuda:
        print("\n" + "=" * 70)
        print("PHASE 1: Loading CUDA Results")
        print("=" * 70)
        cuda_results = load_cuda_results()

    # Phase 2: Run Vulkan and CPU training (each combo in subprocess)
    print("\n" + "=" * 70)
    print("PHASE 2: Vulkan + CPU Training (subprocess per combo)")
    print("=" * 70)

    all_results = {}

    for model_name, model_cfg in MODELS.items():
        print(f"\n{'─'*50}")
        print(f"  {model_name}")
        print(f"{'─'*50}")
        all_results[model_name] = {}

        vulkan_only = model_cfg.get("vulkan_only", False)
        backends = ["vulkan"] if vulkan_only else ["vulkan", "cpu"]

        for dtype_name, dtype_str in DTYPES.items():
            all_results[model_name][dtype_name] = {}

            # Add CUDA results if we have them
            if model_name in cuda_results and dtype_name in cuda_results[model_name]:
                all_results[model_name][dtype_name]["cuda"] = cuda_results[model_name][dtype_name]

            for backend in backends:
                data = run_training_subprocess(model_name, model_cfg, backend, dtype_str)
                all_results[model_name][dtype_name][backend] = data

                tag = f"  {backend:6s} {dtype_name:4s}"
                if "error" in data:
                    print(f"{tag}: ERROR - {data['error'][:80]}")
                else:
                    losses, accs = data["losses"], data["accs"]
                    marker = "OK" if data["converged"] else "FAIL"
                    print(f"{tag}: loss {losses[0]:.4f} -> {losses[-1]:.4f}  "
                          f"acc {accs[0]*100:.1f}% -> {accs[-1]*100:.1f}%  [{marker}]")

        # Save incrementally after each model
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Plot
    plot_results(all_results)


# ── Plotting ──────────────────────────────────────────────────────

def plot_results(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not available")
        return

    output_dir = Path(__file__).parent / "convergence_plots"
    output_dir.mkdir(exist_ok=True)

    dtype_names = ["fp32", "fp16", "bf16"]
    backend_styles = {
        "vulkan": {"color": "#E74C3C", "marker": "o", "ls": "-", "lw": 2.5},
        "cuda":   {"color": "#3498DB", "marker": "s", "ls": "--", "lw": 2.0},
        "cpu":    {"color": "#2ECC71", "marker": "^", "ls": ":", "lw": 2.0},
    }

    # Per-model plots
    for model_name, dtypes in results.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"{model_name} — Training Convergence (batch_size=256)",
                     fontsize=16, fontweight="bold")

        for col, dn in enumerate(dtype_names):
            ax_loss, ax_acc = axes[0, col], axes[1, col]
            bd = dtypes.get(dn, {})
            has_data = False

            for backend in ["vulkan", "cuda", "cpu"]:
                d = bd.get(backend, {})
                if "error" in d or "losses" not in d:
                    continue
                s = backend_styles[backend]
                losses, accs = d["losses"], d["accs"]
                epochs = list(range(1, len(losses) + 1))
                final_acc = accs[-1] * 100
                lbl = f"{backend} ({final_acc:.1f}%)"

                ax_loss.plot(epochs, losses, label=lbl, color=s["color"],
                           marker=s["marker"], linestyle=s["ls"], linewidth=s["lw"], markersize=6)
                ax_acc.plot(epochs, [a*100 for a in accs], label=lbl, color=s["color"],
                          marker=s["marker"], linestyle=s["ls"], linewidth=s["lw"], markersize=6)
                has_data = True

            ax_loss.set_title(dn, fontsize=14, fontweight="bold")
            ax_loss.set_ylabel("Loss" if col == 0 else "", fontsize=12)
            ax_loss.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy (%)" if col == 0 else "", fontsize=12)
            ax_acc.set_xlabel("Epoch")
            for ax in [ax_loss, ax_acc]:
                ax.grid(True, alpha=0.3)
                if has_data:
                    ax.legend(fontsize=9)
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                           transform=ax.transAxes, fontsize=14, color="gray")

        plt.tight_layout()
        safe = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        p = output_dir / f"{safe}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")

    # Summary bar charts
    for metric_key, label, fmt, xlabel in [
        ("losses", "Final Loss", "{:.3f}", "Loss"),
        ("accs", "Final Accuracy", "{:.1f}%", "Accuracy (%)"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(22, max(5, len(results) * 0.9 + 2)))
        fig.suptitle(f"{label} — All Models (batch_size=256)", fontsize=16, fontweight="bold")

        for col, dn in enumerate(dtype_names):
            ax = axes[col]
            ax.set_title(dn, fontsize=14, fontweight="bold")

            grouped = defaultdict(dict)
            for mn in results:
                bd = results[mn].get(dn, {})
                short = mn.split("(")[0].strip()
                if short == "LLM":
                    short = "LLM (SwiGLU)"
                for backend in ["vulkan", "cuda", "cpu"]:
                    d = bd.get(backend, {})
                    if "error" not in d and metric_key in d:
                        v = d[metric_key][-1]
                        if metric_key == "accs":
                            v *= 100
                        grouped[short][backend] = v

            labels_list = list(grouped.keys())
            if not labels_list:
                continue

            x = range(len(labels_list))
            w = 0.25
            for i, (b, c) in enumerate([("vulkan", "#E74C3C"), ("cuda", "#3498DB"), ("cpu", "#2ECC71")]):
                vals = [grouped[m].get(b, 0) for m in labels_list]
                present = [grouped[m].get(b) is not None for m in labels_list]
                pos = [xi + i * w for xi in x]
                ax.barh(pos, vals, w, label=b, color=c, alpha=0.85)
                for j, (v, p) in enumerate(zip(vals, present)):
                    if p and v > 0:
                        ax.text(v + (0.5 if metric_key == "accs" else 0.01),
                               pos[j], fmt.format(v), va="center", fontsize=8)

            ax.set_yticks([xi + w for xi in x])
            ax.set_yticklabels(labels_list, fontsize=10)
            ax.set_xlabel(xlabel)
            if metric_key == "accs":
                ax.set_xlim(0, 110)
            ax.legend(fontsize=10)
            ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        p = output_dir / f"summary_{label.lower().replace(' ', '_')}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
