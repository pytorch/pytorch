#!/usr/bin/env python3
"""Run all CUDA training jobs and save results to JSON.

Must be invoked in a venv where torch_vulkan is NOT installed.
Usage: python cuda_batch_runner.py <output_json_path>
"""
import json
import subprocess
import sys
from pathlib import Path

MODELS = {
    "MLP (all activations)": {
        "cls": "MLPAllActivations",
        "kwargs": {},
        "train_kwargs": {"epochs": 5, "lr": 0.1, "batch_size": 256},
        "half_kwargs": {"epochs": 5, "lr": 0.01, "batch_size": 256},
        "max_samples": 2000,
    },
    "CNN": {
        "cls": "CNNBasic",
        "kwargs": {},
        "train_kwargs": {"epochs": 10, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "half_kwargs": {"epochs": 10, "lr": 0.01, "batch_size": 256, "momentum": 0.9},
        "max_samples": 2000,
    },
    "ResNet": {
        "cls": "ResNetMini",
        "kwargs": {},
        "train_kwargs": {"epochs": 10, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "half_kwargs": {"epochs": 10, "lr": 0.01, "batch_size": 256, "momentum": 0.9},
        "max_samples": 2000,
    },
    "Transformer": {
        "cls": "TransformerClassifier",
        "kwargs": {},
        "train_kwargs": {"epochs": 8, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "half_kwargs": {"epochs": 8, "lr": 0.01, "batch_size": 256, "momentum": 0.9},
        "max_samples": 2000,
    },
    "AdvancedOps": {
        "cls": "AdvancedOpsModel",
        "kwargs": {},
        "train_kwargs": {"epochs": 10, "lr": 0.05, "batch_size": 256, "momentum": 0.9},
        "half_kwargs": {"epochs": 10, "lr": 0.01, "batch_size": 256, "momentum": 0.9},
        "max_samples": 2000,
    },
}

DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def main():
    output_path = sys.argv[1]
    worker_script = Path(__file__).parent / "cuda_worker.py"
    results = {}

    for model_name, cfg in MODELS.items():
        results[model_name] = {}
        for dtype_name, dtype_str in DTYPES.items():
            tkw = cfg["train_kwargs"].copy() if dtype_str == "float32" else cfg["half_kwargs"].copy()
            config = {
                "cls_name": cfg["cls"],
                "kwargs": cfg.get("kwargs", {}),
                "train_kwargs": tkw,
                "max_samples": cfg.get("max_samples", 2000),
                "dtype": dtype_str,
            }
            config_path = f"/tmp/cuda_config_{model_name}_{dtype_name}.json"
            result_path = f"/tmp/cuda_output_{model_name}_{dtype_name}.json"

            with open(config_path, "w") as f:
                json.dump(config, f)

            try:
                result = subprocess.run(
                    [sys.executable, str(worker_script), config_path, result_path],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode != 0:
                    stderr_last = result.stderr.strip().split('\n')[-1] if result.stderr else "?"
                    print(f"  [CUDA] {model_name} {dtype_name}: ERROR - {stderr_last[:100]}")
                    results[model_name][dtype_name] = {"error": stderr_last[:200]}
                    continue

                with open(result_path) as f:
                    data = json.load(f)

                results[model_name][dtype_name] = data
                if "error" in data:
                    print(f"  [CUDA] {model_name} {dtype_name}: ERROR - {data['error'][:80]}")
                else:
                    losses = data["losses"]
                    accs = data["accs"]
                    marker = "OK" if data["converged"] else "FAIL"
                    print(f"  [CUDA] {model_name} {dtype_name}: loss {losses[0]:.4f} -> {losses[-1]:.4f}  "
                          f"acc {accs[0]*100:.1f}% -> {accs[-1]*100:.1f}%  [{marker}]")

            except subprocess.TimeoutExpired:
                print(f"  [CUDA] {model_name} {dtype_name}: TIMEOUT")
                results[model_name][dtype_name] = {"error": "timeout"}
            except Exception as e:
                print(f"  [CUDA] {model_name} {dtype_name}: ERROR - {str(e)[:80]}")
                results[model_name][dtype_name] = {"error": str(e)[:200]}

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCUDA results saved to {output_path}")


if __name__ == "__main__":
    main()
