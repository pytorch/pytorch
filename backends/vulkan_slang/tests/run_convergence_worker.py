#!/usr/bin/env python3
"""Worker script: train a single model/backend/dtype combo and write result to a JSON file.

Usage: python run_convergence_worker.py <config_json>
Config JSON: {"model_name": "...", "model_cfg": {...}, "backend": "vulkan"|"cpu", "dtype_str": "float32"|"float16"|"bfloat16", "output_path": "..."}
"""
import gc
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import torch_vulkan to register the backend
import torch_vulkan

# Import models and training fn from the test file
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from tests.test_mnist_training import (
    MLPAllActivations, CNNBasic, ResNetMini, TransformerClassifier,
    LLMStyleClassifier, AdvancedOpsModel, train_model, load_mnist,
)

CLS_MAP = {
    "MLPAllActivations": MLPAllActivations,
    "CNNBasic": CNNBasic,
    "ResNetMini": ResNetMini,
    "TransformerClassifier": TransformerClassifier,
    "LLMStyleClassifier": LLMStyleClassifier,
    "AdvancedOpsModel": AdvancedOpsModel,
}


def main():
    config = json.loads(sys.argv[1])
    model_cfg = config["model_cfg"]
    backend = config["backend"]
    dtype_str = config["dtype_str"]
    output_path = config["output_path"]

    dtype = getattr(torch, dtype_str)
    device = torch.device(backend)
    tkw = model_cfg["train_kwargs"].copy() if dtype_str == "float32" else model_cfg["half_kwargs"].copy()
    max_samples = model_cfg.get("max_samples", 2000)

    images, labels = load_mnist(train=True, max_samples=max_samples)

    try:
        torch.manual_seed(42)
        model = CLS_MAP[model_cfg["cls"]](**model_cfg.get("kwargs", {}))
        losses, accs = train_model(model, images, labels, device, dtype=dtype, **tkw)
        result = {
            "losses": [float(x) for x in losses],
            "accs": [float(x) for x in accs],
            "converged": losses[-1] < losses[0],
        }
    except Exception as e:
        import traceback
        result = {"error": str(e)[:300]}

    with open(output_path, "w") as f:
        json.dump(result, f)
        f.flush()
        os.fsync(f.fileno())

    # Exit immediately to avoid Vulkan cleanup segfault
    os._exit(0)


if __name__ == "__main__":
    main()
