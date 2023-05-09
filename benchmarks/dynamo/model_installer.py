#!/usr/bin/env python3
import argparse
import os

from typing import Set
from benchmarks import TIMM_MODEL_NAMES, HF_MODELS_FILE_NAME, TORCHBENCH_MODELS_FILE_NAME
from timm_models import TimmRunnner
from huggingface import HuggingfaceRunner
from torchbench import TorchBenchmarkRunner

if __name__ == "__main__":

        # timm = TimmRunnner()
        hugg = HuggingfaceRunner()
        torchbench = TorchBenchmarkRunner()
        # for model_name in TIMM_MODEL_NAMES:
        #         timm.install_model(model_name=model_name)
        # for model_name in HF_MODELS_FILE_NAME:
        #         hugg.install_model(model_name=model_name)
        for model_name in TORCHBENCH_MODELS_FILE_NAME:
                torchbench.install_model(model_name=model_name)
