#!/usr/bin/env python3
# flake8: noqa: F821

import copy
import importlib
import logging
import os
import re
import subprocess
import sys
import warnings


try:
    from .common import (
        BenchmarkRunner,
        download_retry_decorator,
        load_yaml_file,
        main,
        reset_rng_state,
    )
except ImportError:
    from common import (
        BenchmarkRunner,
        download_retry_decorator,
        load_yaml_file,
        main,
        reset_rng_state,
    )

try:       
    from .huggingface_llm_models import models 
except ImportError:
    from huggingface_llm_models import models

import torch
from torch._dynamo.testing import collect_results
from torch._dynamo.utils import clone_inputs


log = logging.getLogger(__name__)

# Enable FX graph caching
if "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ:
    torch._inductor.config.fx_graph_cache = True

# Enable Autograd caching
if "TORCHINDUCTOR_AUTOGRAD_CACHE" not in os.environ:
    torch._functorch.config.enable_autograd_cache = True


class HuggingfaceLLMRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "huggingface_llm"

    @property
    def _config(self):
        return load_yaml_file("huggingface_llm.yaml")

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        extra_args=None,
    ):

        benchmark_cls = models[model_name]
        model, example_inputs = benchmark_cls.get_model_and_inputs(model_name, device)

        batch_size = 1
        
        if (
            self.args.training
            and not self.args.use_eval_mode
            and not (
                self.args.accuracy and model_name in self._config["only_inference"]
            )
        ):
            model.train()
        else:
            model.eval()
        
        self.validate_model(model, example_inputs)
        
        return device, model_name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        yield from models.keys()
    
    def forward_pass(self, mod, inputs, collect_outputs=True):
        with self.autocast(**self.autocast_arg):
            torch.compiler.cudagraph_mark_step_begin()
            res = mod.generate(**inputs)# , generation_config=self.generation_config)
            return res
    
    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        if self.args.dashboard or self.args.accuracy:
            # skipping all accuracy checks
            return list(models.keys())
        return set()

    def pick_grad(self, name, is_training):
        if is_training:
            return torch.enable_grad()
        else:
            return torch.no_grad()


def huggingface_llm_main():
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(HuggingfaceLLMRunner())


if __name__ == "__main__":
    huggingface_llm_main()
