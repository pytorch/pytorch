#!/usr/bin/env python3

import logging
import os
import warnings

import torch


try:
    from .common import BenchmarkRunner, main
except ImportError:
    from common import BenchmarkRunner, main

try:
    from .torchbench_config import TorchBenchConfig
    from .torchbench_model_loader import TorchBenchModelLoader
    from .torchbench_utils import process_train_model_output, setup_torchbench_cwd
except ImportError:
    from torchbench_config import TorchBenchConfig
    from torchbench_model_loader import TorchBenchModelLoader
    from torchbench_utils import process_train_model_output, setup_torchbench_cwd

from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs


# We are primarily interested in tf32 datatype
torch.backends.cuda.matmul.allow_tf32 = True

# Enable FX graph caching
if "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ:
    torch._inductor.config.fx_graph_cache = True

# Enable Autograd caching
if "TORCHINDUCTOR_AUTOGRAD_CACHE" not in os.environ:
    torch._functorch.config.enable_autograd_cache = True


class TorchBenchmarkRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "torchbench"
        self.optimizer = None
        self.config = None
        self.model_loader = None

    def _init_components(self):
        """Initialize config and model loader components after args are available."""
        if self.config is None:
            self.config = TorchBenchConfig(self.args)
            self.model_loader = TorchBenchModelLoader(self.config, self.args)
            self.model_loader._runner = (
                self  # Pass reference for setup_amp and other methods
            )

    @property
    def skip_models(self):
        self._init_components()
        return self.config.skip_models

    @property
    def skip_models_for_cpu(self):
        self._init_components()
        return self.config.skip_models_for_cpu

    @property
    def skip_models_for_cuda(self):
        self._init_components()
        return self.config.skip_models_for_cuda

    @property
    def skip_models_for_freezing_cuda(self):
        self._init_components()
        return self.config.skip_models_for_freezing_cuda

    @property
    def disable_cudagraph_models(self):
        self._init_components()
        return self.config.disable_cudagraph_models

    @property
    def skip_models_for_freezing_cpu(self):
        self._init_components()
        return self.config.skip_models_for_freezing_cpu

    @property
    def slow_models(self):
        self._init_components()
        return self.config.slow_models

    @property
    def very_slow_models(self):
        self._init_components()
        return self.config.very_slow_models

    @property
    def non_deterministic_models(self):
        self._init_components()
        return self.config.non_deterministic_models

    @property
    def get_output_amp_train_process_func(self):
        return process_train_model_output

    @property
    def skip_not_suitable_for_training_models(self):
        self._init_components()
        return self.config.skip_not_suitable_for_training_models

    @property
    def failing_fx2trt_models(self):
        self._init_components()
        return self.config.failing_fx2trt_models

    @property
    def force_amp_for_fp16_bf16_models(self):
        self._init_components()
        return self.config.force_amp_for_fp16_bf16_models

    @property
    def force_fp16_for_bf16_models(self):
        self._init_components()
        return self.config.force_fp16_for_bf16_models

    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        self._init_components()
        return self.config.skip_accuracy_checks_large_models_dashboard

    @property
    def skip_accuracy_check_as_eager_non_deterministic(self):
        self._init_components()
        return self.config.skip_accuracy_check_as_eager_non_deterministic

    @property
    def skip_multiprocess_models(self):
        self._init_components()
        return self.config.skip_multiprocess_models

    @property
    def skip_models_due_to_control_flow(self):
        self._init_components()
        return self.config.skip_models_due_to_control_flow

    @property
    def skip_models_due_to_export_not_supported(self):
        self._init_components()
        return self.config.skip_models_due_to_export_not_supported

    @property
    def guard_on_nn_module_models(self):
        self._init_components()
        return self.config.guard_on_nn_module_models

    @property
    def inline_inbuilt_nn_modules_models(self):
        self._init_components()
        return self.config.inline_inbuilt_nn_modules_models

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        part=None,
        extra_args=None,
    ):
        self._init_components()
        device, name, model, example_inputs, batch_size = self.model_loader.load_model(
            device, model_name, batch_size, part, extra_args
        )
        self.validate_model(model, example_inputs)
        return device, name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        self._init_components()
        # Inject the get_benchmark_indices method into the model loader
        self.model_loader._get_benchmark_indices = self.get_benchmark_indices
        yield from self.model_loader.iter_model_names(args)

    def pick_grad(self, name, is_training):
        if is_training or name in ("maml",):
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def use_larger_multiplier_for_smaller_tensor(self, name):
        self._init_components()
        return self.config.use_larger_multiplier_for_smaller_tensor(name)

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        self._init_components()
        return self.config.get_tolerance_and_cosine_flag(
            is_training, current_device, name
        )

    def compute_loss(self, pred):
        return reduce_to_scalar_loss(pred)

    def forward_pass(self, mod, inputs, collect_outputs=True):
        with self.autocast(**self.autocast_arg):
            if isinstance(inputs, dict):
                return mod(**inputs)
            else:
                return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast(**self.autocast_arg):
            if isinstance(cloned_inputs, dict):
                pred = mod(**cloned_inputs)
            else:
                pred = mod(*cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, None, loss, cloned_inputs)
        return None


def torchbench_main():
    original_dir = setup_torchbench_cwd()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TorchBenchmarkRunner(), original_dir)


if __name__ == "__main__":
    torchbench_main()
