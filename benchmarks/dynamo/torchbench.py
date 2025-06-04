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
    from .torchbench_output_processors import PROCESS_TRAIN_MODEL_OUTPUT
except ImportError:
    from torchbench_output_processors import PROCESS_TRAIN_MODEL_OUTPUT

try:
    from .torchbench_utils import setup_torchbench_cwd
except ImportError:
    from torchbench_utils import setup_torchbench_cwd

try:
    from .torchbench_config import TorchBenchConfig
except ImportError:
    from torchbench_config import TorchBenchConfig

try:
    from .torchbench_model_loader import TorchBenchModelLoader
except ImportError:
    from torchbench_model_loader import TorchBenchModelLoader

try:
    from .torchbench_runner import TorchBenchModelRunner
except ImportError:
    from torchbench_runner import TorchBenchModelRunner


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
        self._tb_config = TorchBenchConfig()
        self._model_loader = None
        self._model_runner = None

    def _ensure_config_setup(self):
        """Ensure config is setup with args if not already done."""
        if hasattr(self, "args") and not hasattr(self._tb_config, "args"):
            self._tb_config.set_args(self.args)
            self._model_loader = TorchBenchModelLoader(self._tb_config, self.args)
            # Setup model runner with autocast settings
            self._model_runner = TorchBenchModelRunner(
                self.autocast, self.autocast_arg, self.grad_scaler
            )
            self._model_runner.set_optimizer_methods(
                self.optimizer_zero_grad, self.optimizer_step
            )

    # Delegate configuration properties to the config object
    @property
    def skip_models(self):
        return self._tb_config.skip_models

    @property
    def skip_models_for_cpu(self):
        return self._tb_config.skip_models_for_cpu

    @property
    def skip_models_for_cuda(self):
        return self._tb_config.skip_models_for_cuda

    @property
    def skip_models_for_freezing_cuda(self):
        return self._tb_config.skip_models_for_freezing_cuda

    @property
    def disable_cudagraph_models(self):
        return self._tb_config.disable_cudagraph_models

    @property
    def skip_models_for_freezing_cpu(self):
        return self._tb_config.skip_models_for_freezing_cpu

    @property
    def slow_models(self):
        return self._tb_config.slow_models

    @property
    def very_slow_models(self):
        return self._tb_config.very_slow_models

    @property
    def non_deterministic_models(self):
        return self._tb_config.non_deterministic_models

    @property
    def get_output_amp_train_process_func(self):
        return PROCESS_TRAIN_MODEL_OUTPUT

    @property
    def skip_not_suitable_for_training_models(self):
        return self._tb_config.skip_not_suitable_for_training_models

    @property
    def failing_fx2trt_models(self):
        return self._tb_config.failing_fx2trt_models

    @property
    def force_amp_for_fp16_bf16_models(self):
        return self._tb_config.force_amp_for_fp16_bf16_models

    @property
    def force_fp16_for_bf16_models(self):
        return self._tb_config.force_fp16_for_bf16_models

    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        self._ensure_config_setup()
        return self._tb_config.skip_accuracy_checks_large_models_dashboard

    @property
    def skip_accuracy_check_as_eager_non_deterministic(self):
        self._ensure_config_setup()
        return self._tb_config.skip_accuracy_check_as_eager_non_deterministic

    @property
    def skip_multiprocess_models(self):
        return self._tb_config.skip_multiprocess_models

    @property
    def skip_models_due_to_control_flow(self):
        return self._tb_config.skip_models_due_to_control_flow

    @property
    def skip_models_due_to_export_not_supported(self):
        return self._tb_config.skip_models_due_to_export_not_supported

    @property
    def guard_on_nn_module_models(self):
        return self._tb_config.guard_on_nn_module_models

    @property
    def inline_inbuilt_nn_modules_models(self):
        return self._tb_config.inline_inbuilt_nn_modules_models

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        part=None,
        extra_args=None,
    ):
        self._ensure_config_setup()

        # Handle sam_fast special case that needs setup_amp
        if model_name == "sam_fast":
            self.args.amp = True
            self.setup_amp()

        device, model_name, model, example_inputs, batch_size = (
            self._model_loader.load_model(
                device, model_name, batch_size, part, extra_args
            )
        )

        # Handle special batch size overrides that the model loader doesn't handle
        if model_name == "vision_maskrcnn":
            batch_size = 1

        self.validate_model(model, example_inputs)
        return device, model_name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        self._ensure_config_setup()
        # Update the model loader to use the right benchmark indices method
        self._model_loader._get_benchmark_indices = (
            lambda total: self.get_benchmark_indices(total)
        )
        return self._model_loader.iter_model_names()

    def pick_grad(self, name, is_training):
        self._ensure_config_setup()
        return self._model_runner.pick_grad(name, is_training)

    def use_larger_multiplier_for_smaller_tensor(self, name):
        return self._tb_config.use_larger_multiplier_for_smaller_tensor(name)

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        self._ensure_config_setup()
        return self._tb_config.get_tolerance_and_cosine_flag(
            is_training, current_device, name, self.args
        )

    def compute_loss(self, pred):
        self._ensure_config_setup()
        return self._model_runner.compute_loss(pred)

    def forward_pass(self, mod, inputs, collect_outputs=True):
        self._ensure_config_setup()
        return self._model_runner.forward_pass(mod, inputs, collect_outputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        self._ensure_config_setup()
        return self._model_runner.forward_and_backward_pass(
            mod, inputs, collect_outputs
        )


def torchbench_main():
    original_dir = setup_torchbench_cwd()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TorchBenchmarkRunner(), original_dir)


if __name__ == "__main__":
    torchbench_main()
