#!/usr/bin/env python3

import gc
import importlib
import logging
import os
import re
import sys
import warnings
from collections import namedtuple

import torch


try:
    from .common import BenchmarkRunner, main
except ImportError:
    from common import BenchmarkRunner, main

from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs

try:
    from .torchbench_output_processors import PROCESS_TRAIN_MODEL_OUTPUT
except ImportError:
    from torchbench_output_processors import PROCESS_TRAIN_MODEL_OUTPUT

try:
    from .torchbench_utils import _reassign_parameters, setup_torchbench_cwd
except ImportError:
    from torchbench_utils import _reassign_parameters, setup_torchbench_cwd

try:
    from .torchbench_config import TorchBenchConfig
except ImportError:
    from torchbench_config import TorchBenchConfig


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

    def _ensure_config_setup(self):
        """Ensure config is setup with args if not already done."""
        if hasattr(self, 'args') and not hasattr(self._tb_config, 'args'):
            self._tb_config.set_args(self.args)

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
        if self.args.enable_activation_checkpointing:
            raise NotImplementedError(
                "Activation checkpointing not implemented for Torchbench models"
            )
        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode
        candidates = [
            f"torchbenchmark.models.{model_name}",
            f"torchbenchmark.canary_models.{model_name}",
            f"torchbenchmark.models.fb.{model_name}",
        ]
        for c in candidates:
            try:
                module = importlib.import_module(c)
                break
            except ModuleNotFoundError as e:
                if e.name != c:
                    raise
        else:
            raise ImportError(f"could not import any of {candidates}")
        benchmark_cls = getattr(module, "Model", None)
        if benchmark_cls is None:
            raise NotImplementedError(f"{model_name}.Model is None")

        if not hasattr(benchmark_cls, "name"):
            benchmark_cls.name = model_name

        cant_change_batch_size = (
            not getattr(benchmark_cls, "ALLOW_CUSTOMIZE_BSIZE", True)
            or model_name in self._tb_config._config["dont_change_batch_size"]
        )
        if cant_change_batch_size:
            batch_size = None
        
        self._ensure_config_setup()
        batch_size = self._tb_config.get_batch_size_for_model(model_name, is_training, batch_size)
        
        # Control the memory footprint for few models
        batch_size = self._tb_config.limit_batch_size_for_accuracy(model_name, batch_size)

        # workaround "RuntimeError: not allowed to set torch.backends.cudnn flags"
        torch.backends.__allow_nonbracketed_mutation_flag = True
        if extra_args is None:
            extra_args = []
        if part:
            extra_args += ["--part", part]

        # sam_fast only runs with amp
        if model_name == "sam_fast":
            self.args.amp = True
            self.setup_amp()

        if model_name == "vision_maskrcnn" and is_training:
            # Output of vision_maskrcnn model is a list of bounding boxes,
            # sorted on the basis of their scores. This makes accuracy
            # comparison hard with torch.compile. torch.compile can cause minor
            # divergences in the output because of how fusion works for amp in
            # TorchInductor compared to eager.  Therefore, instead of looking at
            # all the bounding boxes, we compare only top 4.
            model_kwargs = {"box_detections_per_img": 4}
            benchmark = benchmark_cls(
                test="train",
                device=device,
                batch_size=batch_size,
                extra_args=extra_args,
                model_kwargs=model_kwargs,
            )
            use_eval_mode = True
        elif is_training:
            benchmark = benchmark_cls(
                test="train",
                device=device,
                batch_size=batch_size,
                extra_args=extra_args,
            )
        else:
            benchmark = benchmark_cls(
                test="eval",
                device=device,
                batch_size=batch_size,
                extra_args=extra_args,
            )
        model, example_inputs = benchmark.get_module()
        if model_name in [
            "basic_gnn_edgecnn",
            "basic_gnn_gcn",
            "basic_gnn_sage",
            "basic_gnn_gin",
        ]:
            _reassign_parameters(model)

        # Models that must be in train mode while training
        if is_training and (
            not use_eval_mode or model_name in self._tb_config._config["only_training"]
        ):
            model.train()
        else:
            model.eval()
        gc.collect()
        batch_size = benchmark.batch_size
        if model_name == "torchrec_dlrm":
            batch_namedtuple = namedtuple(
                "Batch", "dense_features sparse_features labels"
            )
            example_inputs = tuple(
                batch_namedtuple(
                    dense_features=batch.dense_features,
                    sparse_features=batch.sparse_features,
                    labels=batch.labels,
                )
                for batch in example_inputs
            )
        # Torchbench has quite different setup for yolov3, so directly passing
        # the right example_inputs
        if model_name == "yolov3":
            example_inputs = (torch.rand(batch_size, 3, 384, 512).to(device),)
        # See https://github.com/pytorch/benchmark/issues/1561
        if model_name == "maml_omniglot":
            batch_size = 5
            assert example_inputs[0].shape[0] == batch_size
        if model_name == "vision_maskrcnn":
            batch_size = 1
        # global current_name, current_device
        # current_device = device
        # current_name = benchmark.name

        if self.args.trace_on_xla:
            # work around for: https://github.com/pytorch/xla/issues/4174
            import torch_xla  # noqa: F401
        self.validate_model(model, example_inputs)
        return device, benchmark.name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        from torchbenchmark import _list_canary_model_paths, _list_model_paths

        models = _list_model_paths()
        models += [
            f
            for f in _list_canary_model_paths()
            if os.path.basename(f) in self._tb_config._config["canary_models"]
        ]
        models.sort()

        start, end = self.get_benchmark_indices(len(models))
        for index, model_path in enumerate(models):
            if index < start or index >= end:
                continue

            model_name = os.path.basename(model_path)
            if (
                not re.search("|".join(args.filter), model_name, re.IGNORECASE)
                or re.search("|".join(args.exclude), model_name, re.IGNORECASE)
                or model_name in args.exclude_exact
                or model_name in self.skip_models
            ):
                continue

            yield model_name

    def pick_grad(self, name, is_training):
        if is_training or name in ("maml",):
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def use_larger_multiplier_for_smaller_tensor(self, name):
        return self._tb_config.use_larger_multiplier_for_smaller_tensor(name)

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        self._ensure_config_setup()
        return self._tb_config.get_tolerance_and_cosine_flag(is_training, current_device, name, self.args)

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
