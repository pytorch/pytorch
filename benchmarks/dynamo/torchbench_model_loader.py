#!/usr/bin/env python3

"""Model loading and setup logic for TorchBench benchmarks."""

import gc
import importlib
import os
import re
from collections import namedtuple

import torch


try:
    from .torchbench_utils import _reassign_parameters
except ImportError:
    from torchbench_utils import _reassign_parameters


class TorchBenchModelLoader:
    """Handles loading and setup of TorchBench models."""

    def __init__(self, config, args):
        self.config = config
        self.args = args

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        part=None,
        extra_args=None,
    ):
        """Load and set up a TorchBench model.

        Args:
            device: Device to load the model on
            model_name: Name of the model to load
            batch_size: Batch size for the model (optional)
            part: Model part to load (optional)
            extra_args: Additional arguments (optional)

        Returns:
            Tuple of (device, model_name, model, example_inputs, batch_size)
        """
        if self.args.enable_activation_checkpointing:
            raise NotImplementedError(
                "Activation checkpointing not implemented for Torchbench models"
            )
        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode

        # Import the model module
        module = self._import_model_module(model_name)
        benchmark_cls = getattr(module, "Model", None)
        if benchmark_cls is None:
            raise NotImplementedError(f"{model_name}.Model is None")

        if not hasattr(benchmark_cls, "name"):
            benchmark_cls.name = model_name

        # Handle batch size configuration
        batch_size = self._configure_batch_size(model_name, batch_size, is_training)

        # Handle special model configurations
        self._handle_special_model_configs(model_name)

        # Set up extra args
        if extra_args is None:
            extra_args = []
        if part:
            extra_args += ["--part", part]

        # Create benchmark instance
        benchmark = self._create_benchmark_instance(
            benchmark_cls,
            model_name,
            device,
            batch_size,
            extra_args,
            is_training,
            use_eval_mode,
        )

        if model_name == "vision_maskrcnn" and is_training:
            use_eval_mode = True

        # Get model and example inputs
        model, example_inputs = benchmark.get_module()

        # Handle model-specific setup
        self._handle_model_specific_setup(
            model_name,
            model,
            example_inputs,
            device,
            batch_size,
            is_training,
            use_eval_mode,
        )

        # Final setup
        gc.collect()
        batch_size = benchmark.batch_size

        # Handle specific input transformations
        example_inputs = self._transform_example_inputs(
            model_name, example_inputs, batch_size, device
        )

        if self.args.trace_on_xla:
            # work around for: https://github.com/pytorch/xla/issues/4174
            import torch_xla  # noqa: F401

        return device, benchmark.name, model, example_inputs, batch_size

    def _import_model_module(self, model_name):
        """Import the model module from TorchBench."""
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
        return module

    def _configure_batch_size(self, model_name, batch_size, is_training):
        """Configure the batch size for the model."""
        cant_change_batch_size = (
            not getattr(
                self._get_benchmark_cls_for_name(model_name),
                "ALLOW_CUSTOMIZE_BSIZE",
                True,
            )
            or model_name in self.config._config["dont_change_batch_size"]
        )
        if cant_change_batch_size:
            batch_size = None
        if (
            batch_size is None
            and is_training
            and model_name in self.config._batch_size["training"]
        ):
            batch_size = self.config._batch_size["training"][model_name]
        elif (
            batch_size is None
            and not is_training
            and model_name in self.config._batch_size["inference"]
        ):
            batch_size = self.config._batch_size["inference"][model_name]

        # Control the memory footprint for few models
        if self.args.accuracy and model_name in self.config._accuracy["max_batch_size"]:
            batch_size = min(
                batch_size, self.config._accuracy["max_batch_size"][model_name]
            )

        return batch_size

    def _get_benchmark_cls_for_name(self, model_name):
        """Get the benchmark class for a model name (used for checking ALLOW_CUSTOMIZE_BSIZE)."""
        try:
            module = self._import_model_module(model_name)
            return getattr(module, "Model", None)
        except (ImportError, AttributeError):
            # Return a dummy class with default behavior if we can't import
            class DummyBenchmark:
                ALLOW_CUSTOMIZE_BSIZE = True

            return DummyBenchmark

    def _handle_special_model_configs(self, model_name):
        """Handle special configurations for specific models."""
        # workaround "RuntimeError: not allowed to set torch.backends.cudnn flags"
        torch.backends.__allow_nonbracketed_mutation_flag = True

        # sam_fast only runs with amp
        if model_name == "sam_fast":
            self.args.amp = True
            # Need to call setup_amp from the runner if available
            if hasattr(self, "_runner") and hasattr(self._runner, "setup_amp"):
                self._runner.setup_amp()

    def _create_benchmark_instance(
        self,
        benchmark_cls,
        model_name,
        device,
        batch_size,
        extra_args,
        is_training,
        use_eval_mode,
    ):
        """Create the benchmark instance with appropriate parameters."""
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
        return benchmark

    def _handle_model_specific_setup(
        self,
        model_name,
        model,
        example_inputs,
        device,
        batch_size,
        is_training,
        use_eval_mode,
    ):
        """Handle model-specific setup after model creation."""
        if model_name in [
            "basic_gnn_edgecnn",
            "basic_gnn_gcn",
            "basic_gnn_sage",
            "basic_gnn_gin",
        ]:
            _reassign_parameters(model)

        # Models that must be in train mode while training
        if is_training and (
            not use_eval_mode or model_name in self.config._config["only_training"]
        ):
            model.train()
        else:
            model.eval()

    def _transform_example_inputs(self, model_name, example_inputs, batch_size, device):
        """Transform example inputs for specific models."""
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
        elif model_name == "yolov3":
            example_inputs = (torch.rand(batch_size, 3, 384, 512).to(device),)
        # See https://github.com/pytorch/benchmark/issues/1561
        elif model_name == "maml_omniglot":
            batch_size = 5
            assert example_inputs[0].shape[0] == batch_size
        elif model_name == "vision_maskrcnn":
            batch_size = 1

        return example_inputs

    def iter_model_names(self, args):
        """Iterate over available model names based on filter criteria."""
        from torchbenchmark import _list_canary_model_paths, _list_model_paths

        models = _list_model_paths()
        models += [
            f
            for f in _list_canary_model_paths()
            if os.path.basename(f) in self.config._config["canary_models"]
        ]
        models.sort()

        start, end = self._get_benchmark_indices(len(models))
        for index, model_path in enumerate(models):
            if index < start or index >= end:
                continue

            model_name = os.path.basename(model_path)
            if (
                not re.search("|".join(args.filter), model_name, re.IGNORECASE)
                or re.search("|".join(args.exclude), model_name, re.IGNORECASE)
                or model_name in args.exclude_exact
                or model_name in self.config.skip_models
            ):
                continue

            yield model_name

    def _get_benchmark_indices(self, total_models):
        """Get start and end indices for benchmark slicing."""
        # This method should be provided by the benchmark runner
        # For now, return full range as default
        return 0, total_models
