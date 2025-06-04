#!/usr/bin/env python3

"""Model loading and setup logic for TorchBench benchmark suite."""

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
        """Load and setup a TorchBench model.

        Args:
            device: Device to load model on
            model_name: Name of the model to load
            batch_size: Batch size to use
            part: Model part (if applicable)
            extra_args: Extra arguments for model initialization

        Returns:
            tuple: (device, model_name, model, example_inputs, batch_size)
        """
        if self.args.enable_activation_checkpointing:
            raise NotImplementedError(
                "Activation checkpointing not implemented for Torchbench models"
            )
        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode

        # Load model class
        benchmark_cls = self._load_model_class(model_name)

        # Setup batch size
        batch_size = self._setup_batch_size(
            benchmark_cls, model_name, batch_size, is_training
        )

        # workaround "RuntimeError: not allowed to set torch.backends.cudnn flags"
        torch.backends.__allow_nonbracketed_mutation_flag = True
        if extra_args is None:
            extra_args = []
        if part:
            extra_args += ["--part", part]

        # Handle special model configurations
        if model_name == "sam_fast":
            self.args.amp = True
            # Note: setup_amp() would need to be called by the runner

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

        # Get model and example inputs
        model, example_inputs = benchmark.get_module()

        # Handle special model parameter setup
        if model_name in [
            "basic_gnn_edgecnn",
            "basic_gnn_gcn",
            "basic_gnn_sage",
            "basic_gnn_gin",
        ]:
            _reassign_parameters(model)

        # Set model mode
        if is_training and (
            not use_eval_mode or model_name in self.config._config["only_training"]
        ):
            model.train()
        else:
            model.eval()

        gc.collect()
        batch_size = benchmark.batch_size

        # Handle special input formats
        example_inputs = self._handle_special_input_formats(
            model_name, example_inputs, batch_size, device
        )

        if self.args.trace_on_xla:
            # work around for: https://github.com/pytorch/xla/issues/4174
            import torch_xla  # noqa: F401

        return device, benchmark.name, model, example_inputs, batch_size

    def iter_model_names(self):
        """Iterate over available model names based on filters."""
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
                not re.search("|".join(self.args.filter), model_name, re.IGNORECASE)
                or re.search("|".join(self.args.exclude), model_name, re.IGNORECASE)
                or model_name in self.args.exclude_exact
                or model_name in self.config.skip_models
            ):
                continue

            yield model_name

    def _load_model_class(self, model_name):
        """Load the model class from TorchBench."""
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

        return benchmark_cls

    def _setup_batch_size(self, benchmark_cls, model_name, batch_size, is_training):
        """Setup appropriate batch size for the model."""
        cant_change_batch_size = (
            not getattr(benchmark_cls, "ALLOW_CUSTOMIZE_BSIZE", True)
            or model_name in self.config._config["dont_change_batch_size"]
        )
        if cant_change_batch_size:
            batch_size = None

        batch_size = self.config.get_batch_size_for_model(
            model_name, is_training, batch_size
        )

        # Control the memory footprint for few models
        batch_size = self.config.limit_batch_size_for_accuracy(model_name, batch_size)

        return batch_size

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
        """Create benchmark instance with appropriate configuration."""
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

        return benchmark

    def _handle_special_input_formats(
        self, model_name, example_inputs, batch_size, device
    ):
        """Handle special input formats for specific models."""
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
            # batch_size is updated to 5 for this model
            assert example_inputs[0].shape[0] == 5

        return example_inputs

    def _get_benchmark_indices(self, total_models):
        """Get start and end indices for benchmark models."""
        # This would normally be implemented in the runner
        # For now, return full range
        return 0, total_models
