#!/usr/bin/env python3

"""Configuration handling for TorchBench benchmark suite."""

try:
    from .common import load_yaml_file
except ImportError:
    from common import load_yaml_file


class TorchBenchConfig:
    """Handles TorchBench configuration from YAML files."""
    
    def __init__(self):
        self._config = load_yaml_file("torchbench.yaml")
        self.args = None  # Will be set by the runner
    
    def set_args(self, args):
        """Set args for configuration properties that depend on command line arguments."""
        self.args = args
    
    @property
    def _skip(self):
        return self._config["skip"]

    @property
    def _batch_size(self):
        return self._config["batch_size"]

    @property
    def _tolerance(self):
        return self._config["tolerance"]

    @property
    def _require_larger_multiplier_for_smaller_tensor(self):
        return self._config["require_larger_multiplier_for_smaller_tensor"]

    @property
    def _accuracy(self):
        return self._config["accuracy"]

    @property
    def skip_models(self):
        return self._skip["all"]

    @property
    def skip_models_for_cpu(self):
        return self._skip["device"]["cpu"]

    @property
    def skip_models_for_cuda(self):
        return self._skip["device"]["cuda"]

    @property
    def skip_models_for_freezing_cuda(self):
        return self._skip["freezing"]["cuda"]

    @property
    def disable_cudagraph_models(self):
        return self._config["disable_cudagraph"]

    @property
    def skip_models_for_freezing_cpu(self):
        return self._skip["freezing"]["cpu"]

    @property
    def slow_models(self):
        return self._config["slow"]

    @property
    def very_slow_models(self):
        return self._config["very_slow"]

    @property
    def non_deterministic_models(self):
        return self._config["non_deterministic"]

    @property
    def skip_not_suitable_for_training_models(self):
        return self._skip["test"]["training"]

    @property
    def failing_fx2trt_models(self):
        return self._config["trt_not_yet_working"]

    @property
    def force_amp_for_fp16_bf16_models(self):
        return self._config["dtype"]["force_amp_for_fp16_bf16_models"]

    @property
    def force_fp16_for_bf16_models(self):
        return self._config["dtype"]["force_fp16_for_bf16_models"]

    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        if self.args and (self.args.dashboard or self.args.accuracy):
            return self._accuracy["skip"]["large_models"]
        return set()

    @property
    def skip_accuracy_check_as_eager_non_deterministic(self):
        if self.args and self.args.accuracy and self.args.training:
            return self._accuracy["skip"]["eager_not_deterministic"]
        return set()

    @property
    def skip_multiprocess_models(self):
        return self._skip["multiprocess"]

    @property
    def skip_models_due_to_control_flow(self):
        return self._skip["control_flow"]

    @property
    def skip_models_due_to_export_not_supported(self):
        return self._skip["export_not_supported"]

    @property
    def guard_on_nn_module_models(self):
        return {
            "vision_maskrcnn",
        }

    @property
    def inline_inbuilt_nn_modules_models(self):
        return {
            "basic_gnn_edgecnn",
            "drq",
            "hf_Reformer",
            "DALLE2_pytorch",
            "hf_BigBird",
            "detectron2_maskrcnn_r_50_fpn",
            "detectron2_maskrcnn_r_101_fpn",
            "vision_maskrcnn",
            "doctr_reco_predictor",
            "hf_T5_generate",
        }

    def use_larger_multiplier_for_smaller_tensor(self, name):
        """Check if model requires larger multiplier for smaller tensors."""
        return name in self._require_larger_multiplier_for_smaller_tensor

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name, args):
        """Get tolerance and cosine flag for model accuracy comparison."""
        tolerance = 1e-4
        cosine = args.cosine
        # Increase the tolerance for torch allclose
        if args.float16 or args.amp:
            if args.freezing and (freezing := self._tolerance["freezing"]):
                higher_fp16 = freezing.get("higher_fp16", None)
                even_higher = freezing.get("even_higher", None)
                if higher_fp16 and name in higher_fp16:
                    return 1e-2, cosine
                elif even_higher and name in even_higher:
                    return 8 * 1e-2, cosine
            if name in self._tolerance["higher_fp16"]:
                return 1e-2, cosine
            elif name in self._tolerance["even_higher"]:
                return 8 * 1e-2, cosine
            return 1e-3, cosine

        if args.bfloat16:
            if name in self._tolerance["higher_bf16"]:
                return 1e-2, cosine

        if is_training and (current_device == "cuda" or current_device == "xpu"):
            tolerance = 1e-3
            if name in self._tolerance["cosine"]:
                cosine = True
            elif name in self._tolerance["higher"]:
                tolerance = 1e-3
            elif name in self._tolerance["even_higher"]:
                tolerance = 8 * 1e-2
        return tolerance, cosine

    def get_batch_size_for_model(self, model_name, is_training, batch_size=None):
        """Get appropriate batch size for a model."""        
        if (
            batch_size is None
            and is_training
            and model_name in self._batch_size["training"]
        ):
            return self._batch_size["training"][model_name]
        elif (
            batch_size is None
            and not is_training
            and model_name in self._batch_size["inference"]
        ):
            return self._batch_size["inference"][model_name]
        
        return batch_size

    def limit_batch_size_for_accuracy(self, model_name, batch_size):
        """Limit batch size for accuracy testing."""
        if self.args and self.args.accuracy and model_name in self._accuracy["max_batch_size"]:
            return min(batch_size, self._accuracy["max_batch_size"][model_name])
        return batch_size