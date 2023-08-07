#!/usr/bin/env python3
import gc
import importlib
import logging
import os
import re
import sys
import warnings
from os.path import abspath, exists

import torch

try:
    from .common import BenchmarkRunner, main
except ImportError:
    from common import BenchmarkRunner, main

from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs

# We are primarily interested in tf32 datatype
torch.backends.cuda.matmul.allow_tf32 = True


def setup_torchbench_cwd():
    original_dir = abspath(os.getcwd())

    os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
    for torchbench_dir in (
        "./torchbenchmark",
        "../torchbenchmark",
        "../torchbench",
        "../benchmark",
        "../../torchbenchmark",
        "../../torchbench",
        "../../benchmark",
    ):
        if exists(torchbench_dir):
            break

    if exists(torchbench_dir):
        torchbench_dir = abspath(torchbench_dir)
        os.chdir(torchbench_dir)
        sys.path.append(torchbench_dir)

    return original_dir


# Some models have large dataset that doesn't fit in memory. Lower the batch
# size to test the accuracy.
USE_SMALL_BATCH_SIZE = {
    "demucs": 4,
    "dlrm": 1024,
    "densenet121": 4,
    "hf_Reformer": 4,
    "hf_T5_base": 4,
    "timm_efficientdet": 1,
    "llama_v2_7b_16h": 1,
}

DETECTRON2_MODELS = {
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "detectron2_maskrcnn_r_101_c4",
    "detectron2_maskrcnn_r_101_fpn",
    "detectron2_maskrcnn_r_50_fpn",
}

SKIP = {
    # https://github.com/pytorch/torchdynamo/issues/101
    "detectron2_maskrcnn",
    # https://github.com/pytorch/torchdynamo/issues/145
    "fambench_xlmr",
    # TIMEOUT, https://github.com/pytorch/pytorch/issues/98467
    "tacotron2",
    "hf_Bert",  # Error: RelaxedUnspecConstraint(L['input_ids'].size()[0]) - inferred constant (4)
    "hf_Bert_large",  # Error: RelaxedUnspecConstraint(L['input_ids'].size()[0]) - inferred constant (4)
    # takes too long, extreme slowdown (< .001)
    "maml",
}

SKIP_FOR_CPU = {
    "hf_T5_generate",  # OOMs
    "cm3leon_generate",  # model is CUDA only
    "nanogpt_generate",  # timeout
    "sam",  # timeout
    "llama_v2_7b_16h",  # model is CUDA only
    "stable_diffusion",  # flaky
}

SKIP_FOR_CUDA = {
    "gat",  # only works on CPU
    "gcn",  # only works on CPU
    "sage",  # only works on CPU
}

# Additional models that are skipped in training
SKIP_TRAIN = {
    # not designed for training
    "pyhpc_equation_of_state",
    "pyhpc_isoneutral_mixing",
    "pyhpc_turbulent_kinetic_energy",
    "maml",
    "llama",
    "llama_v2_7b_16h",
}
SKIP_TRAIN.update(DETECTRON2_MODELS)

# These models support only train mode. So accuracy checking can't be done in
# eval mode.
ONLY_TRAINING_MODE = {
    "tts_angular",
    "tacotron2",
    "demucs",
    "hf_Reformer",
    "pytorch_struct",
    "yolov3",
}
ONLY_TRAINING_MODE.update(DETECTRON2_MODELS)

# Need lower tolerance on GPU. GPU kernels have non deterministic kernels for these models.
REQUIRE_HIGHER_TOLERANCE = {
    "alexnet",
    "attention_is_all_you_need_pytorch",
    "densenet121",
    "hf_Albert",
    "vgg16",
    "mobilenet_v3_large",
    "nvidia_deeprecommender",
    "timm_efficientdet",
}

# These models need >1e-3 tolerance
REQUIRE_EVEN_HIGHER_TOLERANCE = {
    "soft_actor_critic",
    "tacotron2",
}

REQUIRE_HIGHER_FP16_TOLERANCE = {
    "drq",
}

REQUIRE_COSINE_TOLERACE = {
    # Just keeping it here even though its empty, if we need this in future.
}

# non-deterministic output / cant check correctness
NONDETERMINISTIC = {
    # https://github.com/pytorch/pytorch/issues/98355
    "mobilenet_v3_large",
}

# These benchmarks took >600s on an i9-11900K CPU
VERY_SLOW_BENCHMARKS = {
    "hf_BigBird",  # 3339s
    "hf_Longformer",  # 3062s
    "hf_T5",  # 930s
}

# These benchmarks took >60s on an i9-11900K CPU
SLOW_BENCHMARKS = {
    *VERY_SLOW_BENCHMARKS,
    "BERT_pytorch",  # 137s
    "demucs",  # 116s
    "fastNLP_Bert",  # 242s
    "hf_Albert",  # 221s
    "hf_Bart",  # 400s
    "hf_Bert",  # 334s
    "hf_DistilBert",  # 187s
    "hf_GPT2",  # 470s
    "hf_Reformer",  # 141s
    "speech_transformer",  # 317s
    "vision_maskrcnn",  # 99s
}

TRT_NOT_YET_WORKING = {
    "alexnet",
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "mnasnet1_0",
    "squeezenet1_1",
    "shufflenetv2_x1_0",
    "vgg16",
    "resnext50_32x4d",
}

DONT_CHANGE_BATCH_SIZE = {
    "demucs",
    "pytorch_struct",
    "pyhpc_turbulent_kinetic_energy",
    "vision_maskrcnn",  # https://github.com/pytorch/benchmark/pull/1656
}


SKIP_ACCURACY_CHECK_MODELS = {
    # Models too large to have eager, dynamo and fp64_numbers simultaneosuly
    # even for 40 GB machine. We have tested accuracy for smaller version of
    # these models
    "hf_GPT2_large",
    "hf_T5_large",
    "timm_vision_transformer_large",
    "maml",  # accuracy https://github.com/pytorch/pytorch/issues/93847
    "llama_v2_7b_16h",
}

SKIP_ACCURACY_CHECK_AS_EAGER_NON_DETERMINISTIC_MODELS = {
    # Models that deterministic algorithms can not be turned on for eager mode.
    "Background_Matting",
}


MAX_BATCH_SIZE_FOR_ACCURACY_CHECK = {
    "hf_GPT2": 2,
    "pytorch_unet": 2,
}

FORCE_AMP_FOR_FP16_BF16_MODELS = {
    "DALLE2_pytorch",
    "doctr_det_predictor",
    "doctr_reco_predictor",
    "Super_SloMo",
    "tts_angular",
}


class TorchBenchmarkRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "torchbench"
        self.optimizer = None

    @property
    def skip_models(self):
        return SKIP

    @property
    def skip_models_for_cpu(self):
        return SKIP_FOR_CPU

    @property
    def skip_models_for_cuda(self):
        return SKIP_FOR_CUDA

    @property
    def slow_models(self):
        return SLOW_BENCHMARKS

    @property
    def very_slow_models(self):
        return VERY_SLOW_BENCHMARKS

    @property
    def non_deterministic_models(self):
        return NONDETERMINISTIC

    @property
    def skip_not_suitable_for_training_models(self):
        return SKIP_TRAIN

    @property
    def failing_fx2trt_models(self):
        return TRT_NOT_YET_WORKING

    @property
    def force_amp_for_fp16_bf16_models(self):
        return FORCE_AMP_FOR_FP16_BF16_MODELS

    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        if self.args.dashboard or self.args.accuracy:
            return SKIP_ACCURACY_CHECK_MODELS
        return set()

    @property
    def skip_accuracy_check_as_eager_non_deterministic(self):
        if self.args.accuracy and self.args.training:
            return SKIP_ACCURACY_CHECK_AS_EAGER_NON_DETERMINISTIC_MODELS
        return set()

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        part=None,
    ):
        if self.args.enable_activation_checkpointing:
            raise NotImplementedError(
                "Activation checkpointing not implemented for Torchbench models"
            )
        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode
        dynamic_shapes = self.args.dynamic_shapes
        candidates = [
            f"torchbenchmark.models.{model_name}",
            f"torchbenchmark.canary_models.{model_name}",
            f"torchbenchmark.models.fb.{model_name}",
        ]
        for c in candidates:
            try:
                module = importlib.import_module(c)
                break
            except ModuleNotFoundError:
                pass
        else:
            raise ImportError(f"could not import any of {candidates}")
        benchmark_cls = getattr(module, "Model", None)
        if not hasattr(benchmark_cls, "name"):
            benchmark_cls.name = model_name

        cant_change_batch_size = (
            not getattr(benchmark_cls, "ALLOW_CUSTOMIZE_BSIZE", True)
            or model_name in DONT_CHANGE_BATCH_SIZE
        )
        if cant_change_batch_size:
            batch_size = None
        if batch_size is None and is_training and model_name in USE_SMALL_BATCH_SIZE:
            batch_size = USE_SMALL_BATCH_SIZE[model_name]

        # Control the memory footprint for few models
        if self.args.accuracy and model_name in MAX_BATCH_SIZE_FOR_ACCURACY_CHECK:
            batch_size = min(batch_size, MAX_BATCH_SIZE_FOR_ACCURACY_CHECK[model_name])

        # workaround "RuntimeError: not allowed to set torch.backends.cudnn flags"
        torch.backends.__allow_nonbracketed_mutation_flag = True
        extra_args = []
        if part:
            extra_args = ["--part", part]

        if model_name == "vision_maskrcnn" and is_training:
            # Output of vision_maskrcnn model is a list of bounding boxes,
            # sorted on the basis of their scores. This makes accuracy
            # comparison hard with torch.compile. torch.compile can cause minor
            # divergences in the output because of how fusion works for amp in
            # TorchInductor compared to eager.  Therefore, instead of looking at
            # all the bounding boxes, we compare only top 5.
            model_kwargs = {"box_detections_per_img": 5}
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
        model, example_inputs = benchmark.get_module()

        # Models that must be in train mode while training
        if is_training and (not use_eval_mode or model_name in ONLY_TRAINING_MODE):
            model.train()
        else:
            model.eval()
        gc.collect()
        batch_size = benchmark.batch_size

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
        from torchbenchmark import _list_model_paths

        models = _list_model_paths()
        start, end = self.get_benchmark_indices(len(models))
        for index, model_path in enumerate(models):
            if index < start or index >= end:
                continue

            model_name = os.path.basename(model_path)
            if (
                not re.search("|".join(args.filter), model_name, re.I)
                or re.search("|".join(args.exclude), model_name, re.I)
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

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        tolerance = 1e-4
        cosine = self.args.cosine
        # Increase the tolerance for torch allclose
        if self.args.float16 or self.args.amp:
            if name in REQUIRE_HIGHER_FP16_TOLERANCE:
                return 1e-2, cosine
            return 1e-3, cosine
        if is_training and current_device == "cuda":
            tolerance = 1e-3
            if name in REQUIRE_COSINE_TOLERACE:
                cosine = True
            elif name in REQUIRE_HIGHER_TOLERANCE:
                tolerance = 1e-3
            elif name in REQUIRE_EVEN_HIGHER_TOLERANCE:
                tolerance = 8 * 1e-2
        return tolerance, cosine

    def compute_loss(self, pred):
        return reduce_to_scalar_loss(pred)

    def forward_pass(self, mod, inputs, collect_outputs=True):
        with self.autocast():
            return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast():
            pred = mod(*cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return None


def torchbench_main():
    original_dir = setup_torchbench_cwd()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TorchBenchmarkRunner(), original_dir)


if __name__ == "__main__":
    torchbench_main()
