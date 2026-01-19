#!/usr/bin/env python3

import importlib
import logging
import os
import re
import subprocess
import sys
import warnings


try:
    from .common import BenchmarkRunner, download_retry_decorator, load_yaml_file, main
except ImportError:
    from common import BenchmarkRunner, download_retry_decorator, load_yaml_file, main

import torch
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs


# Enable FX graph caching
if "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ:
    torch._inductor.config.fx_graph_cache = True


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    importlib.import_module("timm")
except ModuleNotFoundError:
    print("Installing PyTorch Image Models...")
    pip_install("git+https://github.com/rwightman/pytorch-image-models")
finally:
    from timm import __version__ as timmversion
    from timm.data import resolve_data_config
    from timm.models import create_model

TIMM_MODELS = {}

# Run only this selected group of models, leave this empty to run everything
TORCHBENCH_ONLY_MODELS = [
    m.strip() for m in os.getenv("TORCHBENCH_ONLY_MODELS", "").split(",") if m.strip()
]

filename = os.path.join(os.path.dirname(__file__), "timm_models_list.txt")
with open(filename) as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, batch_size = line.split(" ")
        if TORCHBENCH_ONLY_MODELS and model_name not in TORCHBENCH_ONLY_MODELS:
            continue
        TIMM_MODELS[model_name] = int(batch_size)


# TODO - Figure out the reason of cold start memory spike

BATCH_SIZE_DIVISORS = {
    "beit_base_patch16_224": 2,
    "deit_base_distilled_patch16_224": 2,
    "gluon_xception65": 2,
    "mobilevit_s": 2,
    "swin_base_patch4_window7_224": 2,
}

REQUIRE_HIGHER_TOLERANCE = {
    "inception_v3",
    "mobilenetv3_large_100",
}

REQUIRE_HIGHER_TOLERANCE_FP16_XPU = {
    "botnet26t_256",
}

REQUIRE_HIGHER_TOLERANCE_AMP = {}

REQUIRE_EVEN_HIGHER_TOLERANCE = {
    "deit_base_distilled_patch16_224",
    "vit_base_patch16_siglip_256",
}

# These models need higher tolerance in MaxAutotune mode
REQUIRE_EVEN_HIGHER_TOLERANCE_MAX_AUTOTUNE = {}

REQUIRE_HIGHER_TOLERANCE_FOR_FREEZING = {
    "adv_inception_v3",
}

SCALED_COMPUTE_LOSS = {
    "mobilevit_s",
}

FORCE_AMP_FOR_FP16_BF16_MODELS = {}

SKIP_ACCURACY_CHECK_AS_EAGER_NON_DETERMINISTIC_MODELS = {}

REQUIRE_LARGER_MULTIPLIER_FOR_SMALLER_TENSOR = {
    "inception_v3",
    "mobilenetv3_large_100",
}


def refresh_model_names():
    import glob

    from timm.models import list_models

    def read_models_from_docs():
        models = set()
        # TODO - set the path to pytorch-image-models repo
        for fn in glob.glob("../pytorch-image-models/docs/models/*.md"):
            with open(fn) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if not line.startswith("model = timm.create_model("):
                        continue

                    model = line.split("'")[1]
                    # print(model)
                    models.add(model)
        return models

    def get_family_name(name):
        known_families = [
            "darknet",
            "densenet",
            "dla",
            "dpn",
            "ecaresnet",
            "halo",
            "regnet",
            "efficientnet",
            "deit",
            "mobilevit",
            "mnasnet",
            "convnext",
            "resnet",
            "resnest",
            "resnext",
            "selecsls",
            "vgg",
            "xception",
        ]

        for known_family in known_families:
            if known_family in name:
                return known_family

        if name.startswith("gluon_"):
            return "gluon_" + name.split("_")[1]
        return name.split("_")[0]

    def populate_family(models):
        family = {}
        for model_name in models:
            family_name = get_family_name(model_name)
            if family_name not in family:
                family[family_name] = []
            family[family_name].append(model_name)
        return family

    docs_models = read_models_from_docs()
    all_models = list_models(pretrained=True, exclude_filters=["*in21k"])

    all_models_family = populate_family(all_models)
    docs_models_family = populate_family(docs_models)

    for key in docs_models_family:
        del all_models_family[key]

    chosen_models = set()
    chosen_models.update(value[0] for value in docs_models_family.values())

    chosen_models.update(value[0] for key, value in all_models_family.items())

    filename = "timm_models_list.txt"
    if os.path.exists("benchmarks"):
        filename = "benchmarks/" + filename
    with open(filename, "w") as fw:
        for model_name in sorted(chosen_models):
            fw.write(model_name + "\n")


class TimmRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "timm_models"

    @property
    def _config(self):
        return load_yaml_file("timm_models.yaml")

    @property
    def _skip(self):
        return self._config["skip"]

    @property
    def skip_models_for_cpu(self):
        return self._skip["device"]["cpu"]

    @property
    def skip_models_for_cpu_aarch64(self):
        return self._skip["device"]["cpu_aarch64"]

    @property
    def skip_models(self):
        return self._skip["all"]

    @property
    def force_amp_for_fp16_bf16_models(self):
        return FORCE_AMP_FOR_FP16_BF16_MODELS

    @property
    def force_fp16_for_bf16_models(self):
        return set()

    @property
    def get_output_amp_train_process_func(self):
        return {}

    @property
    def skip_accuracy_check_as_eager_non_deterministic(self):
        if self.args.accuracy and self.args.training:
            return SKIP_ACCURACY_CHECK_AS_EAGER_NON_DETERMINISTIC_MODELS
        return set()

    @property
    def guard_on_nn_module_models(self):
        return {}

    @property
    def inline_inbuilt_nn_modules_models(self):
        return {}

    @download_retry_decorator
    def _download_model(self, model_name):
        model = create_model(
            model_name,
            in_chans=3,
            scriptable=False,
            num_classes=None,
            drop_rate=0.0,
            drop_path_rate=None,
            drop_block_rate=None,
            pretrained=True,
        )
        return model

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        extra_args=None,
    ):
        if self.args.enable_activation_checkpointing:
            raise NotImplementedError(
                "Activation checkpointing not implemented for Timm models"
            )

        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode

        channels_last = self._args.channels_last
        model = self._download_model(model_name)

        if model is None:
            raise RuntimeError(f"Failed to load model '{model_name}'")
        model.to(
            device=device,
            memory_format=torch.channels_last if channels_last else None,
        )

        data_config = resolve_data_config(
            vars(self._args) if timmversion >= "0.8.0" else self._args,
            model=model,
            use_test_size=not is_training,
        )
        input_size = data_config["input_size"]
        recorded_batch_size = TIMM_MODELS[model_name]

        if model_name in BATCH_SIZE_DIVISORS:
            recorded_batch_size = max(
                int(recorded_batch_size / BATCH_SIZE_DIVISORS[model_name]), 1
            )
        batch_size = batch_size or recorded_batch_size

        torch.manual_seed(1337)
        input_tensor = torch.randint(
            256, size=(batch_size,) + input_size, device=device
        ).to(dtype=torch.float32)
        mean = torch.mean(input_tensor)
        std_dev = torch.std(input_tensor)
        example_inputs = (input_tensor - mean) / std_dev

        if channels_last:
            example_inputs = example_inputs.contiguous(
                memory_format=torch.channels_last
            )
        example_inputs = [
            example_inputs,
        ]

        self.loss = torch.nn.CrossEntropyLoss().to(device)

        if model_name in SCALED_COMPUTE_LOSS:
            self.compute_loss = self.scaled_compute_loss

        if is_training and not use_eval_mode:
            model.train()
        else:
            model.eval()

        self.validate_model(model, example_inputs)

        return device, model_name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        # for model_name in list_models(pretrained=True, exclude_filters=["*in21k"]):
        model_names = sorted(TIMM_MODELS.keys())
        start, end = self.get_benchmark_indices(len(model_names))
        for index, model_name in enumerate(model_names):
            if index < start or index >= end:
                continue
            if (
                not re.search("|".join(args.filter), model_name, re.IGNORECASE)
                or re.search("|".join(args.exclude), model_name, re.IGNORECASE)
                or model_name in args.exclude_exact
                or model_name in self.skip_models
            ):
                continue

            yield model_name

    def pick_grad(self, name, is_training):
        if is_training:
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def use_larger_multiplier_for_smaller_tensor(self, name):
        return name in REQUIRE_LARGER_MULTIPLIER_FOR_SMALLER_TENSOR

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        cosine = self.args.cosine
        tolerance = 1e-3

        if self.args.freezing and name in REQUIRE_HIGHER_TOLERANCE_FOR_FREEZING:
            # the conv-batchnorm fusion used under freezing may cause relatively
            # large numerical difference. We need are larger tolerance.
            # Check https://github.com/pytorch/pytorch/issues/120545 for context
            tolerance = 8 * 1e-2

        if is_training:
            from torch._inductor import config as inductor_config

            if name == "beit_base_patch16_224":
                tolerance = 16 * 1e-2
            elif name in REQUIRE_EVEN_HIGHER_TOLERANCE or (
                inductor_config.max_autotune
                and name in REQUIRE_EVEN_HIGHER_TOLERANCE_MAX_AUTOTUNE
            ):
                tolerance = 8 * 1e-2
            elif name in REQUIRE_HIGHER_TOLERANCE or (
                self.args.amp and name in REQUIRE_HIGHER_TOLERANCE_AMP
            ):
                tolerance = 4 * 1e-2
            elif (
                name in REQUIRE_HIGHER_TOLERANCE_FP16_XPU
                and self.args.float16
                and current_device == "xpu"
            ):
                tolerance = 4 * 1e-2
            else:
                tolerance = 1e-2
        return tolerance, cosine

    def compute_loss(self, pred):
        # High loss values make gradient checking harder, as small changes in
        # accumulation order upsets accuracy checks.
        return reduce_to_scalar_loss(pred)

    def scaled_compute_loss(self, pred):
        # Loss values need zoom out further.
        return reduce_to_scalar_loss(pred) / 1000.0

    def forward_pass(self, mod, inputs, collect_outputs=True):
        with self.autocast(**self.autocast_arg):
            return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast(**self.autocast_arg):
            pred = mod(*cloned_inputs)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, None, loss, cloned_inputs)
        return None


def timm_main():
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TimmRunner())


if __name__ == "__main__":
    timm_main()
