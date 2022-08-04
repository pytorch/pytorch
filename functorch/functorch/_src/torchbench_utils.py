import re
import os
import gc
import importlib

SKIP = {
    # non-deterministic output / cant check correctness
    "pyhpc_turbulent_kinetic_energy",
    # https://github.com/pytorch/torchdynamo/issues/101
    "detectron2_maskrcnn",
    # https://github.com/pytorch/torchdynamo/issues/145
    "fambench_xlmr",
    "speech_transformer",
    "fastNLP_Bert",
    # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
    "tacotron2",
    "vision_maskrcnn",
}

# Additional models that are skipped in training
SKIP_TRAIN = {
    # not designed for training
    "pyhpc_equation_of_state",
    "pyhpc_isoneutral_mixing",
    # Unusual training setup
    "opacus_cifar10",
    "maml",
}

# Some models have bad train dataset. We read eval dataset.
# yolov3 - seems to have different number of inputs between eval and train
# timm_efficientdet - loader only exists for eval mode.
ONLY_EVAL_DATASET = {"yolov3", "timm_efficientdet"}

# These models support only train mode. So accuracy checking can't be done in
# eval mode.
ONLY_TRAINING_MODE = {"tts_angular", "tacotron2", "demucs"}


# Some models have large dataset that doesn't fit in memory. Lower the batch
# size to test the accuracy.
USE_SMALL_BATCH_SIZE = {
    "demucs": 4,
    "densenet121": 4,
    "hf_Reformer": 4,
    "timm_efficientdet": 1,
    'timm_nfnet': 64,
}


def iter_model_names(args):
    from torchbenchmark import _list_model_paths

    if args.training:
        SKIP.update(SKIP_TRAIN)

    for model_path in _list_model_paths():
        model_name = os.path.basename(model_path)
        if (
            not re.search("|".join(args.filter), model_name, re.I)
            or re.search("|".join(args.exclude), model_name, re.I)
            or model_name in SKIP
        ):
            continue

        yield model_name

def load_model(device, model_name, is_training, use_eval_mode):
    module = importlib.import_module(f"torchbenchmark.models.{model_name}")
    benchmark_cls = getattr(module, "Model", None)
    if not hasattr(benchmark_cls, "name"):
        benchmark_cls.name = model_name
    batch_size = None
    if is_training and model_name in USE_SMALL_BATCH_SIZE:
        batch_size = USE_SMALL_BATCH_SIZE[model_name]

    if is_training and model_name not in ONLY_EVAL_DATASET:
        benchmark = benchmark_cls(
            test="train", device=device, jit=False, batch_size=batch_size
        )
    else:
        benchmark = benchmark_cls(
            test="eval", device=device, jit=False, batch_size=batch_size
        )
    model, example_inputs = benchmark.get_module()

    # Models that must be in train mode while training
    if is_training and (not use_eval_mode or model_name in ONLY_TRAINING_MODE):
        model.train()
    else:
        model.eval()
    gc.collect()
    return device, benchmark.name, model, example_inputs