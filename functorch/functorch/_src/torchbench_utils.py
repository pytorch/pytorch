import re
import os
import gc
import importlib

SKIP = {}

EXPERIMENT_BATCH_SIZES = {
    "BERT_pytorch":128,
    "LearningToPaint":1024,
    "alexnet":1024,
    "dcgan":1024,
    "densenet121":64,
    "hf_Albert":16,
    "hf_Bart":8, #16
    "hf_Bert":16,
    "hf_GPT2":8, #16
    "hf_T5":4,
    "mnasnet1_0":256,
    "mobilenet_v2":128,
    "mobilenet_v3_large":256,
    "nvidia_deeprecommender":1024,
    "pytorch_unet":4,#8
    "resnet18":512,
    "resnet50":128,
    "resnext50_32x4d":128,
    "shufflenet_v2_x1_0":256, #512
    "squeezenet1_1":256, #512
    "timm_efficientnet":128,
    "timm_regnet":64,
    "timm_resnest":128, #256
    "timm_vision_transformer":256,
    "timm_vovnet":128,
    "vgg16":128, #256
}


def iter_model_names(args):
    from torchbenchmark import _list_model_paths

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
    if model_name not in EXPERIMENT_BATCH_SIZES:
        raise NotImplementedError("not a model in experiment")

    module = importlib.import_module(f"torchbenchmark.models.{model_name}")
    benchmark_cls = getattr(module, "Model", None)
    if not hasattr(benchmark_cls, "name"):
        benchmark_cls.name = model_name
    # if is_training and model_name in USE_SMALL_BATCH_SIZE:
    #     batch_size = USE_SMALL_BATCH_SIZE[model_name]
    batch_size = EXPERIMENT_BATCH_SIZES[model_name]

    if is_training:
        benchmark = benchmark_cls(
            test="train", device=device, jit=False, batch_size=batch_size
        )
    else:
        benchmark = benchmark_cls(
            test="eval", device=device, jit=False, batch_size=batch_size
        )
    model, example_inputs = benchmark.get_module()

    # Models that must be in train mode while training
    if is_training and (not use_eval_mode):
        model.train()
    else:
        model.eval()
    gc.collect()
    return device, benchmark.name, model, example_inputs