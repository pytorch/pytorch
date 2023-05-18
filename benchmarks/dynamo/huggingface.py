#!/usr/bin/env python3
import importlib
import logging
import os
import re
import subprocess
import sys
import warnings

import torch
from common import BenchmarkRunner, download_retry_decorator, main, reset_rng_state

from torch._dynamo.testing import collect_results
from torch._dynamo.utils import clone_inputs

log = logging.getLogger(__name__)


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Disable the flake warnings for the imports. Flake8 does not provide a way to
# disable just warning for the entire file. Disabling flake8 entirely.
# flake8: noqa
imports = [
    "AlbertForPreTraining",
    "AutoConfig",
    "AutoModelForCausalLM",
    "AutoModelForMaskedLM",
    "AutoModelForSeq2SeqLM",
    "BigBirdConfig",
    "BlenderbotForConditionalGeneration",
    "BlenderbotModel",
    "BlenderbotSmallForConditionalGeneration",
    "BlenderbotSmallModel",
    "CLIPModel",
    "CLIPVisionModel",
    "ElectraForPreTraining",
    "GPT2ForSequenceClassification",
    "GPTJForSequenceClassification",
    "GPTNeoForSequenceClassification",
    "HubertForSequenceClassification",
    "LxmertForPreTraining",
    "LxmertForQuestionAnswering",
    "MarianForCausalLM",
    "MarianModel",
    "MarianMTModel",
    "PegasusForConditionalGeneration",
    "PegasusModel",
    "ReformerConfig",
    "ViTForImageClassification",
    "ViTForMaskedImageModeling",
    "ViTModel",
]


try:
    mod = importlib.import_module("transformers")
    for cls in imports:
        if not hasattr(mod, cls):
            raise ModuleNotFoundError
except ModuleNotFoundError:
    print("Installing HuggingFace Transformers...")
    pip_install("git+https://github.com/huggingface/transformers.git#egg=transformers")
finally:
    for cls in imports:
        exec(f"from transformers import {cls}")


# These models contain the models present in huggingface_models_list. It is a
# combination of models supported by HF Fx parser and some manually supplied
# models. For these models, we already know the largest batch size that can fit
# on A100 GPUs - 40 GB.
BATCH_SIZE_KNOWN_MODELS = dict()


# Get the list of models and their batch sizes
MODELS_FILENAME = os.path.join(os.path.dirname(__file__), "huggingface_models_list.txt")
assert os.path.exists(MODELS_FILENAME)
with open(MODELS_FILENAME, "r") as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, batch_size = line.split(",")
        batch_size = int(batch_size)
        BATCH_SIZE_KNOWN_MODELS[model_name] = batch_size
assert len(BATCH_SIZE_KNOWN_MODELS)


SKIP = {
    # Difficult to setup accuracy test because .eval() not supported
    "Reformer",
    # Fails deepcopy
    "BlenderbotForConditionalGeneration",
    "GPTNeoForCausalLM",
    "GPTNeoForSequenceClassification",
    # Fails with even batch size = 1
    "GPTJForCausalLM",
    "GPTJForQuestionAnswering",
}

# TODO - Fails even after fake tensors
BATCH_SIZE_DIVISORS = {
    "AlbertForMaskedLM": 2,
    "AlbertForQuestionAnswering": 2,
    "AllenaiLongformerBase": 2,
    "BartForCausalLM": 2,
    "BartForConditionalGeneration": 2,
    "BertForMaskedLM": 2,
    "BertForQuestionAnswering": 2,
    "BlenderbotForCausalLM": 8,
    # "BlenderbotForConditionalGeneration" : 16,
    "BlenderbotSmallForCausalLM": 4,
    "BlenderbotSmallForConditionalGeneration": 2,
    "CamemBert": 2,
    "DebertaForMaskedLM": 4,
    "DebertaForQuestionAnswering": 2,
    "DebertaV2ForMaskedLM": 4,
    "DebertaV2ForQuestionAnswering": 4,
    "DistilBertForMaskedLM": 2,
    "DistilBertForQuestionAnswering": 2,
    "DistillGPT2": 2,
    "ElectraForCausalLM": 2,
    "ElectraForQuestionAnswering": 2,
    "GPT2ForSequenceClassification": 2,
    # "GPTJForCausalLM" : 2,
    # "GPTJForQuestionAnswering" : 2,
    # "GPTNeoForCausalLM" : 32,
    # "GPTNeoForSequenceClassification" : 2,
    "GoogleFnet": 2,
    "LayoutLMForMaskedLM": 2,
    "LayoutLMForSequenceClassification": 2,
    "M2M100ForConditionalGeneration": 4,
    "MBartForCausalLM": 2,
    "MBartForConditionalGeneration": 2,
    "MT5ForConditionalGeneration": 2,
    "MegatronBertForCausalLM": 4,
    "MegatronBertForQuestionAnswering": 2,
    "MobileBertForMaskedLM": 2,
    "MobileBertForQuestionAnswering": 2,
    "OPTForCausalLM": 2,
    "PLBartForCausalLM": 2,
    "PLBartForConditionalGeneration": 2,
    "PegasusForCausalLM": 4,
    "PegasusForConditionalGeneration": 2,
    "RobertaForCausalLM": 2,
    "RobertaForQuestionAnswering": 2,
    "Speech2Text2ForCausalLM": 4,
    "T5ForConditionalGeneration": 2,
    "T5Small": 2,
    "TrOCRForCausalLM": 2,
    "XGLMForCausalLM": 4,
    "XLNetLMHeadModel": 2,
    "YituTechConvBert": 2,
}

SKIP_ACCURACY_CHECK_MODELS = {
    # Models too large to have eager, dynamo and fp64_numbers simultaneosuly
    # even for 40 GB machine.
    "DebertaV2ForMaskedLM",
    "BlenderbotForCausalLM",
}


REQUIRE_HIGHER_TOLERANCE = {
    "MT5ForConditionalGeneration",
    # AlbertForQuestionAnswering fails in CI GCP A100 but error does not seem
    # harmful.
    "AlbertForQuestionAnswering",
}


SKIP_FOR_CPU = {
    "OPTForCausalLM",  # OOMs
}


def get_module_cls_by_model_name(model_cls_name):
    _module_by_model_name = {
        "Speech2Text2Decoder": "transformers.models.speech_to_text_2.modeling_speech_to_text_2",
        "TrOCRDecoder": "transformers.models.trocr.modeling_trocr",
    }
    module_name = _module_by_model_name.get(model_cls_name, "transformers")
    module = importlib.import_module(module_name)
    return getattr(module, model_cls_name)


def get_sequence_length(model_cls, model_name):
    if model_name.startswith(("Blenderbot",)):
        seq_length = 128
    elif model_name.startswith(("GPT2", "Bart", "T5", "PLBart", "MBart")):
        seq_length = 1024
    elif model_name in ("AllenaiLongformerBase", "BigBird"):
        seq_length = 1024
    elif model_name.startswith("OPT"):
        seq_length = 2048
    elif "Reformer" in model_name:
        seq_length = 4096
    elif model_name.startswith(
        (
            "Albert",
            "Deberta",
            "Layout",
            "Electra",
            "XLNet",
            "MegatronBert",
            "Bert",
            "Roberta",
        )
    ) or model_name in ("DistillGPT2", "GoogleFnet", "YituTechConvBert", "CamemBert"):
        seq_length = 512
    elif model_name in ("TrOCRForCausalLM"):
        seq_length = 256
    elif model_name.startswith("MobileBert"):
        seq_length = 128
    else:
        log.info(
            f"Sequence Length not defined for {model_name}. Choosing 128 arbitrarily"
        )
        seq_length = 128
    return seq_length


def generate_inputs_for_model(
    model_cls, model, model_name, bs, device, include_loss_args=False
):
    # TODO - Check if following values are representative
    num_choices = 3
    num_visual_features = 42
    seq_length = get_sequence_length(model_cls, model_name)
    vocab_size = model.config.vocab_size
    if model_name.endswith("MultipleChoice"):
        input = rand_int_tensor(device, 0, vocab_size, (bs, num_choices, seq_length))
    elif model_name.startswith("Roberta"):
        input = rand_int_tensor(device, 0, 1, (bs, seq_length))
    else:
        input = rand_int_tensor(device, 0, vocab_size, (bs, seq_length))

    if "Bart" in model_name:
        input[:, -1] = model.config.eos_token_id

    input_dict = {"input_ids": input}

    if (
        model_name.startswith("T5")
        or model_name.startswith("M2M100")
        or model_name.startswith("MT5")
        or model_cls
        in [
            BlenderbotModel,
            BlenderbotSmallModel,
            BlenderbotForConditionalGeneration,
            BlenderbotSmallForConditionalGeneration,
            PegasusModel,
            PegasusForConditionalGeneration,
            MarianModel,
            MarianMTModel,
        ]
    ):
        input_dict["decoder_input_ids"] = input

    if model_name.startswith("Lxmert"):
        visual_feat_dim, visual_pos_dim = (
            model.config.visual_feat_dim,
            model.config.visual_pos_dim,
        )
        input_dict["visual_feats"] = torch.randn(
            bs, num_visual_features, visual_feat_dim
        )
        input_dict["visual_pos"] = torch.randn(bs, num_visual_features, visual_pos_dim)

    if include_loss_args:
        if model_name.endswith("PreTraining"):
            if model_cls in [ElectraForPreTraining, LxmertForPreTraining]:
                input_dict["labels"] = rand_int_tensor(device, 0, 1, (bs, seq_length))
            else:
                label_name = (
                    "sentence_order_label"
                    if model_cls in [AlbertForPreTraining]
                    else "next_sentence_label"
                )
                input_dict["labels"] = (
                    rand_int_tensor(device, 0, vocab_size, (bs, seq_length)),
                )
                input_dict[label_name] = rand_int_tensor(device, 0, 1, (bs,))
        elif model_name.endswith("QuestionAnswering"):
            input_dict["start_positions"] = rand_int_tensor(
                device, 0, seq_length, (bs,)
            )
            input_dict["end_positions"] = rand_int_tensor(device, 0, seq_length, (bs,))
        elif (
            model_name.endswith("MaskedLM")
            or model_name.endswith("HeadModel")
            or model_name.endswith("CausalLM")
            or model_name.endswith("DoubleHeadsModel")
        ):
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size, (bs, seq_length)
            )
        elif model_name.endswith("TokenClassification"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, model.config.num_labels - 1, (bs, seq_length)
            )
        elif model_name.endswith("MultipleChoice"):
            input_dict["labels"] = rand_int_tensor(device, 0, num_choices, (bs,))
        elif model_name.endswith("SequenceClassification"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, model.config.num_labels - 1, (bs,)
            )
        elif model_name.endswith("NextSentencePrediction"):
            input_dict["labels"] = rand_int_tensor(device, 0, 1, (bs,))
        elif model_name.endswith("ForConditionalGeneration"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size - 1, (bs, seq_length)
            )
        elif model_name in EXTRA_MODELS:
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size, (bs, seq_length)
            )
        else:
            raise NotImplementedError(
                f"Class {model_name} unsupported for training test "
            )

    return input_dict


def rand_int_tensor(device, low, high, shape):
    return torch.randint(
        low,
        high,
        shape,
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


EXTRA_MODELS = {
    "AllenaiLongformerBase": (
        AutoConfig.from_pretrained("allenai/longformer-base-4096"),
        AutoModelForMaskedLM,
    ),
    "Reformer": (
        ReformerConfig(),
        AutoModelForMaskedLM,
    ),
    "T5Small": (
        AutoConfig.from_pretrained("t5-small"),
        AutoModelForSeq2SeqLM,
    ),
    # "BigBird": (
    #     BigBirdConfig(attention_type="block_sparse"),
    #     AutoModelForMaskedLM,
    # ),
    "DistillGPT2": (
        AutoConfig.from_pretrained("distilgpt2"),
        AutoModelForCausalLM,
    ),
    "GoogleFnet": (
        AutoConfig.from_pretrained("google/fnet-base"),
        AutoModelForMaskedLM,
    ),
    "YituTechConvBert": (
        AutoConfig.from_pretrained("YituTech/conv-bert-base"),
        AutoModelForMaskedLM,
    ),
    "CamemBert": (
        AutoConfig.from_pretrained("camembert-base"),
        AutoModelForMaskedLM,
    ),
}


class HuggingfaceRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "huggingface"

    @property
    def skip_models_for_cpu(self):
        return SKIP_FOR_CPU

    def _get_model_cls_and_config(self, model_name):
        if model_name not in EXTRA_MODELS:
            model_cls = get_module_cls_by_model_name(model_name)
            config_cls = model_cls.config_class
            config = config_cls()

            # NB: some models need a pad token defined to handle BS > 1
            if (
                model_cls
                in [
                    GPT2ForSequenceClassification,
                    GPTNeoForSequenceClassification,
                    GPTJForSequenceClassification,
                ]
                or model_cls.__name__.startswith("Roberta")
                or model_cls.__name__.startswith("Marian")
            ):
                config.pad_token_id = 0

        else:
            config, model_cls = EXTRA_MODELS[model_name]

        return model_cls, config

    @download_retry_decorator
    def _download_model(self, model_name):
        model_cls, config = self._get_model_cls_and_config(model_name)
        if "auto" in model_cls.__module__:
            # Handle auto classes
            model = model_cls.from_config(config)
        else:
            model = model_cls(config)
        return model

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
    ):
        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode
        dtype = torch.float32
        reset_rng_state()
        model_cls, config = self._get_model_cls_and_config(model_name)
        model = self._download_model(model_name)
        model = model.to(device, dtype=dtype)
        if model_name in BATCH_SIZE_KNOWN_MODELS:
            batch_size_default = BATCH_SIZE_KNOWN_MODELS[model_name]
        elif batch_size is None:
            batch_size_default = 16
            log.info(
                f"Batch size not specified for {model_name}. Setting batch_size=16"
            )

        if batch_size is None:
            batch_size = batch_size_default
            if model_name in BATCH_SIZE_DIVISORS:
                batch_size = max(int(batch_size / BATCH_SIZE_DIVISORS[model_name]), 1)
                log.info(
                    f"Running smaller batch size={batch_size} for {model_name}, orig batch_size={batch_size_default}"
                )

        example_inputs = generate_inputs_for_model(
            model_cls, model, model_name, batch_size, device, include_loss_args=True
        )

        # So we can check for correct gradients without eliminating the dropout computation
        for attr in dir(config):
            if "drop" in attr and isinstance(getattr(config, attr), float):
                setattr(config, attr, 1e-30)

        if is_training and not use_eval_mode:
            model.train()
        else:
            model.eval()

        self.validate_model(model, example_inputs)
        return device, model_name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        model_names = list(BATCH_SIZE_KNOWN_MODELS.keys()) + list(EXTRA_MODELS.keys())
        model_names = set(model_names)
        model_names = sorted(model_names)

        start, end = self.get_benchmark_indices(len(model_names))
        for index, model_name in enumerate(model_names):
            if index < start or index >= end:
                continue
            if (
                not re.search("|".join(args.filter), model_name, re.I)
                or re.search("|".join(args.exclude), model_name, re.I)
                or model_name in args.exclude_exact
                or model_name in SKIP
            ):
                continue
            yield model_name

    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        if self.args.dashboard or self.args.accuracy:
            return SKIP_ACCURACY_CHECK_MODELS
        return set()

    def pick_grad(self, name, is_training):
        if is_training:
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        cosine = self.args.cosine
        if is_training:
            if name in REQUIRE_HIGHER_TOLERANCE:
                return 2e-2, cosine
            else:
                return 1e-2, cosine
        return 1e-3, cosine

    def compute_loss(self, pred):
        return pred[0]

    def forward_pass(self, mod, inputs, collect_outputs=True):
        with self.autocast():
            return mod(**inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast():
            pred = mod(**cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return None


def refresh_model_names_and_batch_sizes():
    """
    This function reads the HF Fx tracer supported models and finds the largest
    batch size that could fit on the GPU with PyTorch eager.

    The resulting data is written in huggingface_models_list.txt.

    Note - We only need to run this function if we believe that HF Fx tracer now
    supports more models.
    """
    import transformers.utils.fx as hf_fx

    family = dict()
    lm_seen = set()
    family_seen = set()
    for cls_name in hf_fx._SUPPORTED_MODELS:
        if "For" not in cls_name:
            continue

        model_cls = get_module_cls_by_model_name(cls_name)

        # TODO: AttributeError: '*Config' object has no attribute 'vocab_size'
        if model_cls in [
            CLIPModel,
            CLIPVisionModel,
            # SwinForImageClassification,
            # SwinForImageClassification,
            # SwinForMaskedImageModeling,
            # SwinModel,
            ViTForImageClassification,
            ViTForMaskedImageModeling,
            ViTModel,
        ]:
            continue

        # TODO: AssertionError: Padding_idx must be within num_embeddings
        if model_cls in [MarianForCausalLM, MarianMTModel, MarianModel]:
            continue

        # TODO: "model is not supported yet" from HFTracer
        if model_cls in [HubertForSequenceClassification]:
            continue

        # TODO: shape mismatch in loss calculation
        if model_cls in [LxmertForQuestionAnswering]:
            continue

        family_name = cls_name.split("For")[0]
        if family_name not in family:
            family[family_name] = []
        if cls_name.endswith(("MaskedLM", "CausalLM")) and family_name not in lm_seen:
            family[family_name].append(cls_name)
            lm_seen.add(family_name)
        elif (
            cls_name.endswith(
                ("SequenceClassification", "ConditionalGeneration", "QuestionAnswering")
            )
            and family_name not in family_seen
        ):
            family[family_name].append(cls_name)
            family_seen.add(family_name)
        elif cls_name.endswith("ImageClassification"):
            family[family_name].append(cls_name)

    chosen_models = set()
    for members in family.values():
        chosen_models.update(set(members))

    # Add the EXTRA_MODELS
    chosen_models.update(set(EXTRA_MODELS.keys()))

    for model_name in sorted(chosen_models):
        try:
            subprocess.check_call(
                [sys.executable]
                + sys.argv
                + ["--find-batch-sizes"]
                + [f"--only={model_name}"]
                + [f"--output={MODELS_FILENAME}"]
            )
        except subprocess.SubprocessError:
            log.warning(f"Failed to find suitable batch size for {model_name}")


def huggingface_main():
    # Code to refresh model names and batch sizes
    # if "--find-batch-sizes" not in sys.argv:
    #     refresh_model_names_and_batch_sizes()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(HuggingfaceRunner())


if __name__ == "__main__":
    huggingface_main()
