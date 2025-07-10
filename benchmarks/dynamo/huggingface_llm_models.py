
import copy
import importlib
import time
from contextlib import nullcontext

import torch

imports = [
    "AutoModelForCausalLM",
    "AutoTokenizer",
    "WhisperForConditionalGeneration",
    "WhisperProcessor",
]

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

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

try:
    mod = importlib.import_module("datasets")
    from datasets import load_dataset
except ModuleNotFoundError:
    print("Installing HuggingFace Datasets...")
    pip_install("git+https://github.com/huggingface/datasets.git#egg=datasets")
finally:
    from datasets import load_dataset



class Benchmark:
    @staticmethod
    def get_model_and_inputs( model_name, device):
        raise NotImplementedError("get_model_and_inputs() not implemented")


class WhisperBenchmark(Benchmark):
    @staticmethod
    def get_model_and_inputs(model_name, device):
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(
            device
        )
        model.config.forced_decoder_ids = None

        processor = WhisperProcessor.from_pretrained(model_name)

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = ds[0]["audio"]
        inputs = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt")
        inputs["input_features"] = inputs["input_features"].to(device)

        return model, dict(inputs)


class TextGenerationBenchmark(Benchmark):
    @staticmethod
    def get_model_and_inputs(model_name, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
        ).to(device)

        prompt = "Once upon a time,"

        input_ids = tokenizer(
            prompt, return_tensors="pt"
        ).to(device)
        
        generation_kwargs = {
            "do_sample": False,
            "top_k": 1,
            "max_length": 50,
        }
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.update(**generation_kwargs)
        input_ids["generation_config"] = generation_config

        return model, dict(input_ids)


models: dict[str, Benchmark] = {
    "meta-llama/Llama-3.2-1B": TextGenerationBenchmark,
    "google/gemma-2-2b": TextGenerationBenchmark,
    "google/gemma-3-4b-it": TextGenerationBenchmark,
    "openai/whisper-tiny": WhisperBenchmark,
    "Qwen/Qwen3-0.6B": TextGenerationBenchmark,
}
