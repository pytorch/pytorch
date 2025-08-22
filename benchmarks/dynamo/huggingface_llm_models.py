import subprocess
import sys

import torch


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
except ModuleNotFoundError:
    print("Installing HuggingFace Transformers...")
    pip_install("git+https://github.com/huggingface/transformers.git#egg=transformers")
finally:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )


class Benchmark:
    @staticmethod
    def get_model_and_inputs(model_name, device):
        raise NotImplementedError("get_model_and_inputs() not implemented")


class WhisperBenchmark(Benchmark):
    SAMPLE_RATE = 16000
    DURATION = 30.0  # seconds

    @staticmethod
    def get_model_and_inputs(model_name, device):
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        model.config.forced_decoder_ids = None

        model.generation_config.do_sample = False
        model.generation_config.temperature = 0.0

        num_samples = int(WhisperBenchmark.DURATION * WhisperBenchmark.SAMPLE_RATE)
        audio = torch.randn(num_samples) * 0.1
        inputs = processor(
            audio, sampling_rate=WhisperBenchmark.SAMPLE_RATE, return_tensors="pt"
        )
        inputs["input_features"] = inputs["input_features"].to(device)

        return model, dict(inputs)


class TextGenerationBenchmark(Benchmark):
    INPUT_LENGTH = 1000
    OUTPUT_LENGTH = 2000

    @staticmethod
    def get_model_and_inputs(model_name, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
        ).to(device)
        model.tokenizer = tokenizer

        model.generation_config.do_sample = False
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        model.generation_config.max_new_tokens = 2000
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.temperature = 0.0

        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(1, TextGenerationBenchmark.INPUT_LENGTH),
            device=device,
            dtype=torch.long,
        )

        return model, {"input_ids": input_ids}


HF_LLM_MODELS: dict[str, Benchmark] = {
    "meta-llama/Llama-3.2-1B": TextGenerationBenchmark,
    "google/gemma-2-2b": TextGenerationBenchmark,
    "google/gemma-3-4b-it": TextGenerationBenchmark,
    "openai/whisper-tiny": WhisperBenchmark,
    "Qwen/Qwen3-0.6B": TextGenerationBenchmark,
}
