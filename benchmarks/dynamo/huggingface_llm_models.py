import subprocess
import sys

import torch


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StaticCache,
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
        StaticCache,
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
        inputs = dict(
            processor(
                audio, sampling_rate=WhisperBenchmark.SAMPLE_RATE, return_tensors="pt"
            )
        )
        inputs["input_features"] = inputs["input_features"].to(device)

        decoder_start_token = model.config.decoder_start_token_id
        inputs["decoder_input_ids"] = torch.tensor(
            [[decoder_start_token]], device=device
        )

        return model, inputs


class TextGenerationBenchmark(Benchmark):
    INPUT_LENGTH = 1000
    OUTPUT_LENGTH = 2000

    @staticmethod
    def get_model_and_inputs(
        model_name,
        device,
        batch_size=1,
        prompt_length=INPUT_LENGTH,
        inference_mode="generate",
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        model.eval()

        model.generation_config.do_sample = False
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        model.generation_config.max_new_tokens = TextGenerationBenchmark.OUTPUT_LENGTH
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.temperature = 0.0

        if inference_mode == "prefill":
            total_length = prompt_length + TextGenerationBenchmark.OUTPUT_LENGTH
            text_config = model.config.get_text_config(decoder=True)
            context_length = getattr(text_config, "max_position_embeddings", None)
            if context_length is not None and total_length > context_length:
                raise ValueError(
                    f"prompt length {prompt_length} plus "
                    f"{TextGenerationBenchmark.OUTPUT_LENGTH} generated-token slots exceeds "
                    f"{model_name}'s context length {context_length}"
                )
            cache_capacity = total_length - 1
            model = TextGenerationPrefillModel(
                model,
                batch_size=batch_size,
                prompt_length=prompt_length,
                cache_capacity=cache_capacity,
            )

        input_ids = torch.randint(
            low=0,
            high=tokenizer.vocab_size,
            size=(batch_size, prompt_length),
            device=device,
            dtype=torch.long,
        )
        example_inputs = {"input_ids": input_ids}
        return model, example_inputs


class TextGenerationPrefillModel(torch.nn.Module):
    def __init__(self, model, batch_size, prompt_length, cache_capacity):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.prompt_length = prompt_length
        self.cache_capacity = cache_capacity
        self.cache = None

    def prepare_for_prefill(self, input_ids):
        expected_shape = (self.batch_size, self.prompt_length)
        if tuple(input_ids.shape) != expected_shape:
            raise ValueError(
                f"prefill input_ids must have shape {expected_shape}, got "
                f"{tuple(input_ids.shape)}"
            )

        if self.cache is None:
            cache_shape = self.model._get_static_cache_init_shape()
            if cache_shape is None:
                raise RuntimeError(
                    f"{type(self.model).__name__} cannot preallocate a static cache"
                )
            num_heads, head_dim = cache_shape
            config = self.model.config.get_text_config(decoder=True)
            self.cache = StaticCache(
                config=config,
                max_cache_len=self.cache_capacity,
            )
            self.cache.early_initialization(
                batch_size=self.batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=self.model.dtype,
                device=self.model.device,
            )
            if config.model_type == "qwen3_5_text":
                # KV early initialization skips the hybrid cache's linear layers.
                key_dim = config.linear_num_key_heads * config.linear_key_head_dim
                value_dim = config.linear_num_value_heads * config.linear_value_head_dim
                conv_states = torch.empty(
                    self.batch_size,
                    2 * key_dim + value_dim,
                    config.linear_conv_kernel_dim,
                    dtype=self.model.dtype,
                    device=self.model.device,
                )
                recurrent_shape = (
                    self.batch_size,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                )
                recurrent_states = conv_states.new_empty(recurrent_shape)
                for layer, is_linear in zip(self.cache.layers, self.cache.is_linear):
                    if is_linear:
                        layer.lazy_initialization(
                            conv_states=conv_states, recurrent_states=recurrent_states
                        )
        else:
            self.cache.reset()

    def forward(self, input_ids):
        if self.cache is None:
            raise RuntimeError("prepare_for_prefill() must be called before forward()")
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        return outputs.logits[:, -1, :]


HF_LLM_MODELS: dict[str, Benchmark] = {
    "meta-llama/Llama-3.2-1B": TextGenerationBenchmark,
    "google/gemma-2-2b": TextGenerationBenchmark,
    "google/gemma-3-4b-it": TextGenerationBenchmark,
    "openai/whisper-tiny": WhisperBenchmark,
    "Qwen/Qwen3-0.6B": TextGenerationBenchmark,
    "Qwen/Qwen3.5-0.8B": TextGenerationBenchmark,
    "mistralai/Mistral-7B-Instruct-v0.3": TextGenerationBenchmark,
    "openai/gpt-oss-20b": TextGenerationBenchmark,
}


PREFILL_MODELS = {
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-2b",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-0.8B",
}
