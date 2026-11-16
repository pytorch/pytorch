import dataclasses
import subprocess
import sys
from collections.abc import Callable

import torch


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    from transformers import (
        AutoConfig,
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
        AutoConfig,
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
    # Roughly the token count of a transcript of DURATION seconds of speech.
    LABEL_LENGTH = 128
    DECODE_LENGTH = 128
    # Both registered checkpoints have 448 decoder positions. Reserve the final
    # generation slot and the decoder start token when selecting sweep models.
    MAX_DECODE_LENGTH = 446

    @staticmethod
    def get_model_and_inputs(
        model_name,
        device,
        training=False,
        batch_size=1,
        inference_mode="generate",
        decode_length=DECODE_LENGTH,
    ):
        processor = WhisperProcessor.from_pretrained(model_name)
        # whisper-large-v3-turbo's checkpoint is float16. Start from float32, as
        # whisper-tiny's checkpoint does, so that the precision flags decide the dtype.
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.float32
        ).to(device)
        model.config.forced_decoder_ids = None

        model.generation_config.do_sample = False
        model.generation_config.temperature = 0.0

        num_samples = int(WhisperBenchmark.DURATION * WhisperBenchmark.SAMPLE_RATE)
        audio = torch.randn(batch_size, num_samples) * 0.1
        inputs = dict(
            processor(
                list(audio.numpy()),
                sampling_rate=WhisperBenchmark.SAMPLE_RATE,
                return_tensors="pt",
            )
        )
        inputs["input_features"] = inputs["input_features"].to(device)

        if training:
            # As in fine-tuning, the model derives decoder inputs by shifting labels
            # right; without labels it returns no loss to backpropagate.
            model.config.use_cache = False
            inputs["labels"] = torch.randint(
                model.config.vocab_size,
                (batch_size, WhisperBenchmark.LABEL_LENGTH),
                device=device,
            )
            return model, inputs

        if inference_mode == "encode":
            return WhisperEncodeModel(model.get_encoder()), inputs

        decoder_start_token = model.config.decoder_start_token_id
        inputs["decoder_input_ids"] = torch.full(
            (batch_size, 1), decoder_start_token, device=device
        )
        if inference_mode == "decode":
            # Match generate()'s static cache, which holds one position fewer than
            # the max_target_positions tokens that Whisper generates at most.
            cache_capacity = model.config.max_target_positions - 1
            if 1 + decode_length > cache_capacity:
                raise ValueError(
                    f"decode length {decode_length} plus the decoder start token "
                    f"exceeds {model_name}'s {cache_capacity}-position static cache"
                )
            # Fixed tokens instead of argmax, so eager, compiled, and fp64 runs match.
            inputs["decode_ids"] = torch.randint(
                model.config.vocab_size, (batch_size, decode_length), device=device
            )
            return WhisperDecodeModel(model, cache_capacity), inputs

        return model, inputs


class WhisperEncodeModel(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features):
        return self.encoder(input_features).last_hidden_state


class WhisperDecodeModel(torch.nn.Module):
    def __init__(self, model, cache_capacity):
        super().__init__()
        self.model = model
        self.cache_capacity = cache_capacity
        self.cache = None
        self.encoder_hidden_states = None

    def prepare_for_decode(self, input_features, decoder_input_ids):
        from transformers import EncoderDecoderCache, StaticCache

        config = self.model.config
        if self.cache is None:
            cache_shape = self.model._get_static_cache_init_shape()
            if cache_shape is None:
                raise RuntimeError(
                    f"{type(self.model).__name__} cannot preallocate a static cache"
                )
            num_heads, head_dim = cache_shape
            # The cross-attention cache holds the encoder output.
            caches = [
                StaticCache(config=config, max_cache_len=max_cache_len)
                for max_cache_len in (self.cache_capacity, config.max_source_positions)
            ]
            for cache in caches:
                cache.early_initialization(
                    batch_size=input_features.shape[0],
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dtype=self.model.dtype,
                    device=self.model.device,
                )
            self.cache = EncoderDecoderCache(*caches)
        else:
            self.cache.reset()
        # Encode and prefill the start token outside the compiled forward. The
        # prefill fills the cross-attention cache that every decode step reuses.
        with torch.no_grad():
            encoder = self.model.get_encoder()
            self.encoder_hidden_states = encoder(input_features).last_hidden_state
            self.model(
                encoder_outputs=(self.encoder_hidden_states,),
                decoder_input_ids=decoder_input_ids,
                past_key_values=self.cache,
                use_cache=True,
            )

    def forward(self, decoder_input_ids):
        if self.cache is None:
            raise RuntimeError("prepare_for_decode() must be called before forward()")
        outputs = self.model(
            encoder_outputs=(self.encoder_hidden_states,),
            decoder_input_ids=decoder_input_ids,
            past_key_values=self.cache,
            use_cache=True,
        )
        return outputs.logits[:, -1, :]


class TextGenerationBenchmark(Benchmark):
    INPUT_LENGTH = 1000
    OUTPUT_LENGTH = 2000
    DECODE_LENGTH = 128

    @staticmethod
    def get_model_and_inputs(
        model_name,
        device,
        batch_size=1,
        prompt_length=INPUT_LENGTH,
        inference_mode="generate",
        decode_length=DECODE_LENGTH,
        dtype=None,
        tp_size=1,
    ):
        if inference_mode not in ("generate", "prefill", "decode"):
            raise ValueError(f"unsupported inference mode: {inference_mode}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tp_size > 1:
            model = load_tensor_parallel_model(model_name, dtype, tp_size)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        model.eval()

        model.generation_config.do_sample = False
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        model.generation_config.max_new_tokens = TextGenerationBenchmark.OUTPUT_LENGTH
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.temperature = 0.0

        if inference_mode != "generate":
            total_length = prompt_length + TextGenerationBenchmark.OUTPUT_LENGTH
            # Match the static cache generate() allocates for this request, so prefill
            # and decode attend over the same KV length as generate.
            cache_capacity = total_length - 1
            model = TextGenerationCacheModel(
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
        if inference_mode == "decode":
            # Fixed tokens instead of argmax, so eager, compiled, and fp64 runs match.
            example_inputs["decode_ids"] = torch.randint(
                low=0,
                high=tokenizer.vocab_size,
                size=(batch_size, decode_length),
                device=device,
                dtype=torch.long,
            )
        if tp_size > 1:
            # Every rank of the TP group must serve the same request.
            for tensor in example_inputs.values():
                torch.distributed.broadcast(tensor, src=0)
        return model, example_inputs


def load_tensor_parallel_model(model_name, dtype, tp_size):
    config = AutoConfig.from_pretrained(model_name)
    text_config = config.get_text_config(decoder=True)
    num_heads = text_config.num_attention_heads
    num_kv_heads = getattr(text_config, "num_key_value_heads", None) or num_heads
    if num_heads % tp_size or num_kv_heads % tp_size:
        raise ValueError(
            f"--tp-size={tp_size} must divide {model_name}'s {num_heads} attention "
            f"heads and {num_kv_heads} KV heads"
        )
    with torch.device("meta"):
        meta_model = AutoModelForCausalLM.from_config(config)
    # A tied weight such as lm_head's is the replicated input embedding, so
    # gathering the output of a module planned as sharded would duplicate it.
    tp_plan = {
        name: style
        for name, style in meta_model.tp_plan.items()
        if f"{name}.weight" not in meta_model.all_tied_weights_keys
    }
    # tp_plan shards onto the already initialized process group and places each
    # rank's shard on its LOCAL_RANK device; it rejects device_map.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, tp_plan=tp_plan, tp_size=tp_size
    )
    tp_module = "transformers.integrations.tensor_parallel"
    tp_forward_hooks = [
        hook
        for module in model.modules()
        for hook in module._forward_hooks.values()
        if hook.__module__ == tp_module
    ]
    if model.tp_size != tp_size or not tp_forward_hooks:
        raise RuntimeError(f"{model_name} was not loaded tensor-parallel")
    # The TP plan registers module backward hooks that all-reduce gradients of
    # replicated parameters such as q_norm. They do nothing without autograd, and
    # Dynamo cannot trace module backward hooks unless compiled autograd is on.
    for module in model.modules():
        for key, hook in list(module._backward_hooks.items()):
            if hook.__module__ == tp_module:
                del module._backward_hooks[key]
    return model


# The cache wrappers rely on StaticCache internals from this exact release.
CACHE_TRANSFORMERS_VERSION = "5.14.1"


class TextGenerationCacheModel(torch.nn.Module):
    def __init__(self, model, batch_size, prompt_length, cache_capacity):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.prompt_length = prompt_length
        self.cache_capacity = cache_capacity
        self.cache = None

    def copy_sharing_weights(self):
        # Copies share parameters and module hooks, including HF's TP hooks; each
        # copy allocates its own cache, so requests on one cannot disturb another.
        copy = TextGenerationCacheModel(
            self.model, self.batch_size, self.prompt_length, self.cache_capacity
        )
        return copy.train(self.training)

    def prepare_for_prefill(self, input_ids):
        from transformers import StaticCache

        expected_shape = (self.batch_size, self.prompt_length)
        if tuple(input_ids.shape) != expected_shape:
            raise ValueError(
                f"prompt input_ids must have shape {expected_shape}, got "
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
            if any(self.cache.is_linear):
                # KV early initialization skips the hybrid cache's linear layers.
                # These tensors are shape templates: lazy_initialization allocates
                # zeroed states for each layer.
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
                recurrent_states = conv_states.new_empty(
                    recurrent_shape, dtype=torch.float32
                )
                for layer, is_linear in zip(self.cache.layers, self.cache.is_linear):
                    if is_linear:
                        layer.lazy_initialization(
                            conv_states=conv_states, recurrent_states=recurrent_states
                        )
            if not all(
                initialized
                for layer, is_linear in zip(self.cache.layers, self.cache.is_linear)
                if is_linear
                for initialized in layer.is_conv_states_initialized.values()
            ):
                raise RuntimeError(
                    f"cannot preallocate linear attention cache for {config.model_type}"
                )
        else:
            self.cache.reset()

    def prepare_for_decode(self, input_ids):
        self.prepare_for_prefill(input_ids)
        # Prefill outside the compiled forward: a second input shape would make
        # Dynamo recompile the decode graph with a dynamic sequence dim.
        with torch.no_grad():
            self.model(
                input_ids=input_ids,
                past_key_values=self.cache,
                use_cache=True,
                logits_to_keep=1,
            )

    def forward(self, input_ids):
        if self.cache is None:
            raise RuntimeError(
                "prepare_for_prefill() or prepare_for_decode() must be called before forward()"
            )
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
    "openai/whisper-large-v3-turbo": WhisperBenchmark,
    "Qwen/Qwen3-0.6B": TextGenerationBenchmark,
    "Qwen/Qwen3.5-0.8B": TextGenerationBenchmark,
    "Qwen/Qwen2.5-0.5B-Instruct": TextGenerationBenchmark,
    "google/gemma-3-1b-it": TextGenerationBenchmark,
    "mistralai/Mistral-7B-Instruct-v0.3": TextGenerationBenchmark,
    "openai/gpt-oss-20b": TextGenerationBenchmark,
}


def setup_prefill(model, example_inputs):
    torch.compiler.cudagraph_mark_step_begin()
    model.prepare_for_prefill(example_inputs["input_ids"])


def prefill_forward(model, example_inputs, collect_outputs=True):
    with torch.no_grad():
        return model(example_inputs["input_ids"])


def setup_decode(model, example_inputs):
    torch.compiler.cudagraph_mark_step_begin()
    prompt = {k: v for k, v in example_inputs.items() if k != "decode_ids"}
    model.prepare_for_decode(**prompt)


def decode_forward(model, example_inputs, collect_outputs=True):
    decode_ids = example_inputs["decode_ids"]
    with torch.no_grad():
        for step in range(decode_ids.shape[1]):
            torch.compiler.cudagraph_mark_step_begin()
            logits = model(decode_ids[:, step : step + 1])
    return logits


def encode_forward(model, example_inputs, collect_outputs=True):
    with torch.no_grad():
        return model(example_inputs["input_features"])


@dataclasses.dataclass(frozen=True)
class InferencePhase:
    models: frozenset[str]
    iter_fn: Callable
    # Prepares the cache for each request, outside the timed region.
    setup_fn: Callable | None = None
    # Throughput column and the input whose tokens it counts, if any.
    tokens_column: str | None = None
    tokens_input: str | None = None


# Keyed by --hf-inference-mode; generate is the default and is not a phase.
INFERENCE_PHASES = {
    "prefill": InferencePhase(
        models=frozenset(
            {
                "meta-llama/Llama-3.2-1B",
                "google/gemma-2-2b",
                "google/gemma-3-1b-it",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen3-0.6B",
                "Qwen/Qwen3.5-0.8B",
            }
        ),
        iter_fn=prefill_forward,
        setup_fn=setup_prefill,
        tokens_column="input_tokens_per_second",
        tokens_input="input_ids",
    ),
    # Gemma models are excluded: their sliding-window cache layers track their
    # length in a Python int, so compiled decode recompiles every step.
    "decode": InferencePhase(
        models=frozenset(
            {
                "meta-llama/Llama-3.2-1B",
                "openai/whisper-large-v3-turbo",
                "openai/whisper-tiny",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen3-0.6B",
                "Qwen/Qwen3.5-0.8B",
            }
        ),
        iter_fn=decode_forward,
        setup_fn=setup_decode,
        tokens_column="output_tokens_per_second",
        tokens_input="decode_ids",
    ),
    "encode": InferencePhase(
        models=frozenset({"openai/whisper-large-v3-turbo", "openai/whisper-tiny"}),
        iter_fn=encode_forward,
    ),
}


# gemma-3-1b-it is excluded: its single KV head cannot be sharded.
TP_MODELS = {
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-0.8B",
}
