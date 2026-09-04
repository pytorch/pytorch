"""Decomposed LLM benchmarks for the dynamo benchmark suite.

`huggingface_llm_models.TextGenerationBenchmark` hands the whole generation
loop to `model.generate()`, which folds two very different regimes into a
single end-to-end number: prefill is compute bound and quadratic in the prompt
length, decode is memory-bandwidth bound and linear in the cache depth. This
module owns the loop instead, so each phase can be built and measured on its
own.

Run it through the dynamo benchmark CLI::

    python benchmarks/dynamo/huggingface.py --inference --performance \\
        --llm-mode prefill --prompt-length 4096

or standalone::

    python benchmarks/dynamo/torchbench_llm.py --llm-mode decode
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch


SAMPLING_MODES = ("greedy", "topk", "topp")
LLM_MODES = ("prefill", "decode", "e2e")
DEFAULT_PROMPT_LENGTHS = [128, 512, 1024, 4096, 8192, 32768]
DEFAULT_DECODE_LENGTHS = [256, 1024, 4096, 16384]
DEFAULT_LLM_MODEL = "Qwen/Qwen3-0.6B"

# Chunk size used to bring a KV cache up to a target depth. Filling a 16384
# deep cache one token at a time costs 16384 sequential launches; a handful of
# chunked prefills costs a handful.
CACHE_FILL_CHUNK = 2048

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_DTYPE_ALIASES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Coerce a dtype spelling to a `torch.dtype`.

    Accepts a `torch.dtype` directly, or one of the `_DTYPE_ALIASES` spellings
    with an optional `torch.` prefix. Raises `ValueError` for anything else.
    """
    if isinstance(dtype, torch.dtype):
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"unsupported dtype {dtype}; expected one of {_SUPPORTED_DTYPES}"
            )
        return dtype
    if not isinstance(dtype, str):
        raise ValueError(f"unsupported dtype {dtype!r}; expected a torch.dtype or str")

    key = dtype.strip().lower().removeprefix("torch.")
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of "
            f"{sorted(_DTYPE_ALIASES)} with an optional 'torch.' prefix"
        )
    return _DTYPE_ALIASES[key]


def dtype_name(dtype: torch.dtype | str) -> str:
    """`torch.bfloat16` -> `"bfloat16"`, for JSON-friendly output."""
    return str(resolve_dtype(dtype)).removeprefix("torch.")


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(v) for v in value]


def _normalize_sampling(sampling: Any) -> str:
    raw = str(sampling).strip().lower()
    key = raw.replace("-", "").replace("_", "")
    if key == "nucleus":
        key = "topp"
    # Compare normalized forms, but return canonical from SAMPLING_MODES
    allowed_map = {
        s.lower().replace("-", "").replace("_", ""): s for s in SAMPLING_MODES
    }
    if key not in allowed_map:
        raise ValueError(
            f"unsupported sampling {sampling!r}; expected one of {SAMPLING_MODES}"
        )
    return allowed_map[key]


@dataclasses.dataclass
class LLMBenchmarkConfig:
    """Everything the harness needs to build and measure a phase."""

    model_name: str = DEFAULT_LLM_MODEL
    prompt_lengths: list[int] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_PROMPT_LENGTHS)
    )
    decode_lengths: list[int] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_DECODE_LENGTHS)
    )
    batch_sizes: list[int] = dataclasses.field(default_factory=lambda: [1])
    dtype: torch.dtype | str = torch.bfloat16
    device: str = "cuda"
    sampling: str = "greedy"  # greedy | top-k | top-p
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 32
    warmup: int = 2
    iters: int = 5
    compile: bool = False
    full_logits: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        self.dtype = resolve_dtype(self.dtype)
        self.sampling = _normalize_sampling(self.sampling)
        self.prompt_lengths = _as_int_list(self.prompt_lengths)
        self.decode_lengths = _as_int_list(self.decode_lengths)
        self.batch_sizes = _as_int_list(self.batch_sizes)

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["dtype"] = dtype_name(self.dtype)
        return out


# ---------------------------------------------------------------------------
# Model / config plumbing
# ---------------------------------------------------------------------------
def _text_config(config: Any) -> Any:
    """Unwrap the decoder config of a multimodal wrapper, where one exists."""
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            return getter(decoder=True)
        except TypeError:
            return getter()
    return config


def _head_dim(config: Any) -> int:
    head_dim = getattr(config, "head_dim", None)
    if head_dim:
        return int(head_dim)
    hidden = int(getattr(config, "hidden_size", 0) or 0)
    heads = int(getattr(config, "num_attention_heads", 0) or 0)
    return hidden // heads if heads else 0


def _num_kv_heads(config: Any) -> int:
    kv_heads = getattr(config, "num_key_value_heads", None)
    if kv_heads:
        return int(kv_heads)
    return int(getattr(config, "num_attention_heads", 0) or 0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def count_parameters(model: torch.nn.Module, exclude_embeddings: bool = True) -> int:
    """Parameter count, by default excluding the (possibly tied) vocab tables.

    Embedding tables dominate the parameter count of small models but do almost
    no FLOPs per token, so a roofline built on the raw count is badly skewed.
    """
    total = sum(p.numel() for p in model.parameters())
    if not exclude_embeddings:
        return total

    seen: set[int] = set()
    embedding_params = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            for param in module.parameters(recurse=False):
                if id(param) not in seen:
                    seen.add(id(param))
                    embedding_params += param.numel()

    # An untied lm_head is a second vocab-sized table; a tied one is the object
    # already counted above, which `seen` filters out.
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None) if head is not None else None
    if weight is not None and id(weight) not in seen:
        seen.add(id(weight))
        embedding_params += weight.numel()

    return total - embedding_params


def prefill_flops(
    model_config: Any, num_params: int, batch_size: int, seq_len: int
) -> int:
    """Forward FLOPs for a prompt of `seq_len` tokens.

    Two terms: the dense projections, which are linear in the token count, and
    the attention scores/values, which are quadratic in the sequence length.
    The quadratic term is what makes prefill compute bound at long contexts.
    """
    tokens = int(batch_size) * int(seq_len)
    dense = 2 * int(num_params) * tokens

    layers = int(getattr(model_config, "num_hidden_layers", 0) or 0)
    heads = int(getattr(model_config, "num_attention_heads", 0) or 0)
    attn_dim = heads * _head_dim(model_config)
    # 2 matmuls (QK^T and attn@V), 2 FLOPs per multiply-accumulate.
    attention = 2 * 2 * layers * int(batch_size) * int(seq_len) ** 2 * attn_dim

    return dense + attention


def kv_cache_bytes(
    config: Any, batch_size: int, seq_len: int, dtype: torch.dtype | str
) -> int:
    """Resident bytes of a KV cache holding `seq_len` tokens.

    `config` is the model config: the cache is shaped by the model's layer and
    KV-head geometry, not by the benchmark settings.
    """
    config = _text_config(config)
    layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    itemsize = torch.empty((), dtype=resolve_dtype(dtype)).element_size()
    # 2 for the key and the value tensor.
    return (
        2
        * layers
        * int(batch_size)
        * int(seq_len)
        * _num_kv_heads(config)
        * _head_dim(config)
        * itemsize
    )


def model_weight_bytes(model: torch.nn.Module) -> int:
    """Resident bytes of the model weights."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def peak_bandwidth_gbps(device: str | torch.device) -> float | None:
    """Theoretical peak memory bandwidth in GB/s, or None if unknown.

    Decode is bandwidth bound, so this is the denominator for the achieved
    bandwidth reported by `benchmark_decode`.
    """
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(dev)
    bus_width_bits = int(getattr(props, "memory_bus_width", 0) or 0)
    clock_khz = int(getattr(props, "memory_clock_rate", 0) or 0)
    if not bus_width_bits or not clock_khz:
        return None
    # Double data rate: two transfers per clock.
    bytes_per_cycle = 2 * bus_width_bits / 8
    return bytes_per_cycle * clock_khz * 1e3 / 1e9


def _time_calls(
    fn: Callable[[], Any],
    warmup: int,
    iters: int,
    device: str | torch.device,
) -> dict[str, Any]:
    """Warm up, synchronize, then time `iters` calls with a sync per call."""
    is_cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()

    def sync() -> None:
        if is_cuda:
            torch.cuda.synchronize(device)

    for _ in range(max(int(warmup), 0)):
        fn()
    sync()

    latencies_ms: list[float] = []
    for _ in range(max(int(iters), 1)):
        start = time.perf_counter()
        fn()
        sync()
        latencies_ms.append((time.perf_counter() - start) * 1e3)

    return {
        "warmup": max(int(warmup), 0),
        "iters": len(latencies_ms),
        "latencies_ms": latencies_ms,
        "latency_ms_mean": statistics.fmean(latencies_ms),
        "latency_ms_median": statistics.median(latencies_ms),
        "latency_ms_min": min(latencies_ms),
        "latency_ms_max": max(latencies_ms),
    }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _reset_cache(cache: Any) -> None:
    reset = getattr(cache, "reset", None)
    if callable(reset):
        reset()


def _set_cache_depth(cache: Any, depth: int) -> None:
    """Rewind a static cache's write cursor back to `depth`.

    Static cache layers keep their own monotonic write cursor and ignore the
    `cache_position` passed to the model, so repeatedly timing a single decode
    step would walk off the end of the preallocated tensors. Rewinding between
    timed calls keeps every iteration measuring the same cache depth.
    """
    for layer in getattr(cache, "layers", []) or []:
        cursor = getattr(layer, "cumulative_length", None)
        if isinstance(cursor, torch.Tensor):
            cursor.fill_(depth)
        elif cursor is not None:
            layer.cumulative_length = depth


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
class LLMBenchmark:
    """Owns the generation loop so prefill and decode can be measured apart."""

    def __init__(self, config: LLMBenchmarkConfig | None = None) -> None:
        self.config = config if config is not None else LLMBenchmarkConfig()
        self.model: torch.nn.Module | None = None
        self.tokenizer: Any = None
        self.model_config: Any = None
        self._compiled_generate: Callable[..., torch.Tensor] | None = None

    # -- setup --------------------------------------------------------------
    def setup_model(self, config: LLMBenchmarkConfig | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = config if config is not None else self.config
        self.config = config
        torch.manual_seed(config.seed)

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name, dtype=config.dtype
            )
        except TypeError:
            # transformers < 5 spells it `torch_dtype`.
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name, torch_dtype=config.dtype
            )
        model = model.to(device=config.device, dtype=config.dtype)
        model.eval()
        model.config.use_cache = True

        self.model = model
        self.tokenizer = tokenizer
        self.model_config = _text_config(model.config)
        self._compiled_generate = None
        return model, tokenizer

    def _require_model(self) -> torch.nn.Module:
        if self.model is None:
            self.setup_model(self.config)
        if self.model is None:
            raise AssertionError("setup_model did not populate self.model")
        return self.model

    def _vocab_size(self) -> int:
        sizes = []
        if self.model_config is not None:
            sizes.append(int(getattr(self.model_config, "vocab_size", 0) or 0))
        tokenizer_vocab = getattr(self.tokenizer, "vocab_size", None)
        if tokenizer_vocab:
            sizes.append(int(tokenizer_vocab))
        sizes = [s for s in sizes if s > 0]
        return min(sizes) if sizes else 32000

    # -- cache --------------------------------------------------------------
    def make_cache(
        self, model: torch.nn.Module, max_cache_len: int, batch_size: int = 1
    ):
        """Build a `StaticCache`, tolerating constructor drift across releases.

        transformers >= 5 takes only `(config, max_cache_len)` and infers batch,
        device and dtype lazily on the first write; earlier releases required
        them up front under a few different names. Pass whatever this release
        actually declares and let the rest be inferred.
        """
        from transformers import StaticCache

        config = _text_config(model.config)
        params = inspect.signature(StaticCache.__init__).parameters
        kwargs: dict[str, Any] = {
            "config": config,
            "max_cache_len": int(max_cache_len),
        }
        if "max_batch_size" in params:
            kwargs["max_batch_size"] = int(batch_size)
        elif "batch_size" in params:
            kwargs["batch_size"] = int(batch_size)
        if "device" in params:
            kwargs["device"] = torch.device(self.config.device)
        if "dtype" in params:
            kwargs["dtype"] = self.config.dtype

        cache = StaticCache(**kwargs)

        # Allocate up front where the release supports it: a cache whose
        # tensors already exist has stable data pointers, which is what lets
        # the decode loop be captured rather than re-traced.
        early_init = getattr(cache, "early_initialization", None)
        if callable(early_init) and len(getattr(cache, "layers", []) or []):
            try:
                early_init(
                    int(batch_size),
                    _num_kv_heads(config),
                    _head_dim(config),
                    self.config.dtype,
                    torch.device(self.config.device),
                )
            except TypeError:
                pass

        return cache

    # -- inputs -------------------------------------------------------------
    def build_prefill_inputs(
        self, prompt_length: int, batch_size: int = 1
    ) -> torch.Tensor:
        """Deterministic random `input_ids` of shape `[batch, prompt_length]`.

        Token identity does not affect prefill cost, and a seeded generator
        keeps successive runs comparable without touching global RNG state.
        """
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        input_ids = torch.randint(
            low=0,
            high=self._vocab_size(),
            size=(int(batch_size), int(prompt_length)),
            generator=generator,
            dtype=torch.long,
        )
        return input_ids.to(self.config.device)

    def build_decode_inputs(self, cache_depth: int, batch_size: int = 1):
        """Return `(token, kv_cache, position)` with the cache at `cache_depth`.

        The cache is filled with chunked prefills rather than `cache_depth`
        single-token steps, so setting up a 16384 deep cache costs a handful of
        launches instead of sixteen thousand.
        """
        model = self._require_model()
        cache_depth = int(cache_depth)
        batch_size = int(batch_size)

        # One slot of headroom for the decode step this cache is built for.
        cache = self.make_cache(model, cache_depth + 1, batch_size)
        input_ids = self.build_prefill_inputs(cache_depth, batch_size)

        logits = None
        with torch.no_grad():
            for start in range(0, cache_depth, CACHE_FILL_CHUNK):
                chunk = input_ids[:, start : start + CACHE_FILL_CHUNK]
                cache_position = torch.arange(
                    start, start + chunk.shape[1], device=input_ids.device
                )
                logits, cache = self._forward(
                    model,
                    input_ids=chunk,
                    kv_cache=cache,
                    cache_position=cache_position,
                    full_logits=False,
                )

        if logits is None:
            # cache_depth == 0: nothing was prefilled, so seed the loop with a
            # deterministic token rather than a prediction.
            token = self.build_prefill_inputs(1, batch_size)
        else:
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        position = torch.full(
            (1,), cache_depth, dtype=torch.long, device=input_ids.device
        )
        return token, cache, position

    # -- phases -------------------------------------------------------------
    def _logits_to_keep_name(self, model: Any) -> str | None:
        """Name of the last-position slicing kwarg this model accepts, if any.

        Checks `forward` first, then the callable itself, so plain callables
        that only define `__call__` are handled too; anything unintrospectable
        falls through to None and the manual projection path.
        """
        for target in (getattr(model, "forward", None), model):
            if target is None or not callable(target):
                continue
            try:
                params = inspect.signature(target).parameters
            except (TypeError, ValueError):
                continue
            for name in ("logits_to_keep", "num_logits_to_keep"):
                if name in params:
                    return name
        return None

    def _forward(
        self,
        model: torch.nn.Module,
        *,
        input_ids: torch.Tensor,
        kv_cache: Any,
        cache_position: torch.Tensor,
        full_logits: bool,
    ):
        """Forward pass returning `(logits, kv_cache)`.

        With `full_logits` false, only the last position is projected. A full
        `[batch, seq, vocab]` tensor is 10GB at batch 1, 32768 tokens and a
        151k vocab in bf16, so materializing it is not an option at the top of
        the sweep.
        """
        if full_logits:
            out = model(
                input_ids=input_ids,
                past_key_values=kv_cache,
                cache_position=cache_position,
                use_cache=True,
            )
            return out.logits, out.past_key_values

        keep = self._logits_to_keep_name(model)
        if keep is not None:
            out = model(
                input_ids=input_ids,
                past_key_values=kv_cache,
                cache_position=cache_position,
                use_cache=True,
                **{keep: 1},
            )
            return out.logits, out.past_key_values

        # No slicing kwarg on this release: run the decoder stack directly and
        # project the last position by hand.
        base = getattr(model, "model", None) or getattr(model, "transformer", None)
        head = getattr(model, "lm_head", None)
        if base is None or head is None:
            out = model(
                input_ids=input_ids,
                past_key_values=kv_cache,
                cache_position=cache_position,
                use_cache=True,
            )
            return out.logits[:, -1:, :], out.past_key_values

        hidden = base(
            input_ids=input_ids,
            past_key_values=kv_cache,
            cache_position=cache_position,
            use_cache=True,
        )
        last_hidden = hidden.last_hidden_state[:, -1:, :]
        return head(last_hidden), getattr(hidden, "past_key_values", None) or kv_cache

    def prefill(
        self, model: torch.nn.Module, input_ids: torch.Tensor, kv_cache: Any = None
    ):
        """Prompt pass. Returns `(logits, kv_cache)`."""
        batch_size, seq_len = input_ids.shape
        if kv_cache is None:
            kv_cache = self.make_cache(
                model, seq_len + self.config.max_new_tokens, batch_size
            )
        cache_position = torch.arange(seq_len, device=input_ids.device)
        with torch.no_grad():
            return self._forward(
                model,
                input_ids=input_ids,
                kv_cache=kv_cache,
                cache_position=cache_position,
                full_logits=self.config.full_logits,
            )

    def decode_step(
        self,
        model: torch.nn.Module,
        token: torch.Tensor,
        kv_cache: Any,
        cache_position: torch.Tensor,
    ):
        """One token against an existing cache. Returns `(logits, kv_cache)`."""
        if token.dim() == 1:
            token = token.unsqueeze(-1)
        if cache_position.dim() == 0:
            cache_position = cache_position.reshape(1)
        with torch.no_grad():
            out = model(
                input_ids=token,
                past_key_values=kv_cache,
                cache_position=cache_position,
                use_cache=True,
            )
        return out.logits, out.past_key_values

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Pick the next token. Returns shape `[batch]`."""
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        mode = self.config.sampling
        if mode == "greedy":
            return torch.argmax(logits, dim=-1)

        # Accept both hyphenated and non-hyphenated canonical forms
        norm_mode = str(mode).lower().replace("-", "").replace("_", "")
        if norm_mode == "topk":
            k = max(1, min(int(self.config.top_k), logits.shape[-1]))
            values, indices = torch.topk(logits, k, dim=-1)
            probs = torch.softmax(values, dim=-1)
            picked = torch.multinomial(probs, num_samples=1)
            return indices.gather(-1, picked).squeeze(-1)

        # Nucleus: the smallest prefix of the sorted distribution whose mass
        # reaches top_p, always at least one token.
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        preceding_mass = probs.cumsum(dim=-1) - probs
        keep = preceding_mass < float(self.config.top_p)
        keep[..., 0] = True
        filtered = probs.masked_fill(~keep, 0.0)
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
        picked = torch.multinomial(filtered, num_samples=1)
        return sorted_indices.gather(-1, picked).squeeze(-1)

    def generate(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
        kv_cache: Any = None,
    ) -> torch.Tensor:
        """Prefill plus a fixed number of decode steps.

        The trip count is fixed and there is no EOS early exit, so the loop
        carries no data-dependent control flow and survives
        `torch.compile(dynamic=False)`.
        """
        steps = int(
            self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        )
        if steps <= 0:
            return input_ids

        batch_size, prompt_length = input_ids.shape
        if kv_cache is None:
            kv_cache = self.make_cache(model, prompt_length + steps, batch_size)
        else:
            # The cache cursor is monotonic, so a reused cache has to be
            # rewound or the prompt lands after the previous run's tokens.
            _reset_cache(kv_cache)

        logits, kv_cache = self.prefill(model, input_ids, kv_cache=kv_cache)
        token = self.sample(logits)
        generated = [token]

        cache_position = torch.full(
            (1,), prompt_length, dtype=torch.long, device=input_ids.device
        )
        for _ in range(steps - 1):
            logits, kv_cache = self.decode_step(model, token, kv_cache, cache_position)
            token = self.sample(logits)
            generated.append(token)
            cache_position = cache_position + 1

        return torch.cat([input_ids, torch.stack(generated, dim=1)], dim=1)

    def compiled_generate(self, **compile_kwargs: Any) -> Callable[..., torch.Tensor]:
        """`generate` under `torch.compile`, static shapes by default."""
        compile_kwargs.setdefault("dynamic", False)
        if compile_kwargs == {"dynamic": False}:
            if self._compiled_generate is None:
                self._compiled_generate = torch.compile(self.generate, dynamic=False)
            return self._compiled_generate
        return torch.compile(self.generate, **compile_kwargs)

    # -- measurement --------------------------------------------------------
    def _common_fields(self, mode: str, batch_size: int) -> dict[str, Any]:
        return {
            "mode": mode,
            "model": self.config.model_name,
            "device": self.config.device,
            "dtype": dtype_name(self.config.dtype),
            "batch_size": int(batch_size),
            "compile": bool(self.config.compile),
        }

    def _time(self, fn: Callable[[], Any]) -> dict[str, Any]:
        cfg = self.config
        return _time_calls(fn, cfg.warmup, cfg.iters, cfg.device)

    def benchmark_prefill(
        self, prompt_length: int, batch_size: int = 1
    ) -> dict[str, Any]:
        model = self._require_model()
        prompt_length = int(prompt_length)
        batch_size = int(batch_size)

        input_ids = self.build_prefill_inputs(prompt_length, batch_size)
        cache = self.make_cache(model, prompt_length + 1, batch_size)

        def run() -> None:
            _reset_cache(cache)
            self.prefill(model, input_ids, kv_cache=cache)

        timing = self._time(run)

        num_params = count_parameters(model)
        flops = prefill_flops(self.model_config, num_params, batch_size, prompt_length)
        tokens = batch_size * prompt_length
        seconds = timing["latency_ms_mean"] / 1e3

        result = self._common_fields("prefill", batch_size)
        result.update(timing)
        result.update(
            {
                "prompt_length": prompt_length,
                "tokens": tokens,
                "tokens_per_s": tokens / seconds if seconds else None,
                "num_params": num_params,
                "flops": flops,
                "tflops_per_s": flops / seconds / 1e12 if seconds else None,
                "kv_cache_bytes": kv_cache_bytes(
                    self.model_config, batch_size, prompt_length, self.config.dtype
                ),
                "model_weight_bytes": model_weight_bytes(model),
            }
        )
        return result

    def benchmark_decode(self, cache_depth: int, batch_size: int = 1) -> dict[str, Any]:
        model = self._require_model()
        cache_depth = int(cache_depth)
        batch_size = int(batch_size)

        token, cache, position = self.build_decode_inputs(cache_depth, batch_size)

        def run() -> None:
            # Every timed step must see the same cache depth.
            _set_cache_depth(cache, cache_depth)
            self.decode_step(model, token, cache, position)

        timing = self._time(run)

        seconds = timing["latency_ms_mean"] / 1e3
        weight_bytes = model_weight_bytes(model)
        cache_bytes = kv_cache_bytes(
            self.model_config, batch_size, cache_depth, self.config.dtype
        )
        # A decode step reads the weights once and the whole cache once, which
        # is why the phase is bandwidth bound rather than compute bound.
        moved_bytes = weight_bytes + cache_bytes
        peak = peak_bandwidth_gbps(self.config.device)
        achieved = moved_bytes / seconds / 1e9 if seconds else None

        result = self._common_fields("decode", batch_size)
        result.update(timing)
        result.update(
            {
                "cache_depth": cache_depth,
                "tokens": batch_size,
                "tokens_per_s": batch_size / seconds if seconds else None,
                "num_params": count_parameters(model),
                "model_weight_bytes": weight_bytes,
                "kv_cache_bytes": cache_bytes,
                "bytes_moved": moved_bytes,
                "achieved_bandwidth_gbps": achieved,
                "peak_bandwidth_gbps": peak,
                "bandwidth_utilization": (
                    achieved / peak if achieved is not None and peak else None
                ),
            }
        )
        return result

    def benchmark_e2e(
        self,
        prompt_length: int,
        batch_size: int = 1,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        model = self._require_model()
        prompt_length = int(prompt_length)
        batch_size = int(batch_size)
        steps = int(
            self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        )

        input_ids = self.build_prefill_inputs(prompt_length, batch_size)
        cache = self.make_cache(model, prompt_length + steps, batch_size)
        generate = self.compiled_generate() if self.config.compile else self.generate

        def run() -> None:
            generate(model, input_ids, steps, kv_cache=cache)

        timing = self._time(run)

        seconds = timing["latency_ms_mean"] / 1e3
        new_tokens = batch_size * steps

        result = self._common_fields("e2e", batch_size)
        result.update(timing)
        result.update(
            {
                "prompt_length": prompt_length,
                "max_new_tokens": steps,
                "tokens": new_tokens,
                "tokens_per_s": new_tokens / seconds if seconds else None,
                "sampling": self.config.sampling,
                "num_params": count_parameters(model),
                "model_weight_bytes": model_weight_bytes(model),
                "kv_cache_bytes": kv_cache_bytes(
                    self.model_config,
                    batch_size,
                    prompt_length + steps,
                    self.config.dtype,
                ),
            }
        )
        return result

    def run_sweep(self, mode: str | None = None) -> list[dict[str, Any]]:
        mode = mode or "e2e"
        if mode not in LLM_MODES:
            raise ValueError(f"unsupported mode {mode!r}; expected one of {LLM_MODES}")

        results: list[dict[str, Any]] = []
        for batch_size in self.config.batch_sizes or [1]:
            if mode == "prefill":
                for prompt_length in self.config.prompt_lengths:
                    results.append(self.benchmark_prefill(prompt_length, batch_size))
            elif mode == "decode":
                for cache_depth in self.config.decode_lengths:
                    results.append(self.benchmark_decode(cache_depth, batch_size))
            else:
                for prompt_length in self.config.prompt_lengths:
                    results.append(self.benchmark_e2e(prompt_length, batch_size))
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_llm_args(parser: argparse.ArgumentParser):
    """Add the decomposed-LLM options to a dynamo benchmark parser."""
    group = parser.add_argument_group("decomposed LLM benchmarks")
    group.add_argument(
        "--llm-mode",
        choices=LLM_MODES,
        default=None,
        help="run the decomposed LLM benchmark for a single phase",
    )
    group.add_argument(
        "--prompt-length",
        type=int,
        default=None,
        help="prompt length to benchmark; defaults to the full sweep",
    )
    group.add_argument(
        "--decode-length",
        type=int,
        default=None,
        help="cache depth to decode from; defaults to the full sweep",
    )
    group.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="HuggingFace model id to benchmark",
    )
    group.add_argument(
        "--llm-batch-size",
        type=int,
        default=1,
        help="batch size for the decomposed LLM benchmark",
    )
    group.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=32,
        help="tokens to generate in e2e mode",
    )
    return group


def config_from_args(args: Any) -> LLMBenchmarkConfig:
    """Build a config from parsed dynamo benchmark args."""
    prompt_length = getattr(args, "prompt_length", None)
    decode_length = getattr(args, "decode_length", None)
    batch_size = getattr(args, "llm_batch_size", None) or 1
    max_new_tokens = getattr(args, "llm_max_new_tokens", None) or 32

    return LLMBenchmarkConfig(
        model_name=getattr(args, "llm_model", None) or DEFAULT_LLM_MODEL,
        prompt_lengths=(
            [int(prompt_length)] if prompt_length else list(DEFAULT_PROMPT_LENGTHS)
        ),
        decode_lengths=(
            [int(decode_length)] if decode_length else list(DEFAULT_DECODE_LENGTHS)
        ),
        batch_sizes=[int(batch_size)],
        dtype=torch.float16 if getattr(args, "float16", False) else torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens=int(max_new_tokens),
        compile=bool(getattr(args, "inductor", False)),
    )


def run_llm_benchmark(args: Any) -> list[dict[str, Any]]:
    """Entry point: build the config, sweep the requested phase, print JSON."""
    config = config_from_args(args)
    benchmark = LLMBenchmark(config)
    benchmark.setup_model(config)
    results = benchmark.run_sweep(getattr(args, "llm_mode", None))
    for row in results:
        print(json.dumps(row))
    return results


def main(args: list[str] | None = None) -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float16", action="store_true", help="cast model to fp16")
    add_llm_args(parser)
    return run_llm_benchmark(parser.parse_args(args))


if __name__ == "__main__":
    main()
