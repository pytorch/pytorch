import itertools
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, List, Tuple

from tabulate import tabulate
from tqdm import tqdm

import torch
from torch._inductor.utils import do_bench_using_profiling
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""

    def no_args() -> None:
        func(*args, **kwargs)

    time = do_bench_using_profiling(no_args)
    return time * 1e3


def get_tflops(flop_count: int, time_in_micro: float) -> float:
    time_in_seconds = time_in_micro / 1e6
    flops = flop_count / time_in_seconds
    return flops / 1e12


def flops(batch, headdim, seqlen_q, seq_len_kv, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen_q * seq_len_kv * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int
    embed_dim: int
    is_causal: bool
    dtype: torch.dtype
    backend: SDPBackend
    device: torch.device = torch.device("cuda")
    packed_layout: bool = False

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def asdict(self):
        dict_obj = asdict(self)
        dict_obj["head_dim"] = self.head_dim
        return dict_obj


@dataclass(frozen=True)
class ExperimentResults:
    config: ExperimentConfig
    forward_time: float
    backward_time: float

    @property
    def tflops_forward(self):
        config = self.config
        total_flops = flops(
            config.batch_size,
            config.num_heads,
            config.q_seq_len,
            config.kv_seq_len,
            config.head_dim,
            config.is_causal,
            mode="fwd",
        )
        tflops = get_tflops(flop_count=total_flops, time_in_micro=self.forward_time)
        return tflops if self.config.is_causal is False else tflops / 2

    @property
    def tflops_backward(self):
        config = self.config
        total_flops = flops(
            config.batch_size,
            config.num_heads,
            config.q_seq_len,
            config.kv_seq_len,
            config.head_dim,
            config.is_causal,
            mode="bwd",
        )
        return get_tflops(flop_count=total_flops, time_in_micro=self.backward_time)

    def asdict(self):
        return {
            "forward_time": self.forward_time,
            "backward_time": self.backward_time,
            "tflops_forward": self.tflops_forward,
            "tflops_backward": self.tflops_backward,
        }


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def asdict(self):
        dict1 = asdict(self.config)
        dict2 = self.results.asdict()
        return {**dict1, **dict2}


def get_input(
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.packed_layout:
        assert (
            config.q_seq_len == config.kv_seq_len
        ), "Q and KV seq len must be equal for the packed layout"
        q, k, v = torch.randn(
            config.batch_size,
            config.q_seq_len,
            config.num_heads * config.head_dim * 3,
            dtype=config.dtype,
            device=config.device,
            requires_grad=True,
        ).chunk(3, dim=-1)

        q = q.view(config.batch_size, -1, config.num_heads, config.head_dim).transpose(
            1, 2
        )
        k = k.view(config.batch_size, -1, config.num_heads, config.head_dim).transpose(
            1, 2
        )
        v = v.view(config.batch_size, -1, config.num_heads, config.head_dim).transpose(
            1, 2
        )
        return q, k, v

    q = torch.randn(
        (config.batch_size, config.num_heads, config.q_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    k = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    v = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    return q, k, v


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    q, k, v = get_input(config)
    is_causal = config.is_causal
    context = (
        sdpa_kernel(config.backend) if config.backend is not None else nullcontext()
    )
    with context:
        forward_time = benchmark_cuda_function_in_microseconds(
            scaled_dot_product_attention,
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=None,
        )
        out_torch = scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, attn_mask=None
        )
        dOut = torch.randn_like(out_torch)
        backward_time = benchmark_cuda_function_in_microseconds(
            out_torch.backward, dOut, retain_graph=True
        )

    return ExperimentResults(
        config=config,
        forward_time=forward_time,
        backward_time=backward_time,
    )


def generate_experiment_configs() -> List[ExperimentConfig]:
    batch_sizes = [
        1,
    ]
    num_heads = [16]
    q_kv_seq_lens = [
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (32768, 32768),
        (65536, 65536),
    ]
    embed_dims = [2048]
    backends = [
        None,
    ]  # If set to None, all backends are enabled
    dtypes = [
        torch.bfloat16,
    ]
    is_causal = [True, False]
    packed_layout = [False]
    all_configs = []
    for (
        bsz,
        heads,
        (q_seq_len, kv_seq_len),
        embed_dim,
        causal,
        dtype,
        backend,
        packed,
    ) in itertools.product(
        batch_sizes,
        num_heads,
        q_kv_seq_lens,
        embed_dims,
        is_causal,
        dtypes,
        backends,
        packed_layout,
    ):
        all_configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=heads,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
                embed_dim=embed_dim,
                is_causal=causal,
                dtype=dtype,
                backend=backend,
                packed_layout=packed,
            )
        )

    return all_configs


def print_results(experiments: List[Experiment]):
    table_data = defaultdict(list)
    for experiment in experiments:
        for key, value in experiment.asdict().items():
            table_data[key].append(value)
    del table_data["device"]
    if table_data["backend"][0] is None:
        del table_data["backend"]
    print(tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".3f"))


def main():
    seed = 123
    torch.manual_seed(seed)
    results = []
    for config in tqdm(generate_experiment_configs()):
        results.append(Experiment(config, run_single_experiment(config)))

    print_results(results)


if __name__ == "__main__":
    main()
