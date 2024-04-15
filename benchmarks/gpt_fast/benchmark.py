import argparse
import csv
import dataclasses
import itertools
import os
import time
from typing import Optional, Tuple

import torch
import torch._inductor.config
from mixtral_moe_model import Transformer as MixtralMoE
from mixtral_moe_quantize import (
    WeightOnlyInt8QuantHandler as MixtralMoEWeightOnlyInt8QuantHandler,
)
from model import Transformer as LLaMA
from quantize import WeightOnlyInt8QuantHandler as LLaMAWeightOnlyInt8QuantHandler

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
torch._inductor.config.assert_indirect_indexing = False


@dataclasses.dataclass
class Experiment:
    name: str
    module: type
    mode: Optional[str]
    quantizer: type
    target: float


all_experiments = {
    "llama-7b-fp16": Experiment(
        "Llama-2-7b-chat-hf", LLaMA, "bfloat16", LLaMAWeightOnlyInt8QuantHandler, 104
    ),
    "llama-7b-int8": Experiment(
        "Llama-2-7b-chat-hf", LLaMA, "int8", LLaMAWeightOnlyInt8QuantHandler, 155
    ),
    "mixtral-int8": Experiment(
        "Mixtral-8x7B-v0.1",
        MixtralMoE,
        "int8",
        MixtralMoEWeightOnlyInt8QuantHandler,
        97,
    ),
}

output_filename = "gpt_fast_benchmark.csv"


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@torch.compile(fullgraph=True)
def prefill(
    model: torch.nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


@torch.compile(fullgraph=True, mode="reduce-overhead")
def decode_one_token(
    model: torch.nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: torch.nn.Module,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


@torch.no_grad()
def generate(
    model: torch.nn.Module, prompt: torch.Tensor, max_new_tokens: int, **sampling_kwargs
) -> torch.Tensor:
    device, dtype = prompt.device, prompt.dtype
    T = prompt.size(0)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, _ = decode_n_tokens(
        model, next_token.view(1, -1), input_pos, max_new_tokens - 1, **sampling_kwargs
    )
    seq[T + 1 :] = torch.cat(generated_tokens)
    return seq


def _load_model(x: Experiment, device="cuda", precision=torch.bfloat16):
    with torch.device("meta"):
        model = x.module.from_name(x.name)
    model = model.to(dtype=precision)

    if x.mode == "int8":
        print("Using int8 weight-only quantization!")
        model = x.quantizer(model).convert_for_runtime()

    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = torch.nn.Parameter(
            torch.randn(v.shape, device=device).to(dtype=v.dtype),
            requires_grad=v.requires_grad,
        )
    model.load_state_dict(state_dict, assign=True)
    return model.eval()


def run_experiment(
    x: Experiment,
    num_samples: int = 5,
    max_new_tokens: int = 200,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
    device = "cuda"
    print("Loading model ...")
    t0 = time.time()
    model = _load_model(x)
    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    prompt = torch.tensor(
        [1, 15043, 29892, 590, 1024, 338], device=device, dtype=torch.int32
    )
    prompt_length = prompt.size(0)

    torch.manual_seed(1234)
    model_size = sum(
        p.numel() * p.dtype.itemsize
        for p in itertools.chain(model.parameters(), model.buffers())
    )

    aggregate_metrics = {"tokens_per_sec": []}
    start = -1

    for i in range(start, num_samples):
        device_sync(device=device)  # MKG

        t0 = time.perf_counter()
        y = generate(
            model, prompt, max_new_tokens, temperature=temperature, top_k=top_k
        )

        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue

        device_sync(device=device)  # MKG
        t = time.perf_counter() - t0
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)

    token_per_sec = torch.mean(torch.tensor(aggregate_metrics["tokens_per_sec"])).item()
    print(f"Average tokens/sec: {token_per_sec:.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    return token_per_sec


def output_csv(filename, headers, row):
    if os.path.exists(filename):
        with open(filename) as fd:
            lines = list(csv.reader(fd)) or [[]]
            if headers and len(headers) > len(lines[0]):
                # if prior results failed the header might not be filled in yet
                lines[0] = headers
            else:
                headers = lines[0]
    else:
        lines = [headers]
    lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
    with open(filename, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for line in lines:
            writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))


def main(experiments=None):
    results = []

    if experiments is None:
        experiments = all_experiments
    else:
        experiments = {k: v for k, v in all_experiments.items() if k in experiments}

    for x in experiments.values():
        actual = run_experiment(x)
        percentage = f"{actual / x.target * 100:.2f}%"
        results.append((x, actual, percentage))

    headers = ["name", "mode", "target", "actual", "percentage"]
    rows = [[x[0].name, x[0].mode, x[0].target, x[1], x[2]] for x in results]

    for row in rows:
        output_csv(output_filename, headers, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiment names to run (default: all)",
    )
    args = parser.parse_args()

    main(experiments=args.experiments)
