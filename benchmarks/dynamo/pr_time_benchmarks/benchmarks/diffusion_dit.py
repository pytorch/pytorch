import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.utils import fresh_cache


DIM = 64
NUM_HEADS = 4
MLP_RATIO = 4
NUM_BLOCKS = 4
IMAGE_TOKENS = 16
TEXT_TOKENS = 8

DYNAMIC_IMAGE_TOKENS = 24


def _heads(x):
    return x.unflatten(-1, (NUM_HEADS, -1)).transpose(1, 2)


def _merge_heads(x):
    return x.transpose(1, 2).flatten(-2)


class DiTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(DIM, 3 * DIM)
        self.self_out = nn.Linear(DIM, DIM)
        self.cross_q = nn.Linear(DIM, DIM)
        self.cross_kv = nn.Linear(DIM, 2 * DIM)
        self.cross_out = nn.Linear(DIM, DIM)
        self.mlp = nn.Sequential(
            nn.Linear(DIM, MLP_RATIO * DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(MLP_RATIO * DIM, DIM),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(DIM, 6 * DIM))

    # Functional layer_norm rather than nn.LayerNorm: under dynamic=True Dynamo
    # lifts a module's eps float into a symfloat graph input, which the nested
    # region reuse matcher cannot bind across block instances. A literal eps
    # specializes instead, so the hierarchical rows keep reusing one region.
    def forward(self, image, text, condition):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).unsqueeze(1).chunk(6, dim=-1)
        )

        hidden = F.layer_norm(image, (DIM,)) * (1 + scale_attn) + shift_attn
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            _heads(query), _heads(key), _heads(value)
        )
        image = image + gate_attn * self.self_out(_merge_heads(attended))

        query = self.cross_q(F.layer_norm(image, (DIM,)))
        key, value = self.cross_kv(text).chunk(2, dim=-1)
        attended = F.scaled_dot_product_attention(
            _heads(query), _heads(key), _heads(value)
        )
        image = image + self.cross_out(_merge_heads(attended))

        hidden = F.layer_norm(image, (DIM,)) * (1 + scale_mlp) + shift_mlp
        return image + gate_mlp * self.mlp(hidden)


class HierarchicalDiTBlock(DiTBlock):
    # Marked once on the class, so all instances share a single compile region
    # that Dynamo can stamp out per block instead of retracing each one.
    forward = torch.compiler.nested_compile_region(DiTBlock.forward)


class DiT(nn.Module):
    def __init__(self, block_class, num_blocks):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(DIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.blocks = nn.ModuleList(block_class() for _ in range(num_blocks))
        self.norm_out = nn.LayerNorm(DIM)
        self.proj_out = nn.Linear(DIM, DIM)

    # Takes the blocks as an argument so the regional row can run
    # independently compiled blocks without mutating the model.
    def forward_blocks(self, blocks, image, text, timestep):
        condition = self.t_mlp(timestep)
        for block in blocks:
            image = block(image, text, condition)
        return self.proj_out(self.norm_out(image))

    def forward(self, image, text, timestep):
        return self.forward_blocks(self.blocks, image, text, timestep)


class Benchmark(BenchmarkBase):
    def __init__(self, strategy, *, dynamic=False, num_blocks=NUM_BLOCKS):
        self._strategy = strategy
        self._num_blocks = num_blocks
        super().__init__(
            category="diffusion_dit",
            backend="inductor",
            device="cpu",
            mode="inference",
            dynamic=dynamic,
        )

    def name(self):
        suffix = "dynamic" if self.is_dynamic() else self.backend()
        return f"{self.category()}_{self._strategy}_{suffix}"

    def description(self):
        return f"synthetic DiT, {self._num_blocks} repeated blocks, {self._strategy} compile"

    def _prepare_once(self):
        torch.manual_seed(0)
        torch.set_float32_matmul_precision("high")
        block_class = (
            HierarchicalDiTBlock if self._strategy == "hierarchical" else DiTBlock
        )
        self.model = DiT(block_class, self._num_blocks).eval()
        image_tokens = [IMAGE_TOKENS]
        if self.is_dynamic():
            image_tokens.append(DYNAMIC_IMAGE_TOKENS)
        self.inputs = [
            (
                torch.randn(1, tokens, DIM),
                torch.randn(1, TEXT_TOKENS, DIM),
                torch.randn(1, DIM),
            )
            for tokens in image_tokens
        ]

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        with fresh_cache(), torch.no_grad():
            if self._strategy == "regional":
                blocks = [
                    torch.compile(
                        block,
                        backend=self.backend(),
                        dynamic=self.is_dynamic(),
                        fullgraph=True,
                    )
                    for block in self.model.blocks
                ]
                for inputs in self.inputs:
                    self.model.forward_blocks(blocks, *inputs)
            else:
                compiled = torch.compile(
                    self.model,
                    backend=self.backend(),
                    dynamic=self.is_dynamic(),
                    fullgraph=True,
                )
                for inputs in self.inputs:
                    compiled(*inputs)


def main():
    result_path = sys.argv[1]
    benchmarks = [
        Benchmark("full"),
        Benchmark("regional"),
        Benchmark("hierarchical"),
        Benchmark("hierarchical", dynamic=True),
    ]
    for benchmark in benchmarks:
        benchmark.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
