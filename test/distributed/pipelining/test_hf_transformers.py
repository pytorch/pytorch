# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
End-to-end pipeline-parallelism tests using HuggingFace Transformer models.

These tests verify that torch.distributed.pipelining can correctly:
  - Build a pipeline IR (Pipe object) from real GPT-2 and BERT submodules
  - Split the model at specified transformer-layer boundaries
  - Produce output equivalent to the original (non-pipelined) model

Each wrapper below accepts a plain float tensor (hidden states) rather than
token IDs, which keeps the forward pass free of embedding-lookup branches and
makes it straightforwardly traceable by torch.fx.

See also: test_transformer.py for the analogous handwritten-model tests.
"""

import unittest

import torch
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.testing._internal.common_utils import run_tests, TestCase

try:
    from transformers import BertConfig, BertModel, GPT2Config, GPT2Model

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

skipIfNoTransformers = unittest.skipUnless(
    HAS_TRANSFORMERS,
    "requires the 'transformers' package",
)

# ---------------------------------------------------------------------------
# Tiny model configs – keep tests fast on CPU
# ---------------------------------------------------------------------------

GPT2_CFG = GPT2Config(
    vocab_size=100,
    n_positions=16,
    n_embd=64,
    n_head=2,
    n_layer=4,
    bos_token_id=0,
    eos_token_id=1,
)

BERT_CFG = BertConfig(
    vocab_size=100,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=2,
    intermediate_size=128,
    max_position_embeddings=16,
)

BATCH_SIZE = 2
SEQ_LEN = 8

# ---------------------------------------------------------------------------
# Pipeline-friendly wrappers
# ---------------------------------------------------------------------------


class GPT2StageModel(torch.nn.Module):
    """GPT-2 transformer blocks + final LayerNorm, split-ready.

    Accepts pre-computed hidden states (float tensor) instead of token IDs so
    that torch.fx can trace the forward pass without hitting embedding branches.
    The 'blocks' ModuleList mirrors the naming convention used in split_spec.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        base = GPT2Model(config)
        # Rename h → blocks so split keys are self-explanatory in tests.
        self.blocks = base.h
        self.ln_f = base.ln_f
        n = len(self.blocks)
        self.split_spec = {f"blocks.{n // 2}": SplitPoint.BEGINNING}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            # use_cache=False keeps the output as a plain tensor (no KV-cache
            # tuple), which is required for pipeline tracing.
            x = block(x, use_cache=False)
        return self.ln_f(x)


class BERTStageModel(torch.nn.Module):
    """BERT encoder layers, split-ready.

    Accepts pre-computed hidden states so the forward is fully traceable.
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        base = BertModel(config, add_pooling_layer=False)
        self.layers = base.encoder.layer
        n = len(self.layers)
        self.split_spec = {f"layers.{n // 2}": SplitPoint.BEGINNING}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skipIfNoTransformers
class HFGpt2PipeliningTests(TestCase):
    """Pipeline-parallelism tests using GPT-2 transformer blocks."""

    def _make_model_and_input(self):
        torch.manual_seed(0)
        model = GPT2StageModel(GPT2_CFG)
        model.eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, GPT2_CFG.n_embd)
        return model, x

    def test_pipeline_ir_num_stages(self):
        """pipeline() creates a Pipe with the expected number of stages."""
        model, x = self._make_model_and_input()
        num_stages = 2
        pipe = pipeline(model, (x,), split_spec=model.split_spec)
        self.assertEqual(
            pipe.num_stages,
            num_stages,
            f"Expected {num_stages} stages, got {pipe.num_stages}",
        )

    def test_pipeline_layer_completeness(self):
        """Every GPT-2 block appears in exactly one pipeline stage."""
        model, x = self._make_model_and_input()
        pipe = pipeline(model, (x,), split_spec=model.split_spec)

        original_blocks = {name for name, _ in model.blocks.named_children()}
        pipe_blocks: set = set()
        for stage_idx in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(stage_idx)
            # Collect any 'blocks' submodule children from this stage
            for name, mod in stage_mod.named_modules():
                # Individual GPT-2 blocks are named '0', '1', … inside 'blocks'
                parts = name.split(".")
                if len(parts) >= 2 and parts[-2] == "blocks" and parts[-1].isdigit():
                    pipe_blocks.add(parts[-1])
                elif len(parts) == 1 and parts[0].isdigit():
                    pipe_blocks.add(parts[0])

        self.assertEqual(
            pipe_blocks,
            original_blocks,
            f"Block mismatch: pipe={pipe_blocks} vs original={original_blocks}",
        )

    def test_pipeline_output_equivalence(self):
        """Pipelined GPT-2 produces the same output as the original model."""
        model, x = self._make_model_and_input()
        pipe = pipeline(model, (x,), split_spec=model.split_spec)

        with torch.no_grad():
            ref = model(x)
            out = pipe(x)[0]

        torch.testing.assert_close(
            out,
            ref,
            msg="GPT-2 pipeline output does not match non-pipelined reference",
        )

    def test_pipeline_four_stages(self):
        """GPT-2 can be split into four pipeline stages (one block each)."""
        torch.manual_seed(0)
        model = GPT2StageModel(GPT2_CFG)
        model.eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, GPT2_CFG.n_embd)

        # One split point per block boundary → 4 stages for 4-block model
        n = len(model.blocks)
        split_spec = {
            f"blocks.{i}": SplitPoint.BEGINNING for i in range(1, n)
        }
        pipe = pipeline(model, (x,), split_spec=split_spec)

        self.assertEqual(pipe.num_stages, n)

        with torch.no_grad():
            ref = model(x)
            out = pipe(x)[0]

        torch.testing.assert_close(out, ref)


@skipIfNoTransformers
class HFBertPipeliningTests(TestCase):
    """Pipeline-parallelism tests using BERT encoder layers."""

    def _make_model_and_input(self):
        torch.manual_seed(0)
        model = BERTStageModel(BERT_CFG)
        model.eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, BERT_CFG.hidden_size)
        return model, x

    def test_pipeline_ir_num_stages(self):
        """pipeline() creates a Pipe with the expected number of stages."""
        model, x = self._make_model_and_input()
        num_stages = 2
        pipe = pipeline(model, (x,), split_spec=model.split_spec)
        self.assertEqual(
            pipe.num_stages,
            num_stages,
            f"Expected {num_stages} stages, got {pipe.num_stages}",
        )

    def test_pipeline_layer_completeness(self):
        """Every BERT encoder layer appears in exactly one pipeline stage."""
        model, x = self._make_model_and_input()
        pipe = pipeline(model, (x,), split_spec=model.split_spec)

        original_layers = {name for name, _ in model.layers.named_children()}
        pipe_layers: set = set()
        for stage_idx in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(stage_idx)
            for name, _ in stage_mod.named_modules():
                parts = name.split(".")
                if len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():
                    pipe_layers.add(parts[-1])
                elif len(parts) == 1 and parts[0].isdigit():
                    pipe_layers.add(parts[0])

        self.assertEqual(
            pipe_layers,
            original_layers,
            f"Layer mismatch: pipe={pipe_layers} vs original={original_layers}",
        )

    def test_pipeline_output_equivalence(self):
        """Pipelined BERT produces the same output as the original model."""
        model, x = self._make_model_and_input()
        pipe = pipeline(model, (x,), split_spec=model.split_spec)

        with torch.no_grad():
            ref = model(x)
            out = pipe(x)[0]

        torch.testing.assert_close(
            out,
            ref,
            msg="BERT pipeline output does not match non-pipelined reference",
        )

    def test_pipeline_four_stages(self):
        """BERT encoder can be split into four pipeline stages (one layer each)."""
        torch.manual_seed(0)
        model = BERTStageModel(BERT_CFG)
        model.eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, BERT_CFG.hidden_size)

        n = len(model.layers)
        split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n)
        }
        pipe = pipeline(model, (x,), split_spec=split_spec)

        self.assertEqual(pipe.num_stages, n)

        with torch.no_grad():
            ref = model(x)
            out = pipe(x)[0]

        torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    run_tests()
