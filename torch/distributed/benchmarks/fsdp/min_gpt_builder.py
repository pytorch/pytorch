"""
This file was borrowed from https://github.com/karpathy/minGPT with modifications.

The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import functools
import logging
import math
import os

import torch
import torch.nn as nn

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import functional as F


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPTSmallConfig(GPTConfig):
    """GPT3-small like network roughly 125M params"""

    n_layer = 12
    n_head = 12
    n_embd = 768


class GPTMediumConfig(GPTConfig):
    """GPT3-large like network roughly 350M params"""

    n_layer = 24
    n_head = 16
    n_embd = 1024


class GPTLargeConfig(GPTConfig):
    """GPT3-large like network roughly 760M params"""

    n_layer = 24
    n_head = 16
    n_embd = 1536


class GPTXLConfig(GPTConfig):
    """GPT3-XL like network roughly 1.3B params"""

    n_layer = 24
    n_head = 24
    n_embd = 2064


class GPTXXLConfig(GPTConfig):
    """GPT3-XL like network roughly 2.7B params"""

    n_layer = 32
    n_head = 32
    n_embd = 2560


class GPTXXXLConfig(GPTConfig):
    """GPT3-XL like network roughly 6.7B params"""

    n_layer = 32
    n_head = 32
    n_embd = 4096


class GPT13BConfig(GPTConfig):
    """GPT3-XL like network roughly 13B params"""

    n_layer = 48
    n_head = 48
    n_embd = 5184


class GPT175BConfig(GPTConfig):
    """GPT3-XL like network roughly 175B params"""

    n_layer = 96
    n_head = 96
    n_embd = 12288


class GPT1TConfig(GPTConfig):
    """GPT3-XL like network roughly 1T params"""

    n_layer = 128
    n_head = 128
    n_embd = 25600


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config, device="cpu", dtype=torch.float32):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), f"n_embd={config.n_embd}, n_head={config.n_head}"
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.query = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.value = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # TODO: leave buffer on CPU for now, until we can do meta_tensor.to_empty()
        d = device if torch.device(device).type == "cuda" else "cpu"
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size, device=d, dtype=dtype)
            ).view(1, 1, config.block_size, config.block_size),
        )
        self.n_head = config.n_head

    def reset_parameters(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class EmbeddingStem(nn.Module):
    def __init__(self, config, device="cpu", dtype=torch.float32):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(
            config.vocab_size, config.n_embd, device=device, dtype=dtype
        )
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd, device=device, dtype=dtype)
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.block_size = config.block_size

    def reset_parameters(self):
        self.tok_emb.reset_parameters()

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        return self.drop(token_embeddings + position_embeddings)


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        config,
        device=None,
        dtype=torch.float32,
        wrapper=lambda m: m,
        version="pytorch",
        cpu_offload=False,
    ):
        super().__init__()
        if version == "pytorch" or not cpu_offload:
            self.ln1 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
            self.ln2 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
            self.attn = wrapper(CausalSelfAttention(config, device=device, dtype=dtype))
            self.mlp = nn.Sequential(
                wrapper(
                    nn.Linear(
                        config.n_embd, 4 * config.n_embd, device=device, dtype=dtype
                    )
                ),
                nn.GELU(),
                wrapper(
                    nn.Linear(
                        4 * config.n_embd, config.n_embd, device=device, dtype=dtype
                    )
                ),
                nn.Dropout(config.resid_pdrop),
            )
        else:
            print("fairscale fsdp for block")
            self.ln1 = wrapper(
                nn.LayerNorm(config.n_embd, device=device, dtype=dtype).cpu()
            )
            self.ln2 = wrapper(
                nn.LayerNorm(config.n_embd, device=device, dtype=dtype).cpu()
            )
            self.attn = wrapper(
                CausalSelfAttention(config, device=device, dtype=dtype).cpu()
            )
            self.mlp = nn.Sequential(
                wrapper(
                    nn.Linear(
                        config.n_embd, 4 * config.n_embd, device=device, dtype=dtype
                    ).cpu()
                ),
                nn.GELU(),
                wrapper(
                    nn.Linear(
                        4 * config.n_embd, config.n_embd, device=device, dtype=dtype
                    ).cpu()
                ),
                nn.Dropout(config.resid_pdrop),
            )

    def reset_parameters(self):
        self.attn.reset_parameters()
        for _, m in self.named_modules():
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config, device="cpu", dtype=torch.float32):
        super().__init__()

        # input embedding stem
        self.emb_stem = EmbeddingStem(config, device=device, dtype=dtype)
        # transformer
        self.blocks = nn.Sequential(
            *[Block(config, device=device, dtype=dtype) for _ in range(config.n_layer)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd, device=device, dtype=dtype)
        self.head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False, device=device, dtype=dtype
        )

    def reset_parameters(self):
        for name, m in self.named_children():
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.Linear) or isinstance(m, EmbeddingStem):
                m.reset_parameters()

    def forward(self, idx):
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


def get_gpt_config(model_name):
    assert model_name.startswith("GPT")
    config_class_name = model_name + "Config"
    assert config_class_name in globals()
    return globals()[config_class_name](50000, 2048)


def build(model_name: str, device="cpu"):
    return GPT(get_gpt_config(model_name), device=device)


def get_inputs(batch_size, device):
    return torch.randint(0, 50000, (batch_size, 2048), device=device)


def get_loss(dist_model, inputs):
    out = dist_model(inputs)
    return out.sum() if isinstance(out, torch.Tensor) else out.local_value().sum()


def get_fsdp_wrapping_policy():
    return ModuleWrapPolicy({Block})


def apply_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, Block)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

def get_flops(model_name, batch_size):
    B = batch_size
    s = 2048 # block_size
    conf = get_gpt_config(model_name)
    l = conf.n_layer
    h = conf.n_embd
    V = conf.vocab_size
    return 96 * B * s * l * h * h * (1 + s/6/h + V/16/l/h)