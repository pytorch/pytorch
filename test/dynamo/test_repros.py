"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_rewrite_assert_with_msg and test_rewrite_assert_without_msg)
"""
# Owner(s): ["module: dynamo"]
import collections
import contextlib
import copy
import functools
import inspect
import itertools
import random
import unittest
import weakref
from abc import ABC
from collections import namedtuple
from copy import deepcopy
from functools import wraps
from typing import List

import numpy as np
import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils

import torch._functorch.config
import torch.library

from torch import nn
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.testing import expectedFailureDynamic, rand_strided, same
from torch.nn import functional as F
from torch.testing._internal.common_utils import (
    disable_translation_validation_if_dynamic_shapes,
)


_orig_module_call = torch.nn.Module.__call__

# Custom operator that only supports CPU and Meta
lib = torch.library.Library("test_sample", "DEF")
lib.define("foo(Tensor self) -> Tensor")
lib.impl("foo", torch.sin, "CPU")


requires_cuda = functools.partial(
    unittest.skipIf, not torch.cuda.is_available(), "requires cuda"
)


_GLOBAL_CPU_TENSOR = torch.randn(3)


def exists(val):
    return val is not None


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def is_fx_tracing_test() -> bool:
    """
    Copied from the hpc trainer codebase
    """
    return torch.nn.Module.__call__ is not _orig_module_call


def has_detectron2():
    try:
        from detectron2.layers.mask_ops import _paste_masks_tensor_shape

        return _paste_masks_tensor_shape is not None
    except ImportError:
        return False


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    # from detectron2 mask_ops.py

    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(
            dtype=torch.int32
        )
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(
            dtype=torch.int32
        )
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def cat(tensors, dim=0):
    # from detectron2 wrappers.py
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def shapes_to_tensor(x, device=None):
    # from detectron2 wrappers.py
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all(
            isinstance(t, torch.Tensor) for t in x
        ), "Shape should be tensor during tracing!"
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)


class Boxes:
    # from detectron2 poolers.py
    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = (
            tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        )
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()
        self.tensor = tensor

    def __len__(self) -> int:
        return self.tensor.shape[0]

    @property
    def device(self):
        return self.tensor.device


def convert_boxes_to_pooler_format(box_lists):
    # from detectron2 structures.py
    boxes = torch.cat([x.tensor for x in box_lists], dim=0)
    # __len__ returns Tensor in tracing.
    sizes = shapes_to_tensor([x.__len__() for x in box_lists], device=boxes.device)
    indices = torch.repeat_interleave(
        torch.arange(len(box_lists), dtype=boxes.dtype, device=boxes.device), sizes
    )
    return cat([indices[:, None], boxes], dim=1)


ReformerBackwardOutput = namedtuple(
    "ReformerBackwardOutput",
    ["attn_output", "hidden_states", "grad_attn_output", "grad_hidden_states"],
)
ReformerEncoderOutput = namedtuple(
    "ReformerEncoderOutput",
    ["hidden_states", "all_hidden_states", "all_attentions", "past_buckets_states"],
)


class _ReversibleFunction(torch.autograd.Function):
    # taken from modeling_reformer.py in huggingface
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        layers,
        attention_mask,
        head_mask,
        num_hashes,
        all_hidden_states,
        all_attentions,
        past_buckets_states,
        use_cache,
        orig_sequence_length,
        output_hidden_states,
        output_attentions,
    ):
        all_buckets = ()

        # split duplicated tensor
        hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)

        for layer_id, (layer, layer_head_mask) in enumerate(zip(layers, head_mask)):
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)

            attn_output = layer(attn_output)

        # Add last layer
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)

        # attach params to ctx for backward
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.head_mask = head_mask
        ctx.attention_mask = attention_mask

        # Concatenate 2 RevNet outputs
        return torch.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        grad_attn_output, grad_hidden_states = torch.chunk(
            grad_hidden_states, 2, dim=-1
        )

        # retrieve params from ctx for backward
        attn_output, hidden_states = ctx.saved_tensors

        # create tuple
        output = ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )

        # free memory
        del grad_attn_output, grad_hidden_states, attn_output, hidden_states

        layers = ctx.layers
        all_buckets = ctx.all_buckets
        head_mask = ctx.head_mask
        attention_mask = ctx.attention_mask

        for idx, layer in enumerate(layers[::-1]):
            # pop last buckets from stack
            buckets = all_buckets[-1]
            all_buckets = all_buckets[:-1]

            # backprop
            output = layer.backward_pass(
                next_attn_output=output.attn_output,
                hidden_states=output.hidden_states,
                grad_attn_output=output.grad_attn_output,
                grad_hidden_states=output.grad_hidden_states,
                head_mask=head_mask[len(layers) - idx - 1],
                attention_mask=attention_mask,
                buckets=buckets,
            )

        assert all_buckets == (), "buckets have to be empty after backpropagation"
        grad_hidden_states = torch.cat(
            [output.grad_attn_output, output.grad_hidden_states], dim=-1
        )

        # num of return vars has to match num of forward() args
        # return gradient for hidden_states arg and None for other args
        return (
            grad_hidden_states,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class ReformerEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.5
        self.layer_norm = torch.nn.LayerNorm(512, eps=1.0e-12)
        self.layers = [torch.nn.Linear(256, 256)]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=[None] * 6,
        num_hashes=None,
        use_cache=False,
        orig_sequence_length=64,
        output_hidden_states=False,
        output_attentions=False,
    ):
        # hidden_states and attention lists to be filled if wished
        all_hidden_states = []
        all_attentions = []
        past_buckets_states = [((None), (None)) for i in range(len(self.layers))]

        # concat same tensor for reversible ResNet
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            head_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            past_buckets_states,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions,
        )

        # Apply layer norm to concatenated hidden states
        hidden_states = self.layer_norm(hidden_states)

        # Apply dropout
        hidden_states = torch.nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        return ReformerEncoderOutput(
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
            past_buckets_states=past_buckets_states,
        )


def longformer_chunk(hidden_states, window_overlap=256):
    """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""

    # non-overlapping chunks of size = 2w
    hidden_states = hidden_states.view(
        hidden_states.size(0),
        hidden_states.size(1) // (window_overlap * 2),
        window_overlap * 2,
        hidden_states.size(2),
    )

    # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
    chunk_size = list(hidden_states.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(hidden_states.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)


class PartialT5(torch.nn.Module):
    # Highly simplified T5Attention prefix
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        query_length=None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, 8, 64).transpose(1, 2)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        # (truncated here )
        return scores, value_states


class ChunkReformerFeedForward(torch.nn.Module):
    # simplified from HF modeling_reformer.py
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(256, eps=1e-12)
        self.dense = torch.nn.Linear(256, 256)
        self.output = torch.nn.Linear(256, 256)

    def forward(self, attention_output):
        return apply_chunking_to_forward(
            self.forward_chunk,
            attention_output + 1,
        )

    def forward_chunk(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        return self.output(hidden_states)


def apply_chunking_to_forward(forward_fn, *input_tensors):
    # simplified from HF model_utils.py
    assert len(input_tensors) > 0
    tensor_shape = input_tensors[0].shape[1]
    assert all(input_tensor.shape[1] == tensor_shape for input_tensor in input_tensors)
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError()

    return forward_fn(*input_tensors)


class FakeMamlInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(784, 5)

    def forward(self, x, ignored=None, bn_training=False):
        return self.linear(x.view(x.shape[0], -1))


class PartialMaml(torch.nn.Module):
    # Highly simplified version of maml.meta.Meta.finetuning
    def __init__(self):
        super().__init__()
        self.net = FakeMamlInner()
        self.update_step_test = 10
        self.update_lr = 0.4

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = [
            p[1] - self.update_lr * p[0] for p in zip(grad, net.parameters())
        ]

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        del net

        accs = torch.tensor(corrects) / querysz

        return accs


def softmax_backward_data(parent, grad_output, output, dim, self):
    from torch import _softmax_backward_data

    return _softmax_backward_data(grad_output, output, parent.dim, self.dtype)


class XSoftmax(torch.autograd.Function):
    # transformers.models.deberta.modeling_deberta.XSoftmax
    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        rmask = ~(mask.to(torch.bool))
        output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output, rmask)
        return output

    @staticmethod
    def backward(self, grad_output):
        (output, rmask) = self.saved_tensors
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return inputGrad, None, None


class ModelOutput(collections.OrderedDict):
    """based on file_utils.py in HuggingFace"""

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self):
        return tuple(self[k] for k in self.keys())


def create_rand_mask_from_inputs(
    from_blocked_mask,
    to_blocked_mask,
    rand_attn,
    num_attention_heads,
    num_rand_blocks,
    batch_size,
    from_seq_length,
    from_block_size,
):
    """taken from HF modeling_big_bird.py"""
    num_windows = from_seq_length // from_block_size - 2
    rand_mask = torch.stack(
        [p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)]
    )
    rand_mask = rand_mask.view(
        batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size
    )
    rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
    return rand_mask


class SequentialAppendList(torch.nn.Sequential):
    """from timm/models/vovnet.py"""

    def forward(self, x: torch.Tensor, concat_list: List[torch.Tensor]) -> torch.Tensor:
        for i, module in enumerate(self):
            if i == 0:
                concat_list.append(module(x))
            else:
                concat_list.append(module(concat_list[-1]))
        x = torch.cat(concat_list, dim=1)
        return x, concat_list


class BatchNormAct2d(torch.nn.BatchNorm2d):
    """Taken from timm"""

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        act_layer=torch.nn.ReLU,
        inplace=True,
    ):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.act = act_layer(inplace=inplace)

    @torch.jit.ignore
    def _forward_python(self, x):
        return super().forward(x)

    def forward(self, x):
        if torch.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            x = self._forward_python(x)
        x = self.act(x)
        return x


def get_parameter_dtype(parameter):
    """from huggingface model_utils.py"""
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


class DummyConfig:
    attn_layers = ["local", "lsh", "local", "lsh", "local", "lsh"]
    lsh_attn_chunk_length = 64
    local_attn_chunk_length = 64


def _get_min_chunk_len(config):
    """from hf_Reformer"""
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(  # noqa: C405
        ["lsh", "local"]
    ):
        return min(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )


def _stable_argsort(vector, dim):
    """from hf_Reformer"""
    # this function scales the vector so that torch.argsort is stable.
    # torch.argsort is not stable on its own
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    return torch.argsort(scaled_vector, dim=dim)


def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(buckets):
    """from hf_Reformer"""
    # no gradients are needed
    with torch.no_grad():
        # hash-based sort
        sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

        # create simple indices to scatter to, to have undo sort
        indices = (
            torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
            .view(1, 1, -1)
            .expand(sorted_bucket_idx.shape)
        )

        # get undo sort
        undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

    return sorted_bucket_idx, undo_sorted_bucket_idx


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation, dropout) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(
            self.linear2(self.dropout1(self.activation(self.linear1(x))))
        )


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.ff_block = FeedForwardLayer(d_model, dim_feedforward, activation, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout(x)

    # feed forward block
    def _ff_block(self, x):
        return self.ff_block(x)


class MockModule(torch.nn.Module):
    def inner_fn(self, left, right):
        return tuple(left) == tuple(right)

    def fn(self, tensor):
        if type(tensor) is int:
            return False

        torch.add(tensor, tensor)
        return self.inner_fn(tensor.shape, (1, 2, 3))


class ReproTests(torch._dynamo.test_case.TestCase):
    def test_do_paste_mask(self):
        torch._dynamo.utils.counters.clear()
        opt__do_paste_mask = torch._dynamo.optimize(
            torch._dynamo.testing.CompileCounter()
        )(_do_paste_mask)
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 1,
            427,
            640,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 2,
            427,
            640,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 3,
            612,
            612,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 4,
            612,
            612,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 2,
            427,
            640,
            False,
        )

        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["ok"], 3)
        self.assertEqual(
            torch._dynamo.utils.counters["frames"]["total"],
            torch._dynamo.utils.counters["frames"]["ok"] + 1,
        )

    def test_convert_boxes_to_pooler_format(self):
        boxes1 = [
            Boxes(torch.arange(0, 8).reshape((2, 4))),
            Boxes(torch.arange(8, 16).reshape((2, 4))),
        ]
        boxes2 = [
            Boxes(torch.arange(16, 20).reshape((1, 4))),
            Boxes(torch.arange(20, 24).reshape((1, 4))),
        ]
        correct1 = convert_boxes_to_pooler_format(boxes1)
        correct2 = convert_boxes_to_pooler_format(boxes2)
        fn = convert_boxes_to_pooler_format
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        self.assertTrue(same(opt_fn(boxes1), correct1))
        self.assertTrue(same(opt_fn(boxes2), correct2))

        # repeat_interleave is a dynamic shape operator we do not execute/
        # In the future, we could reduce the frame_count down to 1
        # by guarding on the exact values of `Tensor repeats` arg
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """4""")
            self.assertExpectedInline(cnt.op_count, """10""")
        else:
            self.assertExpectedInline(cnt.frame_count, """4""")
            self.assertExpectedInline(cnt.op_count, """16""")

    def test_boxes_len(self):
        def fn(boxes):
            return len(boxes) + boxes.__len__() + boxes.tensor

        boxes1 = Boxes(torch.arange(0, 8).reshape((2, 4)))
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(boxes1), boxes1.tensor + 4.0))

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """1""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """6""")

    def _reformer(self, nopython):
        input = torch.randn([1, 64, 256])
        model = ReformerEncoder()
        torch.manual_seed(1337)
        correct = copy.deepcopy(model)(input)
        cnt = torch._dynamo.testing.CompileCounter()
        torch.manual_seed(1337)
        opt_model = torch._dynamo.optimize(cnt, nopython=nopython)(model)
        self.assertTrue(same(opt_model(input), correct))
        return cnt

    @requires_cuda()
    def test_sub_alpha_scalar_repro(self):
        @torch.compile(backend="aot_eager")
        def f(x):
            return x.sub(1, alpha=2)

        f(torch.ones(2, device="cuda", dtype=torch.float64))

    # See https://github.com/pytorch/pytorch/issues/97745
    def test_gan_repro_trying_to_backward_through_the_graph_a_second_time(self):
        def f(a, b):
            c = torch.ones(2, 2)
            d = torch.ones(2, 2)
            e = torch.matmul(a, c)
            g_loss = torch.abs(e - d).mean()
            g_loss.backward()
            fake_d_pred = torch.matmul(b, e.detach())
            d_loss = fake_d_pred.mean()
            d_loss.backward()

        a_ref = torch.randn(2, 2, requires_grad=True)
        b_ref = torch.randn(2, 2, requires_grad=True)
        out_ref = f(a_ref, b_ref)

        a_test = a_ref.clone().detach().requires_grad_(True)
        b_test = b_ref.clone().detach().requires_grad_(True)
        out_test = torch.compile(f, backend="aot_eager")(a_test, b_test)

        self.assertEqual(out_ref, out_test)
        self.assertEqual(a_ref.grad, a_test.grad)
        self.assertEqual(b_ref.grad, b_test.grad)

    def test_embedding_backward_broadcasting_decomp(self):
        def f(grad_output, indices):
            num_weights = 10
            padding_idx = 1
            scale_grad_by_freq = True
            return torch.ops.aten.embedding_dense_backward(
                grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
            )

        f_compiled = torch.compile(f, backend="aot_eager")

        grad_output = torch.ones(2, 4, 3, dtype=torch.float16)
        indices = torch.ones(2, 4, dtype=torch.int64)

        out_ref = f(grad_output, indices)
        out_test = f_compiled(grad_output, indices)

        self.assertEqual(out_ref, out_test)

    def test_reformer_eval(self):
        with torch.no_grad():
            cnt = self._reformer(nopython=True)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 11)

    def test_reformer_train(self):
        with torch.enable_grad():
            cnt = self._reformer(nopython=False)
        # cant inline torch.autograd.Function means graph break
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """3""")
            self.assertExpectedInline(cnt.op_count, """10""")
        else:
            self.assertExpectedInline(cnt.frame_count, """3""")
            self.assertExpectedInline(cnt.op_count, """10""")

    @disable_translation_validation_if_dynamic_shapes
    def test_longformer_chunk(self):
        input1 = torch.randn([1, 4096, 1])
        input2 = torch.randn([12, 4096, 64])
        correct1 = longformer_chunk(input1)
        correct2 = longformer_chunk(input2)
        fn = longformer_chunk
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(input1), correct1))
        self.assertTrue(same(opt_fn(input2), correct2))
        self.assertTrue(same(opt_fn(input1), correct1))
        self.assertTrue(same(opt_fn(input2), correct2))

        if torch._dynamo.config.assume_static_by_default:
            if torch._dynamo.config.automatic_dynamic_shapes:
                self.assertExpectedInline(cnt.frame_count, """2""")
                self.assertExpectedInline(cnt.op_count, """14""")
            else:
                self.assertExpectedInline(cnt.frame_count, """2""")
                self.assertExpectedInline(cnt.op_count, """4""")
        else:
            self.assertExpectedInline(cnt.frame_count, """2""")
            self.assertExpectedInline(cnt.op_count, """35""")

    def test_hf_t5_forward(self):
        input = torch.randn([1, 2048, 512])
        model = PartialT5()
        correct = model(input)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(input), correct))

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """12""")

    def test_module_in_skipfiles(self):
        model = nn.Linear(10, 10)
        cnt = torch._dynamo.testing.CompileCounter()
        torch.compile(model, backend=cnt, fullgraph=True)(torch.randn([5, 10]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_function_in_skipfiles(self):
        cnt = torch._dynamo.testing.CompileCounter()
        torch.compile(torch.sin, backend=cnt, fullgraph=True)(torch.randn([5, 10]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_slicing_dynamic_shape(self):
        def fn(y):
            x = torch.ones(8)
            idx = y[0]
            out = x[idx:]
            return (out + 3) * 5

        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(counter)(fn)
        out = opt_fn(torch.ones(10, dtype=torch.long))
        # idx should be 1 -> slicing off [1:] of 8 elem tensor
        self.assertEqual(list(out.shape), [7])

        self.assertEqual(counter.op_count, 2)
        self.assertEqual(counter.frame_count, 1)

        self.assertEqual(list(opt_fn(torch.tensor([4])).shape), [4])

    def test_slicing_dynamic_shape_setitem(self):
        def fn(input_lengths: torch.Tensor, new_ones_1):
            getitem_13 = input_lengths[3]
            new_ones_1[(3, slice(getitem_13, None, None))] = 0
            setitem_13 = new_ones_1
            return (setitem_13,)

        x = torch.randn(10).to(dtype=torch.int64)
        y = torch.randn(10, 204)
        ref = fn(x, y)
        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    # https://github.com/pytorch/pytorch/issues/103620
    @expectedFailureDynamic
    def test_chunk_reformer_ff(self):
        input = torch.randn([1, 4096, 256])
        model = ChunkReformerFeedForward()
        correct = model(input)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(input), correct))

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 4)

    # see: https://github.com/pytorch/pytorch/issues/80067
    # NB: When you remove the expectedFailure, don't forget to
    # uncomment/adjust the assertEqual below
    @unittest.expectedFailure
    @torch._dynamo.config.patch(
        fake_tensor_propagation=True, capture_scalar_outputs=True
    )
    def test_maml_item_capture(self):
        a = torch.randn(5, 1, 28, 28)
        b = torch.zeros(5, dtype=torch.int64)
        c = torch.randn(75, 1, 28, 28)
        d = torch.zeros(75, dtype=torch.int64)
        model = PartialMaml()
        correct = model(a, b, c, d)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize(cnt)(model)
        for _ in range(10):
            self.assertTrue(same(opt_model(a, b, c, d), correct))

        # if torch._dynamo.config.assume_static_by_default:
        #     self.assertExpectedInline(cnt.frame_count, """2""")
        # else:
        #     self.assertExpectedInline(cnt.frame_count, """3""")
        # TODO(jansel): figure out why op count depends on imports
        self.assertIn(cnt.op_count, (36, 35, 34, 29, 28, 27))

    # see: https://github.com/pytorch/pytorch/issues/80067
    @torch._dynamo.config.patch(capture_scalar_outputs=False)
    def test_maml_no_item_capture(self):
        a = torch.randn(5, 1, 28, 28)
        b = torch.zeros(5, dtype=torch.int64)
        c = torch.randn(75, 1, 28, 28)
        d = torch.zeros(75, dtype=torch.int64)
        model = PartialMaml()
        correct = model(a, b, c, d)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize(cnt)(model)
        for _ in range(10):
            self.assertTrue(same(opt_model(a, b, c, d), correct))

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """3""")

    def test_hf_model_output(self):
        ex = ModelOutput(a=torch.randn(10), b=torch.randn(10), c=torch.randn(10))

        def fn1(x):
            return x["a"] + 1

        def fn2(x):
            return x.a + 1

        def fn3(x):
            return x.to_tuple()[0] + 1

        def fn4(x):
            return x[0] + 1

        cnt = torch._dynamo.testing.CompileCounter()
        for fn in (fn1, fn2, fn3, fn4):
            cnt.clear()
            opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
            self.assertTrue(same(opt_fn(ex), ex.a + 1))
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(cnt.op_count, 1)

    @disable_translation_validation_if_dynamic_shapes
    def test_create_rand_mask_from_inputs(self):
        args = [
            torch.randn([1, 64, 64]),
            torch.randn([1, 64, 64]),
            torch.zeros([1, 12, 62, 3], dtype=torch.int64),
            12,
            3,
            1,
            4096,
            64,
        ]
        correct = create_rand_mask_from_inputs(*args)
        fn = create_rand_mask_from_inputs

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(*args), correct))
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """8""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")

    def test_rng_state(self):
        def fn():
            state = torch.get_rng_state()
            before = torch.rand(1000)
            torch.set_rng_state(state)
            after = torch.rand(1000)
            return before, after

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)

        before, after = opt_fn()
        self.assertTrue(same(before, after))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 3)  # rand, rand
        try:
            graph, _ = torch._dynamo.export(fn)
            # See https://github.com/pytorch/pytorch/pull/87490
            self.fail("unexpected export success")
        except torch._dynamo.exc.Unsupported:
            pass

    def test_threading_local(self):
        import threading

        foo = threading.local()
        foo.x = torch.rand(1)

        def f(x):
            return torch.cat([x, foo.x])

        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch._dynamo.optimize(cnt, nopython=True)(f)

        inp = torch.ones(1)
        out = f(inp)
        opt_out = opt_f(inp)
        self.assertEqual(opt_out, out)
        self.assertEqual(cnt.frame_count, 1)

    def test_seq_append_list(self):
        x = torch.randn(4, 10)
        model = SequentialAppendList(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )
        # this one is tricky because it mutates the list provided as an input
        l1 = [x]
        l2 = [x]
        correct, _ = model(x, l1)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        result, l3 = opt_model(x, l2)
        self.assertTrue(same(result, correct))
        self.assertTrue(same(l1, l2))
        self.assertIs(l2, l3)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 5)

    def test_batch_norm_act(self):
        a = torch.randn(5, 1, 28, 28)
        model = BatchNormAct2d(1).eval()
        correct = model(a)
        cnt = torch._dynamo.testing.CompileCounter()
        if not torch._dynamo.config.specialize_int:
            # _local_scalar_dense causes graph break w 0-dim tensor
            opt_model = torch._dynamo.optimize(cnt)(model)
            self.assertTrue(same(opt_model(a), correct))
            return

        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(a), correct))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_get_parameter_dtype(self):
        model = SequentialAppendList(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

        def fn(model, x):
            return x + torch.randn(10, dtype=get_parameter_dtype(model))

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertEqual(opt_fn(model, torch.randn(10)).dtype, torch.float32)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_nn_parameter(self):
        def test_fn():
            a = torch.nn.Parameter(torch.randn(5, 5))
            # Checks that TensorVariable stores the type information correctly
            self.assertTrue(isinstance(a, torch.nn.Parameter))
            return a

        cnt = torch._dynamo.testing.CompileCounter()
        opt_test_fn = torch._dynamo.optimize(cnt)(test_fn)
        out = opt_test_fn()
        self.assertTrue(isinstance(out, torch.nn.Parameter))

    def test_Size(self):
        def test_fn():
            a = torch.randn(4)
            x = torch.Size([1, 2, 3])
            # Checks that SizeVariable return torch.Size object
            assert isinstance(x, torch.Size)
            # Causes graph breaks and checks reconstruction of SizeVariable
            # object
            self.assertIsInstance(x, torch.Size)
            return a

        cnt = torch._dynamo.testing.CompileCounter()
        opt_test_fn = torch._dynamo.optimize(cnt)(test_fn)
        opt_test_fn()

    # See https://github.com/pytorch/pytorch/issues/100067
    def test_copy_weird_strides(self):
        # This test requires inductor's copy() decomp to preserve strides properly.
        def test_fn(a):
            b = torch.zeros(48, 4, 256, 513)
            b[:, 0, 1:256, 1:256] = a
            c = b.view(4, 12, 1024, 513)
            d = c.transpose(2, 1)
            d.add_(1)
            return d

        sh, st, dt, dev, rg = (
            (48, 255, 255),
            (787968, 513, 1),
            torch.float16,
            "cpu",
            True,
        )
        a = rand_strided(sh, st, dt, dev).requires_grad_(rg)
        compiled_f = torch.compile(test_fn, backend="aot_eager_decomp_partition")
        out1 = test_fn(a)
        out2 = compiled_f(a)
        self.assertEqual(out1, out2)

    def test_indexing_with_list(self):
        def test_fn():
            def run_test(tensor, *idx):
                npt = tensor.numpy()
                assert npt[idx].shape == tensor[idx].shape

            x = torch.arange(0, 10)
            cases = [
                [None, None],
                [1, None],
            ]

            for case in cases:
                run_test(x, *case)

            return torch.randn(4)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_test_fn = torch._dynamo.optimize(cnt)(test_fn)
        opt_test_fn()

    def test_reformer_min_chunk_len(self):
        def fn(cfg):
            t = torch.empty(10)
            t.fill_(_get_min_chunk_len(cfg))
            return t[0]

        cfg = DummyConfig()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertEqual(opt_fn(cfg), 64)
        # With unspec int, maximum computation is preserved
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """3""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """4""")

    def test_reformer_sorting(self):
        x = torch.zeros([1, 12, 4096], dtype=torch.int64)
        correct = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(x)
        fn = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(x), correct))
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """14""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """27""")

    def test_recursive_map(self):
        # https://github.com/pytorch/torchdynamo/issues/132
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = v

        def toy_example(a, b, v):
            x = a / (torch.abs(a) + 1)
            if v is not None:
                _recursive_map(v)
            return x * b

        cnt = torch._dynamo.testing.CompileCounter()
        opt_toy_example = torch._dynamo.optimize(cnt)(toy_example)
        opt_toy_example(
            torch.randn(10),
            torch.randn(10),
            {"layer0": {"memory_keys": torch.randn(10)}},
        )
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 4)

    def test_issue175(self):
        n_heads = 2
        d_model = 64
        model = TransformerEncoderLayer(d_model, n_heads)
        inp = torch.randn(1, d_model)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize(cnt, nopython=True)(model)
        opt_model(inp)
        opt_model(inp)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 12)

    def test_exec_import(self):
        def fn1():
            exec("import math")

        def fn2():
            try:
                math.sqrt(4)
                return False
            except NameError:
                return True

        def fn3():
            fn1()
            return fn2()

        self.assertTrue(fn3())
        opt_fn3 = torch._dynamo.optimize("eager")(fn3)
        self.assertTrue(opt_fn3())

    def test_exec_wildcard_import(self):
        # Test that globals are not carried over from frame to frame
        def fn1():
            exec("from torch import *")

        def fn2():
            x = torch.zeros(4)
            for i in range(5):
                x = x + i
            return x

        def fn3():
            fn1()
            return fn2()

        ref = fn3()
        opt_fn3 = torch._dynamo.optimize("eager")(fn3)
        res = opt_fn3()
        self.assertTrue(same(ref, res))

    def test_with_on_graph_break_inst(self):
        def reversible(x):
            print("Hello world")  # Cause graph break so inline fails
            return torch.sin(torch.cos(x))

        def fn(x):
            with torch.enable_grad():
                a = torch.sin(x)
                b = reversible(a)
                c = torch.sigmoid(b)
                c.sum().backward()
                return x.grad

        x = torch.randn(3, requires_grad=True)
        x.grad = None
        with torch.no_grad():
            ref = fn(x)

        x.grad = None
        opt_fn = torch._dynamo.optimize("eager")(fn)
        with torch.no_grad():
            res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_with_on_graph_break_nested(self):
        def reversible(x):
            torch._dynamo.graph_break()  # Cause graph break so inline fails
            return torch.sin(torch.cos(x))

        def fn(x):
            # nested context manager failed previously
            with torch.no_grad():
                with torch.enable_grad():
                    a = torch.sin(x)
                    b = reversible(a)
                    c = torch.sigmoid(b)
                    c.sum().backward()
                    return x.grad

        x = torch.randn(3, requires_grad=True)
        x.grad = None
        with torch.no_grad():
            ref = fn(x)

        x.grad = None
        opt_fn = torch._dynamo.optimize("eager")(fn)
        with torch.no_grad():
            res = opt_fn(x)
        self.assertTrue(same(ref, res))

    # https://github.com/pytorch/torchdynamo/issues/1446
    def test_grad_mode_carrying_correct_state_after_graph_break(self):
        def fn(x):
            with torch.no_grad():
                y = x * 3
                print("Break")
                z = x + 2
            return y, z

        x = torch.randn(3, requires_grad=True)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        y, z = opt_fn(x)
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

    def test_abc_setattr(self):
        # tests that we correctly bail out of __setattr__ calls

        # TODO: does not ensure ABC classes are correctly inferred as ClassVariables
        # (doesn't test the fix for 'super()')

        class BaseModule(torch.nn.Module, ABC):
            def blah(self, x):
                return x + 1

        class Derived(BaseModule):
            def __setattr__(self, name, value) -> None:
                super().__setattr__(name, value)

            def forward(self, x):
                # expect a graph break on __setattr__
                self.foo = 0
                return self.blah(x)

            def blah(self, x):
                return super().blah(x)

        x = torch.randn(3, requires_grad=True)
        mod = Derived()
        opt_mod = torch._dynamo.optimize("eager")(mod)
        opt_mod(x)

        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["ok"], 2)
        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["total"], 2)

    @torch._dynamo.config.patch("suppress_errors", True)
    def test_guard_fail_tensor_bool(self):
        @torch._dynamo.disable(recursive=False)
        def fn():
            condition_shape = (5, 5)
            dtypes = (torch.bool,)
            shapes = (
                (),
                (5,),
                (1, 5),
            )

            tensors = [
                torch.empty(shape, dtype=dtype).fill_(17)
                for shape, dtype in itertools.product(shapes, dtypes)
            ]

            x_vals = (5.0, *tensors)
            y_vals = (6.0, *tensors)

            @torch._dynamo.disable
            def get_expected(condition, x, y):
                x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
                return torch.from_numpy(
                    np.where(condition.cpu().numpy(), x_np, y_np)
                ).to(common_dtype)

            for x, y in zip(x_vals, y_vals):
                condition = torch.empty(*condition_shape, dtype=torch.bool).bernoulli_()
                common_dtype = torch.result_type(x, y)

                def check_equal(condition, x, y):
                    # NumPy aggressively promotes to double, hence cast to output to correct dtype
                    expected = get_expected(condition, x, y)
                    result = torch.where(condition, x, y)
                    assert torch.allclose(expected, result)

                check_equal(condition, x, y)
                check_equal(condition, y, x)

        fn()
        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_fn()

    def test_guard_fail_nested_tuple(self):
        def fn(args):
            return torch.ones(()), args[0] * 2

        # This adds a tensor check on args[1][0] and args[1][1]
        args1 = (torch.ones(1), (torch.ones(1), torch.ones(1)))
        args2 = (torch.ones(1), torch.ones(1))
        opt_fn = torch._dynamo.optimize("eager")(fn)
        ref = opt_fn(args1)
        res = opt_fn(args2)

        self.assertTrue(same(ref, res))

    def test_nullcontext1(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, ctx):
            x = x.sin()
            with ctx:
                x = x.cos()
            x = x.sin()
            return x

        y = torch.randn(10)
        self.assertTrue(same(fn(y, contextlib.nullcontext()), y.sin().cos().sin()))

    def test_nullcontext2(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, ctx):
            x = x.sin()
            with ctx():
                x = x.cos()
            x = x.sin()
            return x

        y = torch.randn(10)
        self.assertTrue(same(fn(y, contextlib.nullcontext), y.sin().cos().sin()))

    def test_no_grad_inline(self):
        @torch.no_grad()
        def a(x):
            return x.sin()

        @torch.compile(backend="eager", fullgraph=True)
        def b(x):
            return a(x).cos()

        y = torch.randn(10)
        self.assertTrue(same(b(y), y.sin().cos()))

    # AssertionError: ABCMeta
    @unittest.expectedFailure
    def test_numpy_list(self):
        @torch._dynamo.disable
        def rand_gen():
            return list(np.array([random.randint(5, 10) for _ in range(10)]))

        def fn(x):
            random_list = rand_gen()
            z = torch.LongTensor(random_list)
            return x * z

        x = torch.ones(10) * 2

        random.seed(0)
        ref0 = fn(x)
        ref1 = fn(x)

        random.seed(0)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res0 = opt_fn(x)
        res1 = opt_fn(x)

        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    def test_primtorch(self):
        @torch._dynamo.optimize("eager")
        def fn(x):
            torch._refs.abs(x)

        fn(torch.randn(3))

    @unittest.expectedFailure
    # inline_call [('inline in skipfiles: bind ...python3.10/inspect.py', 1)]
    def test_primtorch_no_graph_break(self):
        @torch._dynamo.optimize("eager", nopython=True)
        def fn(x):
            torch._refs.abs(x)

        fn(torch.randn(3))

    def test_torch_tensor_ops_no_graph_break(self):
        @torch._dynamo.optimize("eager", nopython=True)
        def fn(x):
            torch.Tensor.abs_(x)

        fn(torch.randn(3))

    @unittest.skipIf(
        not isinstance(torch.ops.aten.abs, torch._ops.OpOverloadPacket),
        "old pt doesn't work",
    )
    def test_torch_ops_aten(self):
        # Picked an op that doesn't show up in the default list
        @torch._dynamo.optimize("eager", nopython=True)
        def fn(x):
            return torch.ops.aten.absolute(x)

        fn(torch.randn(3))

    def test_hf_gelu_inline(self):
        class GELUActivation(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.functional.gelu

            def forward(self, input):
                return self.act(input)

        @torch._dynamo.optimize("eager", nopython=True)
        def fn(x):
            return GELUActivation()(x)

        y = torch.randn(10)
        self.assertTrue(same(fn(y), nn.functional.gelu(y)))

        @torch._dynamo.optimize("eager", nopython=True)
        def fn_returns(x):
            return GELUActivation(), x + 1

        act, _ = fn_returns(y)
        self.assertIsInstance(act, GELUActivation)
        self.assertIs(act.act, nn.functional.gelu)
        self.assertTrue(hasattr(act, "_buffers"))  # check that __init__ got called

    def test_dropout_inline(self):
        @torch._dynamo.optimize("eager")
        def fn(x):
            return torch.nn.Dropout(0.1)(x)

        y = torch.randn(10)
        torch.manual_seed(1337)
        ref = nn.functional.dropout(y, 0.1)
        torch.manual_seed(1337)
        res = fn(y)
        self.assertTrue(same(ref, res))

    def test_setitem_boolean_mask_diff(self):
        def fn(x, b, y):
            x = x.clone()
            x[b] = y
            return x

        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        x = torch.randn(4, requires_grad=True)
        b = torch.tensor([True, False, True, False])
        y = torch.randn(2, requires_grad=True)
        opt_fn(x, b, y)

    def test_setitem_tuple_boolean_mask_diff(self):
        def fn(x, b, y):
            x = x.clone()
            x[:, b] = y
            return x

        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        x = torch.randn(8, 4, requires_grad=True)
        b = torch.tensor([True, False, True, False])
        y = torch.randn(2, requires_grad=True)
        opt_fn(x, b, y)

    def test_torch_tensor_ops(self):
        def fn(x):
            return torch.Tensor.abs_(x)

        x = torch.randn(3)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        y = fn(x)
        y_ = opt_fn(x)
        self.assertTrue(same(y, y_))

    def test_guard_ordering_shape_fail(self):
        # If a function which takes a tensor has an inner function which
        # is compiled and generates a guard on its shape,
        # they are evaluated in the wrong order. So if on a subsequent call
        # an int is passed instead of a tensor, guard evaluation will crash
        # with a "no attribute: shape" error
        m = MockModule()
        opt_m = torch._dynamo.optimize("eager")(m)
        opt_m.fn(torch.ones((5, 5)))
        opt_m.fn(-3)

    def test_tensor_isinstance_tuple(self):
        @torch._dynamo.optimize("eager")
        def fn():
            t = torch.ones(5, 5)
            if not isinstance(t, (int, torch.Tensor)):
                msg = str.format(
                    "{0} is not an instance of {1}",
                    type(t),
                    (int, torch.Tensor),
                )
                raise ValueError(msg)
            return True

        fn()

    def test_isinstance_dtype(self):
        @torch._dynamo.optimize("eager", nopython=True)
        def fn(x):
            isinstance(torch.bfloat16, torch.dtype)
            return x

        fn(torch.randn(3))

    def test_isinstance_storage(self):
        @torch._dynamo.optimize("eager")
        def fn(x):
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            bools = torch.BoolStorage.from_buffer(f, "big")
            assert isinstance(bools, torch.BoolStorage)
            return x

        fn(torch.randn(3))

    def test_dict_list_values(self):
        def inner_fn(args):
            return [x[1].shape for x in args]

        @torch._dynamo.optimize("eager")
        def fn(tensors):
            return inner_fn(zip(itertools.count(), tensors["args"]))

        fn({"args": [torch.ones(5, 5), torch.ones(5, 6), torch.ones(5, 7)]})
        fn({"args": [torch.ones(5, 5)]})

    def test_dict_iter(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = {"my": 1, "const": 2, "dict": 3, "variable": 4}
                tot = 0
                for key in z:
                    tot += z[key]

                return tot

        x = torch.tensor([0])
        model = MyMod()
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
        y = opt_model(x)

        self.assertEqual(y, 10)

    def test_sort_out(self):
        dtype = torch.float32
        device = "cpu"

        def fn():
            tensor = torch.randn((3, 5), dtype=dtype, device=device)[:, 0]
            values1 = torch.tensor(0, dtype=dtype, device=device)
            indices1 = torch.tensor(0, dtype=torch.long, device=device)
            torch.sort(tensor, out=(values1, indices1))
            self.assertEqual(values1.stride(), (1,))
            self.assertEqual(indices1.stride(), (1,))

        fn()
        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_fn()

    def test_sort_out2(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sorted = torch.nn.Buffer(torch.ones(4, 4))
                self.indices = torch.nn.Buffer(torch.ones(4, 4, dtype=torch.long))

            def forward(self, x):
                torch.sort(x, out=(self.sorted, self.indices))
                return (x + 1, self.sorted, self.indices)

        x = torch.randn(4, 4)
        m = MyModule()
        ref = m(x)
        opt_m = torch._dynamo.optimize("eager")(m)
        res = opt_m(x)
        self.assertTrue(same(ref, res))

    def test_sigmoid_out(self):
        dtype = torch.float32
        device = "cpu"

        def fn():
            inp = torch.randn((3, 5), dtype=dtype, device=device)
            out1 = torch.tensor(0, dtype=dtype, device=device)
            torch.sigmoid(inp, out=out1)
            self.assertEqual(out1.numel(), 15)

        fn()
        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_fn()

    def test_sigmoid_out2(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Buffer(torch.ones(4, 4))

            def forward(self, x):
                torch.sigmoid(x, out=self.base)
                return x + self.base

        x = torch.randn(4, 4)
        m = MyModule()
        ref = m(x)
        opt_m = torch._dynamo.optimize("eager")(m)
        res = opt_m(x)
        self.assertTrue(same(ref, res))

    def test_slice_into_list_mutable(self):
        class Mod(torch.nn.Module):
            def forward(self, listy):
                x = listy[3:5]
                for i in range(10):
                    z = torch.abs(torch.randn(10)) + 1
                    x[0] = z
                return x

        m = Mod()
        listy = [torch.randn(10)] * 10

        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        opt_m.forward(listy)

        self.assertEqual(cnt.frame_count, 1)

    def test_vdd_duplicate_error(self):
        def fn(a, dt):
            keys = list(dt._jt_dict.keys())
            p = torch.cos(dt._jt_dict[keys[0]]._value)
            q = torch.sin(a)
            r = torch.sigmoid(dt._jt_dict[keys[0]]._value)
            return p + q + r

        class Value:
            def __init__(self):
                self._value = torch.randn(4)

        class Sample:
            def __init__(self):
                self._jt_dict = {}
                self._jt_dict["POSITION_ID"] = Value()

        a = torch.randn(4)
        sample = Sample()

        ref = fn(a, sample)

        optimized_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = optimized_fn(a, sample)

        self.assertTrue(same(ref, res))

    def test_specialized_stride(self):
        def f():
            e = torch.empty(4)
            x = e[::2]
            return x.stride()

        self.assertEqual(f(), torch._dynamo.optimize("eager")(f)())

    def test_out_none(self):
        # https://github.com/pytorch/pytorch/issues/92814
        def fn(input):
            return torch.nn.functional.normalize(input, dim=0, out=None)

        x = torch.rand([1])
        self.assertEqual(fn(x), torch._dynamo.optimize("eager")(fn)(x))

    @unittest.skipIf(not has_detectron2(), "requires detectron2")
    def test_multi_import(self):
        @torch._dynamo.optimize("eager", nopython=True)
        def to_bitmasks(boxes):
            from detectron2.layers.mask_ops import (
                _paste_masks_tensor_shape,
                paste_masks_in_image,
            )

            if (
                paste_masks_in_image is not None
                and _paste_masks_tensor_shape is not None
            ):
                return boxes + 1

        self.assertTrue((to_bitmasks(torch.zeros(10)) == torch.ones(10)).all())

    def test_multi_dot_import(self):
        def fn1(x):
            return torch.sin(x)

        def fn(x):
            import torch.fx

            _ = torch.fx.symbolic_trace(fn1)
            return x * 2

        x = torch.randn(10)
        fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_relative_import(self):
        try:
            from . import utils as _  # noqa: F401

            def fn(x):
                from .utils import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        except ImportError:

            def fn(x):
                from utils import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        x = torch.randn(10)
        fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(fn)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_relative_import_no_modulename(self):
        try:
            from . import utils as _  # noqa: F401

            def fn(x):
                from . import utils

                return x * 2 * utils.tensor_for_import_testing

        except ImportError:

            def fn(x):
                import utils

                return x * 2 * utils.tensor_for_import_testing

        x = torch.randn(10)
        fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(fn)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_bigbird_unsqueeze_inplace(self):
        def fn(reshape_2):
            view_2 = reshape_2.clone()
            view_2.unsqueeze_(2)
            cat_11 = torch.cat([view_2], dim=2)
            view_13 = cat_11.view((2, 12, 64, -1))
            return (view_13,)

        x = torch.randn(2, 12, 64, 64, requires_grad=True)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_issue1466_size_aot_autograd(self):
        def fn(x):
            # do a tensor op and a size compute
            y = x * 2
            x_size = x.size()
            # trigger a graph break
            print("arf")
            # use the tensor op and size compute
            z = y.view(x_size) + 1
            return z

        x = torch.randn(2, 3, requires_grad=True)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_ellipsis(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lnorm = torch.nn.LayerNorm(
                    (256,), eps=1e-06, elementwise_affine=True
                )
                self.linear = torch.nn.Linear(
                    in_features=256, out_features=256, bias=True
                )

            def forward(self, cat_10):
                lnorm = self.lnorm(cat_10)
                getitem_64 = lnorm[
                    (slice(None, None, None), slice(0, 1, None), Ellipsis)
                ]
                linear = self.linear(getitem_64)
                return (linear,)

        args = [torch.randn(2, 197, 256)]

        mod = Repro()
        opt_mod = torch._dynamo.optimize("eager", nopython=True)(mod)

        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_reinplacing(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_layoutlm_embeddings_x_position_embeddings = (
                    torch.nn.Embedding(1024, 768)
                )
                self.self_layoutlm_embeddings_y_position_embeddings = (
                    torch.nn.Embedding(1024, 768)
                )

            def forward(self, getitem_1, getitem_2, add):
                self_layoutlm_embeddings_x_position_embeddings = (
                    self.self_layoutlm_embeddings_x_position_embeddings(getitem_1)
                )
                self_layoutlm_embeddings_y_position_embeddings = (
                    self.self_layoutlm_embeddings_y_position_embeddings(getitem_2)
                )
                add_1 = add + self_layoutlm_embeddings_x_position_embeddings
                add_2 = add_1 + self_layoutlm_embeddings_y_position_embeddings
                return (add_2,)

        mod = MockModule()
        opt_mod = torch._dynamo.optimize("aot_eager_decomp_partition")(mod)

        args = [
            ((2, 512), (2048, 4), torch.int64, "cpu", False),
            ((2, 512), (2048, 4), torch.int64, "cpu", False),
            ((2, 512, 768), (393216, 768, 1), torch.float32, "cpu", True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        self.assertTrue(same_two_models(mod, opt_mod, args))

    def test_optimized_deepcopy(self):
        # See https://github.com/pytorch/pytorch/pull/88629
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(in_features=2, out_features=3, bias=True)

            def forward(self, x):
                return self.fc(x)

        mod = Foo()
        opt_mod = torch._dynamo.optimize("eager")(mod)
        args = [torch.randn(1, 2)]
        self.assertTrue(same_two_models(mod, opt_mod, args))

    def test_class_member(self):
        class Foo(torch.nn.Module):
            a = 4
            b = torch.ones(3, 4)

            def __init__(self):
                super().__init__()
                self.c = 4

            def forward(self, x):
                return x.cos() + self.a + self.b + self.c

        mod = Foo()
        opt_mod = torch._dynamo.optimize("eager", nopython=True)(mod)
        args = (torch.randn(3, 4),)
        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_named_buffers(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Buffer(torch.ones(3))
                self.y = torch.nn.Buffer(torch.ones(3))

            def forward(self, inp):
                res = 0
                for name, buffer in self.named_buffers():
                    res += buffer.sum()

                return inp.cos() + res

        mod = Foo()
        opt_mod = torch._dynamo.optimize("eager", nopython=True)(mod)
        args = (torch.randn(3, 4),)
        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_requires_grad_guards_with_grad_mode1(self):
        def f(x):
            if x.requires_grad:
                return x + 1
            else:
                return x + 2

        x = torch.ones(2, requires_grad=True)

        f_compiled = torch.compile(f)
        with torch.no_grad():
            # compile an inference graph
            f_compiled(x)

        # Test: we should fail guards and recompile (even though it's still an inference graph)
        out_ref = f(x.detach())
        out = f_compiled(x.detach())

        self.assertEqual(out_ref, out)
        self.assertEqual(out_ref.requires_grad, out.requires_grad)

    def test_requires_grad_guards_with_grad_mode2(self):
        x = torch.ones(2, requires_grad=True)
        x_ref = x.clone().detach().requires_grad_(True)

        m = torch.nn.Linear(2, 2)
        m_compiled = torch.compile(m)

        with torch.no_grad():
            # compile an inference graph
            m_compiled(x)

        # Test: we should fail guards and recompile a training graph
        out_ref = m(x_ref)
        out = m_compiled(x)
        self.assertEqual(out_ref, out)
        self.assertEqual(out_ref.requires_grad, out.requires_grad)

    def test_is_symbolic_tracing(self):
        # Ensure no graph break here
        def fn(x):
            if is_fx_tracing_test():
                return x * 2
            return x * 4

        a = torch.randn(4)
        ref = fn(a)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(a)
        self.assertTrue(same(ref, res))

    def test_tokenization(self):
        from collections import UserDict

        class BatchEncoding(UserDict):
            """
            Copied from tokenization
            """

            def __init__(
                self,
                data,
            ):
                super().__init__(data)

            def __getattr__(self, item: str):
                try:
                    return self.data[item]
                except KeyError as e:
                    raise AttributeError from e

        def tokenization(x):
            encoding = BatchEncoding({"key": x})
            return encoding["key"]

        opt_fn = torch._dynamo.optimize("eager")(tokenization)
        x = torch.rand((1, 4))
        ref = tokenization(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_modules(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 3)

            def forward(self, inp):
                res = torch.zeros(3, 3)
                for mod in self.modules():
                    res += self.fc(inp)
                return res

        mod = Foo()
        args = (torch.ones(3, 4),)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch._dynamo.optimize(cnt, nopython=True)(mod)
        self.assertTrue(same(mod(*args), opt_mod(*args)))
        self.assertEqual(cnt.op_count, 5)
        self.assertEqual(cnt.frame_count, 1)

    def test_tensor_data_kwarg(self):
        # https://github.com/pytorch/pytorch/issues/96278
        def f():
            return torch.tensor(data=[[1.0, -1.0]])

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(f)
        self.assertTrue(same(f(), opt_fn()))
        self.assertEqual(cnt.frame_count, 1)

    @requires_cuda()
    def test_norm_dtype(self):
        def foo(_stack0):
            getitem = _stack0[(slice(None, None, None), -1)]
            _stack0 = None
            normalize = torch.nn.functional.normalize(getitem, p=2, dim=1)
            getitem = None
            return (normalize,)

        args = [((2, 50, 256), (1, 256, 1), torch.float16, "cuda", False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        opt_foo = torch._dynamo.optimize("aot_eager_decomp_partition")(foo)
        with torch.cuda.amp.autocast(enabled=True):
            ref = foo(*args)[0]
            res = foo(*args)[0]
            self.assertEqual(ref.dtype, res.dtype)

            self.assertTrue(same(res, ref))

    def test_for_loop_graph_break(self):
        def inner(x):
            return torch.sin(x)

        def fn(x):
            for _ in range(100):
                inner(x)
                torch._dynamo.graph_break()
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_for_loop_graph_break_before(self):
        # Checks that the backedge is calculated correctly
        def inner(x):
            return torch.sin(x)

        def fn(x):
            torch._dynamo.graph_break()
            for _ in range(100):
                inner(x)
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 100)

    def test_avoid_dupe_specialization(self):
        def f(x, y):
            return (x + y) * 1

        opt_f = torch._dynamo.optimize("aot_eager")(f)

        for b in [True, False]:
            x = torch.randn(4, requires_grad=b)
            y = torch.randn(4, requires_grad=b)
            self.assertEqual(f(x, x), opt_f(x, x))
            self.assertEqual(f(x, y), opt_f(x, y))

    def test_reformer_remove_unused_args(self):
        # This test case is very interesting.  First, let's describe
        # the bug this is testing for.  The bug we fixed is twofold:
        #
        # - We prune GraphArgs that aren't used in the output graph.
        #   However, sometimes it is possible for those GraphArgs to be
        #   utilized in shape guards (you could imagine this happening if
        #   dynamo poked some shape variables without recording them in the
        #   graph.)  If we prune those GraphArgs, we get a
        #   "s1 not in ..." error as we can no longer codegen the
        #   requested guards.
        #
        # - But in practice, Dynamo usually traces size accesses into the
        #   graph, preventing the GraphArg from getting pruned.  So how
        #   come we were running into this in practice with hf_Reformer?
        #   The answer is checkpointing!
        #
        # This brings us to the following test case.  Here's what it does:
        #
        # 1. It traces some operations, and then checkpoints before inlining
        #    the function call to g
        #
        # 2. g traces some more operations (triggering the shape guard
        #    to be created), but then it graph breaks
        #
        # 3. Because you can't graph break in an inlining function, we roll
        #    back to the outer checkpoint ("undoing" the operation that
        #    induced the shape guard) and then immediately generate a
        #    subgraph at that point.
        #
        # If we failed to checkpoint the ShapeEnv, it can still have guards
        # from the aborted speculation, which we will then still attempt to
        # codegen.
        #
        # There's an additional nuance: suppose x is used but y is not.
        # If you create a guard like y == x * 2, you will accidentally avoid
        # the "s1 not in ..." error, as y will get substituted with x * 2,
        # but x is still a GraphArg (it's used) and you don't end up with
        # the error.  This is why we must show y + y == x, not vice versa.
        # Similarly, it is also why we must not do a simple guard like x == y
        #
        # Can we actually demonstrate that checkpointing the ShapeEnv is
        # necessary?  It's not so easy to induce this case.  Dynamo is very
        # eager about adding locals to GraphArgs; any local that is in scope,
        # even if it isn't used, is added to GraphArgs (see also
        # https://github.com/pytorch/torchdynamo/issues/1925 ).  So long
        # as Dynamo eagerly guards in this way, we have an invariant that
        # all locals are guaranteed to show up in GraphArgs before the
        # inlining function call, in which case we will always have enough
        # information to codegen our guards so long as we don't prune the
        # unused GraphArgs away (and indeed, the direct fix for this bug
        # was to make sure we use original GraphArgs).  Non locals,
        # conversely, typically are static, and so won't have guards allocated
        # for them.  That being said, there may still be a way to trigger
        # this error.

        def g(x, y):
            r = torch.cat((y, y)) + x
            print("foo")
            return r

        def f(x, y):
            x = x * 3
            return g(x, y)

        opt_f = torch._dynamo.optimize("aot_eager")(f)

        x = torch.randn(4)
        y = torch.randn(2)
        self.assertEqual(f(x, y), opt_f(x, y))

    def test_swin_base_tensor_attr(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # NB: not a parameter or buffer
                self.t = torch.randn(3)

            def forward(self, x):
                return x + torch.cat((self.t, self.t))

        mod = Foo()
        opt_mod = torch._dynamo.optimize("eager")(mod)
        args = [torch.randn(6)]
        self.assertTrue(same_two_models(mod, opt_mod, args))
        opt_mod(*args)

    def test_pointless_graph_removal(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            with torch.no_grad():
                torch._dynamo.graph_break()
                return x + 1

        fn(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_output_aliases_intermediate(self):
        def f(x):
            intermediate = x.mul(2)
            return intermediate.view(-1), intermediate

        opt_f = torch._dynamo.optimize("aot_eager")(f)

        for b in [True, False]:
            x = torch.randn(4, requires_grad=b)
            out = f(x)
            out_test = opt_f(x)
            self.assertEqual(out[0], out_test[0])
            self.assertEqual(out[1], out_test[1])
            self.assertEqual(out[0].requires_grad, out_test[0].requires_grad)
            self.assertEqual(out[1].requires_grad, out_test[1].requires_grad)
            # test that the aliasing relationship of outputs is preserved
            out[0].mul_(2)
            out_test[0].mul_(2)
            self.assertEqual(out[0], out_test[0])
            self.assertEqual(out[1], out_test[1])

    def test_while_loop_graph_break(self):
        # Repro of tacotron2 cache_size_recompilation
        def inner(x):
            return torch.sin(x)

        def fn(x):
            i = 20
            while i > 10:
                x = inner(x)
                i -= 1
                torch._dynamo.graph_break()
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_nested_while_loop_graph_break(self):
        def inner_loop(x):
            i = 3
            while i > 0:
                i -= 1
                x += 1
                torch._dynamo.graph_break()
            return x

        def inner(x):
            inner_loop(x)
            return torch.sin(x)

        def fn(x):
            i = 20
            while i > 10:
                x = inner(x)
                i -= 1
                torch._dynamo.graph_break()
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_while_loop_graph_break_inside_call_function(self):
        # Repro of huggingface graph break inside loop in `get_parameter_dtype`.
        # Skip only the inner frame that has loop that contains graph break.
        def inner(x):
            for i in range(3):
                x += 1
                torch._dynamo.graph_break()
            return x

        def fn(x):
            x += 2
            inner(x)
            x += 3
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)

    def test_exception_in_dynamo_handling(self):
        hit_handler = False

        # See https://github.com/pytorch/pytorch/pull/96488
        @contextlib.contextmanager
        def ctx():
            try:
                yield
            except RuntimeError:
                nonlocal hit_handler
                hit_handler = True

        @torch._dynamo.optimize("eager")
        def f():
            with ctx():
                h()

        def h():
            raise RuntimeError("boof")

        # Should not error
        f()
        self.assertTrue(hit_handler)

    def test_generator_dealloc(self):
        # See https://github.com/pytorch/pytorch/pull/96488
        #
        # NB: yes, [(...)] is intentional, this is a list containing a
        # generator
        generator_box = [(x for x in [1, 2, 3])]

        counter = torch._dynamo.testing.CompileCounter()

        def g(x):
            return x + 2

        # TODO: This test is pretty delicate.  To test if it's actually doing
        # anything, rebuild eval_frame.c with '#define TORCHDYNAMO_DEBUG 1'
        # and then look at the logs for:
        #
        # TRACE[_custom_eval_frame:650] begin <genexpr> test_repros.py 2276 -1 0 0
        # TRACE[_custom_eval_frame:664] throw <genexpr>
        #
        # This means we're actually hitting the relevant codepath

        # NB: Make sure we don't actually Dynamo this frame; if we do Dynamo
        # this frame, Dynamo actually DOES understand list.clear and will
        # arrange for the generator deallocation to happen when the eval frame
        # handler is disabled, which will prevent the bug from happening (we
        # specifically want to trigger the generator deallocation WHILE the
        # dynamo eval frame handler is active), as that will cause the
        # generator to become exhausted and trigger the throw_flag == TRUE
        # case.
        @torch._dynamo.disable(recursive=False)
        def f(x):
            generator_box.clear()
            return g(x)

        self.assertNoUnraisable(
            lambda: torch._dynamo.optimize(counter)(f)(torch.randn(3))
        )

        # Make sure the x + 2 is captured (a previous incorrect implementation
        # of this fix would have disabled the eval frame callback, which means
        # g wouldn't get traced
        self.assertEqual(counter.op_count, 1)

    def test_error_return_without_exception_set(self):
        # https://github.com/pytorch/pytorch/issues/93781
        @torch.compile
        def f():
            _generator_type = type((_ for _ in ()))

        self.assertNoUnraisable(f)

    def test_rewrite_assert_with_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 3, "First dim need to be 3"
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        cnt = torch._dynamo.testing.CompileCounter()

        opt_f = torch._dynamo.optimize(cnt, nopython=True)(f)
        self.assertTrue(same(f(*args), opt_f(*args)))
        self.assertEqual(cnt.op_count, 6)
        self.assertEqual(cnt.frame_count, 1)

        exported, _ = torch._dynamo.export(f, torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        with self.assertRaisesRegex(RuntimeError, "First dim need to be 3"):
            exported, _ = torch._dynamo.export(f, torch.Tensor([4, 4, 5]))

    def test_not_rewrite_assert_for_other_errors(self):
        def f(x):
            b = x.sin()
            if not x.sum() <= 3:
                raise ValueError("input sum needs to be 3")
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        opt_fn = torch._dynamo.optimize("eager")(f)
        with self.assertRaisesRegex(ValueError, "input sum needs to be 3"):
            opt_fn(*args)

    def test_rewrite_assert_without_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 3
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        exported, _ = torch._dynamo.export(f, torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        with self.assertRaisesRegex(RuntimeError, "assertion error"):
            exported, _ = torch._dynamo.export(f, torch.Tensor([4, 4, 5]))

    def test_rewrite_assert_with_non_string_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 2, x.size()
            return x.cos() + b

        torch._dynamo.utils.counters.clear()
        args = torch.Tensor([3, 4, 5])
        opt_f = torch._dynamo.optimize("eager")(f)
        with self.assertRaisesRegex(AssertionError, "torch.Size"):
            opt_f(args)
        self.assertEqual(
            torch._dynamo.utils.counters["unimplemented"][
                "assert with non-string message"
            ],
            1,
        )

    def test_rewrite_assert_noop(self):
        def f(x):
            b = x.sin()
            assert True
            assert x.dtype == torch.float32
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        exported, _ = torch._dynamo.export(f, torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch._dynamo.optimize(cnt, nopython=True)(f)
        self.assertTrue(same(f(*args), opt_f(*args)))
        # torch._assert shouldn't be in the graph
        self.assertEqual(cnt.op_count, 3)
        self.assertEqual(cnt.frame_count, 1)

        exported, _ = torch._dynamo.export(f, torch.Tensor([4, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

    def test_size_typematch(self):
        def f(x, y):
            if isinstance(x, torch.Size):
                return y + 1
            else:
                return y + 2

        y = torch.zeros(1)
        x1 = torch.Size((3,))
        x2 = (3,)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch._dynamo.optimize(cnt, nopython=True)(f)
        self.assertTrue(same(f(x1, y), opt_f(x1, y)))
        self.assertTrue(same(f(x2, y), opt_f(x2, y)))
        self.assertEqual(cnt.frame_count, 2)

    def test_dict_subclass_contains(self):
        # pattern from huggingface
        class ClassInstantier(collections.OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def f(x, d):
            if "key1" in d:
                x = x + 2
            if "key2" in d:
                x = x + 4
            x = x + 8
            return x

        result = f(torch.ones(8), ClassInstantier({"key1": torch.ones(8)}))
        self.assertTrue(same(result, torch.full([8], 11.0)))

        result = f(torch.ones(8), ClassInstantier({"key2": torch.ones(8)}))
        self.assertTrue(same(result, torch.full([8], 13.0)))

    def test_hf_classinstantier(self):
        # hf activations.py
        class ClassInstantier(collections.OrderedDict):
            def __getitem__(self, key):
                content = super().__getitem__(key)
                cls, kwargs = content if isinstance(content, tuple) else (content, {})
                return cls(**kwargs)

        ACT2CLS = ClassInstantier(
            {
                "relu": (nn.ReLU, {"inplace": False}),
                "tanh": nn.Tanh,
            }
        )

        @torch.compile(fullgraph=True, backend="eager")
        def f(x, act):
            return ACT2CLS[act](x)

        y = torch.randn(10)
        self.assertTrue(same(f(y, "tanh"), torch.tanh(y)))
        self.assertTrue(same(f(y, "relu"), torch.relu(y)))

    def test_ephemeral_module(self):
        # hf activations.py
        class ReLUSquaredActivation(nn.Module):
            def forward(self, input):
                relu_applied = torch.nn.functional.relu(input)
                squared = torch.square(relu_applied)
                return squared

        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            x = x + 0.2
            x = ReLUSquaredActivation()(x)
            x = x + 1
            return x

        y = torch.randn(10)
        self.assertTrue(same(f(y), ReLUSquaredActivation()(y + 0.2) + 1))

    def test_inplace_unsqueeze_input(self):
        def backend(gm, example_inputs):
            self.assertEqual(example_inputs[-1].size(), torch.Size([1, 3, 4]))
            return gm

        @torch.compile(backend=backend)
        def fn(x):
            x.unsqueeze_(0)
            return x + 1

        inputs = [torch.randn(3, 4)]
        self.assertEqual(fn(*inputs).size(), torch.Size([1, 3, 4]))
        self.assertEqual(inputs[0].size(), torch.Size([1, 3, 4]))

    def test_batchnorm_e2e(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(
                    64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                )
                self.conv1 = torch.nn.Conv2d(
                    64,
                    64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=False,
                )

            def forward(self, x):
                x1 = self.bn(x)
                x2 = self.conv1(x1)
                out = torch.nn.functional.relu(x2)
                return (out,)

        torch.manual_seed(1337)

        m_ref = Repro()
        m_test = deepcopy(m_ref)

        @torch._dynamo.optimize("aot_eager_decomp_partition")
        def compiled_fn(x):
            return m_test(x)

        x_ref = torch.randn(2, 64, 32, 32, requires_grad=True)
        x_test = x_ref.clone()

        # Loop multiple times: each iteration the running_mean/var on batchnorm will update,
        # which changes the output of the next iteration
        for _ in range(3):
            ref = m_ref(x_ref)
            res = compiled_fn(x_test)

            self.assertTrue(same(ref, res))

            for r in ref:
                if r.requires_grad:
                    r.sum().backward()
            for r in res:
                if r.requires_grad:
                    r.sum().backward()

            for param_ref, param_test in zip(m_ref.parameters(), m_test.parameters()):
                self.assertTrue(same(param_ref, param_test))
            # Assert running_mean/var
            for buffer_ref, buffer_test in zip(m_ref.buffers(), m_test.buffers()):
                self.assertTrue(same(buffer_ref, buffer_test))

    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_dynamic_shapes_right_side(self):
        def f(x):
            return torch.ones(5 * x.shape[0])

        inp = torch.randn(6, 5)

        gm, _ = torch._dynamo.export(f, torch.randn(4, 5), aten_graph=True)
        self.assertEqual(gm(inp).shape, f(inp).shape)

    @torch._dynamo.config.patch("specialize_int", False)
    def test_maybe_multiply_symint(self):
        # https://github.com/pytorch/pytorch/issues/97346
        from torch._functorch.aot_autograd import aot_module_simplified

        def my_aot_compiler(gm, example_inputs):
            def my_compiler(gm, example_inputs):
                return gm.forward

            # Invoke AOTAutograd
            return aot_module_simplified(gm, example_inputs, fw_compiler=my_compiler)

        def my_example(t1, t2, d):
            out = torch.add(t1, t2, alpha=d)
            return out

        compiled_fn = torch.compile(backend=my_aot_compiler, dynamic=True)(my_example)

        t1 = torch.arange(3, dtype=torch.float32).requires_grad_(True)
        t2 = torch.arange(3, dtype=torch.float32).requires_grad_(True)

        ra = compiled_fn(t1, t2, 5)
        self.assertEqual(ra, torch.tensor([0.0, 6.0, 12.0]))

        ra = compiled_fn(t1, t2, 6)
        self.assertEqual(ra, torch.tensor([0.0, 7.0, 14.0]))

    def test_build_map_unpack_with_call(self):
        def forward_with_cond_scale(x, t, cond_scale, self_cond, other1, other2):
            return x.sin() + t + cond_scale + self_cond + other1 + other2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            d1 = dict(other1=5)
            d2 = dict(other2=4)
            text_cond = {**d1, **d2}
            return forward_with_cond_scale(x, 1, cond_scale=2, self_cond=3, **text_cond)

        self.assertTrue(same(fn(torch.ones(4)), torch.ones(4).sin() + 15))

    def test_graph_break_unsupported_fake(self):
        counter = torch._dynamo.testing.CompileCounter()

        torch._dynamo.config.verbose = True

        @torch._dynamo.optimize(counter)
        def f(x):
            return torch.ops.test_sample.foo(x + 1) + 1

        f(torch.randn(3))

        self.assertEqual(counter.op_count, 2)
        self.assertEqual(counter.frame_count, 2)

    def test_delattr(self):
        class MyObj:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, obj):
            del obj.a
            obj.c = x + 1
            del obj.c
            tmp = MyObj(x + 2, x + 3)
            del tmp.b
            if hasattr(obj, "a"):
                return x + 1
            return tmp

        x = torch.zeros([])
        obj1 = MyObj(x, x)
        obj2 = fn(x, obj1)
        self.assertFalse(hasattr(obj1, "a"))
        self.assertFalse(hasattr(obj1, "c"))
        self.assertFalse(hasattr(obj2, "b"))
        self.assertEqual(obj1.b.item(), 0)
        self.assertEqual(obj2.a.item(), 2)

    def test_delattr_raises(self):
        class MyObj:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        @torch.compile(backend="eager")
        def fn(x, obj):
            del obj.a
            x = x + 1
            obj.a  # will raise
            return x

        x = torch.zeros([])
        obj1 = MyObj(x, x)
        self.assertRaises(AttributeError, lambda: fn(x, obj1))

    def test_attached_attribute_in_dir(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        mod = torch.compile(MyModule(), backend="eager")
        mod.is_compiled = True
        self.assertTrue("is_compiled" in dir(mod))

    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    def test_dynamic_shapes_implicit_guard(self):
        def f(x):
            y = x * x.size(x.shape[0])
            torch.sum(y, [y.shape[0]])
            return y

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(f)
        opt_fn(torch.randn(3, 1, 1, 1, 1))
        self.assertEqual(cnt.frame_count, 1)

    def test_dalle2_maybe(self):
        def normalize(x):
            return x.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, normalize_img):
            lowres_cond_img = x.sin()
            lowres_cond_img = maybe(normalize_img)(lowres_cond_img)
            return lowres_cond_img

        self.assertEqual(fn(torch.ones([]), normalize), torch.ones([]).sin().cos())

    def test_functools_wraps(self):
        def cool_name(x):
            return x.sin()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = x.cos()

            @functools.wraps(cool_name)
            def uncool_name():
                return cool_name(y)

            return uncool_name

        result = fn(torch.ones([]))
        self.assertEqual(result.__name__, "cool_name")
        self.assertEqual(result(), torch.ones([]).cos().sin())

    def test_dynamic_shapes_float_guard(self):
        def f(x):
            return torch.nn.functional.dropout(x, x.shape[0] / 6)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(f)
        opt_fn(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensor_item(self):
        def f(x, y):
            val = y.item()
            return x.sum() + val

        gm, _ = torch._dynamo.export(
            f,
            torch.zeros(6, 4),
            torch.tensor(1),
            aten_graph=True,
        )
        self.assertEqual(
            f(torch.zeros(6, 4), torch.tensor(1)),
            gm(torch.zeros(6, 4), torch.tensor(1)),
        )
        self.assertEqual(
            f(torch.zeros(6, 4), torch.tensor(2)),
            gm(torch.zeros(6, 4), torch.tensor(2)),
        )

    def test_hf_xsoftmax_inference(self):
        def fn(input, mask):
            return XSoftmax.apply(input + 1, mask, 1) + 2

        fn_opt = torch.compile(fn, backend="eager", fullgraph=True)

        inputs = [
            torch.randn(4, 10),
            torch.randn(4, 10) < 0,
        ]
        expected = fn(*inputs)
        actual = fn_opt(*inputs)
        self.assertTrue(same(actual, expected))

    def test_hf_xsoftmax_training(self):
        from torch._dynamo.utils import counters

        counters.clear()

        def fn(input, mask):
            return XSoftmax.apply(input, mask, 1)

        cnt = torch._dynamo.testing.CompileCounter()
        fn_opt = torch.compile(fn, backend=cnt, fullgraph=False)

        torch.manual_seed(1234)
        inputs1 = [
            torch.randn(4, 10, requires_grad=True),
            torch.randn(4, 10) < 0,
        ]
        torch.manual_seed(1234)
        inputs2 = [
            torch.randn(4, 10, requires_grad=True),
            torch.randn(4, 10) < 0,
        ]

        expected = fn(*inputs1)
        actual = fn_opt(*inputs2)
        self.assertTrue(same(actual, expected))
        self.assertEqual(dict(counters["frames"]), {"total": 1, "ok": 1})
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(cnt.frame_count, 1)
        cnt.clear()
        counters.clear()

        expected.sum().backward()
        actual.sum().backward()
        self.assertTrue(same(inputs1[0].grad, inputs2[0].grad))

        # currently we don't capture the backwards frame
        self.assertEqual(cnt.frame_count, 0)
        self.assertEqual(cnt.op_count, 0)
        self.assertEqual(dict(counters["frames"]), {})
        self.assertEqual(dict(counters["graph_break"]), {})

    def test_autograd_function_graph_break(self):
        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                torch._dynamo.graph_break()
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx, gx):
                (x,) = ctx.saved_tensors
                return gx * x.cos()

        x = torch.randn([], requires_grad=True)

        @torch.compile(backend="eager")
        def fn(x):
            return MySin.apply(x)

        y = fn(x)
        self.assertEqual(y, x.sin())

        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, x.cos())

    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_tensor_split(self):
        def f(x):
            return torch.split(x, x.shape[0] // 2, dim=0)[0]

        gm, _ = torch._dynamo.export(
            f,
            torch.zeros(6, 4),
            aten_graph=True,
        )

        self.assertEqual(f(torch.ones(8, 4)), gm(torch.ones(8, 4)))

    def test_optim_state_references_cleared(self):
        model = torch.nn.Linear(2048, 2048, bias=False)
        x = torch.ones(2048)
        state_ref = 0

        optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

        def opt_step():
            optimizer.step()

        compiled_opt_step = torch._dynamo.optimize("eager")(opt_step)

        def compiled_model_step(x):
            optimizer.zero_grad()
            y = model(x)
            torch.sum(y).backward()
            compiled_opt_step()

        compiled_model_step(x)

        # Picked "square_avg" arbitrarily to check that
        # optimizer state tensors are deallocated
        state_ref = weakref.ref(
            optimizer.state[optimizer.param_groups[0]["params"][0]]["square_avg"]
        )
        optimizer = None

        self.assertIsNone(state_ref())

    def test_grad_references_cleared(self):
        model = torch.nn.Linear(2048, 2048, bias=False)
        x = torch.ones(2048)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

        def opt_step():
            optimizer.step()

        compiled_opt_step = torch._dynamo.optimize("eager")(opt_step)

        def compiled_model_step(x):
            optimizer.zero_grad(True)
            y = model(x)
            torch.sum(y).backward()
            compiled_opt_step()

        compiled_model_step(x)
        param_grad_ref = weakref.ref(list(model.parameters())[0].grad)
        optimizer.zero_grad(True)
        self.assertIsNone(param_grad_ref())

    def test_batch_encoding_clone_inputs(self):
        class BatchEncoding(dict):
            """
            Copied from test_tokenization
            """

            def __init__(
                self,
                data,
            ):
                super().__init__(data)

            def __getattr__(self, item: str):
                try:
                    return self.data[item]
                except KeyError as e:
                    raise AttributeError from e

        encoding = BatchEncoding({"key": torch.rand((1, 4))})
        cloned_encoding = torch._dynamo.utils.clone_inputs(encoding)
        self.assertTrue(type(cloned_encoding) is not dict)

    def test_iadd_graph_break(self):
        def fn(x):
            a = ()
            x = torch.sin(x)
            a += (x,)
            return a

        x = torch.randn(4)
        ref = fn(x)

        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_odict_get_item_index_name(self):
        d = {float: torch.float32, np.float16: torch.float16}

        @torch.compile
        def f(x, y1, y2):
            return torch.zeros(5, dtype=d[y1]), torch.zeros(5, dtype=d[y2])

        f(torch.zeros(4), float, np.float16)

    def test_dedup_global(self):
        @torch.compile()
        def f():
            return _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR

        self.assertEqual(f(), _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR)

    @requires_cuda()
    def test_guard_default_device(self):
        try:
            torch.set_default_device("cuda")

            counter = torch._dynamo.testing.CompileCounter()

            @torch._dynamo.optimize(counter)
            def f():
                x = torch.randn(3)
                return x * 2

            self.assertEqual(f().device.type, "cuda")
            self.assertEqual(counter.frame_count, 1)

            torch.set_default_device("cpu")

            self.assertEqual(f().device.type, "cpu")
            self.assertEqual(counter.frame_count, 2)

        finally:
            torch.set_default_device(None)

    def test_list_self_reference(self):
        # Issue - https://github.com/pytorch/pytorch/issues/100150
        root = []
        root[:] = [root, root, None, None]

        @torch._dynamo.optimize("eager")
        def test_bug():
            return root

        test_bug()

    def test_hf_bigbird_unsqueeze(self):
        def torch_bmm_nd(inp_1, inp_2, ndim=None):
            torch._dynamo.graph_break()
            return torch.bmm(inp1, inp2)

        def fn(inp1, inp2, inp3, inp4, c):
            a = torch_bmm_nd(inp1, inp2, 4)
            a.unsqueeze_(2)
            a = a * 2

            b = torch_bmm_nd(inp3, inp4, 4)
            b.unsqueeze_(2)
            l = a + b

            out = torch.cat([a, b, c], dim=2)
            return out, l

        inp1 = torch.rand(1, 64, 448)
        inp2 = torch.rand(1, 448, 64)
        inp3 = torch.rand(1, 64, 448)
        inp4 = torch.rand(1, 448, 64)
        c = torch.rand(1, 64, 1, 64)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        opt_fn(inp1, inp2, inp3, inp4, c)
        self.assertEqual(cnt.frame_count, 3)

    def test_torch_variable_type(self):
        # from torchvision
        def check_type(obj, types_or_checks):
            for type_or_check in types_or_checks:
                if (
                    isinstance(obj, type_or_check)
                    if isinstance(type_or_check, type)
                    else type_or_check(obj)
                ):
                    return True
            return False

        opt_check_type = torch._dynamo.optimize("eager")(check_type)
        ref = check_type(torch.randn(4), [torch.Tensor])
        res = opt_check_type(torch.randn(4), [torch.Tensor])
        self.assertEqual(ref, res)

    # Test for https://github.com/pytorch/pytorch/issues/103132
    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_inference_mode_dynamic_shapes(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, param):
                z = torch.matmul(param, param)
                return z

        model = Repro()
        # Need a 3d tensor to actually cause the error:
        # we go down a path of the C++ matmul decomp that calls sizes().
        inp = torch.randn(4, 4, 4, requires_grad=True)
        model = torch.compile(model, backend="aot_eager", dynamic=True)
        with torch.inference_mode():
            model(inp)

    def test_kwargs_out_list_variable(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, param):
                z = torch.frexp(**param)
                return z

        model = Repro()
        params = {"input": torch.tensor([[0.0, 1, 2, 4]])}
        params["out"] = [
            torch.empty(0, dtype=torch.float32),  # mantissa
            torch.empty(0, dtype=torch.int32),  # exponent
        ]

        model = torch.compile(model, backend="eager")
        mantissa, exponent = model(params)
        ref_mantissa = torch.tensor([[0.0000, 0.5000, 0.5000, 0.5000]])
        ref_exponent = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        self.assertEqual(ref_mantissa, mantissa)
        self.assertEqual(ref_exponent, exponent)

    def test_unspecialized_nn_module_with_torch_variable_attribute(self):
        """
        In this case self.fn = something that should be a TorchVariable.
        When it's not a TorchVariable, dynamo tries to trace through and fails.
        This makes sure that the self.fn is handled as a TorchVariable.
        """

        class UserModule(torch.nn.Module):
            torchdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, **inp):
                return self.fn(**inp)

        inputs = {
            "input": torch.randn([2, 9]).uniform_(0, 1),
            "target": torch.randn([2, 9]).uniform_(0, 1),
            "reduction": "mean",
        }

        mod = UserModule(torch.nn.functional.binary_cross_entropy)
        ref = mod(**inputs)
        res = torch._dynamo.optimize("eager", nopython=True)(mod)(**inputs)
        self.assertEqual(ref, res)

    def test_call_finally_python_3_8(self):
        # Issue - https://github.com/pytorch/pytorch/issues/97811
        def make_fn(g):
            def fn():
                while True:
                    try:
                        print(g)
                        break
                    except Exception as _:
                        break

            return torch.compile(fn, backend="eager")

        make_fn(None)()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
