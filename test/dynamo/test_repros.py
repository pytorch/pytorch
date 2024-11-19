"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_rewrite_assert_with_msg and test_rewrite_assert_without_msg)
"""

# Owner(s): ["module: dynamo"]
import collections
import contextlib
import copy
import dataclasses
import functools
import gc
import importlib
import inspect
import itertools
import os
import random
import unittest
import warnings
import weakref
from abc import ABC
from collections import namedtuple
from copy import deepcopy
from enum import Enum, IntEnum
from functools import wraps
from typing import Any, Dict, Iterator, List, Literal, Tuple, TypedDict
from unittest import mock

import numpy as np

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
import torch._functorch.config
import torch.library
import torch.utils._pytree as pytree
from torch import nn
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.testing import CompileCounter, rand_strided, same, skipIfPy312
from torch._inductor.utils import fresh_inductor_cache
from torch.nn import functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import (
    disable_translation_validation_if_dynamic_shapes,
    instantiate_parametrized_tests,
    parametrize,
    skipIfWindows,
    TEST_WITH_ROCM,
)
from torch.testing._internal.two_tensor import TwoTensor


_orig_module_call = torch.nn.Module.__call__

# Custom operator that only supports CPU and Meta
lib = torch.library.Library("test_sample", "DEF")  # noqa: TOR901
lib.define("foo(Tensor self) -> Tensor")
lib.impl("foo", torch.sin, "CPU")


requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")


_GLOBAL_CPU_TENSOR = torch.randn(3)

HAS_MSGSPEC = importlib.util.find_spec("msgspec")
if HAS_MSGSPEC:
    import msgspec


HAS_OMEGACONG = importlib.util.find_spec("omegaconf")
if HAS_OMEGACONG:
    from omegaconf import OmegaConf


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


def global_fn(x):
    return torch.sin(x)


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


fw_graph = [None]
bw_graph = [None]


def aot_graph_capture_backend(gm, args):
    from functorch.compile import min_cut_rematerialization_partition
    from torch._functorch.aot_autograd import aot_module_simplified

    def fw_compiler(gm, _):
        fw_graph[0] = gm
        return gm

    def bw_compiler(gm, _):
        bw_graph[0] = gm
        return gm

    return aot_module_simplified(
        gm,
        args,
        fw_compiler,
        bw_compiler,
        partition_fn=min_cut_rematerialization_partition,
        keep_inference_input_mutations=True,
    )


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
            all_buckets = all_buckets + (attn_output,)

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

        # free memory
        del grad_attn_output

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
    def __init__(self) -> None:
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


class ListConfig:
    class ValueNode:
        def __init__(self, value):
            self.value = value

        def _dereference_node(self):
            return self

        def _is_missing(self):
            return False

        def _value(self):
            return self.value

    # Based on an example from omegaconfig.listconfig
    class ListIterator(Iterator[Any]):
        def __init__(self, lst: Any, resolve: bool) -> None:
            self.resolve = resolve
            self.iterator = iter(lst.__dict__["_content"])
            self.index = 0

        def __next__(self) -> Any:
            x = next(self.iterator)
            if self.resolve:
                x = x._dereference_node()
                if x._is_missing():
                    raise AssertionError

            self.index = self.index + 1
            if isinstance(x, ListConfig.ValueNode):
                return x._value()
            raise AssertionError

    def __iter__(self):
        return self._iter_ex(True)

    def _iter_ex(self, resolve: bool) -> Iterator[Any]:
        try:
            return ListConfig.ListIterator(self, resolve)
        except Exception:
            raise AssertionError from None

    def __init__(self) -> None:
        self._content = [
            ListConfig.ValueNode(1),
            ListConfig.ValueNode(3),
            ListConfig.ValueNode(torch.tensor([7.0])),
        ]


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
    def __init__(self) -> None:
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
            ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
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
    def __init__(self) -> None:
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
        raise ValueError

    return forward_fn(*input_tensors)


def _validate_model_kwargs(fn, model_kwargs):
    # simplified from transformers.generation.utils._validate_model_kwargs
    unused_model_args = []
    model_args = set(inspect.signature(fn).parameters)
    for key, value in model_kwargs.items():
        if value is not None and key not in model_args:
            unused_model_args.append(key)
    if unused_model_args:
        raise ValueError(
            f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
            " generate arguments will also show up in this list)"
        )


class FakeMamlInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(784, 5)

    def forward(self, x, ignored=None, bn_training=False):
        return self.linear(x.view(x.shape[0], -1))


class PartialMaml(torch.nn.Module):
    # Highly simplified version of maml.meta.Meta.finetuning
    def __init__(self) -> None:
        super().__init__()
        self.net = FakeMamlInner()
        self.update_step_test = 10
        self.update_lr = 0.4

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetuning on the copied model instead of self.net
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
    elif len(attn_types_set) == 2 and attn_types_set == {"lsh", "local"}:
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


class CustomList1(list):
    def __call__(self, x):
        for processor in self:
            x = processor(x)
        return x

    def clear(self):
        pass  # this prevents RestrictedListSubclassVariable from kicking in


class CustomList2(list):
    def __call__(self, x):
        for processor in self:
            x = processor(x)
        return x

    def length_times_10(self):
        return len(self) * 10

    def append_twice(self, x):
        self.extend([x, x])


def _merge_criteria_processor_list(default_list, custom_list):
    # simplified transformers/generation/utils.py
    if len(custom_list) == 0:
        return default_list
    for default in default_list:
        for custom in custom_list:
            if type(custom) is type(default):
                raise ValueError
    default_list.extend(custom_list)
    return default_list


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


class IncByOne:
    def __init__(self, x):
        self.x = x + 1


class IncByTwo:
    def __init__(self, x):
        self.x = x + 2


class ReproTests(torch._dynamo.test_case.TestCase):
    def setUp(self) -> None:
        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(
            install_guard_manager_testing_hook(self.guard_manager_clone_hook_fn)
        )
        super().setUp()

    def tearDown(self) -> None:
        self.exit_stack.close()
        super().tearDown()

    def guard_manager_clone_hook_fn(self, guard_manager_wrapper, f_locals):
        root = guard_manager_wrapper.root
        cloned_root = root.clone_manager(lambda x: True)
        cloned_wrapper = torch._dynamo.guards.GuardManagerWrapper(cloned_root)
        self.assertEqual(str(guard_manager_wrapper), str(cloned_wrapper))
        self.assertTrue(cloned_root.check(f_locals))
        if guard_manager_wrapper.diff_guard_root:
            self.assertTrue(guard_manager_wrapper.diff_guard_root.check(f_locals))

    def test_do_paste_mask(self):
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()
        opt__do_paste_mask = torch.compile(_do_paste_mask, backend=cnt)
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
        # (dynamic shapes, static shapes)
        self.assertIn(cnt.frame_count, (5, 7))
        self.assertIn(cnt.op_count, (92, 106, 119))

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
            self.assertExpectedInline(cnt.op_count, """14""")

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
            self.assertExpectedInline(cnt.op_count, """2""")

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

    @requires_cuda
    def test_sub_alpha_scalar_repro(self):
        @torch.compile(backend="aot_eager")
        def f(x):
            return x.sub(1, alpha=2)

        f(torch.ones(2, device="cuda", dtype=torch.float64))

    # https://github.com/pytorch/pytorch/issues/113010
    def test_out_overload_non_contiguous(self):
        def f(x, y):
            return torch.abs(x, out=y.T)

        f_compiled = torch.compile(f, backend="aot_eager")

        x_ref = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        y_ref = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        x_test = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        y_test = torch.arange(4, dtype=torch.float32).reshape(2, 2)

        out_ref = f(x_ref, y_ref)
        out_test = f_compiled(x_test, y_test)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(y_ref, y_test)

    # https://github.com/pytorch/pytorch/issues/109053
    def test_view_dtype_overload(self):
        def f(x):
            return x.view(torch.int32)

        f_compiled = torch.compile(f, backend="aot_eager")

        x1 = torch.ones(4, requires_grad=True)
        out_ref = f(x1)
        out_test = f_compiled(x1)
        self.assertEqual(out_ref, out_test)

        x2 = torch.ones(4, requires_grad=False)
        out_ref = f(x2)
        out_test = f_compiled(x2)
        self.assertEqual(out_ref, out_test)

    # https://github.com/pytorch/pytorch/issues/90552
    def test_intermediate_leaf_requires_grad(self):
        def f(x):
            leaf = torch.ones(2, requires_grad=True)
            return leaf, leaf * 2

        f_compiled = torch.compile(f, backend="aot_eager")
        x = torch.arange(4, dtype=torch.float32).reshape(2, 2)

        leaf, out = f(x)
        leaf_test, out_test = f_compiled(x)
        out.sum().backward()
        out_test.sum().backward()
        self.assertEqual(leaf.grad, leaf_test.grad)

    # https://github.com/pytorch/pytorch/issues/113263
    def test_unpack_hooks_dont_run_during_tracing(self):
        def f(x, y):
            return x * y

        f_compiled = torch.compile(f, backend="aot_eager")

        pack_count = 0
        unpack_count = 0

        def pack_hook(x):
            nonlocal pack_count
            pack_count += 1
            return x

        # unpack hook shouldn't run during compilation, while we trace the forward
        def unpack_hook(x):
            nonlocal unpack_count
            unpack_count += 1
            return x

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out_test = f_compiled(x, y)
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 0)
            out_test.sum().backward()
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 1)

    # https://github.com/pytorch/pytorch/issues/113263
    def test_unpack_hooks_can_be_disabled(self):
        def f(x, y):
            return x * y

        f_compiled = torch.compile(f, backend="aot_eager")

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        with torch.autograd.graph.disable_saved_tensors_hooks("hooks are disabled"):
            out_test = f_compiled(x, y)
            out_test.sum().backward()

    # https://github.com/pytorch/pytorch/issues/113263
    def test_disabling_unpack_hooks_within_compiled_region(self):
        def g(z):
            with torch.autograd.graph.disable_saved_tensors_hooks("hooks are disabled"):
                return z + 5

        def f(x, y):
            z = x * y
            return g(z)

        f_compiled = torch.compile(f, backend="aot_eager")

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        out_test = f_compiled(x, y)
        out_test.sum().backward()

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

        a_test = a_ref.detach().clone().requires_grad_(True)
        b_test = b_ref.detach().clone().requires_grad_(True)
        out_test = torch.compile(f, backend="aot_eager")(a_test, b_test)

        self.assertEqual(out_ref, out_test)
        self.assertEqual(a_ref.grad, a_test.grad)
        self.assertEqual(b_ref.grad, b_test.grad)

    # https://github.com/pytorch/pytorch/issues/111603
    def test_tuple_enum_as_key_dict(self):
        class MyEnum(Enum):
            A = "a"

        class SomeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x) -> torch.Tensor:
                return self.linear(x[MyEnum.A])

        x = {MyEnum.A: torch.rand(8, 1)}
        model_pytorch = SomeModel()
        model = torch.compile(model_pytorch)
        # Executing twice works
        model(x)
        y = model(x)
        self.assertEqual(y, model_pytorch(x))

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
        expected_op_count = (
            """11""" if torch._dynamo.config.inline_inbuilt_nn_modules else """5"""
        )

        self.assertExpectedInline(cnt.frame_count, """1""")
        self.assertExpectedInline(cnt.op_count, expected_op_count)

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
                self.assertExpectedInline(cnt.op_count, """8""")
            else:
                self.assertExpectedInline(cnt.frame_count, """2""")
                self.assertExpectedInline(cnt.op_count, """4""")
        else:
            self.assertExpectedInline(cnt.frame_count, """2""")
            self.assertExpectedInline(cnt.op_count, """19""")

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
            self.assertExpectedInline(cnt.op_count, """11""")

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

    @torch._dynamo.config.patch(error_on_recompile=True)
    @torch.fx.experimental._config.patch(use_duck_shape=False)
    def test_dynamic_shape_disable_duck_size(self):
        class TestModel(nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x: torch.Tensor, val: int) -> torch.Tensor:
                return x + val

        main_model = TestModel().to(memory_format=torch.channels_last)
        opt_model = torch.compile(main_model, backend="eager", dynamic=True)

        x1 = torch.rand(2, 5, 10, 10).to(memory_format=torch.channels_last)
        x2 = torch.rand(2, 5, 4, 8).to(memory_format=torch.channels_last)

        o1_ref = main_model(x1, 4)
        o1 = opt_model(x1, 4)

        o2_ref = main_model(x2, 20)
        o2 = opt_model(x2, 20)

    def test_chunk_reformer_ff(self):
        input = torch.randn([1, 4096, 256])
        model = ChunkReformerFeedForward()
        correct = model(input)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(input), correct))

        self.assertEqual(cnt.frame_count, 1)
        self.assertLessEqual(cnt.op_count, 10)

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
            self.assertExpectedInline(cnt.frame_count, """4""")
        else:
            self.assertExpectedInline(cnt.frame_count, """5""")

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
        self.assertEqual(cnt.op_count, 2)  # rand, rand
        try:
            graph, _ = torch._dynamo.export(fn)()
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
        self.assertExpectedInline(cnt.frame_count, """1""")
        self.assertExpectedInline(cnt.op_count, """3""")

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
            self.assertExpectedInline(cnt.op_count, """16""")

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

    def test_issue114171(self):
        device = torch.device("cpu")

        def fcnn(in_dim, out_dim, hidden_dim, activation=torch.nn.GELU):
            layers = [
                torch.nn.Linear(in_dim, hidden_dim, device=device),
                activation(),
                torch.nn.Linear(hidden_dim, out_dim, device=device),
            ]
            return torch.nn.Sequential(*layers)

        class testmodel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.interaction_networks = torch.nn.ModuleList(
                    [fcnn(262, 1174, 400) for _ in range(4)]
                )

            def interact(self, x, cycle):
                return self.interaction_networks[cycle](x)

        model = testmodel()
        forward_aot = torch.compile(
            model.interact, fullgraph=True, dynamic=True, backend="eager"
        )

        x = torch.rand([111, 262], device=device)
        y2 = forward_aot(x, 2)  # previously failed

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
        self.assertEqual(12, cnt.op_count)

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

        # Not sure what this test is testing. It was earlier graph breaking on
        # __dict__, so the counter >= 2. With __dict__ support, there is no
        # graph break.
        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)
        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["total"], 1)

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

    @skipIfWindows(
        msg="torch._dynamo.exc.TorchRuntimeError: Failed running call_function <class 'torch.LongTensor'>(*(FakeTensor(..., size=(10,), dtype=torch.int32),), **{}):"  # noqa: B950
    )
    def test_longtensor_list(self):
        for partition in [0, 5, 10]:

            @torch._dynamo.disable
            def rand_gen():
                rand_vals = [random.randint(5, 10) for _ in range(10)]
                # List of tensors mixed with np.arrays
                return list(np.array(rand_vals[:partition])) + [
                    torch.tensor(val) for val in rand_vals[partition:]
                ]

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
            def __init__(self) -> None:
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

    def test_issue111522(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x, y):
            return x + y.a

        class A:
            a = 2

        self.assertEqual(f(torch.zeros(2), A()), torch.full([2], 2.0))

        del A.a

        # graph break on missing attr
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            f(torch.zeros(2), A())

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
            def __init__(self) -> None:
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
            def __init__(self) -> None:
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

    def test_out_root_cell_shape_change(self):
        @torch.compile(backend="eager")
        def fn():
            out = torch.empty(0)

            def run():
                x = torch.zeros(3, 5)
                torch.sigmoid(x, out=out)
                return out.size()

            return run()

        res = fn()
        self.assertEqual((3, 5), res)

    def test_out_nested_cell_shape_change(self):
        @torch.compile(backend="eager")
        def fn():
            def run():
                x = torch.zeros(3, 5)
                out = torch.empty(0)

                def capture():
                    return out  # Force `out` to be a nested cell

                torch.sigmoid(x, out=out)
                return out.size()

            return run()

        res = fn()
        self.assertEqual((3, 5), res)

    def test_out_root_cell_tuple_shape_change(self):
        @torch.compile(backend="eager")
        def fn():
            out1 = torch.empty(0)
            out2 = torch.empty(0, dtype=torch.long)

            def run():
                x = torch.zeros(3, 5)
                torch.sort(x, out=(out1, out2))
                return out1.size(), out2.size()

            return run()

        res = fn()
        self.assertEqual(((3, 5), (3, 5)), res)

    def test_out_nested_cell_tuple_shape_change(self):
        @torch.compile(backend="eager")
        def fn():
            def run():
                x = torch.zeros(3, 5)
                out1 = torch.empty(0)
                out2 = torch.empty(0, dtype=torch.long)

                def capture():
                    # Force `out1` and `out2` to be nested cells
                    return out1, out2

                torch.sort(x, out=(out1, out2))
                return out1.size(), out2.size()

            return run()

        res = fn()
        self.assertEqual(((3, 5), (3, 5)), res)

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

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_issue111918(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, dynamic=True)
        def fn(x):
            x = x + 1
            y = x.item()
            if y > 2:
                return x * 2
            else:
                return x * 3

        x = torch.tensor([3.0])
        fn(x)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 4)

        torch._dynamo.reset()
        fn = torch.compile(fn, fullgraph=True, backend="eager")
        with self.assertRaises(torch._dynamo.exc.UserError):
            fn(x)

    def test_vdd_duplicate_error(self):
        def fn(a, dt):
            keys = list(dt._jt_dict.keys())
            p = torch.cos(dt._jt_dict[keys[0]]._value)
            q = torch.sin(a)
            r = torch.sigmoid(dt._jt_dict[keys[0]]._value)
            return p + q + r

        class Value:
            def __init__(self) -> None:
                self._value = torch.randn(4)

        class Sample:
            def __init__(self) -> None:
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

    def test_multi_import(self):
        if not has_detectron2():
            raise unittest.SkipTest("requires detectron2")

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
            def __init__(self) -> None:
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
            def __init__(self) -> None:
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
            def __init__(self) -> None:
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

            def __init__(self) -> None:
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
            def __init__(self) -> None:
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
        x_ref = x.detach().clone().requires_grad_(True)

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
            def __init__(self) -> None:
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

    def test_omegaconf_listconfig_iter(self):
        obj = ListConfig()
        x = torch.zeros(2)

        def fn():
            y = x
            for i in obj:
                y += i
            return y

        expected = fn()
        actual = torch.compile(fn, fullgraph=True, backend="eager")()
        self.assertEqual(actual, expected)

    def test_user_defined_iter(self):
        class MyIter:
            def __init__(self) -> None:
                self.i = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.i < 3:
                    self.i += 1
                    return self.i
                raise StopIteration

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            for i in MyIter():
                x += i
            return x

        self.assertEqual(fn(torch.zeros(1)), torch.full([1], 6.0))

    def test_stop_iteration_reconstruct(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.sin(), StopIteration(1, 2, 3)

        _, res = fn(torch.ones(1))
        self.assertEqual(str(res), str(StopIteration(1, 2, 3)))

    def test_tensor_data_kwarg(self):
        # https://github.com/pytorch/pytorch/issues/96278
        def f():
            return torch.tensor(data=[[1.0, -1.0]])

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(f)
        self.assertTrue(same(f(), opt_fn()))
        self.assertEqual(cnt.frame_count, 1)

    @requires_cuda
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

    def test_validate_model_kwargs(self):
        cnt = CompileCounter()

        def f1(a, b):
            return torch.sin(a) + torch.cos(b)

        @torch.compile(backend=cnt, fullgraph=True)
        def f2(**kwargs):
            _validate_model_kwargs(f1, kwargs)
            return f1(**kwargs)

        x = torch.randn(10)
        y = torch.randn(10)

        self.assertEqual(f2(a=x, b=y), f1(x, y))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_swin_base_tensor_attr(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
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
            _generator_type = type(_ for _ in ())

        self.assertNoUnraisable(f)

    def common_merge_criteria_processor_list(self, list_cls, fullgraph):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=fullgraph)
        def f(x, left, right):
            combined = _merge_criteria_processor_list(left, right)
            return combined(x)

        l1 = list_cls([torch.nn.ReLU(), torch.nn.Sigmoid()])
        l2 = list_cls([])
        input = torch.randn(16)
        result = f(input, l1, l2)
        self.assertEqual(result, l1(input))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

        cnt.clear()
        l3 = list_cls([torch.nn.SiLU()])
        expected = l3(l1(input))
        result = f(input, l1, l3)
        self.assertEqual(len(l1), 3)
        self.assertEqual(result, expected)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_merge_criteria_processor_list1(self):
        self.common_merge_criteria_processor_list(CustomList1, False)

    def test_merge_criteria_processor_list2(self):
        self.common_merge_criteria_processor_list(CustomList2, True)

    def test_restricted_list_subclass1(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            l = CustomList2()
            l.extend([True])
            l.append(a)
            l.extend([b])
            l.pop(0)
            l.append(l.length_times_10())
            return sum(l)

        x = torch.randn(10)
        y = torch.randn(10)
        self.assertEqual(fn(x, y), x + y + 20)
        self.assertEqual(cnt.op_count, 3)

    def test_restricted_list_subclass2(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            l1 = CustomList2([a + 1])
            l2 = CustomList2([b + 2])
            l1.extend(l2)
            return l1

        x = torch.randn(10)
        y = torch.randn(10)
        z = fn(x, y)
        self.assertEqual(type(z), CustomList2)
        self.assertEqual(len(z), 2)
        self.assertEqual(z.length_times_10(), 20)
        self.assertEqual(list(z), [x + 1, y + 2])

    def test_restricted_list_subclass3(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a: CustomList2, b: CustomList2):
            a.extend(b)
            a.append_twice(b[2] + 1)
            a.append(b[3] + 2)
            return b

        x = torch.randn(10)
        y = torch.randn(10)
        l = CustomList2([x, y])
        self.assertIs(fn(l, l), l)
        self.assertEqual(len(l), 7)
        self.assertIs(l[0], x)
        self.assertIs(l[1], y)
        self.assertIs(l[2], x)
        self.assertIs(l[3], y)
        self.assertEqual(l[4], x + 1)
        self.assertIs(l[5], l[4])
        self.assertEqual(l[6], y + 2)

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

        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

    def test_list_aliasing(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a):
            a.append(torch.sin(a[0]))
            return a

        x = torch.randn(10)
        l = [x]
        self.assertIs(fn(l), l)
        self.assertEqual(len(l), 2)
        self.assertIs(l[0], x)
        self.assertEqual(l[1], torch.sin(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

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

    def test_rewrite_assert_dont_change_bytecode(self):
        def fn(x):
            with torch.no_grad():
                assert x.max() < 5, f"invalid max {x.max()}"
                x = torch.sin(x)
            return x

        x = torch.ones(4)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_rewrite_assert_without_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 3
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        with self.assertRaisesRegex(RuntimeError, "assertion error"):
            exported(torch.Tensor([5, 6, 7]))

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
            torch._dynamo.utils.counters["graph_break"][
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
        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch._dynamo.optimize(cnt, nopython=True)(f)
        self.assertTrue(same(f(*args), opt_f(*args)))
        # torch._assert shouldn't be in the graph
        self.assertEqual(cnt.op_count, 3)
        self.assertEqual(cnt.frame_count, 1)

        exported, _ = torch._dynamo.export(f)(torch.Tensor([4, 4, 5]))
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
            def __init__(self) -> None:
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

        gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.randn(4, 5))
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

    @torch._dynamo.config.patch(verbose=True)
    def test_graph_break_unsupported_fake(self):
        counter = torch._dynamo.testing.CompileCounter()

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

    def test_delsubscr(self):
        @torch.compile(backend="eager")
        def fn(x):
            del x["a"]
            y = x["b"] + 1
            return y

        x = {"a": torch.tensor([1]), "b": torch.tensor([1])}
        result = fn(x)
        self.assertFalse(hasattr(x, "a"))
        self.assertEqual(result.item(), 2)

    def test_delsubscr_raises(self):
        @torch.compile(backend="eager")
        def fn(x):
            del x["a"]
            y = x["a"] + 1  # should raise KeyError
            return y

        x = {"a": torch.tensor([1]), "b": torch.tensor([1])}
        self.assertRaises(KeyError, lambda: fn(x))

    def test_attached_attribute_in_dir(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
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
            aten_graph=True,
        )(
            torch.zeros(6, 4),
            torch.tensor(1),
        )
        self.assertEqual(
            f(torch.zeros(6, 4), torch.tensor(1)),
            gm(torch.zeros(6, 4), torch.tensor(1)),
        )
        self.assertEqual(
            f(torch.zeros(6, 4), torch.tensor(2)),
            gm(torch.zeros(6, 4), torch.tensor(2)),
        )

    def test_dataclass_init_with_default_factory_with_inputs(self):
        @dataclasses.dataclass
        class DClass:
            sharding_contexts: Any = dataclasses.field(default_factory=list)
            a: int = 1

        def fn(x, inp_list):
            d = DClass(inp_list)
            d.sharding_contexts.append(x.sin() + d.a)
            return d

        x = torch.randn(4)
        inp_list1 = [1, 2, 3]
        inp_list2 = [2, 3, 4]
        inp_list3 = [1, 2]
        ref1 = fn(x, inp_list1)
        ref2 = fn(x, inp_list2)
        ref3 = fn(x, inp_list3)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, fullgraph=True)

        opt_ret1 = opt_fn(x, inp_list1)
        opt_ret2 = opt_fn(x, inp_list2)
        opt_ret3 = opt_fn(x, inp_list3)
        self.assertEqual(ref1.sharding_contexts, opt_ret1.sharding_contexts)
        self.assertEqual(ref2.sharding_contexts, opt_ret2.sharding_contexts)
        self.assertEqual(ref3.sharding_contexts, opt_ret3.sharding_contexts)

    def test_list_index(self):
        for i, list_type in enumerate(
            (
                list,
                tuple,
                torch.Size,
                collections.deque,
                namedtuple("FourElems", "one two three four", defaults=[0, 0, 0, 0]),
            )
        ):
            torch._dynamo.reset()
            for index in ([], [2], [0, 3]):

                def f(t):
                    if i == 4:  # namedtuple
                        xs = list_type(1, 2, 3, 4)
                    else:
                        xs = list_type([1, 2, 3, 4])
                    res = xs.index(3, *index)
                    return t + res

                res = torch._dynamo.optimize(backend="eager", nopython=True)(f)(
                    torch.zeros(1)
                )

                self.assertEqual(res, torch.tensor([2.0]))

    def test_list_index_not_found(self):
        def f(t):
            xs = ["bar", "foo", "baz", "buzz"]
            res = xs.index("non-existent")
            return t + res

        # Raising ValueError from item not found is unsupported
        with self.assertRaises(
            torch._dynamo.exc.Unsupported,
        ):
            torch._dynamo.optimize(backend="eager", nopython=True)(f)(torch.zeros(1))

    def test_list_index_tensor_unsupported(self):
        for index in ([], [2], [0, 3]):

            def f(t):
                xs = [torch.tensor([i]) for i in range(4)]
                res = xs.index(torch.tensor([2]), *index)
                return t + res

            with self.assertRaisesRegex(
                torch._dynamo.exc.UserError, "Dynamic control flow is not supported"
            ):
                torch._dynamo.optimize(backend="eager", nopython=True)(f)(
                    torch.zeros(1)
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

    @mock.patch("torch._dynamo.config.guard_nn_modules", True)
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

    def test_jit_trace_errors(self):
        @torch.compile(backend="eager", dynamic=True)
        def f(x):
            return x + 1

        with self.assertRaises(RuntimeError):
            torch.jit.trace(f, torch.randn(3))

    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_tensor_split(self):
        def f(x):
            return torch.split(x, x.shape[0] // 2, dim=0)[0]

        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
        )(
            torch.zeros(6, 4),
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
        param_grad_ref = weakref.ref(next(iter(model.parameters())).grad)
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

        @torch.compile(backend="eager")
        def f(x, y1, y2):
            return torch.zeros(5, dtype=d[y1]), torch.zeros(5, dtype=d[y2])

        f(torch.zeros(4), float, np.float16)

    def test_dedup_global(self):
        @torch.compile()
        def f():
            return _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR

        self.assertEqual(f(), _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR)

    def test_randint_out_dynamic(self):
        def randint_fn(high, size, out):
            return torch.randint(high, size, out=out)

        opt_model = torch.compile(randint_fn)

        out1 = torch.empty(10, dtype=torch.int32)
        opt_model(17, (10,), out1)

        out2 = torch.empty(12, dtype=torch.int32)
        opt_model(17, (12,), out2)

    @requires_cuda
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
            def __init__(self) -> None:
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
            def __init__(self) -> None:
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

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_with_sizes_aot_autograd(self):
        def fn(result, split_sizes):
            rs = torch.ops.aten.split_with_sizes(result, split_sizes.tolist())
            return rs

        example_inputs = (
            torch.randn(32, requires_grad=True),
            torch.tensor((7, 16, 9)),
        )
        actual = torch.compile(fn, fullgraph=True, backend="aot_eager")(*example_inputs)
        expected = fn(*example_inputs)
        self.assertEqual(actual, expected)

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

    def test_call_finally_python_3_8_2(self):
        def f(x):
            while x:
                try:
                    pass
                except Exception as _:
                    continue

        torch.compile(f, backend="eager")(0)

    def test_call_finally_opcode_python_3_8(self):
        def fn():
            try:
                return torch.zeros(4)
            finally:
                return torch.ones(4)  # noqa: SIM107, B012

        result = torch.compile(fn, backend="aot_eager")()
        self.assertEqual(result, torch.ones(4))

    def test_string_format(self):
        s = "temp{i}"

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            if s.format(i=4) == "temp4":
                return torch.sin(x)
            return torch.cos(x)

        x = torch.randn(4)
        self.assertEqual(fn(x), torch.sin(x))

    # Repro of torch._dynamo.exc.InternalTorchDynamoError: 'NoneType' object has no attribute 'guards'
    # due to bad empty list handling
    def test_empty_list_contains_with_jump(self):
        def fn(x, l):
            if x in l:
                return x.cos()
            return x.sin()

        counter = CompileCounter()
        compiled_fn = torch._dynamo.optimize(counter)(fn)(torch.randn([2, 2]), [])
        self.assertEqual(counter.frame_count, 1)

    def test_graph_break_on_jit_isinstance(self):
        @torch.compile(backend="eager")
        def fn(x):
            if torch.jit.isinstance(x, List[str]):
                return x * 2
            return x

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.rand(4)
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_add_sub_alpha_out(self):
        inp = torch.randn(2, 3, 4)
        other = 1
        alpha = 2
        for op in [torch.add, torch.sub]:
            out = torch.zeros(2, 3, 4)
            compile_out = torch.zeros(2, 3, 4)
            op(inp, other, alpha=alpha, out=out)
            compiled_fn = torch.compile(op, dynamic=True)
            compiled_fn(inp, other, alpha=alpha, out=compile_out)
            self.assertTrue(same(out, compile_out))

    def test_negative_shape_guard(self):
        def fn(x):
            if x.size() != (5, 1, 2, 3):
                return x.cos()
            return x.sin()

        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=True)

        x = torch.ones(5, 1, 3, 4)
        x2 = torch.ones(5, 1, 2, 3)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x2), opt_fn(x2))
        self.assertEqual(counter.frame_count, 2)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_deferred_runtime_asserts(self):
        @torch.compile(fullgraph=True)
        def f(x):
            y = x.item()
            torch._check_is_size(y)
            if y >= 0:
                return x * 2
            else:
                return x * 3

        f(torch.tensor([3]))
        self.assertRaises(RuntimeError, lambda: f(torch.tensor([-2])))

    def test_addr_alpha_beta_out(self):
        inp = torch.randn(2, 3)
        vec1 = torch.randn(2)
        vec2 = torch.randn(3)
        alpha = 2
        beta = 5

        out = torch.zeros(2, 3)
        compile_out = torch.zeros(2, 3)

        torch.addr(inp, vec1, vec2, alpha=alpha, beta=beta, out=out)
        compiled_fn = torch.compile(torch.addr, dynamic=True)
        compiled_fn(inp, vec1, vec2, alpha=alpha, beta=beta, out=compile_out)
        self.assertTrue(same(out, compile_out))

    def test_setattr_requires_grad_graph_breaks(self):
        def fn(x):
            z = x + 4
            x.requires_grad = True
            y = x * z
            return y

        for backend in ["count", "eager", "aot_eager"]:
            if backend == "count":
                backend = CompileCounter()
            opt_fn = torch.compile(fn, backend=backend)

            eager = torch.zeros(5)
            compiled = eager.clone()

            out_eager = fn(eager)
            out_opt = opt_fn(compiled)

            self.assertEqual(out_eager, out_opt)

            out_eager.sum().backward()
            out_opt.sum().backward()

            self.assertEqual(eager, compiled)
            if isinstance(backend, CompileCounter):
                self.assertEqual(backend.frame_count, 2)  # graph breaks

    def test_dynamic_shapes_double_not_equal(self):
        # https://github.com/pytorch/pytorch/issues/113393
        def fn(x):
            if x.size() != (5, 1, 2, 3):
                return x.cos()
            return x.sin()

        opt_fn = torch.compile(fn, backend="eager")

        x = torch.ones(5, 1, 2, 3)
        x2 = torch.ones(5, 1, 3, 4)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x2), opt_fn(x2))

    def test_inductor_no_recursionerror_on_for_loops(self):
        def forward(x):
            for _ in range(10000):
                x = 1.0 * x
            return x

        self.assertTrue(
            same(torch.compile(forward)(torch.tensor([1.0])), torch.tensor([1.0]))
        )

    def test_user_defined_object_callable(self):
        # https://github.com/pytorch/pytorch/issues/114019
        class MyCallable:
            def __call__(self, x):
                return x + 1

        def fn(x):
            # Create in graph - will not have source
            return MyCallable()(x)

        fn_opt = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn_opt(torch.zeros(1)), fn(torch.zeros(1)))

    @torch._dynamo.config.patch(log_compilation_metrics=True)
    def test_many_views_with_mutation(self):
        # When symbolic storage offsets were added in #113734, tensors_definitely_do_not_overlap
        # began adding shape guards - a quadratic amount relative to the number of inputs.
        # Test this configuration, and test that a reasonable number of guards are added.
        # Note, when dynamic shapes are turned on, this test fails and we still get quadratic guards.
        def fn(x):
            x[0].relu_()
            return torch.cat(x).sum()

        AMT = 32
        src = torch.rand(16 * (AMT + 1))

        x = [src.as_strided((4, 4), (4, 1), 3 + 16 * i) for i in range(AMT)]

        torch._dynamo.reset()
        torch._dynamo.utils.clear_compilation_metrics()

        res = torch.compile(fn, backend="aot_eager")(x)

        all_metrics = torch._dynamo.utils.get_compilation_metrics()

        total_guards = sum(metric.guard_count for metric in all_metrics)
        self.assertLess(total_guards, AMT * 8)

        total_shape_env_guards = sum(
            metric.shape_env_guard_count for metric in all_metrics
        )
        self.assertLess(total_shape_env_guards, AMT * 8)

    # https://github.com/pytorch/pytorch/issues/118799
    def test_subclass_graph_output_repro(self):
        @torch._dynamo.allow_in_graph
        def to_subclass(x):
            return TwoTensor(x.clone(), x.clone())

        def f(x):
            tmp_subclass = to_subclass(x)
            return tmp_subclass.view(-1)

        x = torch.ones(2)
        out_ref = f(x)
        out_test = torch.compile(f, backend="aot_eager")(x)
        self.assertEqual(out_ref, out_test)

    def test_numpy_tobytes_no_error(self):
        def fn(x):
            x += 1
            z = x.tobytes()
            x += 1
            return z

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        opt_arg, arg = np.array([1, 2]), np.array([1, 2])
        self.assertEqual(opt_fn(opt_arg), fn(arg))
        self.assertEqual(cnt.frame_count, 2)

    def test_numpy_not_ndarray_recompiles(self):
        import torch

        def fn(x=None):
            if x is None:
                x = np.ones(3)
            elif isinstance(x, int):
                x = np.ones(6)
            elif isinstance(x, str):
                x = np.ones(9)
            return x**2

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)

        x = np.zeros((2, 2))

        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(opt_fn(), fn())
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(opt_fn(10), fn(10))
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(opt_fn("10"), fn("10"))
        self.assertEqual(cnt.frame_count, 4)

    @parametrize(
        "backend",
        ["eager", "aot_eager", "inductor"],
    )
    @parametrize(
        "func_name",
        ["func1", "func2", "func3"],
    )
    def test_tensor_set_data(self, backend, func_name):
        # https://github.com/pytorch/pytorch/issues/113030
        def func1(x, y):
            x.data = y
            x.add_(1)
            return x

        def func2(x, y):
            x.data = y
            y.data = torch.zeros([0])
            return x

        def func3(x, y):
            z = x
            x.data = y
            y.data = torch.zeros([0])
            return torch.tensor(x is z)

        funcs = {"func1": func1, "func2": func2, "func3": func3}
        func = funcs[func_name]

        if backend != "eager" and func is func1:
            # add_ not working w/ aot_autograd?
            return

        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        compiled_fn = torch.compile(func, backend=cnt, fullgraph=True)
        requires_grad = func is not func1
        for i in range(0, 5):
            # Inputs
            eager_a = torch.ones([6], requires_grad=requires_grad)
            compiled_a = torch.ones([6], requires_grad=requires_grad)

            eager_b = torch.ones([6], requires_grad=requires_grad)
            compiled_b = torch.ones([6], requires_grad=requires_grad)

            # Eager
            out_eager = func(eager_a, eager_b)
            # Compiled
            out_compiled = compiled_fn(compiled_a, compiled_b)
            self.assertEqual(eager_a, compiled_a)
            self.assertEqual(eager_b, compiled_b)
            self.assertTrue(torch.equal(out_eager, out_compiled))

            # func1 hits a leaf Variable that requires grad is being used in an in-place operation
            if requires_grad:
                bwd_inp_eager = torch.randn([6])
                bwd_inp_compiled = torch.clone(bwd_inp_eager)
                eager_a.backward(bwd_inp_eager)
                compiled_a.backward(bwd_inp_compiled)
                self.assertEqual(eager_a.grad, compiled_a.grad)

        # Prove guarding works - we run the compiled_fn 5 times
        # frame_count should stay at 1.
        self.assertEqual(cnt.frame_count, 1)

    @unittest.skipIf(
        TEST_WITH_ROCM or not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "flash attention not supported",
    )
    def test_flash_attn_backward_mixed_strides(self):
        # in this repro, "grad_out" and "value" are transposed tensors,
        # but "key" and "value" are contiguous
        def gen_inputs(device):
            return (
                torch.randn(
                    2, 513, 16, 64, dtype=torch.float16, device=device
                ).transpose(1, 2),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(
                    2, 513, 16, 64, dtype=torch.float16, device=device
                ).transpose(1, 2),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(2, 16, 513, device=device),
                None,
                None,
                513,
                513,
                0.0,
                False,
                torch.tensor(1, dtype=torch.int64),
                torch.tensor(1, dtype=torch.int64),
            )

        inps_cuda = gen_inputs("cuda")
        inps_meta = gen_inputs("meta")
        (
            out1_ref,
            out2_ref,
            out3_ref,
        ) = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            *inps_cuda, scale=0.125
        )
        from torch._meta_registrations import meta__scaled_dot_product_flash_backward

        out1_test, out2_test, out3_test = meta__scaled_dot_product_flash_backward(
            *inps_meta, scale=0.125
        )

        self.assertEqual(out1_ref.shape, out1_test.shape)
        self.assertEqual(out1_ref.stride(), out1_test.stride())
        self.assertEqual(out2_ref.shape, out2_test.shape)
        self.assertEqual(out2_ref.stride(), out2_test.stride())
        self.assertEqual(out3_ref.shape, out3_test.shape)
        self.assertEqual(out3_ref.stride(), out3_test.stride())

    def test_user_ctor_ctx_manager(self):
        class UserCtxManager:
            def __enter__(self):
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def fn(x, y):
            ucm = UserCtxManager()
            return x * x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(fn)
        x = torch.rand([2, 2])
        opt_fn(x, x)
        self.assertExpectedInline(cnt.frame_count, """1""")

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_arange_in_bounds(self):
        # see https://github.com/pytorch/pytorch/issues/113002
        class PaddingNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, lengths):
                max_seq_len = lengths.max().item()
                row_vector = torch.arange(0, max_seq_len, 1)
                matrix = torch.unsqueeze(lengths, dim=-1)
                mask = row_vector < matrix
                mask = mask.type(torch.float32)
                mask_3d_btd = mask[:, :, None]
                return mask_3d_btd

        model = PaddingNet()
        lengths = torch.tensor([5, 4, 4, 4], dtype=torch.int32)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(model)
        opt_fn(lengths)
        self.assertEqual(cnt.frame_count, 1)

    def test_overlapping_inputs_with_dynamic_shapes_error(self):
        @torch.compile(backend="aot_eager")
        def fn(a, b, c, d, e, f):
            a.mul_(2)
            b.mul_(2)
            c.mul_(2)
            d.mul_(2)
            e.mul_(2)
            f.mul_(2)

            base = torch.ones(2, 20)
            a = base[:, 0:2]
            b = base[:, 2:4]
            c = base[:, 4:6]
            d = base[:, 6:8]
            e = base[:, 8:10]
            f = base[:, 10:12]
            f2 = base[:, 10:14]
            out = fn(a, b, c, d, e, f)
            with self.assertRaisesRegex(
                AssertionError, "is being compiled with dynamic shapes"
            ):
                out2 = fn(a, b, c, d, e, f2)

    def test_user_ctor_ctx_manager_custom_init(self):
        class UserCtxManager:
            def __init__(self, x):
                x[0] = 10

            def __enter__(self):
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def fn(x, y):
            ucm = UserCtxManager(y)
            return x * y[0]

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(fn)
        x = torch.rand([2, 2])
        self.assertEqual(opt_fn(x, [5]), fn(x, [5]))
        self.assertExpectedInline(cnt.frame_count, """1""")

    def test_user_ctor_ctx_manager_custom_init_graph_break(self):
        counter = [0]

        class UserCtxManager:
            def __init__(self, k):
                k[0] += 1

            def __enter__(self):
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def fn(x, counter):
            x = x * x
            ucm = UserCtxManager(counter)
            return x * x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        x = torch.rand([2, 2])
        self.assertEqual(opt_fn(x, counter), fn(x, counter))
        self.assertEqual(counter[0], 2)
        for i in range(0, 10):
            opt_fn(x, counter)
        self.assertEqual(counter[0], 12)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")

    @unittest.expectedFailure
    def test_many_overlapping_inputs_does_not_explode_guards(self):
        from torch._dynamo.backends.common import aot_autograd

        # Before, this was (9702, 0)
        num_shape_guards = None
        num_aot_guards = None
        num_compiles = 0

        def guard_count_backend(gm, *args):
            nonlocal num_shape_guards
            nonlocal num_aot_guards
            nonlocal num_compiles
            num_shape_guards = len(
                torch._guards.TracingContext.try_get().fake_mode.shape_env.guards
            )
            num_aot_guards = len(
                torch._guards.TracingContext.try_get().guards_context.aotautograd_guards
            )
            num_compiles += 1
            return gm

        aot_guard_counter = aot_autograd(fw_compiler=guard_count_backend)

        @torch.compile(backend=aot_guard_counter, dynamic=True)
        def f(*args):
            for a in args:
                a.add_(1)

        x = torch.ones(1000, requires_grad=True)
        args = x.split(10)

        with torch.no_grad():
            f(*args)
        # In this example, there were 4950 guards (roughly (# tensors) ^ 2 // 2),
        # because every pair of aliased inputs needs a guard.
        self.assertTrue(num_aot_guards < 5000)
        # But there are no dynamic shape guards.
        self.assertEqual(num_shape_guards, 0)
        # don't recompile
        with torch.no_grad():
            f(*args)
        self.assertEqual(num_compiles, 1)

    def test_issue134451(self):
        class BoundingBox2DIndex(IntEnum):
            _X = 0
            _Y = 1
            _HEADING = 2
            _LENGTH = 3
            _WIDTH = 4

            @classmethod
            def size(cls):
                return 5

            @classmethod
            @property
            def X(cls):
                return cls._X

            @classmethod
            @property
            def Y(cls):
                return cls._Y

            @classmethod
            @property
            def HEADING(cls):
                return cls._HEADING

            @classmethod
            @property
            def LENGTH(cls):
                return cls._LENGTH

            @classmethod
            @property
            def WIDTH(cls):
                return cls._WIDTH

            @classmethod
            @property
            def POINT(cls):
                # assumes X, Y have subsequent indices
                return slice(cls._X, cls._Y + 1)

            @classmethod
            @property
            def STATE_SE2(cls):
                # assumes X, Y, HEADING have subsequent indices
                return slice(cls._X, cls._HEADING + 1)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._mlp_states = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, BoundingBox2DIndex.size()),
                )

            def forward(self, x):
                agent_states = self._mlp_states(x)
                agent_states[..., BoundingBox2DIndex.POINT] = (
                    agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
                )
                agent_states[..., BoundingBox2DIndex.HEADING] = (
                    agent_states[..., BoundingBox2DIndex.HEADING].tanh() * torch.pi
                )
                return agent_states

        model = SimpleModel().eval()
        input_tensor = torch.randn(1, 10, dtype=torch.float32)
        opt = torch.compile(model.eval(), backend="eager", fullgraph=True)
        actual = opt(input_tensor)
        try:
            expected = model(input_tensor)
        except Exception as e:
            raise unittest.SkipTest("eager failed, requires Python>=3.12") from e
        self.assertEqual(actual, expected)

    def test_invalid_seq_unpack(self):
        def myfn(arg):
            (a, b) = arg

        def fn():
            return myfn((1, 2, 3))

        try:
            torch.compile(fn)()
        except ValueError:
            pass
        else:
            self.fail("expected exception")

    def test_megablocks_moe(self):
        try:
            from megablocks.layers import moe
            from megablocks.layers.arguments import Arguments
        except ImportError as e:
            raise unittest.SkipTest("requires megablocks") from e
        bs, sl, hs, num_experts, top_k = (16, 1024, 512, 1, 1)
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            moe_num_experts=num_experts,
            moe_capacity_factor=1,
            moe_top_k=top_k,
        )
        moe_mlp = moe.MoE(args)
        moe_mlp.cuda(torch.cuda.current_device()).half()
        x = torch.randn(sl, bs, hs).cuda().half()
        out1, _ = moe_mlp(x)
        out2, _ = torch.compile(moe_mlp, backend="eager")(x)
        self.assertEqual(out1, out2)

    def test_udf_classes_reconstruction(self):
        def fn(x):
            o = T(5)
            return o.x + x

        opt_fn = torch.compile(fn, backend="eager")
        T = IncByOne

        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

        # This should recompile
        T = IncByTwo
        self.assertEqual(fn(x), opt_fn(x))

    def test_contains_range_constprop(self):
        def fn(x):
            # dynamo should const prop to False
            if 3 in range(0, 10):
                return x + 1
            else:
                return x + 2

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.zeros(4)
        self.assertEqual(fn(x), opt_fn(x))

    # https://github.com/pytorch/pytorch/issues/104505
    def test_as_strided_on_base_with_mutation_works(self):
        def foo(a):
            f = a.as_strided((2,), (1,), 0)
            f.add_(1.0)
            return a

        a = torch.randn(2, 4)
        a_ref = a.clone()
        out_ref = foo(a_ref)
        f_compiled = torch.compile(foo, backend="aot_eager")
        out = f_compiled(a)
        self.assertEqual(out_ref, out)
        self.assertEqual(a_ref, a)

    # https://github.com/pytorch/pytorch/issues/104505
    def test_as_strided_on_existing_view_banned(self):
        def foo(a):
            e = a.diagonal()
            f = e.as_strided((2,), (1,), 0)
            f.add_(1.0)
            return a

        a = torch.randn(2, 4)
        a_ref = a.clone()
        out_ref = foo(a_ref)
        f_compiled = torch.compile(foo, backend="aot_eager")
        with self.assertRaisesRegex(
            RuntimeError,
            "encountered a mutation on a view chain of length 2, where view 1 was an as_strided",
        ):
            out = f_compiled(a)

    def test_dont_aggressively_write_assert(self):
        record_graph = torch._dynamo.testing.EagerAndRecordGraphs()

        @torch.compile(dynamic=True, backend=record_graph)
        def f(x):
            assert x.shape[0] > 3
            assert x[0].sum() > 0
            assert 1 % (x.shape[0] // 2) != 0
            assert 32 * (x.shape[0] // 2) ** 2 - 16 * (x.shape[0] // 2) != 0
            return x.cos()

        f(torch.ones(6, 4))
        graph = record_graph.graphs[0]
        # It is bit annoying that we generate useless statements for
        # shape guards, but DCE should be able to remove them since t
        # there is no backed assert on them. The reason this is ok is
        # because dynamo will only skip the assert statement, but not
        # the instructions before it.
        self.assertExpectedInline(
            str(graph.code).strip(),
            """\
def forward(self, s0 : torch.SymInt, s1 : torch.SymInt, L_x_ : torch.Tensor):
    l_x_ = L_x_
    getitem_2 = l_x_[0]
    sum_1 = getitem_2.sum();  getitem_2 = None
    gt_1 = sum_1 > 0;  sum_1 = None
    _assert_async = torch._assert_async(gt_1, 'assertion error');  gt_1 = _assert_async = None
    cos = l_x_.cos();  l_x_ = None
    return (cos,)""",
        )
        for node in graph.graph.nodes:
            if "example_value" in node.meta and isinstance(
                node.meta["example_value"], torch._subclasses.fake_tensor.FakeTensor
            ):
                shape_env = node.meta["example_value"].fake_mode.shape_env
                lower_ranges = [val.lower for val in shape_env.var_to_range.values()]
                self.assertTrue(lower_ranges == [4, 2])

        @torch.compile(dynamic=True, backend=record_graph)
        def f_fail(x):
            assert x.shape[0] < 3

        # We graph-break here, so the failure should be eager
        with self.assertRaisesRegex(AssertionError, ""):
            f_fail(torch.ones(6, 4))

    def test_detectron2_instances_cat(self):
        class Instances:
            def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
                self._image_size = image_size
                self._fields: Dict[str, Any] = {}
                for k, v in kwargs.items():
                    self.set(k, v)

            @property
            def image_size(self) -> Tuple[int, int]:
                return self._image_size

            def __setattr__(self, name: str, val: Any) -> None:
                if name.startswith("_"):
                    super().__setattr__(name, val)
                else:
                    self.set(name, val)

            def __getattr__(self, name: str) -> Any:
                if name == "_fields" or name not in self._fields:
                    raise AttributeError(
                        f"Cannot find field '{name}' in the given Instances!"
                    )
                return self._fields[name]

            def __len__(self) -> int:
                for v in self._fields.values():
                    # use __len__ because len() has to be int and is not friendly to tracing
                    return v.__len__()
                raise NotImplementedError("Empty Instances does not support __len__!")

            def set(self, name: str, value: Any) -> None:
                with warnings.catch_warnings(record=True):
                    data_len = len(value)
                if len(self._fields):
                    assert (
                        len(self) == data_len
                    ), f"Adding a field of length {data_len} to a Instances of length {len(self)}"
                self._fields[name] = value

            def get(self, name: str) -> Any:
                return self._fields[name]

            @staticmethod
            def cat(instance_lists: List["Instances"]) -> "Instances":
                assert all(isinstance(i, Instances) for i in instance_lists)
                assert len(instance_lists) > 0
                if len(instance_lists) == 1:
                    return instance_lists[0]

                image_size = instance_lists[0].image_size
                if not isinstance(
                    image_size, torch.Tensor
                ):  # could be a tensor in tracing
                    for i in instance_lists[1:]:
                        assert i.image_size == image_size
                ret = Instances(image_size)
                for k in instance_lists[0]._fields.keys():
                    values = [i.get(k) for i in instance_lists]
                    v0 = values[0]
                    if isinstance(v0, torch.Tensor):
                        values = torch.cat(values, dim=0)
                    elif isinstance(v0, list):
                        values = list(itertools.chain(*values))
                    elif hasattr(type(v0), "cat"):
                        values = type(v0).cat(values)
                    else:
                        raise ValueError(
                            f"Unsupported type {type(v0)} for concatenation"
                        )
                    ret.set(k, values)
                return ret

        instances = [
            Instances((16, 16), a=torch.randn(16, 16), b=torch.randn(16, 16))
            for _ in range(3)
        ]

        @torch.compile(backend="eager", fullgraph=True)
        def fn(instances):
            return instances[0].cat(instances)

        actual = fn(instances)
        expected = instances[0].cat(instances)
        self.assertEqual(type(actual), type(expected))
        self.assertEqual(actual.__dict__, expected.__dict__)

    def test_weakref_construction(self):
        def fn(x, y):
            x_weak = weakref.ref(x)
            return x_weak() * y

        x = torch.randn(4)
        y = torch.randn(4)

        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_weakref(self):
        def fn(x_weak, weight, y):
            if x_weak is not None and x_weak() is not weight:
                return torch.sin(y)
            return torch.cos(y)

        weight = torch.randn(4)
        y = torch.randn(4)
        x_weak = weakref.ref(weight)

        ref = fn(x_weak, weight, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x_weak, weight, y)
        self.assertEqual(ref, res)

    def test_weakref_reconstruct(self):
        def fn(x_weak, weight, y):
            y = torch.sin(y)
            referent = x_weak()
            torch._dynamo.graph_break()
            if referent is not weight:
                return torch.sin(y)
            return torch.cos(y)

        weight = torch.randn(4)
        y = torch.randn(4)
        x_weak = weakref.ref(weight)

        ref = fn(x_weak, weight, y)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        res = opt_fn(x_weak, weight, y)
        self.assertEqual(ref, res)
        self.assertEqual(cnt.frame_count, 2)

    def test_weakref_del(self):
        def fn(x_weak, y):
            x = x_weak()
            if x is not None:
                return torch.sin(y)
            return torch.cos(y)

        weight = torch.randn(4)
        x_weak = weakref.ref(weight)
        y = torch.randn(4)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        ref = fn(x_weak, y)
        res = opt_fn(x_weak, y)
        self.assertEqual(ref, res)

        del weight
        gc.collect()
        ref = fn(x_weak, y)
        res = opt_fn(x_weak, y)
        self.assertEqual(ref, res)

    #     @torch._functorch.config.patch(
    #         recompute_views=True,
    #     )
    #     def test_storage_resize_forward_full_graph(self):
    #         class TestModule(torch.nn.Module):
    #             def __init__(self) -> None:
    #                 super().__init__()
    #                 self.param = torch.nn.Parameter(torch.randn(4, 4))

    #             def forward(self, x):
    #                 self.param.untyped_storage().resize_(
    #                     self.param.numel() * self.param.itemsize
    #                 )
    #                 with torch.no_grad():
    #                     torch._foreach_copy_([self.param], [x])
    #                 out = torch.matmul(self.param, self.param)
    #                 self.param.untyped_storage().resize_(0)
    #                 return out

    #         def post_accumulate_grad_hook(param):
    #             param.untyped_storage().resize_(0)

    #         # Beginning of backward, resize and put data into the param
    #         def pre_backward_hook(module, grad) -> None:
    #             module.param.untyped_storage().resize_(
    #                 self.param.numel() * self.param.itemsize
    #             )
    #             with torch.no_grad():
    #                 # simulates loading data into param from allgather
    #                 module.param.fill_(2)

    #         def post_forward_hook(module, args, output):
    #             output.register_hook(functools.partial(pre_backward_hook, module))

    #         x = torch.randn(4, 4)

    #         mod_ref = TestModule()
    #         mod_test = deepcopy(mod_ref)

    #         # Start the param off with zero storage size to mimic fsdp
    #         mod_ref.param.untyped_storage().resize_(0)
    #         mod_test.param.untyped_storage().resize_(0)

    #         # Resize storage at beginning of backward
    #         # Free storage at end of backward
    #         mod_ref.register_forward_hook(post_forward_hook, prepend=False)
    #         mod_ref.param.register_post_accumulate_grad_hook(post_accumulate_grad_hook)
    #         mod_test.register_forward_hook(post_forward_hook, prepend=False)
    #         mod_test.param.register_post_accumulate_grad_hook(post_accumulate_grad_hook)

    #         mod_test = torch.compile(mod_test, backend=aot_graph_capture_backend)

    #         out_ref = mod_ref(x)
    #         out_test = mod_test(x)
    #         self.assertExpectedInline(
    #             str(fw_graph[0].code.strip()),
    #             """\
    # def forward(self, primals_1, primals_2):
    #     _foreach_copy = torch.ops.aten._foreach_copy.default([primals_1], [primals_2]);  primals_1 = primals_2 = None
    #     getitem = _foreach_copy[0];  _foreach_copy = None
    #     mm = torch.ops.aten.mm.default(getitem, getitem)
    #     return [mm, getitem]""",
    #         )
    #         self.assertEqual(out_ref, out_test)

    def test_super_in_staticmethod(self):
        class A:
            @staticmethod
            def foo():
                return super().__init__()

        def fn(obj):
            return obj.foo()

        obj = A()

        try:
            fn(obj)
        except Exception as e:
            orig_str = str(e)
        self.assertIn("no arguments", orig_str)

        try:
            torch.compile(backend="eager")(fn)(obj)
        except Exception as e:
            compiled_str = str(e)
        self.assertEqual(orig_str, compiled_str)

    def test_super_staticmethod(self):
        class Parent:
            @staticmethod
            def greet():
                return 5

        class Child(Parent):
            @staticmethod
            def greet(x):
                return x * super(Child, Child).greet()

        child = Child()

        def fn(x):
            return child.greet(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_super_diamond(self):
        class A:
            def __init__(self):
                super().__init__()
                self.a = 5

        class Nothing:
            pass

        class B(Nothing, A):
            def __init__(self):
                super().__init__()
                self.b = 10

            def run(self, x):
                return self.a * self.b * x

        def fn(x):
            b = B()
            return b.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_vc_bumped_in_inference_graph(self):
        @torch.compile
        def f(x):
            return x.mul_(2)

        x = torch.randn(4)
        vc_before = x._version
        f(x)
        vc_after = x._version
        self.assertTrue(vc_after > vc_before)

    def test_nn_module_callable(self):
        class M(nn.Module):
            def forward(self, x):
                return x.sin()

        def f(m):
            return callable(m)

        res = torch.compile(f, fullgraph=True)(M())
        self.assertTrue(res)

    def test_stk_sdd_is_transposed(self):
        trigger_graph_break = False

        def _is_transposed(x):
            return (
                not x.is_contiguous()
                and x.stride()[0] == 1
                and x.stride()[1] == x.size()[0]
            )

        class SDD(torch.autograd.Function):
            @staticmethod
            def forward(ctx, lhs, rhs):
                ctx.save_for_backward(lhs, rhs)
                out = torch.full_like(lhs, 1.0, dtype=lhs.dtype, device=lhs.device)
                return out

            @staticmethod
            def backward(ctx, dy):
                saved_tensors = ctx.saved_tensors
                lhs, rhs = saved_tensors[:2]
                trans_a = _is_transposed(lhs)
                trans_b = _is_transposed(rhs)
                dlhs = None
                if ctx.needs_input_grad[0]:
                    dlhs = torch.full_like(lhs, 1.0 if trans_a else 2.0)
                drhs = None
                if ctx.needs_input_grad[1]:
                    drhs = torch.full_like(rhs, 1.0 if trans_b else 2.0)
                if trigger_graph_break:
                    if _is_transposed(dy):
                        return dlhs + 1, drhs + 1, None, None
                return dlhs, drhs, None, None

        x1 = torch.randn((8, 8), requires_grad=True)
        y1 = torch.randn((8, 8)).transpose(0, 1).requires_grad_(True)
        x2 = torch.randn((8, 8), requires_grad=True)
        y2 = torch.randn((8, 8)).transpose(0, 1).requires_grad_(True)

        SDD.apply(x1, y1).sum().backward()

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return SDD.apply(x2, y2)

        fn().sum().backward()

        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(y1.grad, y2.grad)

        trigger_graph_break = True
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            fn().sum().backward()

    def test_partially_initialized_module_property(self):
        class Matrix(torch.nn.Module):
            def __init__(self, data):
                super().__init__()
                self._data = data
                self.foo = 10 * self.blocking

            @property
            def data(self):
                return self._data

            @property
            def blocking(self):
                return self.data.shape[1]

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return Matrix(torch.randn(10, 20))

        v = fn()
        self.assertEqual(v.foo, 200)
        self.assertEqual(v.data.shape, (10, 20))
        self.assertEqual(type(v), Matrix)

    def test_classmethod_with_slots(self):
        class Mock:
            __slots__ = ("_a",)

            def __init__(self):
                self._a = 2

            @classmethod
            def _m(cls):
                return 3

            def run(self, x):
                return torch.sin(x) * self._a * self._m()

        def fn(x):
            mock = Mock()
            return mock.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_nn_parametrize(self):
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(10, 10))

            def forward(self, x):
                return self.param @ x

        class Parametrization(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        m = Module()
        torch.nn.utils.parametrize.register_parametrization(
            m, "param", Parametrization()
        )

        sin_found = False

        def backend(gm, _):
            nonlocal sin_found
            for node in gm.graph.nodes:
                if node.target is torch.sin:
                    sin_found = True
            return gm

        opt_m = torch.compile(m, backend=backend, fullgraph=True)
        inp = torch.randn(10, 10)
        self.assertEqual(m(inp), opt_m(inp))
        self.assertTrue(sin_found)

        torch.nn.utils.parametrize.remove_parametrizations(m, "param")
        sin_found = False
        self.assertEqual(m(inp), opt_m(inp))
        self.assertFalse(sin_found)

    def test_nn_module_property_closure(self):
        x = torch.randn(10, 10)

        class Mod(torch.nn.Module):
            @property
            def y(self):
                return torch.ones(10, 10) + x

            def forward(self, x):
                return x @ self.y

        mod = Mod()

        def fn(x):
            return mod(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        inp = torch.randn(10, 10)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_global_fn_mutation(self):
        def foo(x, y):
            return global_fn(x) + y

        x = torch.ones(1)
        y = torch.ones(1)

        opt = torch.compile(foo, fullgraph=True, backend="eager")
        self.assertEqual(opt(x, y), foo(x, y))

        # Change global_fn
        global global_fn

        def new_fn(x):
            return torch.cos(x)

        global_fn = new_fn
        self.assertEqual(opt(x, y), foo(x, y))

    # ref https://github.com/pytorch/pytorch/issues/123974
    def test_list_reverse(self):
        def ladder(x):
            trail = x.size(-1)
            assert trail > 2
            weights = []
            for s in [trail, trail - 1, trail - 2]:
                weights.append(torch.ones(s, s - 1))

            for w in weights:
                x = x @ w

            weights.reverse()

            for w in weights:
                x = x @ w.t()

            return x

        data = torch.randn(3, 4)
        opt_ladder = torch.compile(ladder, fullgraph=True, backend="eager")
        self.assertEqual(opt_ladder(data), ladder(data))

    def test_trace_functional_tensor_with(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        def f(a, tmp):
            a_view = a.view(-1)
            with torch.no_grad():
                a.set_(tmp)
                a_view.mul_(2)
            return a + tmp

        fake_mode = FakeTensorMode()
        with FunctionalTensorMode():
            inp = torch.ones(3, 3, requires_grad=True)
            inp = fake_mode.from_tensor(inp, static_shapes=True)
            inp = FunctionalTensor.to_functional(inp)

            tmp = torch.ones(3, 3, requires_grad=True)
            tmp = fake_mode.from_tensor(tmp, static_shapes=True)
            tmp = FunctionalTensor.to_functional(tmp)

            opt_f = torch.compile(f, backend="eager")
            with self.assertRaisesRegex(
                RuntimeError, "cannot mutate tensors with frozen storage"
            ):
                opt_f(inp, tmp)

    def test_const_dict_keyerror(self):
        d = {}

        def fn(x):
            try:
                y = d[0]
            except KeyError:
                y = 1
            return x + y

        opt_fn = torch.compile(fn, backend="eager")
        inp = torch.randn(3, 3)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_dict_tag_guard(self):
        class Foo:
            def __init__(self) -> None:
                self.scalar = 10

        def fn(d, x):
            return d["a"] * d["b"] * d["c"].scalar * x

        foo = Foo()

        d = {"a": 2, "b": 3, "c": foo}

        opt_fn = torch.compile(fn, backend="eager")
        inp = torch.randn(3, 3)
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

        d["a"] = 4
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

        # Check that recompilation happens
        foo.scalar = 12
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

    def test_nonconst_issubclass(self):
        def fn(x):
            if issubclass(x.__class__, np.ndarray):
                return 1
            return 0

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(np.ones([3, 3]))

    def test_issue126128(self):
        def fn():
            x = torch.randn(1, 10)
            y = torch.randn(10, 1)
            return torch.mm(x, y).sum()

        def fn2():
            x = torch.randn(10, 100)
            y = torch.randn(100, 10)
            return torch.mm(x, y).sum()

        with fresh_inductor_cache():
            torch.compile(fn)()

        torch.compile(fn2)()

    def test_jit_script_defaults(self):
        @torch.jit.script
        def fast_cos(x, c: float = 2.0):
            return torch.cos(x) * c

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fast_cos = fast_cos

            def forward(self, x):
                return self.fast_cos(x)

        mod = Mod()
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(mod(x), opt_mod(x))

    def test_enum(self):
        class ExplicitEnum(str, Enum):
            @classmethod
            def _missing_(cls, value):
                raise ValueError(
                    f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
                )

        class PaddingStrategy(ExplicitEnum):
            LONGEST = "longest"
            MAX_LENGTH = "max_length"
            DO_NOT_PAD = "do_not_pad"

        def fn(x):
            a = PaddingStrategy("longest")
            if a == PaddingStrategy.LONGEST:
                return torch.sin(x)
            return torch.cos(x)

        x = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_hasattr_builtin(self):
        class MyClass:
            foo: int = 1

        def func(x, m):
            if getattr(type(m), "foo", 0):
                return x + MyClass.foo
            return x

        opt_func = torch.compile(func, backend="eager", fullgraph=True)
        m = MyClass()
        x = torch.zeros(())
        self.assertEqual(func(x, m), opt_func(x, m))
        self.assertEqual(func(x, 0), opt_func(x, 0))

    def test_grad(self):
        def fn(x, y):
            x._grad = y
            return x.grad.data

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, y), opt_fn(x, y))

    def test_nn_module_stack_bc(self):
        from torch._dynamo.mutation_guard import GenerationTracker

        def compiler(gm, *args):
            module_stacks = [
                node.meta.get("nn_module_stack", None) for node in gm.graph.nodes
            ]
            module_stacks, _ = pytree.tree_flatten(module_stacks)
            module_stacks = [x for x in module_stacks if isinstance(x, str)]
            for stack in module_stacks:
                self.assertTrue("_module" not in stack)
            return gm.forward

        class SubMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.submod1 = SubMod()
                self.submod2 = SubMod()

            def forward(self, x):
                return self.submod1(x) + self.submod2(x)

        mod = Mod()
        opt_mod = torch.compile(mod, backend=compiler)
        opt_mod(torch.randn(2, 2))

        with torch._dynamo.config.patch(inline_inbuilt_nn_modules=True):
            mod = Mod()
            opt_mod = torch.compile(mod, backend=compiler)
            opt_mod(torch.randn(2, 2))

        # an example similar to Pippy usecase
        mod = Mod()
        GenerationTracker.tag(mod.submod1)
        GenerationTracker.mark_class_dynamic(type(mod.submod1))
        mod = Mod()
        opt_mod = torch.compile(mod, backend=compiler)
        opt_mod(torch.randn(2, 2))

    def test_is_make_fx_tracing(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            torch.nn.modules.activation._is_make_fx_tracing()
            return torch.sin(x)

        fn(torch.rand(4))

    def test_negative_floor_div_solve(self):
        class CompiledClass(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nums = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                self.t = 5

            def forward(self):
                self.num = self.nums[self.t // 12]
                self.t += 1
                return self.num

        m = CompiledClass()
        m = torch.compile(m, backend="eager")

        # the first call works
        m()
        # the second call causes a failure
        m()

    # https://github.com/pytorch/pytorch/issues/121621
    def test_tensor_random(self):
        def random_op(tensor, params):
            res = tensor.random_(**params)
            return res

        random_op = torch.compile(random_op)
        params = {"from": -10, "to": 10}
        tensor = torch.randn([2, 3])
        res = random_op(tensor, params)

    # https://github.com/pytorch/pytorch/issues/131019
    def test_tensor_uniform(self):
        def uniform_op(tensor, params):
            res = tensor.uniform_(**params)
            return res

        uniform_op = torch.compile(uniform_op)
        params = {"from": -10, "to": 10}
        tensor = torch.randn([2, 3])
        res = uniform_op(tensor, params)

    def test_data_attr_mutation_after_saved_for_bw(self):
        def f(x):
            out = x.sin()
            x.data.mul_(2)
            return out

        x = torch.randn(4, requires_grad=True)
        x_test = x.detach().clone().requires_grad_(True)

        out = f(x)
        out_test = torch.compile(f, backend="aot_eager")(x_test)
        self.assertEqual(out, out_test)

        out.sum().backward()
        out_test.sum().backward()
        self.assertEqual(x.grad, x_test.grad)

    # https://github.com/pytorch/pytorch/issues/128072
    def test_map_with_multiple_args(self):
        def f(a, b):
            return a[0] * b[0] + a[1] * b[1]

        def gen_inps(len_x, len_y):
            x = [torch.randn(5) for _ in range(len_x)]
            y = [torch.randn(5) for _ in range(len_y)]
            return x, y

        def g(x, y):
            return map(f, x, y)

        opt_g = torch.compile(g, fullgraph=True, backend="eager")

        inps = gen_inps(3, 3)
        self.assertEqual(type(g(*inps)), type(opt_g(*inps)))
        self.assertEqual(tuple(g(*inps)), tuple(opt_g(*inps)))

        inps = gen_inps(3, 5)
        self.assertEqual(type(g(*inps)), type(opt_g(*inps)))
        self.assertEqual(tuple(g(*inps)), tuple(opt_g(*inps)))

    def test_staticmethod_allow_in_graph(self):
        class MyClass:
            i = 3

            @staticmethod
            def foo_inner(x):
                return torch.mul(x, MyClass.i)

            # if dynamo inlines with fullgraph, will error
            # verify that dynamo doesn't inline
            @staticmethod
            @torch._dynamo.allow_in_graph
            def foo1(x):
                torch._dynamo.graph_break()
                return MyClass.foo_inner(x)

        @torch.compile(backend="eager", fullgraph=True)
        def f_bad(x):
            return MyClass.foo1(x)

        f_bad(torch.ones(2, 2))

    def test_guard_with_tuple_mutation(self):
        class Foo:
            def __init__(self) -> None:
                self.x = 10

        foo = Foo()
        d = {
            "a": 2,
            "b": (foo,),
        }

        def fn(x, d):
            return x * d["a"] * d["b"][0].x

        opt_fn = torch.compile(fn, backend="eager")
        inp = torch.randn(3, 3)
        self.assertEqual(fn(inp, d), opt_fn(inp, d))
        d["b"][0].x = 12
        self.assertEqual(fn(inp, d), opt_fn(inp, d))

    def test_compile_complex_conj(self):
        def f(x):
            return torch.mul(x, 2j)

        x_ref = torch.randn(4, 2, requires_grad=True)
        x_test = x_ref.detach().clone().requires_grad_(True)

        out_ref = f(torch.view_as_complex(x_ref))
        out_test = torch.compile(f, backend="aot_eager")(torch.view_as_complex(x_test))
        self.assertEqual(out_ref, out_test)

        torch.view_as_real(out_ref).sum().backward()
        torch.view_as_real(out_test).sum().backward()
        self.assertEqual(x_ref.grad, x_test.grad)

    # https://github.com/pytorch/pytorch/issues/132200
    def test_partitioner_cse_respects_mutation_boundaries(self):
        set_available = hasattr(torch.ops, "fsdp") and hasattr(torch.ops.fsdp, "set_")
        if not set_available:
            return

        @torch.compile(backend="aot_eager_decomp_partition")
        def f(x, l):
            # z0 and z1 can be CSEd
            z0 = x.sin()
            z1 = x.sin()
            y = x + 1
            torch.ops.fsdp.copy_.default(x, y)
            # z3 and z3 can be CSEd with each other,
            # but *not* with z0/z1 (they cross a mutation boundary)
            z2 = x.sin()
            z3 = x.sin()
            return z0, z1, z2, z3, l**2

        x = torch.randn(3)
        x_clone = x.clone()
        l = torch.randn(3, requires_grad=True)
        z0, z1, z2, z3, _ = f(x, l)

        # the partitioner runs CSE. We expect that of the 4 sin() ops above:
        # - the first 2 are CSE'd
        # - the last 2 are CSE'd
        # - the set_() op in the middle is a mutation barrier, preventing CSE
        self.assertEqual(z0, (x_clone).sin())
        self.assertEqual(z1, (x_clone).sin())
        self.assertEqual(z2, (x_clone + 1).sin())
        self.assertEqual(z3, (x_clone + 1).sin())

    # https://github.com/pytorch/pytorch/issues/132197
    def test_fsdp_set_input_mutation_applied_when_input_gets_no_gradients(self):
        set_available = hasattr(torch.ops, "fsdp") and hasattr(torch.ops.fsdp, "set_")
        if not set_available:
            return

        @torch.compile(backend="aot_eager_decomp_partition")
        def f(x, l):
            z = x.sin()
            y = x + 1
            # graph input has its storage mutated
            torch.ops.fsdp.copy_.default(x, y)
            z2 = x.sin()
            return z2, l**2

        x = torch.randn(3)
        x_test = x.clone()
        l = torch.randn(3, requires_grad=True)
        result, _ = f(x, l)
        result_test, _ = torch.compile(f, backend="aot_eager_decomp_partition")(
            x_test, l
        )

        self.assertEqual(result, result_test)
        self.assertEqual(x, x_test)

    def test_changing_stride(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x, y):
            return x * y

        for i in range(1, 4):
            x = torch.randn(4, i)

            # create a view for i > 1
            if i == 1:
                x1 = x
            else:
                x1 = x[:, 0:1]

            y = torch.randn(4, 1)
            print(x1.shape, y.shape)
            fn(x1, y)

        self.assertTrue(cnt.frame_count <= 2)

    @torch._dynamo.config.patch(guard_nn_modules=False)
    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    def test_inlining_cornercase(self):
        """
        nn.Modules can be mapped to either NNModuleVariable or UnspecializedNNModuleVariable. For NNModuleVariable, the
        tensor attributes become part of the Dynamo graph. For unspecialized, they are lifted as inputs.

        But there is a cornercase. Suppose you have NNModuleVariable with a submodule that is
        UnspecializedNNModuleVariable. Today, Dynamo will still consider the submodule as specialized (courtesy of
        guard.source().is_nn_module()). In retrospect, this is a mistake but there are dependencies of export and also
        cudagraphs which make it harder to fix the corner case right away. The long term solution is
        inline_inbuilt_nn_modules anyways, so we might have to live with this cornercase in the short term.

        We are starting to annotate the source of each nn module more precisely - NNModuleVariable attribute is marked
        as NNModuleSource, UnspecilaizedNNModuleVariable attribute is marked as UnspecializedNNModuleSource. But this
        changes the behavior for the cornercase. And fails some tests which have unfortunately relied on this behavior.


        To solve this, we tag the source only when inline_inbuilt_nn_module flag is turned on.

        In this test, we purposely turn the flag off, testing that the tagging is disabled.
        """

        class SubMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)
                self.a = torch.randn(1, 1)
                self.counter = 0
                self.multipliers = [2.2, 3.3]

            def forward(self, x):
                self.counter += 1
                return (
                    self.linear(x) * self.a * self.multipliers[0] * self.multipliers[1]
                )

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submod = SubMod()

            def forward(self, x):
                return self.submod(x)

        mod = Mod()
        opt_mod = torch.compile(mod, backend="eager")

        x = torch.randn(1, 1)
        ref = mod(x)
        res = opt_mod(x)

        mod.submod.multipliers = [3.3, 4.4]
        # Since guard_nn_modules is False, this will not recompile
        with torch._dynamo.config.patch(error_on_recompile=True):
            ref = mod(x)
            res = opt_mod(x)

    def test_optimized_module_training(self):
        mod = torch.nn.Linear(3, 3)
        mod.eval()

        opt_mod = torch.compile(mod, backend="eager")
        self.assertFalse(opt_mod.training)

        opt_mod.train()
        self.assertTrue(opt_mod.training)
        self.assertTrue(mod.training)

        mod.eval()
        self.assertFalse(opt_mod.training)

    @requires_cuda
    def test_memleak_when_graph_input_has_tensor_attr(self):
        @torch.compile(backend="eager")
        def f(x):
            x.add_(1)

        mem_before = torch.cuda.memory_allocated()

        x = torch.ones(2, device="cuda")
        x.foo = torch.zeros(2, device="cuda")
        f(x)
        del x.foo
        del x
        mem_after = torch.cuda.memory_allocated()
        self.assertEqual(mem_before, mem_after)

        # check when non-tensor data structure attribute contains a tensor
        @torch.compile(backend="eager")
        def f(x):
            x.add_(1)

        mem_before = torch.cuda.memory_allocated()
        x = torch.ones(2, device="cuda")
        x.foo = [torch.zeros(2, device="cuda") for _ in range(5)]
        f(x)
        del x.foo
        del x
        mem_after = torch.cuda.memory_allocated()
        self.assertEqual(mem_before, mem_after)

        # check with tensor refcycle
        @torch.compile(backend="eager")
        def g(x, y):
            return x + y

        mem_before = torch.cuda.memory_allocated()
        x = torch.ones(2, device="cuda")
        y = torch.zeros(2, device="cuda")
        x.foo = [y]
        y.foo = [x]
        g(x, y)
        del x.foo
        del y.foo
        del x
        del y
        mem_after = torch.cuda.memory_allocated()
        self.assertEqual(mem_before, mem_after)

    def test_os_fspath(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            os.fspath(".")
            return torch.sin(x)

        fn(torch.randn(4))

    @requires_cuda
    # This test will fail as flip in combination with particular input lenghts
    # produces weird results.
    # This is under investigations in
    # https://github.com/pytorch/pytorch/issues/131805
    @unittest.skip("Skip this flip test for the moment. It is under investigation")
    def test_flip_bad_accuracy(self):
        import torch
        import torch._dynamo.config
        import torch._functorch.config
        import torch._inductor.config
        import torch._inductor.inductor_prims
        import torch.fx.experimental._config

        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, arg0_1):
                rev = torch.ops.prims.rev.default(arg0_1, [0])
                arg0_1 = None
                slice_1 = torch.ops.aten.slice.Tensor(rev, 0, 0, -1, 2)
                slice_2 = torch.ops.aten.slice.Tensor(rev, 0, 1, 9223372036854775807, 2)
                add_1 = torch.ops.aten.add.Tensor(slice_1, slice_2)
                slice_1 = slice_2 = None
                slice_3 = torch.ops.aten.slice.Tensor(add_1, 0, 0, -1, 2)
                slice_4 = torch.ops.aten.slice.Tensor(
                    add_1, 0, 1, 9223372036854775807, 2
                )
                add_2 = torch.ops.aten.add.Tensor(slice_3, slice_4)
                slice_3 = slice_4 = None
                slice_5 = torch.ops.aten.slice.Tensor(add_2, 0, 0, -1, 2)
                slice_6 = torch.ops.aten.slice.Tensor(
                    add_2, 0, 1, 9223372036854775807, 2
                )
                add_3 = torch.ops.aten.add.Tensor(slice_5, slice_6)
                slice_5 = slice_6 = None
                slice_9 = torch.ops.aten.slice.Tensor(add_2, 0, 0, 1)
                add_2 = None
                unsqueeze = torch.ops.aten.unsqueeze.default(slice_9, 1)
                slice_9 = None
                unsqueeze_1 = torch.ops.aten.unsqueeze.default(add_3, 1)
                add_3 = None
                cat = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1], 1)
                unsqueeze = unsqueeze_1 = None
                view = torch.ops.aten.view.default(cat, [2])
                cat = None
                slice_10 = torch.ops.aten.slice.Tensor(view, 0, 0, -1)
                slice_11 = torch.ops.aten.slice.Tensor(
                    add_1, 0, 2, 9223372036854775807, 2
                )
                add_5 = torch.ops.aten.add.Tensor(slice_10, slice_11)
                slice_10 = slice_11 = None
                slice_12 = torch.ops.aten.slice.Tensor(add_1, 0, 0, 1)
                add_1 = None
                cat_1 = torch.ops.aten.cat.default([slice_12, add_5])
                slice_12 = add_5 = None
                unsqueeze_2 = torch.ops.aten.unsqueeze.default(cat_1, 1)
                cat_1 = None
                unsqueeze_3 = torch.ops.aten.unsqueeze.default(view, 1)
                view = None
                cat_2 = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], 1)
                unsqueeze_2 = unsqueeze_3 = None
                view_1 = torch.ops.aten.view.default(cat_2, [4])
                cat_2 = None
                slice_13 = torch.ops.aten.slice.Tensor(
                    rev, 0, 2, 9223372036854775807, 2
                )
                add_6 = torch.ops.aten.add.Tensor(view_1, slice_13)
                slice_13 = None
                slice_14 = torch.ops.aten.slice.Tensor(rev, 0, 0, 1)
                rev = None
                cat_3 = torch.ops.aten.cat.default([slice_14, add_6])
                slice_14 = add_6 = None
                constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                    view_1, [0, 1], 0.0
                )
                view_1 = None
                unsqueeze_4 = torch.ops.aten.unsqueeze.default(cat_3, 1)
                cat_3 = None
                unsqueeze_5 = torch.ops.aten.unsqueeze.default(constant_pad_nd, 1)
                constant_pad_nd = None
                cat_4 = torch.ops.aten.cat.default([unsqueeze_4, unsqueeze_5], 1)
                unsqueeze_4 = unsqueeze_5 = None
                view_2 = torch.ops.aten.view.default(cat_4, [10])
                cat_4 = None
                slice_15 = torch.ops.aten.slice.Tensor(view_2, 0, 0, 9)
                view_2 = None
                rev_1 = torch.ops.prims.rev.default(slice_15, [0])
                slice_15 = None
                return (rev_1,)

        mod = Repro()
        x = torch.arange(9, device=torch.device("cuda"))

        @torch.compile
        def f(x):
            return mod(x)

        out = f(x)
        self.assertEqual(torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0]), out[0])

    # https://github.com/pytorch/pytorch/issues/88813
    def test_return_value_duplication_tensor(self) -> None:
        def fn(val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return val * 2, val * 2

        x = torch.randn(2, requires_grad=True)

        expect = fn(x)
        self.assertNotEqual(
            expect[0].untyped_storage().data_ptr(),
            expect[1].untyped_storage().data_ptr(),
        )

        actual = torch.compile(fn, backend="aot_eager")(x)
        self.assertNotEqual(
            actual[0].untyped_storage().data_ptr(),
            actual[1].untyped_storage().data_ptr(),
        )

    # https://github.com/pytorch/pytorch/issues/114344
    def test_return_value_duplication_mixed_grad(self) -> None:
        def fn(val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                out0 = val + 1
            out1 = val + 1
            return out0, out1

        x = torch.randn(2, requires_grad=True)

        with torch.enable_grad():
            expect = fn(x)
            actual = torch.compile(fn, backend="aot_eager")(x)

            self.assertEqual(expect[0].requires_grad, actual[0].requires_grad)
            self.assertEqual(expect[1].requires_grad, actual[1].requires_grad)

    # https://github.com/pytorch/pytorch/pull/134726#discussion_r1738774371
    def test_return_value_duplication_scalar(self) -> None:
        def fn(val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x, y = val * 2, val * 2
            return x[0], y[0]

        x = torch.randn(2, requires_grad=True)

        expect = fn(x)
        self.assertNotEqual(
            expect[0].untyped_storage().data_ptr(),
            expect[1].untyped_storage().data_ptr(),
        )

        actual = torch.compile(fn, backend="aot_eager")(x)
        self.assertNotEqual(
            actual[0].untyped_storage().data_ptr(),
            actual[1].untyped_storage().data_ptr(),
        )

    def test_torch_compile_in_compile_frame(self):
        def gn(x, c=None):
            if c is None:
                c = 2
            return c * x

        def outer_func(x):
            return torch.compile(gn, backend="eager")(x)

        compile_outer = torch.compile(outer_func, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = outer_func(x)
        res = compile_outer(x)
        self.assertEqual(ref, res)

    # https://github.com/pytorch/pytorch/issues/136640
    def test_inductor_dynamic_shapes_broadcasting(self) -> None:
        def fn(x, y):
            x_view = x.view(-1, 4)
            y_view = y.view(-1, 4)
            return x_view * y_view

        x = torch.randn(4)
        y = torch.randn(8)
        out_ref = fn(x, y)
        out_test = torch.compile(fn, dynamic=True)(x, y)
        self.assertEqual(out_ref, out_test)

    # https://github.com/pytorch/pytorch/issues/119162
    def test_inductor_rng_default_dtype(self) -> None:
        @torch.compile
        def fn():
            tmp = torch.randn(4, 4, dtype=torch.bfloat16)
            return tmp

        try:
            old = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            out = fn()
        finally:
            torch.set_default_dtype(old)
        # output dtype should be float32
        self.assertEqual(out.dtype, torch.bfloat16)

    @unittest.skipIf(not HAS_MSGSPEC, "missing msgspec package")
    def test_c_defined_metaclass(self):
        class User(msgspec.Struct):
            """A new type describing a User"""

            name: str
            value: int

        def fn(x):
            u = User("alice", 10)
            return x * u.value

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x), opt_fn(x))

    @unittest.skipIf(not HAS_OMEGACONG, "missing omegaconf package")
    def test_omegaconf_dictconfig(self):
        def fn(cfg, x):
            a = cfg["foo"].a * x
            b = cfg.bar["b"] * a
            cfg.__dict__["baz"] = 4
            return b * cfg.baz

        config = OmegaConf.create({"foo": {"a": 3}, "bar": {"b": 5}})

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        ref = fn(config, x)
        cloned_config = copy.deepcopy(config)
        res = opt_fn(cloned_config, x)

        self.assertEqual(fn(config, x), opt_fn(config, x))
        self.assertEqual(cloned_config.baz, 4)

    # https://github.com/pytorch/pytorch/issues/136257
    def test_overwriting_params(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 2)
                self.fc2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        class ZeROOrderedDict(collections.OrderedDict):
            def __init__(self, parent_module=None, *args, **kwargs):
                """A replacement for ``collections.OrderedDict`` to detect external ZeRO params.

                Args:
                    parent_module (``collections.OrderedDict``): the collection to replace
                """

                super().__init__(*args, **kwargs)
                self._parent_module = parent_module

            def __getitem__(self, key):
                param = super().__getitem__(key)

                # Params can be registered as None (e.g., bias)
                if param is None:
                    return param

                # do something here
                return param

        def inject_parameters(module, cls):
            for module in module.modules():  # noqa: B020
                if cls == ZeROOrderedDict:
                    new_param = cls(parent_module=module)
                else:
                    new_param = cls()

                for key, param in module._parameters.items():
                    new_param[key] = param
                module._parameters = new_param

        model = M()

        inject_parameters(model, ZeROOrderedDict)

        model = torch.compile(model, backend="eager", fullgraph=True)

        x = torch.ones(2)
        with torch.no_grad():
            y = model(x)

    def test_typed_dict(self):
        class LlavaImagePixelInputs(TypedDict):
            type: Literal["pixel_values"]
            data: torch.Tensor
            """Shape: `(batch_size, num_channels, height, width)`"""

        def fn(x, y):
            obj = LlavaImagePixelInputs(type=int, data=y)
            out = x * obj["data"]
            obj["data"] = 3
            return out * obj["data"]

        x, y = torch.randn(4), torch.randn(4)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)

        self.assertEqual(ref, res)

    def test_typed_dict_total(self):
        class LlavaImagePixelInputs(TypedDict):
            type: Literal["pixel_values"]
            data: torch.Tensor
            """Shape: `(batch_size, num_channels, height, width)`"""

        def fn(x, y):
            obj = LlavaImagePixelInputs(data=y, total=False)
            return x * obj["data"]

        x, y = torch.randn(4), torch.randn(4)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)

        self.assertEqual(ref, res)

    @skipIfPy312  # listcomp bytecode is optimized
    def test_listcomp(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._num = 4

            @torch._dynamo.disable(recursive=False)
            def forward(self, x):
                values = [i * torch.cos(x) for i in range(self._num)]
                return sum(values)

        mod = Module()

        def fn(x):
            return mod(x)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnt.frame_count, 1)
        # Ensure that the listcomp is fully compiled
        self.assertEqual(cnt.op_count, 8)

    # https://github.com/pytorch/pytorch/issues/140266
    def test_distributions_subclass(self):
        import torch
        from torch.distributions import Categorical

        class SubCateg(Categorical):
            ...

        @torch.compile(backend="eager", fullgraph=True)
        def make_dist_and_execute(t, d):
            categ = d(logits=t)
            a = categ.log_prob(categ.sample()) + categ.probs + categ.logits
            return a

        for _ in range(2):
            make_dist_and_execute(torch.randn(10), SubCateg)

    def test_tensor_split_within_device_cm(self):
        @torch.compile(fullgraph=True)
        def split(x):
            return x.split(4, 0)

        x = torch.zeros(12)
        res = split(x)

        with torch.device("cpu"):
            self.assertEqual(res, split(x))

    def test_method_overriding(self):
        class DilateConv(torch.nn.Module):
            def __init__(
                self,
                dilate_func=None,
            ):
                super().__init__()
                self.dilate_func = dilate_func

            def forward(self, x):
                return self.dilate_func() * torch.sin(x)

        class MainModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = DilateConv(self.dilate_func)
                self.a = 4

            def dilate_func(self):
                return self.a

            def forward(self, x):
                return self.mod(x)

        mod = MainModule()

        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = mod(x)
        res = opt_mod(x)
        self.assertEqual(ref, res)


instantiate_parametrized_tests(ReproTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
