"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_rewrite_assert_with_msg and test_rewrite_assert_without_msg)
"""

# Owner(s): ["module: dynamo"]
# This module intentionally re-exports shared imports for the split test shards.
import collections
import contextlib  # noqa: F401
import copy  # noqa: F401
import dataclasses  # noqa: F401
import functools  # noqa: F401
import gc  # noqa: F401
import importlib
import inspect
import itertools  # noqa: F401
import logging  # noqa: F401
import os  # noqa: F401
import random  # noqa: F401
import sys  # noqa: F401
import types  # noqa: F401
import typing  # noqa: F401
import unittest
import warnings  # noqa: F401
import weakref  # noqa: F401
from abc import ABC  # noqa: F401
from collections import defaultdict, namedtuple  # noqa: F401
from collections.abc import Iterator
from copy import deepcopy
from enum import Enum, IntEnum  # noqa: F401
from functools import wraps
from typing import Any, Literal, TypedDict  # noqa: F401
from unittest import mock  # noqa: F401

import numpy as np  # noqa: F401

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
import torch._functorch.config
import torch.distributed as dist  # noqa: F401
import torch.library
import torch.utils._pytree as pytree  # noqa: F401
from torch import nn
from torch._dynamo.backends.debugging import ExplainWithBackend  # noqa: F401
from torch._dynamo.debug_utils import same_two_models  # noqa: F401
from torch._dynamo.testing import (  # noqa: F401
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    expectedFailureDynamic,
    rand_strided,
    same,
    skipIfNotPy312,
    skipIfPy312,
)
from torch._inductor.utils import fresh_cache  # noqa: F401
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (  # noqa: F401
    AuxRequest,
    create_block_mask,
    flex_attention,
)
from torch.profiler import profile, ProfilerActivity  # noqa: F401
from torch.testing._internal.common_cuda import (  # noqa: F401
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FP8,
    SM70OrLater,
    TEST_CUDA,
)
from torch.testing._internal.common_device_type import (  # noqa: F401
    E4M3_MAX_POS,
    e4m3_type,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (  # noqa: F401
    instantiate_parametrized_tests,
    parametrize,
    serialTest,
    skipIfHpu,
    skipIfWindows,
    TEST_WITH_ROCM,
    xfailIfS390X,
)
from torch.testing._internal.logging_utils import (  # noqa: F401
    LoggingTestCase,
    make_logging_test,
)
from torch.testing._internal.two_tensor import TwoTensor  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: F401


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
else:
    msgspec = None


HAS_OMEGACONG = importlib.util.find_spec("omegaconf")
if HAS_OMEGACONG:
    from omegaconf import OmegaConf
else:
    OmegaConf = None

HAS_CUDA = torch.cuda.is_available()


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
    if not isinstance(tensors, (list, tuple)):
        raise AssertionError(f"Expected list or tuple, got {type(tensors)}")
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def shapes_to_tensor(x, device=None):
    # from detectron2 wrappers.py
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        if not all(isinstance(t, torch.Tensor) for t in x):
            raise AssertionError("Shape should be tensor during tracing!")
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
        if not (tensor.dim() == 2 and tensor.size(-1) == 4):
            raise AssertionError(
                f"Expected 2D tensor with last dim 4, got {tensor.size()}"
            )
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

        for layer in layers:
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
            if len(past_key_value) != 2:
                raise AssertionError(
                    f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
                )
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
    if len(input_tensors) == 0:
        raise AssertionError("Expected at least one input tensor")
    tensor_shape = input_tensors[0].shape[1]
    if not all(input_tensor.shape[1] == tensor_shape for input_tensor in input_tensors):
        raise AssertionError("All input tensors must have the same shape[1]")
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
        output, _ = self.saved_tensors
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

    def forward(self, x: torch.Tensor, concat_list: list[torch.Tensor]) -> torch.Tensor:
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
