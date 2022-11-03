# Owner(s): ["module: dynamo"]
import collections
import copy
import inspect
import itertools
import random
import unittest
from abc import ABC
from collections import namedtuple
from copy import deepcopy
from typing import List
from unittest.mock import patch

import numpy as np
import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch import nn
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.testing import rand_strided, requires_static_shapes, same
from torch.nn import functional as F

try:
    import torch._refs

    HAS_REFS = True
except ImportError:
    HAS_REFS = False


def ifdyn(count1, count2):
    if torch._dynamo.config.dynamic_shapes:
        return count1
    else:
        return count2


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
            [isinstance(t, torch.Tensor) for t in x]
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
        super(PartialT5, self).__init__()
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
        super(FakeMamlInner, self).__init__()
        self.linear = torch.nn.Linear(784, 5)

    def forward(self, x, ignored=None, bn_training=False):
        return self.linear(x.view(x.shape[0], -1))


class PartialMaml(torch.nn.Module):
    # Highly simplified version of maml.meta.Meta.finetuning
    def __init__(self):
        super(PartialMaml, self).__init__()
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
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters()))
        )

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


class ModelOutput(collections.OrderedDict):
    """based on file_utils.py in HuggingFace"""

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
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

    def __init__(self, *args):
        super(SequentialAppendList, self).__init__(*args)

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
        super(BatchNormAct2d, self).__init__(
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
    elif len(attn_types_set) == 2 and attn_types_set == set(["lsh", "local"]):
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
        super(FeedForwardLayer, self).__init__()
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
        super(TransformerEncoderLayer, self).__init__()
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


class TestModule(torch.nn.Module):
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
        # Graph break because of dynamic slicing
        self.assertEqual(
            torch._dynamo.utils.counters["frames"]["total"],
            torch._dynamo.utils.counters["frames"]["ok"] + 1,
        )

    @patch.object(torch._dynamo.config, "fake_tensor_propagation", True)
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
        self.assertEqual(cnt.frame_count, ifdyn(2, 4))
        self.assertEqual(cnt.op_count, ifdyn(9, 10))

    def test_boxes_len(self):
        def fn(boxes):
            return len(boxes) + boxes.__len__() + boxes.tensor

        boxes1 = Boxes(torch.arange(0, 8).reshape((2, 4)))
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(boxes1), boxes1.tensor + 4.0))

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, ifdyn(6, 1))

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

    def test_reformer_eval(self):
        with torch.no_grad():
            cnt = self._reformer(nopython=True)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 10)

    def test_reformer_train(self):
        with torch.enable_grad():
            cnt = self._reformer(nopython=False)
        # cant inline torch.autograd.Function means graph break
        self.assertEqual(cnt.frame_count, 4)
        self.assertEqual(cnt.op_count, 10)

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

        # Dyn recompiles are due to changes in hidden_state (Should we be guarding on this?)
        self.assertEqual(cnt.frame_count, ifdyn(4, 2))
        self.assertEqual(cnt.op_count, ifdyn(76, 4))

    def test_hf_t5_forward(self):
        input = torch.randn([1, 2048, 512])
        model = PartialT5()
        correct = model(input)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(input), correct))

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, ifdyn(13, 11))

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

        expected_ops = ifdyn(5, 4)
        expected_frame = ifdyn(1, 2)

        self.assertEqual(expected_ops, expected_ops)
        self.assertEqual(expected_frame, expected_frame)

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

    @requires_static_shapes
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
    @patch.object(torch._dynamo.config, "fake_tensor_propagation", False)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
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

        self.assertEqual(cnt.frame_count, ifdyn(3, 2))
        # TODO(jansel): figure out why op count depends on imports
        self.assertIn(cnt.op_count, (36, 35, 29, 28))

    # see: https://github.com/pytorch/pytorch/issues/80067
    @patch.object(torch._dynamo.config, "fake_tensor_propagation", False)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", False)
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

        self.assertEqual(cnt.frame_count, ifdyn(5, 4))
        # TODO(jansel): figure out why op count depends on imports
        self.assertIn(cnt.op_count, (31, 36, 35, 29, 28))

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

    @requires_static_shapes
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
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 8)

    # TODO: make set_rng_state work with FakeTensor/aot_autograd
    @patch.object(torch._dynamo.config, "fake_tensor_propagation", False)
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
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 4)  # rand, rand
        graph, _ = torch._dynamo.export(fn)

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
        if not torch._dynamo.config.specialize_int_float:
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

    @patch.object(torch._dynamo.config, "fake_tensor_propagation", True)
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
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_reformer_sorting(self):
        x = torch.zeros([1, 12, 4096], dtype=torch.int64)
        correct = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(x)
        fn = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(x), correct))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, ifdyn(28, 14))

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

        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["ok"], 3)
        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["total"], 3)

    def test_guard_fail_tensor_bool(self):
        @torch._dynamo.skip
        def fn():
            condition_shape = (5, 5)
            dtypes = (torch.bool,)
            shapes = (
                (),
                (5,),
                (1, 5),
            )

            tensors = list(
                [
                    torch.empty(shape, dtype=dtype).fill_(17)
                    for shape, dtype in itertools.product(shapes, dtypes)
                ]
            )

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

    @unittest.skipIf(not HAS_REFS, "requires recent PT version")
    @unittest.expectedFailure
    def test_primtorch(self):
        @torch._dynamo.optimize("eager", nopython=True)
        def fn(x):
            torch._refs.abs(x)

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

    def test_guard_ordering_shape_fail(self):
        # If a function which takes a tensor has an inner function which
        # is compiled and generates a guard on its shape,
        # they are evaluated in the wrong order. So if on a subsequent call
        # an int is passed instead of a tensor, guard evaluation will crash
        # with a "no attribute: shape" error
        m = TestModule()
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

    # AssertionError: ABCMeta
    @unittest.expectedFailure
    def test_isinstance_storage(self):
        @torch._dynamo.optimize("eager")
        def fn(x):
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            bools = torch.BoolStorage.from_buffer(f, "big")
            self.assertTrue(isinstance(bools, torch.BoolStorage))
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

    @patch.object(torch._dynamo.config, "fake_tensor_propagation", False)
    def test_specialized_stride(self):
        def f():
            e = torch.empty(4)
            x = e[::2]
            return x.stride()

        self.assertEqual(f(), torch._dynamo.optimize("eager")(f)())

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
            from . import test_functions as _  # noqa: F401

            def fn(x):
                from .test_functions import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        except ImportError:

            def fn(x):
                from test_functions import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        x = torch.randn(10)
        fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(fn)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_relative_import_no_modulename(self):
        try:
            from . import test_functions as _  # noqa: F401

            def fn(x):
                from . import test_functions

                return x * 2 * test_functions.tensor_for_import_testing

        except ImportError:

            def fn(x):
                import test_functions

                return x * 2 * test_functions.tensor_for_import_testing

        x = torch.randn(10)
        fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(fn)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    # This doesn't work without fake tensors but I don't care
    @patch.object(torch._dynamo.config, "fake_tensor_propagation", True)
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
        opt_mod = torch._dynamo.optimize("aot_inductor_debug")(mod)

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
                self.register_buffer("x", torch.ones(3))
                self.register_buffer("y", torch.ones(3))

            def forward(self, inp):
                res = 0
                for name, buffer in self.named_buffers():
                    res += buffer.sum()

                return inp.cos() + res

        mod = Foo()
        opt_mod = torch._dynamo.optimize("eager", nopython=True)(mod)
        args = (torch.randn(3, 4),)
        self.assertTrue(same(mod(*args), opt_mod(*args)))

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

    @patch.object(torch._dynamo.config, "rewrite_assert_with_torch_assert", True)
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

        with self.assertRaisesRegex(AssertionError, ""):
            exported, _ = torch._dynamo.export(f, torch.Tensor([4, 4, 5]))

    # TODO (tmanlaibaatar) handle data-dependent fstring in assert statement.
    @patch.object(torch._dynamo.config, "rewrite_assert_with_torch_assert", True)
    def test_rewrite_assert_with_fstring_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 3, f"First dim need to be {x[0]}"
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "generic_jump"):
            exported, _ = torch._dynamo.export(f, torch.Tensor([3, 4, 5]))

    @patch.object(torch._dynamo.config, "rewrite_assert_with_torch_assert", True)
    def test_rewrite_assert_without_msg(self):
        def f(x):
            b = x.sin()
            assert x[0] == 3
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        exported, _ = torch._dynamo.export(f, torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        with self.assertRaisesRegex(AssertionError, ""):
            exported, _ = torch._dynamo.export(f, torch.Tensor([4, 4, 5]))

    @patch.object(torch._dynamo.config, "rewrite_assert_with_torch_assert", True)
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

    @patch.object(torch._dynamo.config, "rewrite_assert_with_torch_assert", False)
    def test_not_rewrite_assert(self):
        def f(x):
            b = x.sin()
            assert x[0] == 3
            return x.cos() + b

        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "generic_jump"):
            torch._dynamo.export(f, torch.Tensor([3, 4, 5]))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
