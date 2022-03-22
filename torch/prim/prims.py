from typing import Sequence
from numbers import Number

import torch
from torch.overrides import (
    has_torch_function, has_torch_function_unary, handle_torch_function
)
from torch.prim.tracing import (
    TensorLikes, DimProxy, ShapeProxy, TensorProxy
)
import torch.prim.utils as utils
from torch.fx import wrap


# prim
# prim contains basic operations that always appear in traces

# Data movement operators

def convert_element_type_meta(t: TensorProxy, dtype: torch.dtype):
    # Type checks (not redundant with convert_element_type)
    assert isinstance(t, TensorProxy)

    output_name = t.ctx.tensor_name()
    node = t.ctx.graph.create_node(
        'call_function',
        convert_element_type,
        name=output_name,
        args=(t, dtype))
    t.node.users[node] = None

    return TensorProxy(t.ctx, node, name=output_name, tensor=t, dtype=dtype)

# TODO: comment
def convert_element_type(t, dtype: torch.dtype):
    # Type checks
    assert isinstance(t, TensorLikes)
    assert isinstance(dtype, torch.dtype)

    # Requires the conversion actually occur
    assert t.dtype is not torch.dtype

    # Meta/Proxy impl
    if isinstance(t, TensorProxy):
        return convert_element_type_meta(t, dtype)

    # ATen impl
    return t.to(dtype)

def device_put_meta(t: TensorProxy, device: torch.device):
    # Type checks (not redundant with device_put)
    assert isinstance(t, TensorProxy)

    output_name = t.ctx.tensor_name()
    node = t.ctx.graph.create_node(
        'call_function',
        device_put,
        name=output_name,
        args=(t, device))
    t.node.users[node] = None

    return TensorProxy(t.ctx, node, output_name, tensor=t, device=device)

# TODO: comment
def device_put(t, device):
    # Type checks
    assert isinstance(t, TensorLikes)
    assert isinstance(device, (str, torch.device))

    # Type conversion
    if isinstance(device, str):
        device = torch.device(device)

    # Requires the conversion actually occur
    assert t.device != device

    # Meta/Proxy impl
    if isinstance(t, TensorProxy):
        return device_put_meta(t, device)

    # ATen impl
    return t.to(device)

# Shape Ops

def expand_meta(t: TensorProxy, shape: Sequence):
    # Type checks (not redundant with expand)
    assert isinstance(t, TensorProxy)
    assert isinstance(shape, ShapeProxy)

    output_name = t.ctx.tensor_name()
    node = t.ctx.graph.create_node(
        'call_function',
        expand,
        name=output_name,
        args=(t, shape))
    t.node.users[node] = None
    shape.node.users[node] = None


    return TensorProxy(t.ctx, node, output_name, shape=shape, dtype=t.dtype, device=t.device)

# TODO: comment
def expand(t, shape: Sequence):
    # Type checking
    assert isinstance(t, TensorLikes)
    assert isinstance(shape, Sequence)

    if not utils.requires_broadcasting(t, shape):
        raise ValueError("Tensor was not expanded!")

    # Meta/Proxy impl
    if isinstance(t, TensorProxy):
        return expand_meta(t, shape)

    # ATen impl
    return t.expand(shape)


# Elementwise Binary Ops
def add_meta(a: TensorProxy, b: TensorProxy):
    # Type checks (not redundant with add)
    assert isinstance(a, TensorProxy)
    assert isinstance(b, TensorProxy)

    output_name = a.ctx.tensor_name()
    node = a.ctx.graph.create_node(
        'call_function',
        add,
        name=output_name,
        args=(a, b))
    a.node.users[node] = None
    b.node.users[node] = None

    return TensorProxy(a.ctx, node, output_name, tensor=a)

def add(a, b):
    # Type checks
    assert isinstance(a, TensorLikes)
    assert isinstance(b, TensorLikes)

    utils.elementwise_binary_checks(a, b)

    # Meta/Proxy impl
    if isinstance(a, TensorProxy):
        return add_meta(a, b)

    # ATen impl
    return torch.add(a, b)

