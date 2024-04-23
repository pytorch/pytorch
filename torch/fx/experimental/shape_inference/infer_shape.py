import copy
from collections import defaultdict

import torch
from torch._dynamo.source import LocalSource
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.shape_inference.infer_symbol_values import (
    infer_symbol_values,
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.utils import _pytree


"""
This is the function that runs shape inference. It will modify the input graph module so that shapes are annotated.
"""


def infer_shape(gm, input_tensors):
    # Prepare environments
    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)

    flatten_inputs, spec = _pytree.tree_flatten(input_tensors)
    dim_count = 1
    for input_tensor in flatten_inputs:
        dim_count += input_tensor.dim() - 1

    sample = {f"s{i}": 2 for i in range(dim_count)}
    init_symints = [
        mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)
        for k, v in sample.items()
    ]
    symints = copy.deepcopy(init_symints)
    symbol_to_idx_dict = {f"s{i}": i for i in range(dim_count)}
    padding_constraints = defaultdict(list)  # type: ignore[var-annotated]

    complete_flag = False
    allowed_try_times = dim_count * 2

    while not complete_flag and allowed_try_times > 0:
        # Create symbolic input tensors
        with fake_mode:
            sym_tensors = []
            i = 1
            for input_tensor in flatten_inputs:
                curr_dim = input_tensor.dim()
                desired_size = [symints[0]] + [
                    symints[ii] for ii in range(i, i + curr_dim - 1)
                ]
                sym_tensor = torch.randn(desired_size)
                sym_tensors.append(sym_tensor)
                i += curr_dim - 1
            sym_tensors = _pytree.tree_unflatten(sym_tensors, spec)
        try:
            with fake_mode:
                make_fx(
                    gm,
                    tracing_mode="symbolic",
                    _allow_non_fake_inputs=True,
                    pre_dispatch=True,
                    _allow_fake_constant=True,
                )(*sym_tensors)
            complete_flag = True
            return (gm, input_tensors, fake_mode, symints[0])
        except RuntimeError as e:
            if e:
                infer_symbol_values(
                    symints,
                    init_symints,
                    symbol_to_idx_dict,
                    padding_constraints,
                    str(e),
                )
                allowed_try_times -= 1
        except ValueError as e:
            if e:
                infer_symbol_values(
                    symints,
                    init_symints,
                    symbol_to_idx_dict,
                    padding_constraints,
                    str(e),
                )
                allowed_try_times -= 1


def mksym(shape_env, value, source, dynamic_dim):
    return shape_env.create_symintnode(
        shape_env.create_symbol(
            value,
            source=source,
            dynamic_dim=dynamic_dim,
        ),
        hint=value,
        source=source,
    )
