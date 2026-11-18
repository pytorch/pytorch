"""
This module provides TVM backend integration for TorchDynamo.

Apache TVM is a deep learning compiler framework that can optimize and execute
models on various hardware backends. The backend compiles graphs through TVM's
relax frontend and can be used with torch.compile():

    model = torch.compile(model, backend="tvm")

Tuning and target selection are configured through a TVM pipeline:

    pipeline = tvm.relax.get_pipeline("static_shape_tuning", target="llvm", total_trials=2000)
    model = torch.compile(model, backend="tvm", options={"pipeline": pipeline})
"""

import importlib.util
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import torch
from torch import fx

from .common import fake_tensor_unsupported
from .registry import register_backend


@register_backend
@fake_tensor_unsupported  # type: ignore[arg-type]
def tvm(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    options: MappingProxyType[str, Any] | None = None,
) -> Callable[..., Any]:
    try:
        from tvm.relax.frontend.torch import relax_dynamo  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "Please install apache-tvm to use the tvm backend. "
            "See https://tvm.apache.org/docs/install/index.html for instructions."
        ) from e

    pipeline = options.get("pipeline", None) if options else None
    return relax_dynamo(pipeline=pipeline)(gm, example_inputs)


def has_tvm() -> bool:
    # avoid the heavy tvm import just to check availability
    return importlib.util.find_spec("tvm") is not None
