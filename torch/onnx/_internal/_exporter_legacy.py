# mypy: allow-untyped-defs
from __future__ import annotations


__all__ = [
    "enable_fake_mode",
]


import contextlib
import dataclasses
import logging
from typing import Any, TYPE_CHECKING

import torch
import torch._ops
from torch.onnx._internal.fx import patcher as patcher


# We can only import onnx from this module in a type-checking context to ensure that
# 'import torch.onnx' continues to work without having 'onnx' installed. We fully
# 'import onnx' inside of dynamo_export (by way of _assert_dependencies).
if TYPE_CHECKING:
    import io

    from torch._subclasses import fake_tensor

log = logging.getLogger(__name__)


@dataclasses.dataclass
class ONNXFakeContext:
    """A dataclass used to store context for model export using FakeTensor.

    This dataclass stores the FakeTensorMode instance used to convert
    real tensors and model parameters into fake tensors. This :attr:`ONNXFakeContext.fake_mode` is
    reused internally during tracing of a :class:`torch.nn.Module` into a FX :class:`GraphModule`.
    """

    fake_mode: fake_tensor.FakeTensorMode
    """The fake tensor mode used for tracing model using fake tensors and parameters."""

    state_dict_paths: tuple[str | io.BytesIO | dict[str, Any]] | None = None
    """List of paths of files that contain the model :meth:`state_dict`"""


@contextlib.contextmanager
def enable_fake_mode():
    """Enable fake mode for the duration of the context.

    Internally it instantiates a :class:`torch._subclasses.fake_tensor.FakeTensorMode` context manager
    that converts user input and model parameters into :class:`torch._subclasses.fake_tensor.FakeTensor`.

    A :class:`torch._subclasses.fake_tensor.FakeTensor`
    is a :class:`torch.Tensor` with the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a ``meta`` device. Because
    there is no actual data being allocated on the device, this API allows for
    initializing and exporting large models without the actual memory footprint needed for executing it.

    It is highly recommended to initialize the model in fake mode when exporting models that
    are too large to fit into memory.

    .. note::
        This function does not support torch.onnx.export(..., dynamo=True, optimize=True).
        Please call ONNXProgram.optimize() outside of the function after the model is exported.

    Example::

        # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> import torch
        >>> class MyModel(torch.nn.Module):  # Model with a parameter
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.weight = torch.nn.Parameter(torch.tensor(42.0))
        ...     def forward(self, x):
        ...         return self.weight + x
        >>> with torch.onnx.enable_fake_mode():
        ...     # When initialized in fake mode, the model's parameters are fake tensors
        ...     # They do not take up memory so we can initialize large models
        ...     my_nn_module = MyModel()
        ...     arg1 = torch.randn(2, 2, 2)
        >>> onnx_program = torch.onnx.export(my_nn_module, (arg1,), dynamo=True, optimize=False)
        >>> # Saving model WITHOUT initializers (only the architecture)
        >>> onnx_program.save(
        ...     "my_model_without_initializers.onnx",
        ...     include_initializers=False,
        ...     keep_initializers_as_inputs=True,
        ... )
        >>> # Saving model WITH initializers after applying concrete weights
        >>> onnx_program.apply_weights({"weight": torch.tensor(42.0)})
        >>> onnx_program.save("my_model_with_initializers.onnx")

    .. warning::
        This API is experimental and is *NOT* backward-compatible.

    """
    from torch._subclasses import fake_tensor
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    # This overrides the internal `FakeTensorMode` instance created by `torch._dynamo.export`[1].
    # It is a good idea to keep them in sync (constructor args) to maintain the same default behavior
    # [1] `torch/_dynamo/output_graph.py::InstructionTranslator::OutputGraph.__init__`
    # Mixed fake/real tensors are only allowed when `torch.onnx.dynamo_export` is not called within `FakeTensorMode`
    # This is needed because models can create new parameters during `forward(self, *args, **kwargs)` run
    fake_mode = fake_tensor.FakeTensorMode(
        allow_non_fake_inputs=not torch._guards.detect_fake_mode(),
        shape_env=ShapeEnv(
            allow_scalar_outputs=False, allow_dynamic_output_shape_ops=False
        ),
    )
    # The patcher is needed for when user calls `fake_model.load_state_dict(...)` within fake mode
    patcher_context = patcher.ONNXTorchPatcher()
    fake_context = ONNXFakeContext(fake_mode=fake_mode)
    with fake_mode, patcher_context:
        yield fake_context
    fake_context.state_dict_paths = tuple(
        patcher_context.paths,
    )  # type: ignore[assignment]
