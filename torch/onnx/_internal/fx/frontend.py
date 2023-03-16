from __future__ import annotations

import abc
import copy
import functools

import inspect

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch._dynamo
import torch.fx
from torch._functorch import aot_autograd
from torch.fx.experimental import proxy_tensor
from torch.onnx._internal import _beartype

DecompositionTableType = Dict[Callable, Callable]


class FxFrontend(abc.ABC):
    """Base class for all FX frontends.

    Providing a common interface for all FX frontends. To allow onnx exporter to easily
    experiment with and swap between different frontends.
    """

    @_beartype.beartype
    def __init__(self, *config_args, **config_kwargs):
        """Initialize the frontend with config arguments."""
        ...

    @abc.abstractmethod
    @_beartype.beartype
    def _trace(self, model: Callable, *args, **kwargs) -> torch.fx.GraphModule:
        ...

    @_beartype.beartype
    def trace(self, model: Callable, *args, **kwargs) -> torch.fx.GraphModule:
        """Capture the model and return a torch.fx.GraphModule.

        The returned GraphModule also takes the same args and kwargs as the model.

        Args:
            model: The model to be captured.
            args: The args to be passed to the model.
            kwargs: The kwargs to be passed to the model.

        Returns:
            A torch.fx.GraphModule object that takes the same args and kwargs as the model.

        Example:

            >>> import torch
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
            >>> from torch.onnx._internal.fx import frontend
            >>> def model(x, y=2):
            >>>     return x + y
            >>> class CustomFrontend(frontend.FxFrontend):
            >>>     def _trace(self, model, *args, **kwargs):
            >>>         return torch.fx.symbolic_trace(model)
            >>>
            >>> fx_frontend = CustomFrontend()
            >>> graph_module = fx_frontend.trace(model, torch.randn(2), y=torch.randn(2))
            >>> fx_output = graph_module(torch.randn(3), y=torch.randn(3))
        """
        # args will be converted to symbolic tensor. Let's copy to avoid side effects.
        # FIXME: Can this copy be avoided?
        args = copy.deepcopy(args)
        kwargs = copy.deepcopy(kwargs)
        return self._trace(model, *args, **kwargs)

    @property
    def name(self) -> str:
        """Returns the name of the frontend class formatted with configurations."""
        return f"{self.__class__.__name__}"


class FxFrontendUnpackKwargs:
    """A wrapper class for FxFrontend to unpack kwargs.

    This class is used to unpack kwargs to the end of the args list. This is required by
    most fx tracing methods, except for dynamo export.
    """

    def __init__(self, fx_frontend: FxFrontend):
        self._fx_frontend = fx_frontend

    @classmethod
    def bind_args_kwargs(
        cls, model: Callable, *args, **kwargs
    ) -> inspect.BoundArguments:
        """Bind args and kwargs to the model signature.

        Apply default values to the bound kwargs, and expect all kwargs are mapped to
        bound.args after binding.

        Args:
            model: The model to be captured.
            args: The args to be passed to the model.
            kwargs: The kwargs to be passed to the model.

        Returns:
            A BoundArguments object that contains the bound args and kwargs.

        Raises:
            AssertionError: If bound.kwargs is not empty.
        """
        if isinstance(model, torch.nn.Module):
            signature = inspect.signature(model.forward)
        else:
            signature = inspect.signature(model)

        # We hope the input kwargs will be mapped to bound.args after binding.
        # If not, we will raise an error.
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        # kwargs are not handled.
        assert not bound.kwargs

        return bound

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_{self._fx_frontend.name}"

    @_beartype.beartype
    def trace(
        self, model: Callable, *args, **kwargs
    ) -> Tuple[torch.fx.GraphModule, Tuple[Any, ...]]:
        """Capture the model and return a tuple of torch.fx.GraphModule and formatted arguments.

        The new formatted arguments are constructed from the original args and kwargs,
        with kwargs unpacked to the end of the args. The returned GraphModule takes
        args in the new format.

        TODO: Obviously, it is not ideal to alter the original args and kwargs format.
        Most of the fx tracing methods used by fx onnx exporter today has this restriction.

        Args:
            model: The model to be captured.
            args: The args to be passed to the model.
            kwargs: The kwargs to be passed to the model.

        Returns:
            A tuple of torch.fx.GraphModule object and formatted arguments.

        Example:

            >>> import torch
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
            >>> from torch.onnx._internal.fx import frontend
            >>> def model(x, y=2):
            >>>     return x + y
            >>> class CustomFrontend(frontend.FxFrontend):
            >>>     def _trace(self, model, *args, **kwargs):
            >>>         return torch.fx.symbolic_trace(model)
            >>>
            >>> fx_frontend = frontend.FxFrontendUnpackKwargs(CustomFrontend())
            >>> graph_module, new_awrgs = fx_frontend.trace(model, torch.randn(2), y=torch.randn(2))
            >>> fx_output = graph_module(torch.randn(3), torch.randn(3))
        """
        bound = self.bind_args_kwargs(model, *args, **kwargs)
        graph_module = self._fx_frontend.trace(model, *bound.args)
        return graph_module, bound.args


class DynamoExport(FxFrontend):
    @_beartype.beartype
    def __init__(self, *, tracing_mode: str, aten_graph: bool):
        self.tracing_mode = tracing_mode
        self.aten_graph = aten_graph

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_{self.tracing_mode}_aten_graph_{self.aten_graph}"

    @_beartype.beartype
    def _trace(self, model: Callable, *args, **kwargs) -> torch.fx.GraphModule:
        torch._dynamo.reset()
        graph_module, guards = torch._dynamo.export(
            model,
            tracing_mode=self.tracing_mode,
            aten_graph=self.aten_graph,
            *args,
            **kwargs,
        )
        torch._dynamo.reset()

        assert graph_module is not None
        return graph_module


class _GraphCaptureCompiler:
    def __init__(self):
        self.captured_graph: Optional["torch.fx.GraphModule"] = None
        self.captured_graph_count = 0

    def compile(self, graph_module: "torch.fx.GraphModule", _):
        assert self.captured_graph_count == 0
        self.captured_graph = graph_module
        self.captured_graph_count += 1
        return graph_module


class AOTAutogradFrontend(FxFrontend):
    """Frontend based on 'aot_autograd.aot_module_simplified'.

    See Note [Fake Modules and AOTAutograd] in torch/_functorch/aot_autograd.py
    `aot_module_simplified` requires real tensors.
    """

    @_beartype.beartype
    def __init__(self, *, dynamic: bool = True):
        self.dynamic = dynamic

    @property
    def name(self) -> str:
        dynamic = "dynamic" if self.dynamic else "static"
        return f"{self.__class__.__name__}_{dynamic}"

    @_beartype.beartype
    def _trace(self, model: Callable, *args, **kwargs) -> torch.fx.GraphModule:
        compiler = _GraphCaptureCompiler()

        backend = functools.partial(
            aot_autograd.aot_module_simplified, fw_compiler=compiler.compile
        )

        torch._dynamo.reset()
        torch.compile(backend=backend, dynamic=self.dynamic, fullgraph=True)(model)(
            *args, **kwargs
        )
        torch._dynamo.reset()

        assert compiler.captured_graph is not None
        return compiler.captured_graph


class DynamoOptimize(FxFrontend):
    @_beartype.beartype
    def __init__(self, *, dynamic: bool):
        self.dynamic = dynamic

    @property
    def name(self) -> str:
        dynamic = "dynamic" if self.dynamic else "static"
        return f"{self.__class__.__name__}_{dynamic}"

    @_beartype.beartype
    def _trace(self, model: Callable, *args, **kwargs) -> torch.fx.GraphModule:
        compiler = _GraphCaptureCompiler()

        torch._dynamo.reset()
        torch._dynamo.optimize(compiler.compile, nopython=True, dynamic=self.dynamic)(
            model
        )(*args, **kwargs)
        torch._dynamo.reset()

        assert compiler.captured_graph is not None
        return compiler.captured_graph


class MakeFx(FxFrontend):
    @_beartype.beartype
    def __init__(
        self,
        *,
        tracing_mode: str,
        decomposition_table: Optional[DecompositionTableType] = None,
    ):
        self.tracing_mode = tracing_mode
        self.decomposition_table = decomposition_table

    @_beartype.beartype
    def _trace(self, model: Callable, *args, **kwargs) -> torch.fx.GraphModule:
        assert (
            not kwargs
        ), f"kwargs are not supported in {self.name}. Try wrapping this class with 'FxFrontendUnpackKwargs'"

        return proxy_tensor.make_fx(
            model,
            tracing_mode=self.tracing_mode,
            _allow_non_fake_inputs=True,
            decomposition_table=self.decomposition_table,
        )(*args)
