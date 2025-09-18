import abc
import builtins
import dataclasses
import importlib
import inspect
import logging
import pickle
import types
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
import torch.fx
from torch._dynamo.graph_utils import _graph_device_type
from torch._dynamo.precompile_context import SystemInfo

from . import convert_frame
from .hooks import Hooks


if TYPE_CHECKING:
    from .guards import GuardManagerWrapper
    from .package import SourceInfo


log = logging.getLogger(__name__)


class SerializableCallable(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def serialize_compile_artifacts(cls, fn: Any) -> bytes:
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> Any:
        pass


def bind_locals(
    signature: inspect.Signature, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return bound_arguments.arguments


@dataclass
class CompileArtifacts:
    signature: inspect.Signature
    bytecode: types.CodeType
    guard_manager: Optional["GuardManagerWrapper"]
    guards_state: bytes
    import_sources: dict[str, str]
    backend_id: str
    compiled_fn: SerializableCallable
    original_code: types.CodeType
    closure: Optional[tuple[Any, ...]]
    source_info: "SourceInfo"
    device_type: str
    system_info: SystemInfo = dataclasses.field(default_factory=SystemInfo.current)

    def check_compatibility(self) -> None:
        current_system = SystemInfo.current()
        current_system.check_compatibility(self.system_info, self.device_type)


@dataclass
class AOTCompiledFunction:
    _artifacts: CompileArtifacts

    def guard_check(self, *args: Any, **kwargs: Any) -> bool:
        f_locals = bind_locals(self._artifacts.signature, *args, **kwargs)
        assert self._artifacts.guard_manager is not None
        return self._artifacts.guard_manager.check(f_locals)

    def __post_init__(self) -> None:
        self._artifacts.check_compatibility()

        import_sources = {
            alias: importlib.import_module(module_name)
            for alias, module_name in self._artifacts.import_sources.items()
        }
        f_globals = {
            **import_sources,
            self._artifacts.backend_id: self._artifacts.compiled_fn,
        }
        self.fn = types.FunctionType(
            self._artifacts.bytecode, f_globals, closure=self._artifacts.closure
        )

        if self._artifacts.guard_manager is None:
            guards_state = pickle.loads(self._artifacts.guards_state)
            self._artifacts.guard_manager = torch._dynamo.guards.CheckFunctionManager(
                self._artifacts.original_code,
                guards_state.output_graph,
                shape_code_parts=guards_state.shape_code_parts,
                runtime_global_scope=f_globals,
            ).guard_manager

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        assert self._artifacts.guard_manager is not None
        if not self.guard_check(*args, **kwargs):
            f_locals = bind_locals(self._artifacts.signature, *args, **kwargs)
            reason = str(self._artifacts.guard_manager.check_verbose(f_locals))
            raise RuntimeError(f"GuardManager check failed, reason: {reason}")
        return self.fn(*args, **kwargs)

    def source_info(self) -> "SourceInfo":
        return self._artifacts.source_info

    def save_compiled_function(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(type(self).serialize(self))

    @classmethod
    def serialize(cls, fn: "AOTCompiledFunction") -> bytes:
        from torch._dynamo.package import SerializedCode

        state = fn._artifacts.__dict__.copy()
        state["guard_manager"] = None
        state["bytecode"] = SerializedCode.from_code_object(state["bytecode"])
        compiled_fn = state["compiled_fn"]
        state["compiled_fn"] = (
            type(compiled_fn).deserialize_compile_artifacts,
            type(compiled_fn).serialize_compile_artifacts(compiled_fn),
        )
        state["original_code"] = SerializedCode.from_code_object(state["original_code"])
        return pickle.dumps(state)

    @classmethod
    def deserialize(cls, data: bytes) -> "AOTCompiledFunction":
        from torch._dynamo.package import SerializedCode

        state = pickle.loads(data)
        state["bytecode"] = SerializedCode.to_code_object(state["bytecode"])
        deserializer, compiled_fn_state = state["compiled_fn"]
        state["compiled_fn"] = deserializer(compiled_fn_state)
        state["original_code"] = SerializedCode.to_code_object(state["original_code"])

        artifacts = CompileArtifacts(**state)
        return cls(artifacts)


class BundledAOTAutogradSerializableCallable(SerializableCallable):
    """
    Represents a serializable callable generated by compile_fx.
    This class wraps around the compiled function generated by AOTAutograd.

    TODO: Instead of using PrecompileContext to grab it from AOTAutograd,
    this object should be what's *returned* by aot_module_simplified.
    We'll do that refactor in a later PR.
    """

    def __init__(self, compiled_fn: Any) -> None:
        """
        Takes in a BundledAOTAutogradCacheArtifact, which is the serialized form
        of a compiled function generated by AOTAutograd.
        """
        assert hasattr(compiled_fn, "serialize")
        self.compiled_fn = compiled_fn

    def __getattr__(self, attr: Any) -> Any:
        if hasattr(self, attr):
            return getattr(super(), attr)
        else:
            return getattr(self.compiled_fn, attr)

    @classmethod
    def serialize_compile_artifacts(
        cls, fn: "BundledAOTAutogradSerializableCallable"
    ) -> bytes:
        with torch._functorch.config.patch("bundled_autograd_cache", True):
            result = pickle.dumps(fn.compiled_fn.serialize())
            return result

    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> Any:
        from torch._functorch._aot_autograd.autograd_cache import (
            deserialize_bundled_cache_entry,
        )

        compiled_fn = deserialize_bundled_cache_entry(data)
        return cls(compiled_fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.compiled_fn(*args, **kwargs)


def aot_compile_fullgraph(
    model: Any,
    example_inputs: tuple[tuple[Any, ...], dict[str, Any]],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
) -> AOTCompiledFunction:
    from torch._dynamo.guards import CheckFunctionManager
    from torch._dynamo.package import SourceInfo
    from torch._dynamo.utils import dynamo_timed, get_metrics_context
    from torch._guards import compile_context, CompileContext, TracingContext

    args, kwargs = example_inputs
    if hasattr(model, "__self__"):
        fn = model.__func__
        args = (model.__self__,) + args
    elif inspect.isfunction(model):
        fn = model
    else:
        raise RuntimeError(f"Unsupported model code type {model}")

    signature = inspect.signature(fn)
    f_locals = bind_locals(signature, *args, **kwargs)
    if fn.__code__.co_freevars or fn.__closure__:
        assert len(fn.__closure__) == len(fn.__code__.co_freevars)
        f_locals.update(
            {
                name: cell.cell_contents
                for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)
            }
        )

    with (
        compile_context(CompileContext(convert_frame.get_compile_id({}))),
        get_metrics_context(),
        dynamo_timed("fullgraph_capture"),
    ):
        capture_output = convert_frame.fullgraph_capture(
            convert_frame.FrameInfo(
                fn.__code__,
                fn.__globals__,
                f_locals,
                builtins.__dict__,
                closure=fn.__closure__ or (),  # type: ignore[arg-type]
            )
        )
        dynamo_output = capture_output.dynamo_output

        if not hooks.guard_filter_fn:
            from torch._dynamo.types import GuardFilterEntry

            def new_guard_filter_fn(
                guard_entries: list[GuardFilterEntry],
            ) -> list[bool]:
                return [
                    (
                        not (
                            g.is_global
                            or g.guard_type
                            in CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
                        )
                    )
                    for g in guard_entries
                ]

            hooks.guard_filter_fn = new_guard_filter_fn

        check_fn = dynamo_output.build_guards(
            fn.__code__, hooks=hooks, save=True, strict_error=True
        )

        assert check_fn.guards_state is not None

        backend_input = capture_output.backend_input
        assert backend_input is not None
        backend_input.graph_module._backend_id = backend_input.backend_id  # type: ignore[assignment]
        output_graph = dynamo_output.tracer_output.output_graph
        assert output_graph is not None
        device_type = _graph_device_type(output_graph.current_tracer.graph)
        import_sources = output_graph.import_sources
        with (
            torch._guards.tracing(TracingContext(backend_input.fake_mode)),
            torch._functorch.config.patch(
                {
                    "bundled_autograd_cache": True,
                    "force_non_lazy_backward_lowering": True,
                }
            ),
        ):
            compiled_fn = backend(
                backend_input.graph_module, backend_input.example_inputs
            )
            # If Inductor backend is used, grab the compiled_fn from PrecompileContext
            # TODO: this should be replaced once we make the backend return the SerializableCallable directly.
            if isinstance(backend, torch._TorchCompileInductorWrapper):
                compiled_fn = BundledAOTAutogradSerializableCallable(compiled_fn)

        if not isinstance(compiled_fn, SerializableCallable):
            if hasattr(backend, "compiler_fn"):
                compiler_fn = backend.compiler_fn
            else:
                compiler_fn = backend
            raise RuntimeError(
                f"Compiled function type {type(compiled_fn)} (produced "
                + f"from backend {compiler_fn}) does not implement SerializableCallable."
            )

        source_info = SourceInfo(inlined_sources=set())
        for traced_code in output_graph.traced_code:
            source_info.add_code(traced_code)

        artifacts = CompileArtifacts(
            signature=signature,
            bytecode=dynamo_output.bytecode,
            guard_manager=check_fn.guard_manager,
            guards_state=check_fn.guards_state,
            import_sources=import_sources,
            backend_id=backend_input.backend_id,
            compiled_fn=compiled_fn,
            original_code=fn.__code__,
            closure=fn.__closure__,
            source_info=source_info,
            device_type=device_type,
        )
        aot_compiled_fn = AOTCompiledFunction(_artifacts=artifacts)

    return aot_compiled_fn


@dataclass
class ModelInput:
    """
    WIP type: represents a single model input
    Which consists of a tuple of arguments and a set of contexts in which to run the model.

    For each ModelInput, we'll compile one full graph of the model, and then use the guards generated
    to dispatch between the compiled graphs.


    """

    args: tuple[Any]
    kwargs: dict[str, Any]
    contexts: list[AbstractContextManager[Any]]


@dataclass
class AOTCompiledModel:
    # Represents a single forward function of a model along with dispatch
    # compiled_results is serializable. We require the model to deserialize again.
    model: torch.nn.Module
    compiled_results: list[AOTCompiledFunction]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for result in self.compiled_results:
            if result.guard_check(self.model, *args, **kwargs):
                return result(self.model, *args, **kwargs)
        # All guards failed, just run one of them and throw the guard check error.
        return self.compiled_results[0](self.model, *args, **kwargs)

    def serialize(self) -> bytes:
        data: list[bytes] = []
        for result in self.compiled_results:
            data.append(AOTCompiledFunction.serialize(result))
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, model: torch.nn.Module, data: bytes) -> "AOTCompiledModel":
        from torch._dynamo.utils import get_metrics_context
        from torch._guards import compile_context, CompileContext

        results: list[bytes] = pickle.loads(data)
        compiled_results = []
        for result in results:
            with (
                compile_context(CompileContext(convert_frame.get_compile_id({}))),
                get_metrics_context(),
            ):
                compiled_results.append(AOTCompiledFunction.deserialize(result))
        return cls(model, compiled_results)


def aot_compile_module(
    model: torch.nn.Module,
    inputs: list[ModelInput],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
) -> AOTCompiledModel:
    """
    Compiles a single nn.Module with any number of inputs, and returns a compiled forward function.
    """

    def compile_single_graph(model_input: ModelInput) -> AOTCompiledFunction:
        example_inputs = (model_input.args, model_input.kwargs)
        orig_forward = model.forward
        with ExitStack() as stack:
            for ctx in model_input.contexts:
                stack.enter_context(ctx)
            return aot_compile_fullgraph(
                orig_forward,
                example_inputs,
                hooks=hooks,
                backend=backend,
            )

    compiled_results = []
    for model_input in inputs:
        log.info("Compiling input %s..", model_input)
        compiled_results.append(compile_single_graph(model_input))

    assert len(compiled_results) > 0

    return AOTCompiledModel(model, compiled_results)
