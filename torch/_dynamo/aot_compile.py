import dataclasses
import importlib
import inspect
import io
import logging
import pickle
import types
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.fx
from torch._dynamo.graph_utils import _graph_device_type
from torch._dynamo.package import SystemInfo

from . import convert_frame
from .aot_compile_types import (
    BundledAOTAutogradSerializableCallable,
    SerializableCallable,
)
from .hooks import Hooks


if TYPE_CHECKING:
    from .guards import GuardManagerWrapper
    from .package import SourceInfo


log = logging.getLogger(__name__)


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
    argdefs: Optional[tuple[Any, ...]]
    source_info: "SourceInfo"
    device_type: str
    backend_name: str
    system_info: SystemInfo = dataclasses.field(default_factory=SystemInfo.current)

    def check_compatibility(self) -> None:
        current_system = SystemInfo.current()
        current_system.check_compatibility(self.system_info, self.device_type)


class AOTCompilePickler(pickle.Pickler):
    @classmethod
    def _unpickle_cell(cls, val: Any) -> Any:
        def _() -> Any:
            return val

        assert _.__closure__ is not None
        return _.__closure__[0]

    # pyrefly: ignore [bad-override]
    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, type((lambda x: lambda: x)(0).__closure__[0])):  # type: ignore[index] # noqa: PLC3002
            return type(self)._unpickle_cell, (obj.cell_contents,)
        return NotImplemented


@dataclass
class AOTCompiledFunction:
    _artifacts: CompileArtifacts
    _guard_check_enabled: bool = True

    def guard_check(self, *args: Any, **kwargs: Any) -> bool:
        f_locals: dict[str, Any] = {}
        if self._artifacts.closure:
            assert self._artifacts.bytecode.co_freevars and len(
                self._artifacts.closure
            ) == len(self._artifacts.bytecode.co_freevars)
            f_locals = {
                name: cell.cell_contents
                for name, cell in zip(
                    self._artifacts.bytecode.co_freevars, self._artifacts.closure
                )
            }
        f_locals.update(bind_locals(self._artifacts.signature, *args, **kwargs))
        assert self._artifacts.guard_manager is not None
        return self._artifacts.guard_manager.check(f_locals)

    def __post_init__(self) -> None:
        from .package import load_guard_manager, load_guards_state

        self._artifacts.check_compatibility()

        import_sources = {
            alias: importlib.import_module(module_name)
            for alias, module_name in self._artifacts.import_sources.items()
        }
        f_globals = {
            **import_sources,
            self._artifacts.backend_id: self._artifacts.compiled_fn,
        }
        # pyrefly: ignore [read-only]
        self.fn = types.FunctionType(
            self._artifacts.bytecode,
            f_globals,
            closure=self._artifacts.closure,
            argdefs=self._artifacts.argdefs,
        )

        if self._artifacts.guard_manager is None:
            guards_state = load_guards_state(self._artifacts.guards_state)
            self._artifacts.guard_manager = load_guard_manager(
                guards_state,
                self._artifacts.original_code,
                f_globals,
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        assert self._artifacts.guard_manager is not None
        if self._guard_check_enabled and not self.guard_check(*args, **kwargs):
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
        buf = io.BytesIO()
        pickler = AOTCompilePickler(buf)
        pickler.dump(state)
        return buf.getvalue()

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

    def disable_guard_check(self) -> None:
        self._guard_check_enabled = False


def aot_compile_fullgraph(
    model: Any,
    example_inputs: tuple[tuple[Any, ...], dict[str, Any]],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
) -> AOTCompiledFunction:
    from torch._dynamo.guards import CheckFunctionManager
    from torch._dynamo.package import SourceInfo
    from torch._dynamo.utils import dynamo_timed, get_metrics_context
    from torch._guards import TracingContext

    args, kwargs = example_inputs

    with (
        get_metrics_context(),
        dynamo_timed("fullgraph_capture"),
    ):
        capture_output = convert_frame.fullgraph_capture(model, args, kwargs)
        graph_capture_output = capture_output.graph_capture_output
        assert graph_capture_output.output_graph is not None

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

        fn, _ = convert_frame.get_traced_fn(model)
        check_fn = graph_capture_output.build_guards(
            fn.__code__, hooks=hooks, save=True, strict_error=True
        )

        assert check_fn.guards_state is not None

        backend_input = capture_output.backend_input
        assert backend_input is not None
        backend_input.graph_module._backend_id = backend_input.backend_id  # type: ignore[assignment]
        device_type = _graph_device_type(backend_input.graph_module.graph)
        tracing_context = TracingContext(backend_input.fake_mode)
        tracing_context.tensor_to_context = backend_input.tensor_to_context
        with (
            torch._guards.tracing(tracing_context),
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
        for traced_code in graph_capture_output.traced_code:
            source_info.add_code(traced_code)

        artifacts = CompileArtifacts(
            signature=convert_frame._get_signature(fn),
            bytecode=graph_capture_output.bytecode,
            guard_manager=check_fn.guard_manager,
            guards_state=check_fn.guards_state,
            import_sources=graph_capture_output.import_sources,
            backend_id=backend_input.backend_id,
            compiled_fn=compiled_fn,
            original_code=fn.__code__,
            closure=fn.__closure__,
            argdefs=fn.__defaults__,
            source_info=source_info,
            device_type=device_type,
            backend_name=getattr(backend, "compiler_name", "unknown"),
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
