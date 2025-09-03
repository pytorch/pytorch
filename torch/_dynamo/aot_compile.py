import abc
import builtins
import contextlib
import importlib
import inspect
import pickle
import types
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.fx

from . import convert_frame
from .hooks import Hooks


class SerializableCallable(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def serialize_compile_artifacts(cls, fn: Any) -> bytes:
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> Any:
        pass


class BundledAOTAutogradSerializableCallable(SerializableCallable):
    def __init__(self, compiled_fn, backend_id=None, serialized=None):
        from torch._dynamo.precompile_context import PrecompileContext
        self.compiled_fn = compiled_fn
        if serialized is None:
            assert backend_id is not None
            bundled = PrecompileContext.serialize_artifact_by_key(backend_id)
            self.serialized = bundled.content
            self.compiled_fn = bundled.after_deserialization()

    def __getattr__(self, attr):
        if hasattr(self, attr):
            return getattr(super(), attr)
        else:
            return getattr(self.compiled_fn, attr)

    @classmethod
    def serialize_compile_artifacts(cls, fn: Any) -> bytes:
        return fn.serialized

    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> Any:
        from torch._functorch._aot_autograd.autograd_cache import BundledAOTAutogradCacheArtifact
        compiled_fn = BundledAOTAutogradCacheArtifact("key", data).after_deserialization()
        return cls(compiled_fn, data)

    def __call__(self, *args):
        return self.compiled_fn(*args)

def bind_locals(
    signature: inspect.Signature, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return bound_arguments.arguments


@dataclass
class ModelInput:
    args: tuple[Any]
    kwargs: dict[str, Any]
    contexts: list[contextlib.AbstractContextManager[Any]]

@dataclass
class CompileArtifacts:
    signature: inspect.Signature
    bytecode: types.CodeType
    guard_manager: Optional[torch._dynamo.guards.GuardManagerWrapper]
    guards_state: bytes
    import_sources: dict[str, str]
    backend_id: str
    compiled_fn: SerializableCallable
    original_code: types.CodeType


    def guard_check(self, *args: Any, **kwargs: Any) -> bool:
        f_locals = bind_locals(self.signature, *args, **kwargs)
        assert self.guard_manager is not None
        return self.guard_manager.check(f_locals)

    def __post_init__(self) -> None:
        import_sources = {
            alias: importlib.import_module(module_name)
            for alias, module_name in self.import_sources.items()
        }
        f_globals = {**import_sources, self.backend_id: self.compiled_fn}
        self.fn = types.FunctionType(self.bytecode, f_globals)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        f_locals = bind_locals(self.signature, *args, **kwargs)
        assert self.guard_manager is not None
        if not self.guard_manager.check(f_locals):
            reason = str(self.guard_manager.check_verbose(f_locals))
            raise RuntimeError(f"GuardManager check failed, reason: {reason}")
        return self.fn(*args, **kwargs)

    def save_compiled_function(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(type(self).serialize(self))


    @classmethod
    def serialize(cls, artifacts: "CompileArtifacts") -> bytes:
        from torch._dynamo.package import SerializedCode

        state = artifacts.__dict__.copy()
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
    def deserialize(cls, data: bytes) -> "CompileArtifacts":
        from torch._dynamo.package import SerializedCode

        state = pickle.loads(data)
        state["bytecode"] = SerializedCode.to_code_object(state["bytecode"])
        deserializer, compiled_fn_state = state["compiled_fn"]
        state["compiled_fn"] = deserializer(compiled_fn_state)
        state["original_code"] = SerializedCode.to_code_object(state["original_code"])
        return cls(**state)


def aot_compile_fullgraph(
    model: Any,
    example_inputs: tuple[tuple[Any, ...], dict[str, Any]],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
) -> CompileArtifacts:
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
                closure=(),  # type: ignore[arg-type]
            )
        )
        dynamo_output = capture_output.dynamo_output
        check_fn = dynamo_output.build_guards(fn.__code__, hooks=hooks, save=True)
        assert check_fn.guards_state is not None

    backend_input = capture_output.backend_input
    backend_input.graph_module._backend_id = backend_input.backend_id
    output_graph = dynamo_output.tracer_output.output_graph
    assert output_graph is not None
    import_sources = output_graph.import_sources
    with torch._guards.tracing(TracingContext(backend_input.fake_mode)):
        compiled_fn = backend(backend_input.graph_module, backend_input.example_inputs)

    # Inductor Backend
    if isinstance(backend, torch._TorchCompileInductorWrapper):
        compiled_fn = BundledAOTAutogradSerializableCallable(compiled_fn, backend_input.backend_id)

    if not isinstance(compiled_fn, SerializableCallable):
        if hasattr(backend, "compiler_fn"):
            compiler_fn = backend.compiler_fn
        else:
            compiler_fn = backend
        raise RuntimeError(
            f"Compiled function type {type(compiled_fn)} (produced "
            + f"from backend {compiler_fn}) does not implement SerializableCallable."
        )
    compile_artifacts = CompileArtifacts(
        signature=signature,
        bytecode=dynamo_output.bytecode,
        guard_manager=check_fn.guard_manager,
        guards_state=check_fn.guards_state,
        import_sources=import_sources,
        backend_id=backend_input.backend_id,
        compiled_fn=compiled_fn,
        original_code=fn.__code__,
    )
    return compile_artifacts
