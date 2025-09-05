import abc
import builtins
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
    guard_manager: Optional[torch._dynamo.guards.GuardManagerWrapper]
    guards_state: bytes
    import_sources: dict[str, str]
    backend_id: str
    compiled_fn: SerializableCallable
    original_code: types.CodeType
    closure: Optional[tuple[Any, ...]]

    def compiled_function(self) -> Any:
        import_sources = {
            alias: importlib.import_module(module_name)
            for alias, module_name in self.import_sources.items()
        }
        f_globals = {**import_sources, self.backend_id: self.compiled_fn}
        core = types.FunctionType(self.bytecode, f_globals, closure=self.closure)

        def optimized_call(*args: Any, **kwargs: Any) -> Any:
            f_locals = bind_locals(self.signature, *args, **kwargs)
            assert self.guard_manager is not None
            if not self.guard_manager.check(f_locals):
                reason = str(self.guard_manager.check_verbose(f_locals))
                raise RuntimeError(f"GuardManager check failed, reason: {reason}")
            return core(*args, **kwargs)

        if self.guard_manager is None:
            guards_state = pickle.loads(self.guards_state)
            self.guard_manager = torch._dynamo.guards.CheckFunctionManager(
                self.original_code,
                guards_state.output_graph,
                shape_code_parts=guards_state.shape_code_parts,
                runtime_global_scope=f_globals,
            ).guard_manager

        def save_compiled_function(path: str) -> None:
            with open(path, "wb") as f:
                f.write(type(self).serialize(self))

        optimized_call.save_compiled_function = save_compiled_function  # type: ignore[attr-defined]
        return optimized_call

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
) -> Any:
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
        check_fn = dynamo_output.build_guards(
            fn.__code__, hooks=hooks, save=True, strict_error=True
        )
        assert check_fn.guards_state is not None

    backend_input = capture_output.backend_input
    output_graph = dynamo_output.tracer_output.output_graph
    assert output_graph is not None
    import_sources = output_graph.import_sources
    with torch._guards.tracing(TracingContext(backend_input.fake_mode)):
        compiled_fn = backend(backend_input.graph_module, backend_input.example_inputs)

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
        closure=fn.__closure__,
    )
    return compile_artifacts.compiled_function()
