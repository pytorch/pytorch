import dataclasses
import importlib
import inspect
import io
import logging
import os
import pickle
import tempfile
import types
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.fx
from torch._dynamo.convert_frame import GraphRuntimeEnv
from torch._dynamo.graph_utils import _graph_device_type
from torch._dynamo.package import emits_native_code, SystemInfo

from . import convert_frame
from .aot_compile_types import (
    BundledAOTAutogradSerializableCallable,
    SerializableCallable,
)
from .hooks import Hooks


if TYPE_CHECKING:
    from .guards import GuardManagerWrapper
    from .package import SerializedCode, SourceInfo


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
    guard_manager: Optional["GuardManagerWrapper"]
    guards_state: bytes
    backend_id: str
    compiled_fn: SerializableCallable
    original_code: types.CodeType
    runtime_env: GraphRuntimeEnv
    source_info: "SourceInfo"
    device_type: str
    backend_name: str
    system_info: SystemInfo = dataclasses.field(default_factory=SystemInfo.current)

    @property
    def emits_native_code(self) -> bool:
        from torch._dynamo.package import emits_native_code

        return emits_native_code(self.backend_name)

    def check_compatibility(self) -> None:
        # The CACHED info is the receiver, matching _DynamoCacheEntry: the skip
        # for an artifact predating cpu_codegen_target keys off self, and every
        # mismatch message labels self "cached" and the argument "current".
        # Determining the codegen target runs the C++ toolchain, so only pay for
        # it when this artifact actually records one to compare against.
        check_codegen = self.emits_native_code
        current = SystemInfo.current(
            cpu_codegen=(
                check_codegen
                and self.device_type == "cpu"
                and self.system_info.cpu_codegen_target is not None
            )
        )
        self.system_info.check_compatibility(
            current, self.device_type, check_codegen=check_codegen
        )


class AOTCompilePickler(pickle.Pickler):
    def __init__(self, external_data: dict[str, object], buf: io.BytesIO) -> None:
        super().__init__(buf)
        self.external_data = external_data
        self.id_map: dict[int, str] = {
            id(value): key for key, value in external_data.items()
        }
        self.errors = {}

    def persistent_id(self, obj: object) -> int | str | None:
        if id(obj) in self.id_map:
            return self.id_map[id(obj)]
        elif isinstance(obj, torch.nn.Module):
            self.errors[id(obj)] = obj
            return id(obj)
        else:
            return None

    @classmethod
    def _unpickle_cell(cls, val: object) -> object:
        def _() -> object:
            return val

        if _.__closure__ is None:
            raise AssertionError("closure must not be None")
        return _.__closure__[0]

    @classmethod
    # pyrefly: ignore [implicit-any]
    def _unpickle_bound_method(cls, func: Callable, base: object) -> types.MethodType:
        return types.MethodType(func, base)

    @classmethod
    def _unpickle_module(cls, name: str) -> types.ModuleType:
        return importlib.import_module(name)

    @classmethod
    def _unpickle_code(cls, serialized_code: "SerializedCode") -> types.CodeType:
        from torch._dynamo.package import SerializedCode

        return SerializedCode.to_code_object(serialized_code)

    @classmethod
    def _unpickle_nested_function(
        cls,
        code: types.CodeType,
        module: str,
        qualname: str,
        argdefs: tuple[object, ...] | None,
        closure: tuple[types.CellType, ...] | None,
    ) -> types.FunctionType:
        f_globals = importlib.import_module(module).__dict__
        return types.FunctionType(code, f_globals, qualname, argdefs, closure)

    # pyrefly: ignore [bad-override]
    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, type((lambda x: lambda: x)(0).__closure__[0])):  # type: ignore[index] # noqa: PLC3002
            return type(self)._unpickle_cell, (obj.cell_contents,)
        elif inspect.iscode(obj):
            from torch._dynamo.package import SerializedCode

            return type(self)._unpickle_code, (SerializedCode.from_code_object(obj),)

        elif inspect.ismodule(obj):
            return type(self)._unpickle_module, (obj.__name__,)
        elif inspect.ismethod(obj):
            """
            By default, pickle will call getattr() directly on the self object
            for pickling bounded methods, this is not what we want, instead we
            always want to serialize the original function and the self object
            in their original form.
            """
            func = obj.__func__
            method_self = obj.__self__
            inner_func = getattr(method_self, func.__name__)
            if inspect.ismethod(inner_func):
                inner_func = inner_func.__func__
            if func is not inner_func:
                return type(self)._unpickle_bound_method, (func, method_self)
        elif inspect.isfunction(obj):
            if "<locals>" in obj.__qualname__:
                return type(self)._unpickle_nested_function, (
                    obj.__code__,
                    obj.__module__,
                    obj.__qualname__,
                    obj.__defaults__,
                    obj.__closure__,
                )

        return NotImplemented


class AOTCompileUnpickler(pickle.Unpickler):
    def __init__(self, external_data: dict[str, object], file: io.BytesIO) -> object:
        super().__init__(file)
        self.external_data = external_data

    def persistent_load(self, key: str) -> object:
        if key not in self.external_data:
            raise RuntimeError(
                f"Missing required external reference to data: {key}. "
                "Please load AOT compiled function with "
                "`external_data=<external data dictionary>`"
                f"{self.external_data}"
            )
        return self.external_data[key]


@dataclass
class AOTCompileSaveResult:
    serialized_data: bytes


def atomic_write_binary(file_path: str, data: bytes):
    dir_name = os.path.dirname(file_path) or "."

    with tempfile.NamedTemporaryFile(
        dir=dir_name, delete=False, mode="wb"
    ) as temp_file:
        temp_path = temp_file.name
        temp_file.write(data)
        temp_file.flush()
        os.fsync(temp_file.fileno())

    os.replace(temp_path, file_path)


@dataclass
class AOTCompiledFunction:
    _artifacts: CompileArtifacts
    _guard_check_enabled: bool = True
    _extra_globals: dict[str, object] | None = None
    # Scope used ONLY to resolve guards, kept separate from _extra_globals so
    # that supplying it cannot rewire what the compiled bytecode reads. It is
    # held by reference, not copied, so guards track the loading process's
    # globals as they change.
    _guard_globals: dict[str, object] | None = None

    def prepare_f_locals(self, *args: object, **kwargs: object) -> dict[str, object]:
        f_locals: dict[str, object] = {}
        env = self._artifacts.runtime_env
        if env.closure:
            if not env.bytecode.co_freevars or len(env.closure) != len(
                env.bytecode.co_freevars
            ):
                raise AssertionError("closure length must match co_freevars length")
            f_locals = {
                name: cell.cell_contents
                for name, cell in zip(env.bytecode.co_freevars, env.closure)
            }
        f_locals.update(bind_locals(self._artifacts.signature, *args, **kwargs))
        return f_locals

    def guard_check(self, *args: Any, **kwargs: Any) -> bool:
        f_locals = self.prepare_f_locals(*args, **kwargs)
        if self._artifacts.guard_manager is None:
            raise AssertionError("guard_manager must not be None")
        return self._artifacts.guard_manager.check(f_locals)

    def __post_init__(self) -> None:
        from .package import load_guard_manager, load_guards_state

        self._artifacts.check_compatibility()

        self.fn = self._artifacts.runtime_env.forward_callable(
            self._artifacts.backend_id,
            self._artifacts.compiled_fn,
            extra_globals=self._extra_globals,
        )

        if self._artifacts.guard_manager is None:
            guards_state = load_guards_state(self._artifacts.guards_state)
            # The guard manager keeps this dict by reference, so handing it the
            # loading process's scope rather than a copy is what makes a global
            # rebound after load redirect dispatch instead of silently serving
            # whichever graph matched at load time. There is deliberately no
            # fallback to the serialized scope: a name the loading process does
            # not have must fail the guard, not resolve against the value baked
            # into the artifact. The graph's own globals stay as serialized.
            guard_scope = self._guard_globals
            if guard_scope is None:
                guard_scope = self.fn.__globals__
            else:
                # Seed the artifact's own __import_* aliases. Dynamo MINTS those
                # at trace time (symbolic_convert.import_source writes them into
                # the tracing process's globals) and roots guards at them -- every
                # child nn.Module call guards its hook dicts through
                # G['__import_torch_dot_nn_dot_modules_dot_module']. A process
                # that only LOADS never traced, so its module dict has none and
                # the guard KeyErrors on every call. setdefault, so only the
                # synthetic names are added: a real global the loading process
                # already has keeps its live value, which is the point of using
                # the live scope at all.
                for (
                    alias,
                    module_name,
                ) in self._artifacts.runtime_env.import_sources.items():
                    guard_scope.setdefault(alias, importlib.import_module(module_name))
            self._artifacts.guard_manager = load_guard_manager(
                guards_state,
                self._artifacts.original_code,
                guard_scope,
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._artifacts.guard_manager is None:
            raise AssertionError("guard_manager must not be None")
        if self._guard_check_enabled and not self.guard_check(*args, **kwargs):
            f_locals = self.prepare_f_locals(*args, **kwargs)
            reason = str(self._artifacts.guard_manager.check_verbose(f_locals))
            raise RuntimeError(f"GuardManager check failed, reason: {reason}")
        return self.fn(*args, **kwargs)

    def source_info(self) -> "SourceInfo":
        return self._artifacts.source_info

    def save_compiled_function(
        self, path: str, external_data: dict[str, Any] | None = None
    ) -> AOTCompileSaveResult:
        result = type(self).serialize(self, external_data)
        atomic_write_binary(path, result.serialized_data)
        return result

    @classmethod
    def serialize(
        cls, fn: "AOTCompiledFunction", external_data: dict[str, Any] | None = None
    ) -> AOTCompileSaveResult:
        from torch._dynamo.package import SerializedCode

        state = fn._artifacts.__dict__.copy()
        state["guard_manager"] = None
        state["runtime_env"] = dataclasses.replace(
            state["runtime_env"],
            bytecode=SerializedCode.from_code_object(state["runtime_env"].bytecode),
        )
        compiled_fn = state["compiled_fn"]
        state["compiled_fn"] = (
            type(compiled_fn).deserialize_compile_artifacts,
            type(compiled_fn).serialize_compile_artifacts(compiled_fn),
        )
        state["original_code"] = SerializedCode.from_code_object(state["original_code"])
        buf = io.BytesIO()
        pickler = AOTCompilePickler(external_data or {}, buf)
        pickler.dump(state)
        if pickler.errors:
            raise RuntimeError(
                f"Failed to serialize the following objects: {list(pickler.errors.values())}\n"
                "Please mark these as external data by using `external_data={'key': ...}`"
            )
        return AOTCompileSaveResult(serialized_data=buf.getvalue())

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        f_globals: dict[str, object] | None = None,
        external_closure_data: dict[str, Any] | None = None,
        guard_globals: dict[str, object] | None = None,
    ) -> "AOTCompiledFunction":
        from torch._dynamo.package import SerializedCode

        f = io.BytesIO(data)
        f.seek(0)
        unpickler = AOTCompileUnpickler(external_closure_data or {}, f)
        state = unpickler.load()
        f.close()
        state["runtime_env"] = dataclasses.replace(
            state["runtime_env"],
            bytecode=SerializedCode.to_code_object(state["runtime_env"].bytecode),
        )
        deserializer, compiled_fn_state = state["compiled_fn"]
        with torch._inductor.config.patch(enable_autograd_for_aot=True):
            state["compiled_fn"] = deserializer(compiled_fn_state)
        state["original_code"] = SerializedCode.to_code_object(state["original_code"])

        artifacts = CompileArtifacts(**state)
        return cls(artifacts, _extra_globals=f_globals, _guard_globals=guard_globals)

    def disable_guard_check(self) -> None:
        self._guard_check_enabled = False


def aot_compile_fullgraph(
    model: Any,
    example_inputs: tuple[tuple[Any, ...], dict[str, Any]],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
    dynamic: bool | None = None,
) -> AOTCompiledFunction:
    from torch._dynamo.guards import CheckFunctionManager
    from torch._dynamo.package import SourceInfo
    from torch._dynamo.utils import dynamo_timed, get_metrics_context
    from torch._dynamo.variables.torch_function import (
        torch_function_mode_stack_state_mgr,
    )
    from torch._guards import TracingContext

    args, kwargs = example_inputs

    dynamic_ctx = nullcontext()
    if dynamic is not None:
        from torch._dynamo.eval_frame import set_enable_dynamic

        dynamic_ctx = set_enable_dynamic(dynamic)

    with (
        get_metrics_context(),
        dynamo_timed("fullgraph_capture"),
        torch._functorch.config.patch(strict_autograd_cache=True),
        dynamic_ctx,
        torch_function_mode_stack_state_mgr,
    ):
        capture_output = convert_frame.fullgraph_capture(model, args, kwargs)
        graph_capture_output = capture_output.graph_capture_output
        if graph_capture_output.output_graph is None:
            raise AssertionError("output_graph must not be None")

        if not hooks.guard_filter_fn:
            from torch._dynamo.types import GuardFilterEntry

            def new_guard_filter_fn(
                guard_entries: Sequence[GuardFilterEntry],
            ) -> Sequence[bool]:
                # NB: dropping every global guard is what
                # torch.compiler.skip_guard_on_globals_unsafe does explicitly,
                # and the "unsafe" in that name applies here too: a dropped
                # global guard does not fail, it silently reuses a graph traced
                # under a different global value. Narrowing this default needs
                # guard construction to resolve arbitrary global references
                # first (today they can raise KeyError on G['...']), so callers
                # who need a specific global guarded must pass guard_filter_fn.
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

        backend_input = capture_output.backend_input
        if backend_input is None:
            raise AssertionError("backend_input must not be None")
        backend_input.graph_module._backend_id = backend_input.backend_id  # type: ignore[assignment]
        device_type = _graph_device_type(backend_input.graph_module.graph)
        if (
            backend_input.fake_mode.shape_env
            is not graph_capture_output.output_graph.shape_env
        ):
            raise AssertionError(
                "fake_mode.shape_env must be the same as output_graph.shape_env"
            )
        tracing_context = TracingContext(backend_input.fake_mode)
        tracing_context.tensor_to_context = backend_input.tensor_to_context
        with (
            torch._guards.tracing(tracing_context),
            torch._functorch.config.patch(
                {
                    "strict_autograd_cache": True,
                    "bypass_autograd_cache_key": True,
                    "bundled_autograd_cache": True,
                    "force_non_lazy_backward_lowering": True,
                    "force_autograd_cache": True,
                }
            ),
        ):
            compiled_fn = backend(
                backend_input.graph_module, backend_input.example_inputs
            )
            # If Inductor backend or AOTAutograd-based backend is used,
            # wrap the compiled_fn for serialization.
            # TODO: this should be replaced once we make the backend return the SerializableCallable directly.
            if (
                isinstance(backend, torch._TorchCompileInductorWrapper)
                or (
                    hasattr(backend, "compiler_fn")
                    and isinstance(
                        backend.compiler_fn, torch._dynamo.backends.common.AotAutograd
                    )
                )
                or (
                    hasattr(compiled_fn, "serialize")
                    and compiled_fn.serialize is not None
                )
            ):
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

        # Temporarily restore the mode stack so guard expressions that
        # reference modes can evaluate, matching the compile_inner path.
        build_guards_ctx = ExitStack()
        if torch_function_mode_stack_state_mgr.stack:
            build_guards_ctx.enter_context(
                torch_function_mode_stack_state_mgr.temp_restore_stack()
            )
        with build_guards_ctx:
            check_fn = graph_capture_output.build_guards(
                fn.__code__, hooks=hooks, save=True, strict_error=True
            )

        if check_fn.guards_state is None:
            raise AssertionError("guards_state must not be None")

        source_info = SourceInfo(inlined_sources=set())
        for traced_code in graph_capture_output.traced_code:
            source_info.add_code(traced_code)

        backend_name = getattr(backend, "compiler_name", "unknown")
        artifacts = CompileArtifacts(
            signature=convert_frame._get_signature(fn),
            guard_manager=check_fn.guard_manager,
            guards_state=check_fn.guards_state,
            backend_id=backend_input.backend_id,
            compiled_fn=compiled_fn,
            original_code=fn.__code__,
            runtime_env=graph_capture_output.get_runtime_env(),
            source_info=source_info,
            device_type=device_type,
            backend_name=backend_name,
            # The field's default_factory would run the C++ toolchain probe on
            # every capture; only an artifact that can hold CPU native code has
            # a baked vector width to record.
            system_info=SystemInfo.current(
                cpu_codegen=(emits_native_code(backend_name) and device_type == "cpu")
            ),
        )
        aot_compiled_fn = AOTCompiledFunction(
            _artifacts=artifacts, _extra_globals=fn.__globals__
        )

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
        # disable_guard_check() is the escape hatch for an artifact whose guards
        # fail on the serving machine for a reason the caller judges benign. A
        # result that opted out accepts anything, but only after a real match has
        # been looked for, so dispatch is unchanged when nobody opted out.
        for result in self.compiled_results:
            if not result._guard_check_enabled:
                return result(self.model, *args, **kwargs)
        raise RuntimeError(self._no_match_message(*args, **kwargs))

    def _no_match_message(self, *args: Any, **kwargs: Any) -> str:
        # Report why every compiled input was rejected, not just the first one,
        # so it is clear which ModelInput is missing.
        lines = [
            f"No AOT compiled graph matched this call. Tried "
            f"{len(self.compiled_results)} compiled input(s):"
        ]
        for i, result in enumerate(self.compiled_results):
            guard_manager = result._artifacts.guard_manager
            if guard_manager is None:
                lines.append(f"  [{i}] <guards unavailable>")
                continue
            # A guard that raises while being re-evaluated for this report must
            # not replace the report. The call did not match, and that -- along
            # with every other entry's reason -- is what the caller has to hear.
            try:
                f_locals = result.prepare_f_locals(self.model, *args, **kwargs)
                reason = guard_manager.check_verbose(f_locals)
            except Exception as e:
                lines.append(f"  [{i}] <guard check raised {type(e).__name__}: {e}>")
                continue
            # Report just the failing guard: GuardDebugInfo's repr is multi-line
            # and would break the per-entry layout into an unreadable blob.
            parts = getattr(reason, "verbose_code_parts", None) or [str(reason)]
            lines.append(f"  [{i}] {'; '.join(str(p) for p in parts)}")
        lines.append(
            "Add a ModelInput covering this call, or check whether a guard that "
            "distinguishes it was dropped by guard_filter_fn."
        )
        return "\n".join(lines)

    def serialize(self) -> bytes:
        data: list[bytes] = []
        for result in self.compiled_results:
            data.append(AOTCompiledFunction.serialize(result).serialized_data)
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, model: torch.nn.Module, data: bytes) -> "AOTCompiledModel":
        from torch._dynamo.utils import get_metrics_context
        from torch._guards import compile_context, CompileContext

        # Guards on globals resolve against the traced function's global scope,
        # which is not reconstructible from the serialized bytecode alone: a
        # global that was specialized away never appears in it.
        #
        # Resolve from model.forward, which is what aot_compile_module traces.
        # Passing the model instead would go through get_traced_fn's nn.Module
        # branch and, whenever a forward hook is registered, hand back
        # Module._wrapped_call_impl and torch/nn/modules/module.py's namespace.
        # Only needed when a global guard survived the filter, which requires a
        # caller-supplied guard_filter_fn. get_traced_fn refuses anything that
        # is not a function or bound method -- a forward rebound to a partial,
        # a callable object -- so failing here would stop such a model loading
        # an artifact that has no global guards at all. Fall back to the old
        # behaviour for those instead: no scope, guards resolve as before.
        try:
            traced_fn, _ = convert_frame.get_traced_fn(model.forward)
            guard_globals = traced_fn.__globals__
        except RuntimeError:
            # Log rather than swallow: if this model DOES carry a surviving
            # global guard, it now resolves against self.fn.__globals__ -- the
            # bug this change exists to fix -- and that should be visible.
            # info, not warning: the overwhelmingly common case is a model with
            # no surviving global guard at all, where this is harmless.
            log.info(
                "Could not resolve a guard scope from %s.forward; global guards "
                "will resolve against the reconstructed scope instead",
                type(model).__name__,
                exc_info=True,
            )
            guard_globals = None

        results: list[bytes] = pickle.loads(data)
        compiled_results = []
        for result in results:
            with (
                compile_context(CompileContext(convert_frame.get_compile_id({}))),
                get_metrics_context(),
            ):
                compiled_results.append(
                    AOTCompiledFunction.deserialize(result, guard_globals=guard_globals)
                )
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

    # pyrefly: ignore [implicit-any]
    compiled_results = []
    for model_input in inputs:
        log.info("Compiling input %s..", model_input)
        compiled_results.append(compile_single_graph(model_input))

    if len(compiled_results) == 0:
        raise AssertionError("Expected at least one compiled result")

    return AOTCompiledModel(model, compiled_results)
