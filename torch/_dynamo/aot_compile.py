import builtins
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
from torch._dynamo.graph_utils import _graph_device_types
from torch._dynamo.package import (
    emits_native_code,
    FunctionPicklerBase,
    SerializedCode,
    SystemInfo,
)

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
    # device_type keeps the collapsed accelerator-wins value for BC; a mixed
    # cpu+accelerator graph still emits native CPU code, so keep the full set.
    device_types: frozenset[str] = frozenset()

    def check_compatibility(self) -> None:
        # The cached info is the receiver so mismatch messages label self
        # "cached", matching _DynamoCacheEntry.check_versions. This also sets
        # which side the triton_version/gpu_name guards read off, so with the
        # cached info as receiver those two exempt the current host, not the
        # artifact -- the correct direction for a compatibility check.
        device_types = self.device_types or frozenset((self.device_type,))
        check_codegen = emits_native_code(self.backend_name)
        current = SystemInfo.current(
            cpu_codegen=(
                check_codegen
                and "cpu" in device_types
                and self.system_info.cpu_codegen_target is not None
            )
        )
        for device_type in sorted(device_types):
            self.system_info.check_compatibility(
                current, device_type, check_codegen=check_codegen
            )


class AOTCompilePickler(FunctionPicklerBase):
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

    # pyrefly: ignore [bad-override]
    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, types.CellType):
            return self._reduce_cell(obj)
        elif inspect.iscode(obj):
            return type(self)._unpickle_code, (SerializedCode.from_code_object(obj),)
        elif inspect.ismodule(obj):
            return type(self)._unpickle_python_module, (obj.__name__,)
        elif inspect.ismethod(obj):
            reduced = self._reduce_bound_method(obj)
            if reduced is not None:
                return reduced
        elif inspect.isfunction(obj) and "<locals>" in obj.__qualname__:
            # The runtime env has to RUN this function, so unlike the guard
            # pickler nothing it holds is pruned -- except its annotations and
            # type params. The runtime assigns those back verbatim and never
            # evaluates them, so a value this pickler cannot serialize (a
            # <locals> annotation class, a PEP 695 function-scoped TypeVar) is
            # dropped rather than left to fail the whole dump. Known limitation:
            # the top-level function's own annotations ride on
            # CompileArtifacts.signature, which serialize() dumps unpruned, so
            # this only protects the nested functions reached here.
            return self._reduce_function(
                obj,
                defaults=obj.__defaults__,
                kwdefaults=obj.__kwdefaults__,
                closure=obj.__closure__,
                attributes=obj.__dict__,
                annotations=self._pickleable_annotations(obj),
                type_params=self._pickleable_type_params(obj),
            )

        return NotImplemented

    def _dumps_cleanly(self, value: Any) -> bool:
        # "does it pickle?" has no cheaper predicate than trying. A throwaway
        # pickler of this exact class keeps external_data/persistent_id behaviour
        # identical to the real dump. RecursionError is re-raised, not treated as
        # unpicklable, to match the guard side's deliberate carve-out.
        probe = type(self)(self.external_data, io.BytesIO())
        try:
            probe.dump(value)
        except RecursionError:
            raise
        except Exception:
            return False
        # persistent_id records nn.Module instances rather than raising, so such
        # a value dumps here but would poison the real serialize(); treat it as
        # unpicklable so it is pruned now instead of failing the whole dump later.
        return not probe.errors

    def _pickleable_annotations(self, obj: Any) -> dict[str, Any]:
        # resolve=True first turns a 3.14 FORWARDREF proxy into a real value (or
        # drops the whole set when a TYPE_CHECKING-only name will not resolve).
        # Below 3.14 it hands back __annotations__ raw. Either way a value can
        # still be unpicklable -- a <locals> class resolves fine yet pickle
        # cannot reference it -- so probe each and keep only the ones that dump.
        return {
            name: value
            for name, value in self._read_raw_annotations(obj, resolve=True).items()
            if self._dumps_cleanly(value)
        }

    def _pickleable_type_params(self, obj: Any) -> tuple[Any, ...] | None:
        # A PEP 695 function-scoped TypeVar pickles to its bare name and then
        # fails the module lookup, so drop the whole tuple when any element will
        # not dump. Ordinary functions carry (), which dumps and is kept.
        type_params = getattr(obj, "__type_params__", None)
        if type_params and not all(self._dumps_cleanly(p) for p in type_params):
            return None
        return type_params


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
    # Guard-only scope, held by reference; kept apart from _extra_globals so it
    # cannot rewire what the compiled bytecode reads.
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
            # No fallback to the serialized scope: a name the loading process
            # lacks must fail the guard rather than resolve to a baked-in value.
            guard_scope = self._guard_globals
            if guard_scope is None:
                guard_scope = self.fn.__globals__
            else:
                # Dynamo mints __import_* aliases and a __builtins_dict___N key
                # into the tracing process's globals and roots guards at them;
                # a process that only loads never traced, so seed them here.
                # Mirrors the precompile load path in package.py.
                from .output_graph import get_builtins_dict
                from .utils import CleanupHook

                import_sources = self._artifacts.runtime_env.import_sources
                for alias, module_name in import_sources.items():
                    # A pre-reset compile may still own the alias via a
                    # CleanupHook; drop it so it can't delete the binding once
                    # collected, even when we leave an existing value in place.
                    # See _install_global.
                    CleanupHook.disown(guard_scope, alias)
                    if alias not in guard_scope:
                        guard_scope[alias] = importlib.import_module(module_name)
                builtins_key = (
                    guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
                )
                if builtins_key:
                    # A pre-reset compile's CleanupHook may still own this name
                    # even when we leave its value alone; drop it so it can't
                    # delete the binding once collected.
                    CleanupHook.disown(guard_scope, builtins_key)
                    if builtins_key not in guard_scope:
                        # A caller-supplied f_globals need not carry
                        # __builtins__; exec would seed it, so fall back to the
                        # real builtins here.
                        if "__builtins__" not in guard_scope:
                            guard_scope["__builtins__"] = builtins.__dict__
                        guard_scope[builtins_key] = get_builtins_dict(guard_scope)
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
        *,
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
        # A graph naming no device lowers to CPU code.
        graph_devices = _graph_device_types(backend_input.graph_module.graph)
        device_types = graph_devices or frozenset(("cpu",))
        device_type = next((d for d in sorted(device_types) if d != "cpu"), "cpu")
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
            # The codegen probe runs the C++ toolchain; only pay for it when the
            # artifact can hold native CPU code.
            system_info=SystemInfo.current(
                cpu_codegen=(emits_native_code(backend_name) and "cpu" in device_types)
            ),
            device_types=device_types,
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

    args: tuple[object, ...]
    kwargs: dict[str, object]
    contexts: Sequence[AbstractContextManager[object]]


@dataclass
class AOTCompiledModel:
    # Represents a single forward function of a model along with dispatch
    # compiled_results is serializable. We require the model to deserialize again.
    model: torch.nn.Module
    compiled_results: list[AOTCompiledFunction]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for result in self.compiled_results:
            if not result._guard_check_enabled:
                continue
            if result.guard_check(self.model, *args, **kwargs):
                return result(self.model, *args, **kwargs)
        # A result that opted out via disable_guard_check() accepts anything,
        # but only after a real match has been sought.
        for result in self.compiled_results:
            if not result._guard_check_enabled:
                return result(self.model, *args, **kwargs)
        raise RuntimeError(self._no_match_message(*args, **kwargs))

    def _no_match_message(self, *args: Any, **kwargs: Any) -> str:
        lines = [
            f"No AOT compiled graph matched this call. Tried "
            f"{len(self.compiled_results)} compiled input(s):"
        ]
        for i, result in enumerate(self.compiled_results):
            # __post_init__ always leaves a live artifact with a populated
            # guard_manager (only serialize() nulls it, on a copy).
            guard_manager = result._artifacts.guard_manager
            if guard_manager is None:
                raise AssertionError("live artifact must have a guard_manager")
            # A guard that raises here must not replace the whole report.
            try:
                f_locals = result.prepare_f_locals(self.model, *args, **kwargs)
                reason = guard_manager.check_verbose(f_locals)
            except Exception as e:
                lines.append(f"  [{i}] <guard check raised {type(e).__name__}: {e}>")
                continue
            parts = reason.verbose_code_parts or [str(reason)]
            joined = "; ".join(str(p) for p in parts).replace("\n", " ")
            lines.append(f"  [{i}] {joined}")
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
        """Rebuild the compiled forward of ``model`` from ``serialize()`` output.

        Guards on globals are evaluated, by reference, against the live
        ``__globals__`` of the function ``model.forward`` resolves to. That dict
        is mutated: the ``__import_*`` aliases and the ``__builtins_dict___N``
        key the artifact recorded at capture are inserted (never overwriting an
        existing key) so guards rooted at them resolve in a process that never
        traced. A guarded global the dict
        lacks fails the guard; there is no fallback to the serialized scope.
        The compiled bytecode itself still reads the globals serialized with
        the artifact, not this dict. Only when ``model.forward`` is neither a
        function nor a bound method is there no live scope to use, and guards
        then resolve against the reconstructed one, with a warning.
        """
        from torch._dynamo.utils import get_metrics_context
        from torch._guards import compile_context, CompileContext

        # Resolve from model.forward, not the model: for a hooked module
        # get_traced_fn would return Module._wrapped_call_impl and nn.Module's
        # namespace.
        forward = model.forward
        try:
            traced_fn, _ = convert_frame.get_traced_fn(forward)
            guard_globals = traced_fn.__globals__
        except (RuntimeError, AttributeError):
            log.warning(
                "%s.forward is %r, from which no live guard scope could be "
                "resolved (not a plain function or a bound method with an "
                "importable __globals__); global guards on this artifact "
                "resolve against the scope reconstructed from the serialized "
                "bytecode instead",
                type(model).__name__,
                forward,
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
