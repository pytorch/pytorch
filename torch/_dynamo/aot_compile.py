import dataclasses
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
from torch._dynamo.package import FunctionPicklerBase, SerializedCode, SystemInfo

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

    def check_compatibility(self) -> None:
        current_system = SystemInfo.current()
        current_system.check_compatibility(self.system_info, self.device_type)


@dataclasses.dataclass
class _ProbeState:
    """Shared by an AOTCompilePickler and the throwaway probe picklers its
    _dumps_cleanly spawns, so the whole probe tree sees one memo."""

    # id(value) -> picklable; without the memo a probe tree is exponential.
    cache: dict[int, bool] = dataclasses.field(default_factory=dict)
    inflight: set[int] = dataclasses.field(default_factory=set)
    # id(value) -> the unmarked nn.Modules the probe reached inside it, so the
    # warning can name the actual reason and the offending modules.
    unmarked_modules: dict[int, list[Any]] = dataclasses.field(default_factory=dict)
    # id(value) -> the exception type that failed its probe, so the warning
    # tells a value that does not pickle from a reducer bug (an AttributeError
    # out of this file's own machinery, a RecursionError).
    failures: dict[int, str] = dataclasses.field(default_factory=dict)
    # id(function) -> its picklable __dict__ entries; a function that closes
    # over itself is reduced twice, and the second pass must not re-probe or
    # re-warn.
    attributes: dict[int, dict[str, Any]] = dataclasses.field(default_factory=dict)
    # Whether a probe short-circuited on an in-flight id; such a verdict is
    # not cached as final but parked (as unpicklable) for the rest of the
    # probe tree.
    leaned: bool = False
    parked: set[int] = dataclasses.field(default_factory=set)


class AOTCompilePickler(FunctionPicklerBase):
    def __init__(
        self,
        external_data: dict[str, object],
        buf: io.BytesIO,
        *,
        probe_state: _ProbeState | None = None,
    ) -> None:
        super().__init__(buf)
        self.external_data = external_data
        self.id_map: dict[int, str] = {
            id(value): key for key, value in external_data.items()
        }
        self.errors = {}
        # A probe pickler shares its parent's state; only the real dump reports
        # what it drops, since a probe's verdict may not be final.
        self._probing = probe_state is not None
        self._probe_state = probe_state or _ProbeState()

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
            receiver = obj.__self__
            # A receiver in external_data is served by persistent_id, so it is
            # the LIVE object at load and pickle's default getattr(receiver,
            # name) resolves on it; the shared reducer's __getattr__ gate would
            # instead pickle __func__ (every nn.Module defines __getattr__),
            # rebuilding a local subclass's method by value and failing on its
            # __class__ cell. An unmarked nn.Module is recorded in errors and
            # fails serialize() anyway, so for it this only keeps the dump going.
            live = id(receiver) in self.id_map or isinstance(receiver, torch.nn.Module)
            reduced = self._reduce_bound_method(obj, receiver_is_live=live)
            if reduced is not None:
                return reduced
        elif inspect.isfunction(obj) and not self._fqn_resolves(obj):
            # The runtime env has to RUN this function, so unlike the guard
            # pickler nothing it holds is pruned -- except __dict__ entries that
            # will not pickle. The runtime assigns those back and never forces
            # the pruned ones, so a value this pickler cannot serialize (a
            # __dict__ entry like the __wrapped__ functools.wraps stashes, which
            # can drag an unrelated lock/Module in) is dropped rather than left
            # to fail the whole dump.
            return self._reduce_function(
                obj,
                defaults=obj.__defaults__,
                kwdefaults=obj.__kwdefaults__,
                closure=obj.__closure__,
                attributes=self._pickleable_attributes(obj),
                annotations={},
                doc=obj.__doc__,
                type_params=None,
                globals_snapshot=None,
            )

        return NotImplemented

    def _warn_dropped(self, obj: Any, slot: str, value: Any) -> None:
        # The body may read a pruned attribute (`with helper.lock:`), so the
        # drop is a warning that names the fix, not a silent debug line; the
        # user can hand the object over as external data and it is kept. It
        # names the exception type too: a reducer bug then reads as one in a
        # bug report instead of as the user's value not pickling. The
        # function is named by its code object: functools.wraps overwrites
        # __qualname__ with the wrappee's, which would make the two drops of a
        # wrapper and its wrappee indistinguishable.
        if self._probing:
            return
        code = obj.__code__
        modules = self._probe_state.unmarked_modules.get(id(value))
        if modules is not None:
            names = ", ".join(type(m).__name__ for m in modules)
            reason = f"it holds nn.Module(s) not marked as external data ({names})"
        else:
            failure = self._probe_state.failures.get(id(value))
            reason = (
                f"it does not pickle ({failure})" if failure else "it does not pickle"
            )
        # co_qualname is 3.11+; the bare co_name on 3.10 cannot tell a wraps
        # wrapper from its wrappee, but __qualname__ could not either.
        log.warning(
            "dropping %s.%s (%s) from the artifact: %s; pass it in external_data to keep it (function defined at %s:%d)",
            getattr(code, "co_qualname", code.co_name),
            slot,
            type(value).__name__,
            reason,
            code.co_filename,
            code.co_firstlineno,
        )

    def _pickleable_attributes(self, obj: Any) -> dict[str, Any]:
        # Memoized for the REAL dump only, where every verdict consulted is
        # final, so a function reduced twice (it closes over itself) is neither
        # re-probed nor re-warned. A probe's answer may lean on an in-flight
        # value and must not be reused. A snapshot of the items: a probe runs
        # user __reduce__ code that may write back onto the function.
        state = self._probe_state
        if not self._probing and id(obj) in state.attributes:
            return state.attributes[id(obj)]
        attributes = {}
        for name, value in list(obj.__dict__.items()):
            if self._dumps_cleanly(value):
                attributes[name] = value
            else:
                self._warn_dropped(obj, name, value)
        if not self._probing:
            state.attributes[id(obj)] = attributes
        return attributes

    def _dumps_cleanly(self, value: Any) -> bool:
        # "does it pickle?" has no cheaper predicate than trying. A throwaway
        # pickler of this exact class keeps external_data/persistent_id behaviour
        # identical to the real dump. The cache stops a value from being probed
        # twice, not from being dumped again inside an ancestor's probe, so the
        # total work is the reachable bytes times the nesting depth, and user
        # __reduce__ code runs once per probe that reaches it; a tensor stashed
        # on a function is serialized into the throwaway buffer too (a transient
        # copy of its storage, and its hook warning fires once more). A
        # recursion overflow counts as unpicklable (the value is pruned) rather
        # than re-raising: a deep-but-finite value in an optional slot must not
        # fail a save that has nothing wrong with it; the guard pickler makes
        # the opposite call for the same condition, since it has a bypass to
        # fall back to and this pickler does not.
        if self._is_literal(value):
            return True
        state = self._probe_state
        vid = id(value)
        cached = state.cache.get(vid)
        if cached is not None:
            return cached
        if vid in state.parked:
            return False
        if vid in state.inflight:
            # Re-entered mid-probe (a value whose attributes reach back to
            # itself). Say picklable to break the cycle -- pickle's memo handles
            # the reference -- and record the lean so a verdict computed on top
            # of it is not cached as final.
            state.leaned = True
            return True
        # Every probed value is reachable from the object being dumped, which
        # the REAL pickler's memo keeps alive until dump() returns, so an id is
        # not reused within one dump; the cache lives as long as this pickler,
        # one per serialize(). (A function a user __reduce__ manufactures inside
        # a probe is not held that way; a later object at its address would
        # inherit its verdict.)
        probe = type(self)(self.external_data, io.BytesIO(), probe_state=state)
        state.inflight.add(vid)
        leaned_before = state.leaned
        state.leaned = False
        try:
            probe.dump(value)
        except Exception as exc:
            # No %r of the value: a repr can raise or be huge.
            log.debug(
                "pruning an unpicklable %s from a nested function: %s",
                type(value).__name__,
                exc,
            )
            state.failures[vid] = type(exc).__name__
            result = False
        else:
            # persistent_id records an unmarked nn.Module rather than raising, so
            # such a value dumps here but would fail the real serialize(); treat
            # it as unpicklable so it is pruned now instead of failing the whole
            # dump later.
            result = not probe.errors
            if not result:
                state.unmarked_modules[vid] = list(probe.errors.values())
                log.debug(
                    "pruning unmarked nn.Module(s) %s from a nested function",
                    list(probe.errors.values()),
                )
        finally:
            state.inflight.discard(vid)
            # The lean travels back through this shared flag because the nested
            # probe is reached through pickle's own dump stack (probe.dump ->
            # reducer_override -> _pickleable_attributes -> _dumps_cleanly), so
            # no return value of the child can reach this frame; restore it here
            # so an aborting dump cannot leave the child's value behind.
            leaned = state.leaned
            state.leaned = leaned_before or leaned
        # A False that leaned on an in-flight True may be a false negative, so
        # it is not cached as final. It is parked for the rest of this probe
        # tree -- re-deriving it is exponential on a cyclic cluster -- and
        # dropped when the tree finishes, so the real dump never consults it.
        # Consulting a park is not a lean: it can make a probe over-prune, and
        # over-pruning CAN flip a probe's verdict False -> True, and the real
        # dump may read that True straight from the cache. It cannot hurt: the
        # real dump never reuses a probe's ATTRIBUTE SET (the memo above is
        # gated on not _probing), so every prunable edge below such a value is
        # re-decided by an outermost probe of its own, and the non-prunable
        # slots (defaults, kwdefaults, closure) are traversed identically in
        # every probe, so a failure there would have failed the earlier probe
        # too. A True, or a False that leaned on nothing, is final. So is the
        # OUTERMOST probe's verdict, leaned or not: the only in-flight id it can
        # lean on is its own, and that lean is exact because pickle's memo
        # resolves the back-reference; the caller acts on it irrevocably.
        if result or not leaned or not state.inflight:
            state.cache[vid] = result
        else:
            state.parked.add(vid)
        if not state.inflight:
            state.parked.clear()
            state.leaned = False
        return result


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
            self._artifacts.guard_manager = load_guard_manager(
                guards_state,
                self._artifacts.original_code,
                self.fn.__globals__,
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
    ) -> "AOTCompiledFunction":
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
        return cls(artifacts, _extra_globals=f_globals)

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
            backend_name=getattr(backend, "compiler_name", "unknown"),
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
            if result.guard_check(self.model, *args, **kwargs):
                return result(self.model, *args, **kwargs)
        # All guards failed, just run one of them and throw the guard check error.
        return self.compiled_results[0](self.model, *args, **kwargs)

    def serialize(self) -> bytes:
        data: list[bytes] = []
        for result in self.compiled_results:
            data.append(AOTCompiledFunction.serialize(result).serialized_data)
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

    # pyrefly: ignore [implicit-any]
    compiled_results = []
    for model_input in inputs:
        log.info("Compiling input %s..", model_input)
        compiled_results.append(compile_single_graph(model_input))

    if len(compiled_results) == 0:
        raise AssertionError("Expected at least one compiled result")

    return AOTCompiledModel(model, compiled_results)
