"""
This module provides the infrastructure for creating and managing compile package
for torch.compile. We mainly have two abstractions here:
  - Precompile: Compiled artifacts related to one particular recompile.
  - CompilePackage: Overarching data structure for store and lookup a list of precompiles.

This is different from the typical global caching system in the sense that compile package is
always saved/loaded via user API calls. This means the caching behavior is always under
user control explicitly so that a stronger guarantee can be provided about cache hit for a
specific compiled model. Users can load the compile package from a different process or even
host but cautions should be taken that compile package will only check a subset of the original
Dynamo guards so there might be soundness problems.
"""

import contextlib
import dataclasses
import functools
import glob
import importlib
import io
import logging
import os
import pickle
import platform
import shutil
import types
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
import torch._inductor.package
from torch._guards import Source

from .bytecode_transformation import get_code_keys


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCacheEntry
    from torch._inductor.output_code import CompiledAOTI


@dataclasses.dataclass(frozen=True)
class SerializedCode:
    co_argcount: int
    co_posonlyargcount: int
    co_kwonlyargcount: int
    co_nlocals: int
    co_stacksize: int
    co_flags: int
    co_code: bytes
    co_consts: tuple[Any, ...]
    co_names: tuple[str, ...]
    co_varnames: tuple[str, ...]
    co_filename: str
    co_name: str
    co_firstlineno: int
    co_linetable: bytes
    co_cellvars: tuple[str, ...]
    co_freevars: tuple[str, ...]
    co_qualname: Optional[str] = None
    co_exceptiontable: Optional[bytes] = None


class _GraphState:
    """
    Stores the compiled artifacts per compiled FX graph. This includes:
      - AOTI compiled kernels.
      - AOTAuograd wrappers.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.forward_aoti: Optional[CompiledAOTI] = None
        self.backward_aoti: Optional[CompiledAOTI] = None
        self.aot_autograd = None
        self.aot_config = None
        self.compile_fx_kwargs = None

    def add_forward_aoti(self, aoti: "CompiledAOTI") -> None:
        assert self.forward_aoti is None
        self.forward_aoti = aoti

    def add_backward_aoti(self, aoti: "CompiledAOTI") -> None:
        assert self.backward_aoti is None
        self.backward_aoti = aoti

    def add_aot_autograd(self, aot_autograd: "AOTAutogradCacheEntry") -> None:
        assert self.aot_autograd is None
        self.aot_autograd = aot_autograd

    def add_aot_config(self, aot_config: "AOTConfig") -> None:
        assert self.aot_config is None
        self.aot_config = aot_config

    def update_aot_autograd_joint(self, aot_autograd: "AOTAutogradCacheEntry") -> None:
        assert self.aot_autograd is not None
        assert aot_autograd.compiled_bw is not None
        self.aot_autograd = aot_autograd

    def add_compile_fx_kwargs(self, compile_fx_kwargs) -> None:
        assert self.compile_fx_kwargs is None
        self.compile_fx_kwargs = compile_fx_kwargs

    @functools.cached_property
    def _callable(self):
        class PrecompiledFunction:
            def __init__(self, callback):
                self.callback = callback

            def load(self, *args, **kwargs):
                return self.callback

            def post_compile(self, compiled_func, *args, **kwargs):
                # TODO Does this work if we just remove the boxing?
                if not hasattr(compiled_func, "_boxed_call"):
                    from torch._functorch._aot_autograd.utils import make_boxed_func

                    compiled_func = make_boxed_func(compiled_func)
                return compiled_func

        entry = dataclasses.replace(
            self.aot_autograd, compiled_fw=PrecompiledFunction(self.forward_aoti)
        )
        if self.backward_aoti is not None:  # TODO Test this.
            entry = dataclasses.replace(
                entry, compiled_bw=PrecompiledFunction(self.backward_aoti)
            )

        # TODO logging should be removed? but currently needed to make this not throwing
        with torch._dynamo.utils.dynamo_timed(
            "backend_compile", log_pt2_compile_event=True
        ):
            compiled_fn = entry.wrap_post_compile(
                None, self.aot_config, self.compile_fx_kwargs
            )

        @torch._dynamo.decorators.disable
        def forward(*args):
            return compiled_fn(list(args))

        return forward


def _load_graph_states(load_path: str, graph_states: _GraphState) -> None:
    """
    precondition: graph_states is empty.
    """
    from torch._inductor.output_code import CompiledAOTI

    root = os.path.join(load_path, graph_states.name)

    with open(root + ".aot_autograd", "rb") as f:
        aot_autograd = pickle.load(f)
    graph_states.add_aot_autograd(aot_autograd)

    with open(root + ".aot_config", "rb") as f:
        from torch._functorch._aot_autograd.autograd_cache import AOTConfig

        aot_config = pickle.load(f)

        def _(*args, **kwargs):
            raise RuntimeError("NYI")

        graph_states.add_aot_config(AOTConfig(_, _, _, {}, **aot_config))

    with open(root + ".compile_fx_kwargs", "rb") as f:
        cfk = pickle.load(f)
        graph_states.add_compile_fx_kwargs(cfk)

    forward_aoti_path = root + ".forward.pt2"
    forward_aoti = torch._inductor.package.load_package(forward_aoti_path, "forward")

    def current_callable(args: tuple[Any, ...]) -> Any:
        return forward_aoti.loader.run(list(args))  # type: ignore[attr-defined]

    graph_states.add_forward_aoti(CompiledAOTI(forward_aoti_path, current_callable))

    assert not os.path.exists(root + ".backward.pt2"), "TODO NYI"


def _save_graph_states(save_path: str, graph_states: _GraphState) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert graph_states.aot_autograd is not None
    with open(os.path.join(save_path, f"{graph_states.name}.aot_autograd"), "wb") as f:
        pickle.dump(graph_states.aot_autograd, f)
    with open(os.path.join(save_path, f"{graph_states.name}.aot_config"), "wb") as f:
        fields = [
            "num_params_buffers",
            "aot_id",
            "keep_inference_input_mutations",
            "is_export",
            "no_tangents",
            "dynamic_shapes",
            "aot_autograd_arg_pos_to_source",
            "static_input_indices",
            "enable_log",
            "pre_dispatch",
        ]
        pickle.dump(
            {field: getattr(graph_states.aot_config, field) for field in fields}, f
        )

    with open(
        os.path.join(save_path, f"{graph_states.name}.compile_fx_kwargs"), "wb"
    ) as f:
        cfk = graph_states.compile_fx_kwargs.copy()
        assert "package" in cfk
        cfk["package"] = None
        pickle.dump(cfk, f)

    assert graph_states.forward_aoti is not None
    src = graph_states.forward_aoti.filename
    dst = os.path.join(save_path, f"{graph_states.name}.forward.pt2")
    if isinstance(src, str) and src.endswith(".pt2"):
        # Already packed aoti file.
        shutil.copy(src, dst)
    else:
        # Unpacked aoti files.
        assert isinstance(src, list)
        torch._inductor.package.package_aoti(dst, {"forward": src})

    assert graph_states.backward_aoti is None, "TODO NYI"


class SerializedGuard:
    source: Source
    create_fn: Callable[["GuardBuilderBase", "Guard"], None]
    metadata: object

    def __init__(self, guard):
        self.source = guard.originating_source
        if isinstance(guard.create_fn, functools.partial):
            self.create_fn = guard.create_fn.func
        else:
            self.create_fn = guard.create_fn
        self.metadata = None


def _construct_raw_module(state):
    mod = torch.nn.Module()
    mod.__setstate__(state)
    return mod


class ScopePickler(pickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, torch.Tensor):
            return torch.tensor, ([],)  # TODO use FakeTensor?
        elif isinstance(obj, torch.nn.Module):
            if obj.__class__.__getstate__ == torch.nn.Module.__getstate__:
                return _construct_raw_module, (obj.__getstate__(),)

        if type(obj).__qualname__ != type(obj).__name__:
            raise RuntimeError(
                f"Type {type(obj)} for object {obj} cannot be saved "
                + "into torch.compile() package since it's defined in local scope. "
                + "Please define the class at global scope (top level of a module)."
            )

        return NotImplemented


def serialize_local_scope(scope) -> bytes:
    buf = io.BytesIO()
    ScopePickler(buf).dump(scope)
    return buf.getvalue()


@dataclass
class GuardsState:
    guards: list[SerializedGuard]
    global_state: str
    f_code: SerializedCode  # TODO deduplicate this.
    input_source_to_sizes_strides: dict[Source, tuple[torch.Size, tuple[int, ...]]]
    local_scope: bytes
    torch_function_mode_stack: bytes


# TODO Potentially GuardBuilder can be split into the first and
#      second half, and serialier should read data from the return
#      value of the first half.
#      For prototyping we just leave this as a separate class.
class GuardSerializer:
    def __init__(self, output_graph):
        self.output_graph = output_graph

    def _get(self, guard):
        from torch._dynamo.guards import _get_closure_vars

        return eval(
            guard.name,
            {"L": self.output_graph.local_scope, "G": self.output_graph.global_scope},
            _get_closure_vars(),
        )

    def serialize(self, guard):
        out = SerializedGuard(guard)
        create_fn = guard.create_fn
        kwargs = {}
        if isinstance(create_fn, functools.partial):
            kwargs = create_fn.keywords
            assert create_fn.args == ()
            create_fn = create_fn.func
        if hasattr(self, create_fn.__name__):
            getattr(self, create_fn.__name__)(guard, out, **kwargs)
        else:
            raise NotImplementedError(f"Serializing guard {create_fn}")
        return out

    def TENSOR_MATCH(self, guard, out, value=None):
        value = value or guard.obj_weakref
        out.metadata = (value().device, torch.empty_like(value(), device="meta"))

    def TYPE_MATCH(self, guard, out):
        t = guard.guarded_class_weakref()
        out.metadata = (t.__module__, t.__qualname__)

    def CONSTANT_MATCH(self, guard, out):
        value = self._get(guard)
        if value in (True, False, None):
            out.metadata = value
        else:
            raise NotImplementedError(f"Unknown constant value: {value}")

    def SHAPE_ENV(self, guard, out):
        out.metadata = guard.code_list

    def DETERMINISTIC_ALGORITHMS(self, guard, out):
        pass

    def DEFAULT_DEVICE(self, guard, out):
        pass

    def GRAD_MODE(self, guard, out):
        pass

    def TORCH_FUNCTION_STATE(self, guard, out):
        pass

    def SEQUENCE_LENGTH(self, guard, out):
        value = self._get(guard)
        out.metadata = len(value), type(value)

    def HASATTR(self, guard, out):
        pass


# TODO Custom guard builder with deserialized values from disk.
class _CustomGuardBuilder(torch._dynamo.guards.GuardBuilder):
    def __init__(
        self, f_code, local_scope, global_scope, guard_manager, check_fn_manager
    ):
        super().__init__(
            f_code,
            lambda obj, name: id(obj),
            None,
            None,
            local_scope,
            global_scope,
            guard_manager,
            check_fn_manager,
        )


class GuardDeserializer:
    def __init__(self, builder: _CustomGuardBuilder):
        self.builder = builder

    def deserialize(self, serialized_guard):
        from torch._guards import Guard

        guard = Guard(
            originating_source=serialized_guard.source,
            create_fn=serialized_guard.create_fn,
        )
        metadata = serialized_guard.metadata
        if hasattr(self, guard.create_fn.__name__):
            getattr(self, guard.create_fn.__name__)(guard, metadata)
        else:
            raise NotImplementedError(f"Deserializing guard {create_fn}")
        return guard

    def TENSOR_MATCH(self, guard, metadata):
        # self.builder.set(guard.name, metadata)  # Needed for example_value
        device, meta_tensor = metadata
        assert isinstance(meta_tensor, torch.Tensor)
        self.builder.TENSOR_MATCH(
            guard, value=torch.empty_like(meta_tensor, device=device)
        )

    def TYPE_MATCH(self, guard, metadata):
        from torch._dynamo.guards import get_verbose_code_parts
        from torch._dynamo.source import AttrSource
        from torch._guards import Guard

        module, qualname = metadata
        cls_source = AttrSource(guard.originating_source, "__class__")

        # TODO Should we just create a new type of FQN-matching guard for this?
        def add_guard(attr, val):
            source = AttrSource(cls_source, attr)
            guard = Guard(source, None)
            ref = self.builder.arg_ref(guard)
            code = [f"{ref} == {val!r}"]
            self.builder.get_guard_manager(guard).add_equals_match_guard(
                val, get_verbose_code_parts(code, guard)
            )

        add_guard("__module__", module)
        add_guard("__qualname__", qualname)

    def CONSTANT_MATCH(self, guard, metadata):
        # self.builder.set(guard.name, metadata)
        self.builder.CONSTANT_MATCH(guard)

    def SHAPE_ENV(self, guard, metadata):
        # TODO Share the same helper function from _dynamo/guards.py
        from torch._dynamo.guards import _get_closure_vars
        from torch.fx.experimental.symbolic_shapes import SYMPY_INTERP

        if metadata is not None:
            assert isinstance(metadata, list)
            self.builder.add_python_lambda_leaf_guard_to_root(
                metadata,
                (),
                closure_vars={**SYMPY_INTERP, **_get_closure_vars()},
            )

    def DETERMINISTIC_ALGORITHMS(self, guard, metadata):
        pass

    def DEFAULT_DEVICE(self, guard, metadata):
        pass

    def GRAD_MODE(self, guard, metadata):
        pass

    def TORCH_FUNCTION_STATE(self, guard, metadata):
        pass

    def SEQUENCE_LENGTH(self, guard, metadata):
        from torch._dynamo.guards import get_verbose_code_parts

        # Mostly copying from GuardBuilder.SEQUENCE_LENGTH
        ref = self.builder.arg_ref(guard)
        value_len, value_type = metadata
        if not issubclass(value_type, dict):
            # C++ DICT_LENGTH checks for type
            self.TYPE_MATCH(guard, (value_type.__module__, value_type.__qualname__))

        code = []
        if value_len == 0:
            code.append(f"not {ref}")
        else:
            code.append(f"len({ref}) == {value_len}")

        if issubclass(value_type, dict):
            self.builder.get_guard_manager(guard).add_dict_length_check_guard(
                value_len, get_verbose_code_parts(code, guard)
            )
        else:
            self.builder.get_guard_manager(guard).add_length_check_guard(
                value_len, get_verbose_code_parts(code, guard)
            )

    def HASATTR(self, guard, metadata):
        self.builder.HASATTR(guard)


class _Precompile:
    """
    A precompile contains all the serializable information associated with a single
    compilation in torch.compile(). To restore an execution of compiled code, we will
    need to serialize the following (not exhaustive):
      - AOTI compiled code and kernels for forward and backward graph.
      - AOTAutograd wrappers for things like input mutation.
      - Dynamo bytecode for mapping Python inputs/outputs.
      - Dynamo guards for cache keys.
    """

    def __init__(self) -> None:
        self.dynamo_code: Optional[types.CodeType] = None
        self.guards_state: Optional[GuardsState] = None
        self.import_sources = {}
        self.graph_states = []

    def add_import_source(self, alias: str, module_name: str) -> None:
        self.import_sources[alias] = module_name

    def add_dynamo_code(self, dynamo_code: types.CodeType) -> None:
        assert self.dynamo_code is None
        self.dynamo_code = dynamo_code

    def add_guards_state(self, guards_state: GuardsState) -> None:
        assert self.guards_state is None
        self.guards_state = guards_state

    def add_graph_state(self, graph_state: _GraphState) -> None:
        self.graph_states.append(graph_state)

    def check_globals(self, f_globals: dict[str, Any]) -> None:
        f_globals = f_globals or {}
        global_names = set(f_globals.keys()).intersection(self.dynamo_code.co_names)
        serialized_names = {
            *self.import_sources.keys(),
            *(c.name for c in self.graph_states),
        }
        unserialized_names = global_names - serialized_names
        assert len(unserialized_names) == 0, (
            f"Global variable names not serialized: {unserialized_names}"
        )

    @functools.cached_property
    def global_scope(self):
        f_globals = {g.name: g._callable for g in self.graph_states}
        for alias, module_name in self.import_sources.items():
            f_globals[alias] = importlib.import_module(module_name)
        return f_globals

    @functools.cached_property
    def guard_manager(self):
        from torch._dynamo.guards import CheckFunctionManager, GuardManagerWrapper
        from torch._dynamo.output_graph import OutputGraph
        from torch._guards import GuardsSet

        guards_out = GuardsSet()
        gm = GuardManagerWrapper()
        check_fn_manager = CheckFunctionManager.__new__(CheckFunctionManager)
        check_fn_manager.guard_manager = gm
        output_graph = OutputGraph.__new__(OutputGraph)
        output_graph.guard_on_key_order = set()
        output_graph.export = False
        output_graph.input_source_to_sizes_strides = (
            self.guards_state.input_source_to_sizes_strides
        )
        check_fn_manager.output_graph = output_graph

        f_code = types.CodeType(
            *[getattr(self.guards_state.f_code, key) for key in get_code_keys()]
        )

        builder = _CustomGuardBuilder(
            f_code,
            pickle.loads(self.guards_state.local_scope),
            self.global_scope,
            gm,
            check_fn_manager,
        )
        deserializer = GuardDeserializer(builder)
        for serialized_guard in self.guards_state.guards:
            guards_out.add(deserializer.deserialize(serialized_guard))

        # TODO CheckFunctionManager probably needs refactor as well to
        #      compile check_fn from saved data directly.
        check_fn_manager.torch_function_mode_stack = pickle.loads(
            self.guards_state.torch_function_mode_stack
        )
        check_fn_manager.output_graph = None  # FIXME aot_autograd guards are skipped.
        check_fn_manager.compile_check_fn(builder, guards_out, None)

        return gm

    @functools.cached_property
    def _callable(self):
        assert self.dynamo_code is not None

        return types.FunctionType(self.dynamo_code, globals=self.global_scope)

    def __call__(self, *args, **kwargs):
        # Only used to debugging
        return self._callable(*args, **kwargs)


def _load_precompile(
    package: "_CompilePackage", load_path: str, index: int
) -> _Precompile:
    with package.precompile_context():
        precompile = package.current_precompile

        root = os.path.join(load_path, f"{index}")
        for graph_states in glob.glob(os.path.join(root, "*.aot_autograd")):
            graph_states_name = os.path.basename(graph_states).rsplit(".", 1)[0]
            with package.graph_state_context(graph_states_name):
                _load_graph_states(root, package.current_graph_state)

        with open(root + ".import_sources", "rb") as f:
            import_sources = pickle.load(f)

        for alias, module_name in import_sources.items():
            precompile.add_import_source(alias, module_name)

        with open(root + ".dynamo_code", "rb") as f:
            serialized_code = pickle.load(f)
        precompile.add_dynamo_code(
            types.CodeType(*[getattr(serialized_code, key) for key in get_code_keys()])
        )

        with open(root + ".guards_state", "rb") as f:
            guards_state = pickle.load(f)
        precompile.add_guards_state(guards_state)

        return precompile


def _save_precompile(save_path: str, index: int, precompile: _Precompile) -> None:
    assert precompile.dynamo_code is not None
    serialized_code = SerializedCode(
        **{key: getattr(precompile.dynamo_code, key) for key in get_code_keys()}
    )
    with open(os.path.join(save_path, f"{index}.dynamo_code"), "wb") as f:
        pickle.dump(serialized_code, f)

    for graph_states in precompile.graph_states:
        _save_graph_states(os.path.join(save_path, str(index)), graph_states)

    with open(os.path.join(save_path, f"{index}.import_sources"), "wb") as f:
        pickle.dump(precompile.import_sources, f)

    with open(os.path.join(save_path, f"{index}.guards_state"), "wb") as f:
        pickle.dump(precompile.guards_state, f)


def _load_precompile_entries(fn, precompiles):
    from torch._C._dynamo.eval_frame import (
        _load_precompile_entry,
        _reset_precompile_entry,
    )

    assert callable(fn)
    code = fn.__code__
    _reset_precompile_entry(code)
    for precompile in precompiles:
        _load_precompile_entry(
            code,
            precompile.guard_manager,
            precompile.dynamo_code,
            precompile.global_scope,
        )


class _CompilePackage:
    """
    The main entry point of compile package system. This data structure should be created
    per torch.compile() call and propagated through the layers to collect compiled
    artifacts from Dynamo, AOTAutograd and Inductor. This essentially maintains a
    list of (guards, compiled code) which will be looked up in order when a set of
    new inputs are passed to compiled object.
    """

    def __init__(self, path: Optional[str] = None):
        self._innermost_fn = None
        self.path = path
        if self.path is not None and self.path.endswith(".pt2"):
            self.unimplemented("single file package")
        self._precompiles: list[_Precompile] = []
        self._current_precompile: Optional[_Precompile] = None
        self._current_graph_state: Optional[_GraphState] = None

    def set_fn(self, fn):
        from .eval_frame import innermost_fn

        self._innermost_fn = innermost_fn(fn)

    @property
    def current_graph_state(self) -> _GraphState:
        assert self._current_graph_state is not None
        return self._current_graph_state

    @contextlib.contextmanager
    def graph_state_context(self, name: str) -> Generator[None, None, None]:
        """
        Set up the current graph context, should be tied to a full recompilation
        cycle.
        """
        assert self._current_graph_state is None
        graph_state = _GraphState(name)
        self._current_graph_state = graph_state
        try:
            yield
        finally:
            self._current_graph_state = None
            self.current_precompile.add_graph_state(graph_state)

    @property
    def current_precompile(self) -> _Precompile:
        """
        Used to access the current precompile object within different compilation
        phases.
        """
        assert self._current_precompile is not None
        return self._current_precompile

    @contextlib.contextmanager
    def precompile_context(self) -> Generator[None, None, None]:
        """
        Set up the current precompile context, should be tied to a full recompilation
        cycle.
        """
        assert self._current_precompile is None
        precompile = _Precompile()
        self._current_precompile = precompile
        try:
            yield
        finally:
            self._current_precompile = None
            self._precompiles.append(precompile)

    def unimplemented(self, msg: str) -> None:
        raise NotImplementedError(
            f"Feature not implemented yet for compile package: {msg}."
        )

    def save(self) -> None:
        """
        Implementation of torch.compile().save_stikcy_cache().
        """
        assert self._current_precompile is None
        path = self.path
        if len(self._precompiles) == 0:
            logger.warning("No compiled models found for compile package.")
        else:
            # TODO Inductor packaging currently doesn't support things like metadata read/write,
            #      to unblock we will have a custom directory for now.
            os.makedirs(path)
            with open(os.path.join(path, "PACKAGE_INFO"), "wb") as f:
                pickle.dump(
                    {
                        "python_version": platform.python_version(),
                        "torch_version": torch.__version__,
                    },
                    f,
                )
            for i, precompile in enumerate(self._precompiles):
                _save_precompile(path, i, precompile)

    def load(self) -> None:
        """
        Implementation of torch.compile().load_stikcy_cache().
        """
        assert self._current_precompile is None
        path = self.path
        if not os.path.exists(path) or not os.path.isdir(path):
            raise RuntimeError(f"Compile package path '{path}' doesn't exist.")
        # TODO Check PACKAGE_INFO
        dynamo_codes = glob.glob(os.path.join(path, "*.dynamo_code"))
        self._precompiles.clear()
        for i in range(len(dynamo_codes)):
            _load_precompile(self, path, i)
        assert len(self._precompiles) == len(dynamo_codes)
        _load_precompile_entries(self._innermost_fn, self._precompiles)

    def reset(self, state: Optional[list[_Precompile]] = None) -> Any:
        # TODO not needed, remove this method.
        assert self._current_precompile is None
        _precompiles = self._precompiles
        self._precompiles = state or []
        assert all(isinstance(p, _Precompile) for p in self._precompiles)
        return _precompiles
