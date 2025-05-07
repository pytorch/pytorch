# mypy: allow-untyped-defs

"""
Core guard system for Dynamo that detects when compiled code needs to be recompiled due to
changes in program state. Guards are conditions that must remain true for previously-compiled
code to be valid for reuse.

This module provides the infrastructure for creating, managing and checking guards, including:
- Guard creation and composition
- Guard state management and invalidation
- Guard checking and failure handling
- Utilities for guard optimization and debugging
- Integration with Dynamo's compilation caching

The guard system is critical for Dynamo's ability to efficiently reuse compiled code while
maintaining correctness by detecting when recompilation is necessary due to changes in
program state, tensor properties, or control flow.
"""

from __future__ import annotations

import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import io
import logging
import math
import pickle
import sys
import textwrap
import types
import warnings
import weakref
from contextlib import contextmanager
from copy import deepcopy
from inspect import currentframe
from typing import Any, Callable, NoReturn, Optional, TYPE_CHECKING, Union
from weakref import ReferenceType

import torch
import torch.overrides
import torch.utils._device
from torch._C._dynamo.eval_frame import code_framelocals_names
from torch._C._dynamo.guards import (
    check_obj_id,
    check_type_id,
    dict_version,
    DictGuardManager,
    install_no_tensor_aliasing_guard,
    install_object_aliasing_guard,
    install_storage_overlapping_guard,
    install_symbolic_shape_guard,
    profile_guard_manager,
    RootGuardManager,
)
from torch._dynamo.source import (
    get_global_source_name,
    IndexedSource,
    is_from_flatten_script_object_source,
    is_from_local_source,
    is_from_optimizer_source,
    TensorProperty,
    TensorPropertySource,
)
from torch._dynamo.utils import CompileEventLogger, get_metrics_context
from torch._guards import (
    CompileContext,
    CompileId,
    DuplicateInputs,
    Guard,
    GuardBuilderBase,
    GuardEnvExpr,
    GuardSource,
    Source,
    StorageOverlap,
)
from torch._logging import structured
from torch._utils_internal import justknobs_check
from torch.fx.experimental.symbolic_shapes import (
    EqualityConstraint,
    is_symbolic,
    SYMPY_INTERP,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef

from . import config, convert_frame, exc, mutation_guard
from .eval_frame import set_guard_error_hook
from .source import (
    AttrProxySource,
    AttrSource,
    CallFunctionNoArgsSource,
    CallMethodItemSource,
    ChainedSource,
    ConstantSource,
    ConstDictKeySource,
    DefaultsSource,
    DictGetItemSource,
    FlattenScriptObjectSource,
    FloatTensorSource,
    FSDPNNModuleSource,
    GenericAttrSource,
    GetItemSource,
    GlobalSource,
    GlobalStateSource,
    GlobalWeakRefSource,
    GradSource,
    ListGetItemSource,
    LocalSource,
    NNModuleSource,
    NumpyTensorSource,
    OptimizerSource,
    ScriptObjectQualifiedNameSource,
    ShapeEnvSource,
    SubclassAttrListSource,
    TorchFunctionModeStackSource,
    TupleIteratorGetItemSource,
    TypeSource,
    UnspecializedBuiltinNNModuleSource,
    UnspecializedNNModuleSource,
    UnspecializedParamBufferSource,
    WeakRefCallSource,
)
from .types import (  # noqa: F401
    CacheEntry,
    DynamoFrameType,
    ExtraState,
    GuardedCode,
    GuardFail,
    GuardFilterEntry,
    GuardFn,
)
from .utils import (
    builtin_dict_keys,
    common_constant_types,
    dict_keys,
    get_custom_getattr,
    get_torch_function_mode_stack,
    get_torch_function_mode_stack_at,
    guard_failures,
    istype,
    key_is_id,
    key_to_id,
    normalize_range_iter,
    orig_code_map,
    tensor_always_has_static_shape,
    tuple_iterator_getitem,
    tuple_iterator_len,
    unpatched_nn_module_getattr,
    verify_guard_fn_signature,
)


guard_manager_testing_hook_fn: Optional[Callable[[Any, Any], Any]] = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from sympy import Symbol

    from torch._dynamo.output_graph import OutputGraphGuardsState


log = logging.getLogger(__name__)
guards_log = torch._logging.getArtifactLogger(__name__, "guards")
recompiles_log = torch._logging.getArtifactLogger(__name__, "recompiles")
recompiles_verbose_log = torch._logging.getArtifactLogger(
    __name__, "recompiles_verbose"
)
verbose_guards_log = torch._logging.getArtifactLogger(__name__, "verbose_guards")


class GuardManagerWrapper:
    """
    A helper class that contains the root guard manager. An instance of this
    class is stored in the Dynamo cache entry, so that the cache entry can
    access the RootGuardManager stored in the "root" attribute and directly call
    the check_nopybind from C++.
    """

    def __init__(self, root=None):
        if root is None:
            self.root = RootGuardManager()
        else:
            self.root = root

        self.diff_guard_root = None
        self.closure_vars = None
        self.args = None
        self.code_parts = []
        self.verbose_code_parts = None
        self.global_scope = None
        self.guard_fail_fn = None
        self.cache_entry = None
        self.extra_state = None
        self.id_matched_objs = {}
        self.no_tensor_aliasing_sources = []

        self.printed_relational_guards = set()

        self.diff_guard_sources: OrderedSet[str] = OrderedSet()

    @contextmanager
    def _preserve_printed_relational_guards(self):
        self.printed_relational_guards = set()
        try:
            yield
        finally:
            self.printed_relational_guards = set()

    def collect_diff_guard_sources(self):
        # At the time of finalize, we have only marked guard managers with
        # TENSOR_MATCH guards as diff guard managers. So, we do a tree traversal
        # and collect all the nodes in the tree (branches) that lead to tensor
        # guards.

        # After a recompilation, some of guard managers will have a fail_count >
        # 0, so we collect them as well. Later on, we accumulate the diff guard
        # sources for all the guard managers.

        def visit_dict_manager(node):
            is_diff_guard_node = (
                node.get_source() in self.diff_guard_sources or node.fail_count() > 0
            )
            for idx, (key_mgr, val_mgr) in sorted(
                node.get_key_value_managers().items()
            ):
                is_diff_guard_node |= visit(key_mgr) | visit(val_mgr)

            if is_diff_guard_node:
                self.diff_guard_sources.add(node.get_source())

            return is_diff_guard_node

        def visit_manager(node):
            assert not isinstance(node, DictGuardManager)

            is_diff_guard_node = (
                node.get_source() in self.diff_guard_sources or node.fail_count() > 0
            )
            for child_mgr in node.get_child_managers():
                is_diff_guard_node |= visit(child_mgr)

            if is_diff_guard_node:
                self.diff_guard_sources.add(node.get_source())

            return is_diff_guard_node

        def visit(node):
            if node is None:
                return False
            if isinstance(node, DictGuardManager):
                return visit_dict_manager(node)
            return visit_manager(node)

        visit(self.root)

        return self.diff_guard_sources

    def finalize(self):
        self.collect_diff_guard_sources()
        self.populate_diff_guard_manager()

    def populate_diff_guard_manager(self):
        self.diff_guard_root = self.clone_with_chosen_sources(self.diff_guard_sources)

        # Ensure that that C++ side points to the updated diff guard manager.
        # When a new GuardManagerWrapper is created, it does not have a
        # cache_entry attribute, so it relies on the CacheEntry constructor to
        # set the diff_guard_root in C++.  But once it is saved in the Dynamo
        # cache, C++ side adds a cache_entry attribute. On recompiles, this
        # cache_entry is visible, so we update the C++ side to point to the
        # update guard manager.
        if self.cache_entry:
            self.cache_entry.update_diff_guard_root_manager()

    def clone_with_chosen_sources(self, chosen_sources):
        def filter_fn(node_mgr):
            return node_mgr.get_source() in chosen_sources

        return self.root.clone_manager(filter_fn)

    def get_guard_lines(self, guard):
        guard_name = guard.__class__.__name__
        parts = guard.verbose_code_parts()
        parts = [guard_name + ": " + part for part in parts]
        return parts

    def get_manager_line(self, guard_manager, accessor_str=None):
        source = guard_manager.get_source()
        t = guard_manager.__class__.__name__
        s = t + ": source=" + source
        if accessor_str:
            s += ", " + accessor_str
        return s

    def construct_dict_manager_string(self, mgr, body):
        for idx, (key_mgr, val_mgr) in sorted(mgr.get_key_value_managers().items()):
            body.writeline(f"KeyValueManager pair at index={idx}")
            with body.indent():
                if key_mgr:
                    body.writeline(f"KeyManager: {self.get_manager_line(key_mgr)}")
                    self.construct_manager_string(key_mgr, body)

                if val_mgr:
                    body.writeline(f"ValueManager: {self.get_manager_line(val_mgr)}")
                    self.construct_manager_string(val_mgr, body)

    def construct_manager_string(self, mgr, body):
        with body.indent():
            for guard in mgr.get_leaf_guards():
                if isinstance(guard, torch._C._dynamo.guards.RelationalGuard):  # type: ignore[attr-defined]
                    if guard not in self.printed_relational_guards:
                        self.printed_relational_guards.add(guard)
                        body.writelines(self.get_guard_lines(guard))
                    else:
                        body.writelines(
                            [
                                guard.__class__.__name__,
                            ]
                        )
                else:
                    body.writelines(self.get_guard_lines(guard))

            # This works for both DictGuardManager and SubclassedDictGuardManager
            if isinstance(mgr, DictGuardManager):
                self.construct_dict_manager_string(mgr, body)

            # General case of GuardManager/RootGuardManager
            for accessor, child_mgr in zip(
                mgr.get_accessors(), mgr.get_child_managers()
            ):
                body.writeline(
                    self.get_manager_line(child_mgr, f"accessed_by={accessor.repr()}")
                )
                self.construct_manager_string(child_mgr, body)

    def __str__(self):
        from torch._inductor.utils import IndentedBuffer

        class IndentedBufferWithPrefix(IndentedBuffer):
            def prefix(self):
                return "| " * (self._indent * self.tabwidth)

            def writeline(self, line, skip_prefix=False):
                if skip_prefix:
                    super().writeline(line)
                else:
                    super().writeline("+- " + line)

        with self._preserve_printed_relational_guards():
            body = IndentedBufferWithPrefix()
            body.tabwidth = 1
            body.writeline("", skip_prefix=True)
            body.writeline("TREE_GUARD_MANAGER:", skip_prefix=True)
            body.writeline("RootGuardManager")
            self.construct_manager_string(self.root, body)
            if hasattr(self.root, "get_epilogue_lambda_guards"):
                for guard in self.root.get_epilogue_lambda_guards():
                    body.writelines(self.get_guard_lines(guard))
            return body.getvalue()

    def check(self, x):
        # Only needed for debugging purposes.
        return self.root.check(x)

    def check_verbose(self, x):
        # Only needed for debugging purposes.
        return self.root.check_verbose(x)

    def populate_code_parts_for_debugging(self):
        # This should be called when the guard manager is fully populated
        tensor_aliasing_guard_seen = False

        def get_code_parts(leaf_guard):
            code_parts = []
            for verbose_code_part in leaf_guard.verbose_code_parts():
                code_part = verbose_code_part.split("#")[0].rstrip()
                code_parts.append(code_part)
            return code_parts

        def visit(mgr):
            nonlocal tensor_aliasing_guard_seen
            for guard in mgr.get_leaf_guards():
                if isinstance(guard, torch._C._dynamo.guards.NO_TENSOR_ALIASING):  # type: ignore[attr-defined]
                    if not tensor_aliasing_guard_seen:
                        self.code_parts.extend(get_code_parts(guard))
                        tensor_aliasing_guard_seen = True
                else:
                    self.code_parts.extend(get_code_parts(guard))

            for child_mgr in mgr.get_child_managers():
                visit(child_mgr)

        visit(self.root)


def from_numpy(a):
    # If not numpy array, piggy back on e.g. tensor guards to check type
    # Re-enable torch function since we disable it on leaf guards
    # we need it to properly construct the tensor if a default device is set
    with torch.overrides._enable_torch_function():
        return torch.as_tensor(a) if isinstance(a, (np.generic, np.ndarray)) else a


# For user stack printing
@functools.lru_cache(None)
def uninteresting_files():
    import torch._dynamo.external_utils
    import torch._dynamo.polyfills

    mods = [torch._dynamo.external_utils, torch._dynamo.polyfills]

    from torch._dynamo.polyfills.loader import POLYFILLED_MODULES

    mods.extend(POLYFILLED_MODULES)

    return {inspect.getfile(m) for m in mods}


_CLOSURE_VARS: Optional[dict[str, object]] = None


def _get_closure_vars():
    global _CLOSURE_VARS
    if _CLOSURE_VARS is None:
        _CLOSURE_VARS = {
            "___check_type_id": check_type_id,
            "___check_obj_id": check_obj_id,
            "___odict_getitem": collections.OrderedDict.__getitem__,
            "___key_to_id": key_to_id,
            "___dict_version": dict_version,
            "___dict_contains": lambda a, b: dict.__contains__(b, a),
            "___tuple_iterator_len": tuple_iterator_len,
            "___normalize_range_iter": normalize_range_iter,
            "___tuple_iterator_getitem": tuple_iterator_getitem,
            "___get_torch_function_mode_stack_at": get_torch_function_mode_stack_at,
            "__math_isnan": math.isnan,
            "__numpy_isnan": None if np is None else np.isnan,
            "inf": float("inf"),
            "__load_module": importlib.import_module,
            "utils_device": torch.utils._device,
            "device": torch.device,
            "___from_numpy": from_numpy,
            "___as_tensor": torch._as_tensor_fullprec,
            "torch": torch,
            "inspect": inspect,
        }
    return _CLOSURE_VARS


def _ast_unparse(node: ast.AST) -> str:
    return ast.unparse(node).replace("\n", "")


strip_function_call = torch._C._dynamo.strip_function_call


def get_verbose_code_part(code_part: str, guard: Guard) -> str:
    extra = ""
    if guard is not None:
        if guard.user_stack:
            for fs in reversed(guard.user_stack):
                if fs.filename not in uninteresting_files():
                    extra = f"  # {format_frame(fs, line=True)}"
                    break
        elif guard.stack:
            summary = guard.stack.summary()
            if len(summary) > 0:
                extra = f"  # {format_frame(summary[-1])}"
            else:
                extra = "  # <unknown>"
    return f"{code_part:<60}{extra}"


def get_verbose_code_parts(
    code_parts: Union[str | list[str]], guard: Guard
) -> list[str]:
    if not isinstance(code_parts, list):
        code_parts = [code_parts]
    return [get_verbose_code_part(code_part, guard) for code_part in code_parts]


def convert_to_concrete_values(size_or_stride):
    converted: list[Optional[int]] = []
    for dim in size_or_stride:
        if not is_symbolic(dim):
            converted.append(dim)
        else:
            assert isinstance(dim, torch.SymInt)
            converted.append(dim.node.maybe_as_int())
    return converted


def get_tensor_guard_code_part(value, name, sizes, strides, pytype, dispatch_keys):
    dispatch_key = (
        dispatch_keys | torch._C._dispatch_tls_local_include_set()
    ) - torch._C._dispatch_tls_local_exclude_set()
    dtype = value.dtype
    device_index = value.device.index
    requires_grad = value.requires_grad
    guard_str = (
        f"check_tensor({name}, {pytype.__qualname__}, {dispatch_key}, {dtype}, "
        f"device={device_index}, requires_grad={requires_grad}, size={sizes}, stride={strides})"
    )
    return guard_str


def get_key_index(dct, key):
    # Ensure that we call dict.keys and not value.keys (which can call
    # overridden keys method). In the C++ guards, we relied on PyDict_Next
    # to traverse the dictionary, which uses the internal data structure and
    # does not call the overridden keys method.
    return list(builtin_dict_keys(dct)).index(key)


def get_key_index_source(source, index):
    return f"list(dict.keys({source}))[{index}]"


def raise_local_type_error(obj: Any) -> NoReturn:
    raise TypeError(
        f"Type {type(obj)} for object {obj} cannot be saved "
        + "into torch.compile() package since it's defined in local scope. "
        + "Please define the class at global scope (top level of a module)."
    )


@dataclasses.dataclass(frozen=True)
class NNModuleAttrAccessorInfo:
    # Represents where is the attr name is present in the nn module attribute
    # access

    # Tells that the attribute can be accessed via __dict__
    present_in_generic_dict: bool = False

    # Either the actual name or _parameters/_buffers/_modules
    l1_key: Optional[str] = None

    # Actual paramter/buffer/submodule name
    l2_key: Optional[str] = None


def getitem_on_dict_manager(
    source, base_guard_manager, base_example_value, example_value, guard_manager_enum
):
    base_source_name = source.base.name()
    if isinstance(source.index, ConstDictKeySource):
        index = source.index.index
    else:
        assert isinstance(base_example_value, dict)
        index = get_key_index(base_example_value, source.index)

    key_source = get_key_index_source(base_source_name, index)

    # Ensure that we call dict.keys and not value.keys (which can call
    # overridden keys method). In the C++ guards, we relied on PyDict_Next
    # to traverse the dictionary, which uses the internal data structure and
    # does not call the overridden keys method.
    key_example_value = list(builtin_dict_keys(base_example_value))[index]
    if isinstance(key_example_value, (int, str)):
        value_source = f"{base_source_name}[{key_example_value!r}]"
    else:
        value_source = f"{base_source_name}[{key_source}]"
    if not isinstance(source.index, ConstDictKeySource):
        # We have to insert a key manager guard here
        # TODO - source debug string is probably wrong here.
        base_guard_manager.get_key_manager(
            index=index,
            source=key_source,
            example_value=source.index,
            guard_manager_enum=GuardManagerType.GUARD_MANAGER,
        ).add_equals_match_guard(
            source.index, [f"{key_source} == {key_example_value!r}"]
        )

    return base_guard_manager.get_value_manager(
        index=index,
        source=value_source,
        example_value=example_value,
        guard_manager_enum=guard_manager_enum,
    )


def match_on_id_for_tensor(guard):
    source = guard.originating_source
    # For numpy tensors, always use TENSOR_MATCH because __from_numpy leads
    # to a new tensor everytime and therefore id differs.
    if isinstance(source, NumpyTensorSource):
        return False

    if guard.is_specialized_nn_module():
        return True

    return source.is_dict_key() and not isinstance(source, GradSource)


# The ready to eval generated code (possibly multiple parts) for a guard, plus
# the original guard object that created it for provenance
@dataclasses.dataclass
class GuardCodeList:
    code_list: list[str]
    guard: Guard


class GuardManagerType(enum.Enum):
    GUARD_MANAGER = 1
    DICT_GUARD_MANAGER = 2


@functools.lru_cache(None)
def code_framelocals_names_reversed_cached(code: types.CodeType):
    return list(reversed(code_framelocals_names(code)))


class GuardBuilder(GuardBuilderBase):
    def __init__(
        self,
        f_code: types.CodeType,
        id_ref: Callable[[Any, str], str],
        source_ref: Callable[[Source], str],
        lookup_weakrefs: Callable[[object], ReferenceType[object]],
        local_scope: dict[str, object],
        global_scope: dict[str, object],
        guard_manager: GuardManagerWrapper,
        check_fn_manager: CheckFunctionManager,
        serialization_mode: Optional[str] = None,
    ):
        self.f_code = f_code
        self.id_ref = id_ref
        self.source_ref = source_ref
        self.lookup_weakrefs = lookup_weakrefs
        self.scope: dict[str, dict[str, object]] = {"L": local_scope, "G": global_scope}
        self.scope["__builtins__"] = builtins.__dict__.copy()
        for (
            name,
            package_module,
        ) in torch.package.package_importer._package_imported_modules.items():
            name = name.replace(">", "_").replace("<", "_").replace(".", "_dot_")
            # Write the package module into the scope so that we can import it
            self.scope["__builtins__"][name] = package_module
            # Write the demangled name to the scope so that we can use it
            self.scope[name] = package_module
        self.guard_manager = guard_manager

        self.argnames: list[str] = []
        # Code is python expression strings generated for each guard
        self.code: list[GuardCodeList] = []
        # shape_env_code is only used by builder and is used for
        # shape env code.  This exists only because we need to make sure
        # shape env guards get run after tensor match guards (since the
        # tensor match guards make sure we actually have tensors)
        self.shape_env_code: list[GuardCodeList] = []

        # Collect the guard managers and debug info to insert no tensor aliasing
        # guards.
        self.no_tensor_aliasing_names: list[str] = []
        self.no_tensor_aliasing_guard_managers: list[GuardManagerWrapper] = []

        self.check_fn_manager: CheckFunctionManager = check_fn_manager

        # Collect the ids of dicts which need key order guarding. source_name is
        # not sufficient because for nn modules, we can have different sources
        # to access the same object - self._module["param"] is same as
        # self.param.
        self.key_order_guarded_dict_ids = set()
        for source in self.check_fn_manager.output_graph.guard_on_key_order:
            self.key_order_guarded_dict_ids.add(id(self.get(source.name())))

        # Keep track of weak references of objects with ID_MATCH guard. This
        # info is stored alongside optimized_code and guard_manager and is used to
        # limit the number of cache entries with same ID_MATCH'd object.
        self.id_matched_objs: dict[str, ReferenceType[object]] = {}

        # Save the guard managers to avoid repeatedly traversing sources.
        self._cached_guard_managers: dict[
            str, torch._C._dynamo.guards.GuardManager
        ] = {}
        self._cached_duplicate_input_guards: set[tuple[str, str]] = set()
        self.serialization_mode = serialization_mode

    def guard_on_dict_keys_and_ignore_order(self, example_value, guard):
        dict_mgr = self.get_guard_manager(guard)
        if isinstance(dict_mgr, DictGuardManager):
            raise NotImplementedError(
                "Not expecting a DictGuardManager. Seems like Dynamo incorrectly "
                f"added the dict to tx.output.guard_on_key_order for {guard.name}"
            )

        # Iterate over the dicts and install a dict_getitem_manager.
        dict_source = guard.originating_source.name()

        # Ensure that we call dict.keys and not value.keys (which can call
        # overridden keys method). In the C++ guards, we relied on PyDict_Next
        # to traverse the dictionary, which uses the internal data structure and
        # does not call the overridden keys method.
        for key in builtin_dict_keys(example_value):
            value = example_value[key]
            value_source = DictGetItemSource(guard.originating_source, index=key)
            guard_manager_enum = self.get_guard_manager_type(
                value_source, example_value
            )
            dict_mgr.dict_getitem_manager(
                key=key,
                source=f"{dict_source}[{key!r}]",
                example_value=value,
                guard_manager_enum=guard_manager_enum,
            )

    def guard_on_dict_keys_and_order(self, value, guard):
        # Add key managers for the DictGuardManager. Then add either an
        # ID_MATCH or EQUALS_MATCH guard on the key.
        dict_mgr = self.get_guard_manager(guard)
        if not isinstance(dict_mgr, DictGuardManager):
            raise NotImplementedError(
                "Expecting a DictGuardManager. Seems like Dynamo forgot "
                f"to set the right guard manager enum for {guard.name}"
            )
        assert isinstance(dict_mgr, DictGuardManager)

        # Ensure that we call dict.keys and not value.keys (which can call
        # overridden keys method). In the C++ guards, we relied on PyDict_Next
        # to traverse the dictionary, which uses the internal data structure and
        # does not call the overridden keys method.
        for idx, key in enumerate(builtin_dict_keys(value)):
            key_source = get_key_index_source(guard.name, idx)
            key_manager = dict_mgr.get_key_manager(
                index=idx,
                source=key_source,
                example_value=key,
                guard_manager_enum=GuardManagerType.GUARD_MANAGER,
            )
            if key_is_id(key):
                # Install ID_MATCH guard
                id_val = self.id_ref(key, key_source)
                key_manager.add_id_match_guard(
                    id_val,
                    get_verbose_code_parts(
                        f"__check_obj_id({key_source}, {id_val})", guard
                    ),
                )
            else:
                # Install EQUALS_MATCH guard
                key_manager.add_equals_match_guard(
                    key, get_verbose_code_parts(f"{key_source} == {key!r}", guard)
                )

    @staticmethod
    def _get_generic_dict_manager_example_value(example_value):
        # due to a bug in 3.13.0 (introduced by https://github.com/python/cpython/pull/116115,
        # reported in https://github.com/python/cpython/issues/125608,
        # fixed by https://github.com/python/cpython/pull/125611), we cannot take
        # advantage of __dict__ versions to speed up guard checks.
        if (
            config.issue_3_13_0_warning
            and sys.version_info >= (3, 13)
            and sys.version_info < (3, 13, 1)
        ):
            warnings.warn(
                "Guards may run slower on Python 3.13.0. Consider upgrading to Python 3.13.1+.",
                RuntimeWarning,
            )
            return None
        return example_value

    def getattr_on_nn_module(
        self,
        source,
        base_guard_manager,
        base_example_value,
        example_value,
        base_source_name,
        source_name,
        guard_manager_enum,
    ):
        """
        This tries to avoid calling the expensive nn module custom getattr method by
        checking if the attribute is accessible via __dict__. For attributes that
        are not accessible via __dict__ (like descriptors), we fallback to
        PyObject_GetAttr.

        There are two cases that we optimize for
        1) attributes present directly in __dict__, e.g training.
        2) parameters/buffers/modules - they can be accessed via _parameters,
        _buffers, _modules keys in __dict__. For example, mod.linear can be
        accessed as mod.__dict__["_parameters"]["linear"]

        The most common and expensive case for nn module guards is of type
        mod.submod1.submod2.submod3.training. We avoid the python getattr of nn
        modules by going through the __dict__.
        """

        def getitem_on_dict_mgr(
            mgr, key, source_name, base_example_value, example_value, guard_manager_enum
        ):
            if isinstance(mgr, DictGuardManager):
                # Case where the user code relies on key order, e.g.,
                # named_parameters
                index = get_key_index(base_example_value, key)

                # Install the key manager and add equals match guard
                key_source = f"list(dict.keys({source_name}))[{index!r}]"
                mgr.get_key_manager(
                    index=index,
                    source=key_source,
                    example_value=key,
                    guard_manager_enum=GuardManagerType.GUARD_MANAGER,
                ).add_equals_match_guard(key, [f"{key_source} == {key!r}"])

                # Install the value manager
                return mgr.get_value_manager(
                    index=index,
                    source=source_name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )
            else:
                return mgr.dict_getitem_manager(
                    key=key,
                    source=source_name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )

        attr_name = source.member
        mod_dict = base_example_value.__dict__

        all_class_attribute_names: set[str] = set()
        for x in inspect.getmro(base_example_value.__class__):
            all_class_attribute_names.update(x.__dict__.keys())

        accessor_info = NNModuleAttrAccessorInfo(False, None, None)

        if attr_name in mod_dict:
            accessor_info = NNModuleAttrAccessorInfo(True, attr_name, None)
        elif "_parameters" in mod_dict and attr_name in mod_dict["_parameters"]:
            accessor_info = NNModuleAttrAccessorInfo(True, "_parameters", attr_name)
        elif "_buffers" in mod_dict and attr_name in mod_dict["_buffers"]:
            accessor_info = NNModuleAttrAccessorInfo(True, "_buffers", attr_name)
        elif (
            attr_name not in all_class_attribute_names
            and "_modules" in mod_dict
            and attr_name in mod_dict["_modules"]
        ):
            # Check test_attr_precedence test - instance attributes always take precedence unless its an nn.Module.
            accessor_info = NNModuleAttrAccessorInfo(True, "_modules", attr_name)

        if not accessor_info.present_in_generic_dict:
            # The attribute can be accessed by __getattribute__ call, so rely on
            # PyObject_GetAttr
            return base_guard_manager.getattr_manager(
                attr=source.member,
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        else:
            assert accessor_info.l1_key
            l1_key = accessor_info.l1_key
            l2_key = accessor_info.l2_key

            # Set source strings for debug info
            mod_dict_source = f"{base_source_name}.__dict__"
            l1_source_name = l2_source_name = None
            l1_value = l2_value = None
            l1_guard_manager_enum = l2_guard_manager_enum = None
            if l2_key:
                l1_source = AttrSource(source.base, l1_key)
                l1_source_name = l1_source.name()
                l1_value = mod_dict[l1_key]
                # do not guard on key order for _parameters etc unless the user code
                # actually needs the key order (e.g. calling named_parameters)
                l1_guard_manager_enum = self.get_guard_manager_type(l1_source, l1_value)

                l2_source_name = source_name
                l2_value = example_value
                l2_guard_manager_enum = self.get_guard_manager_type(
                    source, example_value
                )
            else:
                l1_source_name = source_name
                l1_value = example_value
                l1_guard_manager_enum = self.get_guard_manager_type(
                    source, example_value
                )

            # Get __dict__ accessor. No need to guard on dict key order, so use base
            # Guard Manager
            mod_generic_dict_manager = base_guard_manager.get_generic_dict_manager(
                source=mod_dict_source,
                example_value=self._get_generic_dict_manager_example_value(mod_dict),
                guard_manager_enum=GuardManagerType.GUARD_MANAGER,
            )

            l1_mgr = getitem_on_dict_mgr(
                mgr=mod_generic_dict_manager,
                key=l1_key,
                source_name=l1_source_name,
                base_example_value=mod_dict,
                example_value=l1_value,
                guard_manager_enum=l1_guard_manager_enum,
            )

            if l2_key:
                return getitem_on_dict_mgr(
                    mgr=l1_mgr,
                    key=l2_key,
                    source_name=l2_source_name,
                    base_example_value=l1_value,
                    example_value=l2_value,
                    guard_manager_enum=l2_guard_manager_enum,
                )
            return l1_mgr

    def requires_key_order_guarding(self, source):
        source_name = source.name()
        if source_name == "":
            return False
        obj_id = id(self.get(source_name))
        return obj_id in self.key_order_guarded_dict_ids

    def get_guard_manager_type(self, source, example_value):
        guard_manager_enum = GuardManagerType.GUARD_MANAGER
        if self.requires_key_order_guarding(source):
            # Fix this if condition
            if isinstance(example_value, dict_keys):
                guard_manager_enum = GuardManagerType.DICT_GUARD_MANAGER
            else:
                assert isinstance(example_value, dict)
                guard_manager_enum = GuardManagerType.DICT_GUARD_MANAGER
        return guard_manager_enum

    def manager_guards_on_keys(self, mgr_enum):
        return mgr_enum == GuardManagerType.DICT_GUARD_MANAGER

    def get_global_guard_manager(self):
        return self.guard_manager.root.globals_dict_manager(
            f_globals=self.scope["G"],
            source="G",
            example_value=self.scope["G"],
            guard_manager_enum=GuardManagerType.GUARD_MANAGER,
        )

    def get_guard_manager_from_source(self, source):
        root_guard_manager = self.guard_manager.root

        example_value = None
        source_name = source.name()

        if source_name != "" and source_name in self._cached_guard_managers:
            return self._cached_guard_managers[source_name]

        if source_name != "":
            example_value = self.get(source_name)

        guard_manager_enum = self.get_guard_manager_type(source, example_value)

        # Get base manager related information
        base_source_name = None
        base_example_value = None
        base_guard_manager = None
        base_guard_manager_enum = GuardManagerType.GUARD_MANAGER
        if isinstance(source, ChainedSource):
            base_source_name = source.base.name()
            base_example_value = self.get(base_source_name)
            base_guard_manager = self.get_guard_manager_from_source(source.base)
            base_guard_manager_enum = self.get_guard_manager_type(
                source.base, base_example_value
            )

        # Use istype instead of isinstance to check for exact type of source.
        if istype(source, LocalSource):
            # Refer to index in the frame's localsplus directly.
            # NOTE: name order for a code object doesn't change.
            # NOTE: we need to find the LAST matching index because <= 3.10 contains
            # duplicate names in the case of cells: a name can be both local and cell
            # and will take up 2 slots of the frame's localsplus. The correct behavior
            # is to refer to the cell, which has a higher index.
            framelocals_names_reversed = code_framelocals_names_reversed_cached(
                self.f_code
            )
            framelocals_idx = (
                len(framelocals_names_reversed)
                - framelocals_names_reversed.index(source.local_name)
                - 1
            )
            out = root_guard_manager.framelocals_manager(
                key=(source.local_name, framelocals_idx),
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, GlobalSource):
            # Global manager accepts a dict but it is not a DictGuardManager
            # because globals dict is big and we typically guard on a very
            # selected items on globals.
            out = self.get_global_guard_manager().dict_getitem_manager(
                key=source.global_name,
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, GlobalWeakRefSource):
            out = self.get_global_guard_manager().global_weakref_manager(
                global_name=source.global_name,
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, GlobalStateSource):
            # Don't do anything here. We guard on global state completely in
            # C++. So just return the root mgr.
            return root_guard_manager
        elif istype(source, ShapeEnvSource):
            return root_guard_manager
        elif istype(source, TypeSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.type_manager(
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(
            source,
            (
                OptimizerSource,
                NNModuleSource,
                UnspecializedNNModuleSource,
                UnspecializedBuiltinNNModuleSource,
                FSDPNNModuleSource,
            ),
        ):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager
        elif istype(source, TorchFunctionModeStackSource):
            out = root_guard_manager.lambda_manager(
                python_lambda=lambda _: get_torch_function_mode_stack_at(
                    source._get_index()
                ),
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, GradSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.grad_manager(
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, GenericAttrSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.generic_getattr_manager(
                attr=source.member,
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, (AttrSource, UnspecializedParamBufferSource)):
            assert base_guard_manager  # to make mypy happy

            if (
                isinstance(base_example_value, torch.nn.Module)
                and get_custom_getattr(base_example_value)
                is unpatched_nn_module_getattr
            ):
                out = self.getattr_on_nn_module(
                    source,
                    base_guard_manager,
                    base_example_value,
                    example_value,
                    base_source_name,
                    source_name,
                    guard_manager_enum,
                )
            else:
                out = base_guard_manager.getattr_manager(
                    attr=source.member,
                    source=source_name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )
        elif istype(source, DictGetItemSource):
            assert base_guard_manager  # to make mypy happy
            assert isinstance(base_example_value, (dict, collections.OrderedDict))
            if isinstance(base_guard_manager, DictGuardManager):
                assert self.manager_guards_on_keys(base_guard_manager_enum)
                out = getitem_on_dict_manager(
                    source,
                    base_guard_manager,
                    base_example_value,
                    example_value,
                    guard_manager_enum,
                )
            else:
                if isinstance(source.index, ConstDictKeySource):
                    raise RuntimeError(
                        "Expecting clean index here. Likely Dynamo forgot to mark"
                        " a dict as guard_on_key_order"
                    )
                out = base_guard_manager.dict_getitem_manager(
                    key=source.index,
                    source=source_name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )
        elif istype(source, TensorPropertySource):
            out = getattr(
                base_guard_manager,
                f"tensor_property_{source.prop.name.lower()}_manager",
            )(
                idx=source.idx,
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, IndexedSource):
            assert base_guard_manager  # to make mypy happy

            out = base_guard_manager.indexed_manager(
                idx=source.idx,
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, ListGetItemSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.list_getitem_manager(
                key=source.index,
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, GetItemSource):
            assert base_guard_manager  # to make mypy happy
            assert not isinstance(
                base_example_value, (dict, collections.OrderedDict)
            ), "Use DictGetItemSource"
            if isinstance(base_example_value, list) and not source.index_is_slice:
                out = base_guard_manager.list_getitem_manager(
                    key=source.index,
                    source=source_name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )
            elif isinstance(base_example_value, tuple) and not source.index_is_slice:
                out = base_guard_manager.tuple_getitem_manager(
                    key=source.index,
                    source=source_name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )
            else:
                index = source.index
                if source.index_is_slice:
                    index = source.unpack_slice()
                out = base_guard_manager.getitem_manager(
                    key=index,
                    source=source_name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )
        elif istype(source, DefaultsSource):
            assert base_guard_manager  # to make mypy happy
            assert callable(base_example_value)
            if not source.is_kw:
                out = base_guard_manager.func_defaults_manager(
                    source=base_source_name,
                    example_value=base_example_value.__defaults__,
                    guard_manager_enum=GuardManagerType.GUARD_MANAGER,
                ).getitem_manager(
                    key=source.idx_key,
                    source=source_name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )
            else:
                # kwdefauts is a dict, so use a DictGuardManager
                kwdefaults = base_example_value.__kwdefaults__
                assert base_source_name is not None
                kw_source = base_source_name + ".__kwdefaults__"

                # kwdefaults is a dict. No need to guard on dict order.
                dict_mgr = base_guard_manager.func_kwdefaults_manager(
                    source=kw_source,
                    example_value=kwdefaults,
                    guard_manager_enum=GuardManagerType.GUARD_MANAGER,
                )
                assert not isinstance(dict_mgr, DictGuardManager)

                out = dict_mgr.dict_getitem_manager(
                    key=source.idx_key,
                    source=source_name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )
        elif istype(source, NumpyTensorSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.lambda_manager(
                python_lambda=from_numpy,
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, SubclassAttrListSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.lambda_manager(
                python_lambda=lambda x: x.__tensor_flatten__()[0],
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, FlattenScriptObjectSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.lambda_manager(
                python_lambda=lambda x: x.__obj_flatten__(),
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, ScriptObjectQualifiedNameSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.lambda_manager(
                python_lambda=lambda x: x._type().qualified_name(),
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, AttrProxySource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.lambda_manager(
                python_lambda=lambda x: x.get_base(),
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, CallMethodItemSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.lambda_manager(
                python_lambda=lambda x: x.item(),
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, FloatTensorSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.lambda_manager(
                python_lambda=lambda x: torch._as_tensor_fullprec(x),
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, TupleIteratorGetItemSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.tuple_iterator_getitem_manager(
                index=source.index,
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif isinstance(source, ConstDictKeySource):
            if not isinstance(base_guard_manager, DictGuardManager):
                raise AssertionError(
                    "ConstDictKeySource can only work on DictGuardManager"
                )
            out = base_guard_manager.get_key_manager(
                index=source.index,
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, WeakRefCallSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.weakref_call_manager(
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        elif istype(source, CallFunctionNoArgsSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.call_function_no_args_manager(
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        else:
            raise AssertionError(
                f"missing guard manager builder {source} - {source.name()}"
            )

        self._cached_guard_managers[source.name()] = out
        return out

    def get_guard_manager(self, guard: Guard):
        return self.get_guard_manager_from_source(guard.originating_source)

    def add_python_lambda_leaf_guard_to_root(
        self,
        code_parts,
        verbose_code_parts,
        closure_vars=None,
        is_epilogue=True,
    ):
        if closure_vars is None:
            closure_vars = _get_closure_vars()
        # Adds a lambda leaf guard to the root guard manager. It wraps the
        # code_parts in a function object which is then passed on to the leaf
        # guard.
        make_guard_fn_args = ", ".join(closure_vars.keys())
        _guard_body, pycode = build_guard_function(code_parts, make_guard_fn_args)
        out: dict[str, Any] = {}
        globals_for_guard_fn = {"G": self.scope["G"]}
        guards_log.debug("Python shape guard function:\n%s", pycode)
        exec(pycode, globals_for_guard_fn, out)
        guard_fn = out["___make_guard_fn"](*closure_vars.values())
        if is_epilogue:
            # Epilogue guards are run after all the other guards have finished.
            # If epilogue guards contain a getattr or getitem access, one of the
            # other guards would fail preventing the epilogue guards to run.
            self.guard_manager.root.add_epilogue_lambda_guard(
                guard_fn, verbose_code_parts
            )
        else:
            self.guard_manager.root.add_lambda_guard(guard_fn, verbose_code_parts)

    # Warning: use this with care!  This lets you access what the current
    # value of the value you are guarding on is.  You probably don't want
    # to actually durably save this value though (because it's specific
    # to this frame!)  Instead, you should be reading out some property
    # (like its type) which is what you permanently install into the
    # guard code.
    def get(self, name: str, closure_vars: Optional[dict[str, Any]] = None) -> Any:
        if closure_vars is None:
            closure_vars = _get_closure_vars()
        return eval(name, self.scope, closure_vars)

    # Registers the usage of the source name referenced by the
    # string (or stored in the Guard) as being guarded upon.  It's important
    # to call this before generating some code that makes use of 'guard',
    # because without this call, we won't actually bind the variable
    # you reference in the actual guard closure (oops!)
    def arg_ref(self, guard: Union[str, Guard]) -> str:
        name: str
        if isinstance(guard, str):
            name = guard
        else:
            name = guard.name
        base = strip_function_call(name)
        if base not in self.argnames:
            is_valid = torch._C._dynamo.is_valid_var_name(base)
            if is_valid:
                if is_valid == 2:
                    log.warning("invalid var name: %s", guard)
                self.argnames.append(base)

        return name

    def _guard_on_attribute(self, guard: Guard, attr_name: str, guard_fn):
        attr_source = AttrSource(guard.originating_source, attr_name)
        # Copy the stack info
        new_guard = Guard(
            attr_source, guard_fn, stack=guard.stack, user_stack=guard.user_stack
        )
        new_guard.create(self)

    # Note: the order of the guards in this file matters since we sort guards on the same object by lineno
    def HASATTR(self, guard: Guard):
        source = guard.originating_source
        if isinstance(source, NNModuleSource):
            source = source.base
        assert isinstance(source, AttrSource), f"invalid source {guard.name}"
        base_source = source.base
        base = base_source.name()
        attr = source.member

        ref = self.arg_ref(base)
        val = hasattr(self.get(base), attr)
        code = None
        if val:
            code = f"hasattr({ref}, {attr!r})"
        else:
            code = f"not hasattr({ref}, {attr!r})"
        self._set_guard_export_info(
            guard, [code], provided_guarded_object=self.get(base)
        )

        base_manager = self.get_guard_manager_from_source(base_source)
        if val:
            # Just install a getattr manager. GetAttrGuardAccessor itself
            # acts as hasattr guard.
            example_value = self.get(source.name())
            base_example_value = self.get(base)
            guard_manager_enum = self.get_guard_manager_type(source, example_value)

            # if the base value is nn.Module, check if we can speedup the
            # guard by going through __dict__ attrs.
            if (
                isinstance(base_example_value, torch.nn.Module)
                and get_custom_getattr(base_example_value)
                is unpatched_nn_module_getattr
            ):
                return self.getattr_on_nn_module(
                    source,
                    base_manager,
                    base_example_value,
                    example_value,
                    base,
                    source.name(),
                    guard_manager_enum,
                )
            else:
                base_manager.getattr_manager(
                    attr=attr,
                    source=guard.name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )
        else:
            base_manager.add_no_hasattr_guard(attr, get_verbose_code_parts(code, guard))

    def NOT_PRESENT_IN_GENERIC_DICT(self, guard: Guard, attr=None) -> None:
        assert attr is not None
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        assert isinstance(val, torch.nn.Module)

        base_manager = self.get_guard_manager(guard)

        mod_dict_source = f"{guard.name}.__dict__"
        mod_generic_dict_manager = base_manager.get_generic_dict_manager(
            source=mod_dict_source,
            example_value=self._get_generic_dict_manager_example_value(val.__dict__),
            guard_manager_enum=GuardManagerType.GUARD_MANAGER,
        )

        code = f"not ___dict_contains({attr!r}, {ref}.__dict__)"
        mod_generic_dict_manager.add_dict_contains_guard(
            False, attr, get_verbose_code_parts(code, guard)
        )

    def TYPE_MATCH(self, guard: Guard) -> None:
        # ___check_type_id is same as `id(type(x)) == y`
        value = self.get(guard.name)
        t = type(value)
        if self.serialization_mode == "save":
            if t.__qualname__ != t.__name__:
                raise_local_type_error(value)

        obj_id = self.id_ref(t, f"type({guard.name})")
        code = f"___check_type_id({self.arg_ref(guard)}, {obj_id})"
        self._set_guard_export_info(guard, [code])

        self.get_guard_manager(guard).add_type_match_guard(
            obj_id, get_verbose_code_parts(code, guard)
        )

    def DICT_VERSION(self, guard: Guard):
        if self.serialization_mode == "save":
            raise RuntimeError("DICT_VERSION guard cannot be serialized.")
        # ___check_dict_version is same as `dict_version(x) == y`
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        version = dict_version(self.get(guard.name))
        code = f"___dict_version({ref}) == {version}"
        self._set_guard_export_info(guard, [code])

        # TODO(anijain2305) - Delete this when DictGuardManager uses tags
        # for dicts.
        self.get_guard_manager(guard).add_dict_version_guard(
            val, get_verbose_code_parts(code, guard)
        )

    def DICT_CONTAINS(self, guard: Guard, key: str, invert: bool):
        dict_ref = self.arg_ref(guard)

        maybe_not = "not " if invert else ""
        code = f"{maybe_not}___dict_contains({key!r}, {dict_ref})"
        self._set_guard_export_info(guard, [code])

        self.get_guard_manager(guard).add_dict_contains_guard(
            not invert, key, get_verbose_code_parts(code, guard)
        )

    def BOOL_MATCH(self, guard: Guard):
        # checks val == True or val == False
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        assert istype(val, bool)
        code = [f"{ref} == {val!r}"]
        self._set_guard_export_info(guard, code)

        if val:
            self.get_guard_manager(guard).add_true_match_guard(
                get_verbose_code_parts(code, guard)
            )
        else:
            self.get_guard_manager(guard).add_false_match_guard(
                get_verbose_code_parts(code, guard)
            )

    def NONE_MATCH(self, guard: Guard):
        # checks `val is None`
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        assert val is None
        code = [f"{ref} is None"]
        self._set_guard_export_info(guard, code)

        self.get_guard_manager(guard).add_none_match_guard(
            get_verbose_code_parts(code, guard)
        )

    def ID_MATCH(self, guard: Guard):
        if self.serialization_mode == "save":
            raise RuntimeError("ID_MATCH guard cannot be serialized.")
        # ___check_obj_id is same as `id(x) == y`
        if isinstance(guard.originating_source, TypeSource):
            # optional optimization to produce cleaner/faster guard code
            return self.TYPE_MATCH(
                Guard(guard.originating_source.base, GuardBuilder.TYPE_MATCH)  # type: ignore[arg-type]
            )

        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        id_val = self.id_ref(val, guard.name)
        code = f"___check_obj_id({ref}, {id_val})"
        self._set_guard_export_info(guard, [code])

        self.get_guard_manager(guard).add_id_match_guard(
            id_val, get_verbose_code_parts(code, guard)
        )

        # Keep track of ID_MATCH'd objects. This will be used to modify the
        # cache size logic
        if isinstance(guard.originating_source, LocalSource):
            # TODO(anijain2305) - This is currently restricted to nn.Module objects
            # because many other ID_MATCH'd objects fail - like DeviceMesh.
            # Increase the scope of ID_MATCH'd objects.
            if isinstance(val, torch.nn.Module):
                local_name = guard.originating_source.local_name
                weak_id = self.lookup_weakrefs(val)
                if weak_id is not None:
                    self.id_matched_objs[local_name] = weak_id

    def NOT_NONE_MATCH(self, guard: Guard, value=None):
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        assert isinstance(val, torch.Tensor)
        code = f"{ref} is not None"
        self._set_guard_export_info(guard, [code])

        self.get_guard_manager(guard).add_not_none_guard(
            get_verbose_code_parts(code, guard)
        )

    def DISPATCH_KEY_SET_MATCH(self, guard: Guard):
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        assert isinstance(val, torch._C.DispatchKeySet)
        code_parts = f"{ref}.raw_repr() == {val!r}.raw_repr()"

        self.get_guard_manager(guard).add_dispatch_key_set_guard(
            val, get_verbose_code_parts(code_parts, guard)
        )

    def NAME_MATCH(self, guard: Guard):
        self._guard_on_attribute(guard, "__name__", GuardBuilder.EQUALS_MATCH)

    def DUAL_LEVEL(self, guard: Guard):
        # Invalidate dual level if current dual level is different than the one
        # in the fx graph
        dual_level = self.check_fn_manager.output_graph.dual_level
        code = [f"torch.autograd.forward_ad._current_level == {dual_level}"]
        self._set_guard_export_info(guard, [code])
        # TODO(anijain2305) - Consider this moving this guard to C++
        forward_ad = torch.autograd.forward_ad

        def fn(x):
            return forward_ad._current_level == dual_level

        self.guard_manager.root.add_lambda_guard(
            fn, get_verbose_code_parts(code, guard)
        )

    def FUNCTORCH_STACK_MATCH(self, guard: Guard):
        # Invalidate functorch code if current level is different than
        # the one when FX graph was generated
        cis = self.check_fn_manager.output_graph.functorch_layers
        states = [ci.get_state() for ci in cis]
        code = [f"torch._functorch.pyfunctorch.compare_functorch_state({states})"]
        self._set_guard_export_info(guard, code)

        # TODO(anijain2305) - Consider this moving this guard to C++
        compare_fn = torch._functorch.pyfunctorch.compare_functorch_state

        def fn(x):
            return compare_fn(states)

        self.guard_manager.root.add_lambda_guard(
            fn, get_verbose_code_parts(code, guard)
        )

    def TENSOR_SUBCLASS_METADATA_MATCH(self, guard: Guard):
        value = self.get(guard.name)
        original_metadata = deepcopy(self.get(guard.name).__tensor_flatten__()[1])
        if hasattr(value, "__metadata_guard__"):
            verify_guard_fn_signature(value)

            def metadata_checker(x):
                return value.__metadata_guard__(
                    original_metadata, x.__tensor_flatten__()[1]
                )

        else:

            def metadata_checker(x):
                return x.__tensor_flatten__()[1] == original_metadata

        global_name = f"___check_metadata_{id(metadata_checker)}_c{CompileContext.current_compile_id()}"
        self.get_guard_manager(guard).add_lambda_guard(
            metadata_checker, get_verbose_code_parts(global_name, guard)
        )

    def EQUALS_MATCH(self, guard: Guard):
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        if np:
            np_types: tuple[type[Any], ...] = (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
            )
        else:
            np_types = ()

        ok_mutable_types = (list, set)

        ok_types = tuple(
            common_constant_types
            | {
                type,
                tuple,
                frozenset,
                slice,
                range,
                dict_keys,
                torch.Size,
                *np_types,
                *ok_mutable_types,
            }
        )

        if torch.distributed.is_available():
            from torch.distributed.device_mesh import DeviceMesh
            from torch.distributed.tensor.placement_types import (
                Partial,
                Replicate,
                Shard,
            )

            ok_types = ok_types + (
                Shard,
                Replicate,
                Partial,
                DeviceMesh,
            )

        import torch.utils._pytree as pytree

        assert istype(val, ok_types) or pytree.is_constant_class(type(val)), (
            f"Unexpected type {type(val)}"
        )

        # Special case for nan because float("nan") == float("nan") evaluates to False
        if istype(val, float) and math.isnan(val):
            self.TYPE_MATCH(guard)
            code = []
            code.append(f"__math_isnan({ref})")
            self._set_guard_export_info(guard, code)

            self.get_guard_manager(guard).add_lambda_guard(
                _get_closure_vars()["__math_isnan"],
                get_verbose_code_parts(code, guard),
            )
            return

        # Python math library doesn't support complex nan, so we need to use numpy
        if istype(val, complex) and np.isnan(val):
            self.TYPE_MATCH(guard)
            code = []
            code.append(f"__numpy_isnan({ref})")
            self._set_guard_export_info(guard, code)

            self.get_guard_manager(guard).add_lambda_guard(
                _get_closure_vars()["__numpy_isnan"],
                get_verbose_code_parts(code, guard),
            )
            return

        # Construct a debug string to put into the c++ equals match guard.
        code = [f"{ref} == {val!r}"]
        if istype(val, ok_mutable_types):
            # C++ guards perform a pointer equality check to speedup guards, but the assumption is that the object
            # is immutable. For a few corner cases like sets and lists, we make a deepcopy to purposefully fail the
            # pointer equality check.
            val = deepcopy(val)
        self.get_guard_manager(guard).add_equals_match_guard(
            val, get_verbose_code_parts(code, guard)
        )
        self._set_guard_export_info(guard, code)
        return

    def CONSTANT_MATCH(self, guard: Guard):
        val = self.get(guard.name)
        if istype(val, bool):
            self.BOOL_MATCH(guard)
        elif val is None:
            self.NONE_MATCH(guard)
        elif istype(val, types.CodeType):
            self.ID_MATCH(guard)
        else:
            self.EQUALS_MATCH(guard)

    def NN_MODULE(self, guard: Guard):
        self.ID_MATCH(guard)
        val = self.get(guard.name)
        if hasattr(val, "training"):
            assert istype(val.training, bool)
            self._guard_on_attribute(guard, "training", GuardBuilder.CONSTANT_MATCH)
        else:
            exc.unimplemented_v2(
                gb_type="Attempted to guard on uninitialized nn.Module",
                context="",
                explanation="Attempted to setup an NN_MODULE guard on uninitialized "
                f"nn.Module subclass `{type(val)}`.",
                hints=[
                    "Ensure the `nn.Module` subclass instance has called `super().__init__()`.",
                ],
            )

    def FUNCTION_MATCH(self, guard: Guard):
        """things like torch.add and user defined functions"""
        return self.ID_MATCH(guard)

    def CLOSURE_MATCH(self, guard: Guard):
        """matches a closure by __code__ id."""
        val = self.get(guard.name)
        # Strictly only want user-defined functions
        if type(val) == types.FunctionType and hasattr(val, "__code__"):
            self._guard_on_attribute(guard, "__code__", GuardBuilder.HASATTR)
            self._guard_on_attribute(guard, "__code__", GuardBuilder.FUNCTION_MATCH)
        else:
            self.FUNCTION_MATCH(guard)

    def BUILTIN_MATCH(self, guard: Guard):
        return self.FUNCTION_MATCH(guard)

    def PYMODULE_MATCH(self, guard: Guard):
        return self.FUNCTION_MATCH(guard)

    def SEQUENCE_LENGTH(self, guard):
        # This guard is used to check length of PySequence objects like list,
        # tuple, collections.deque etc
        ref = self.arg_ref(guard)
        value = self.get(guard.name)

        if not isinstance(value, dict):
            # C++ DICT_LENGTH checks for type
            self.TYPE_MATCH(guard)

        code = []
        if len(value) == 0:
            code.append(f"not {ref}")
        else:
            code.append(f"len({ref}) == {len(value)}")

        self._set_guard_export_info(guard, code)
        if isinstance(value, dict):
            self.get_guard_manager(guard).add_dict_length_check_guard(
                len(value), get_verbose_code_parts(code, guard)
            )
        else:
            self.get_guard_manager(guard).add_length_check_guard(
                len(value), get_verbose_code_parts(code, guard)
            )

    def TUPLE_ITERATOR_LEN(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)

        code = []
        code.append(f"___tuple_iterator_len({ref}) == {tuple_iterator_len(value)}")
        self._set_guard_export_info(guard, code)

        t = type(value)
        obj_id = self.id_ref(t, f"type({guard.name})")

        self.get_guard_manager(guard).add_tuple_iterator_length_guard(
            tuple_iterator_len(value), obj_id, get_verbose_code_parts(code, guard)
        )

    def RANGE_ITERATOR_MATCH(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)

        code = []
        normalized_range_iter = normalize_range_iter(value)
        code.append(f"___normalize_range_iter({ref}) == {normalized_range_iter}")
        self._set_guard_export_info(guard, code)

        t = type(value)
        obj_id = self.id_ref(t, f"type({guard.name})")

        start, stop, step = normalized_range_iter
        self.get_guard_manager(guard).add_range_iterator_match_guard(
            start, stop, step, obj_id, get_verbose_code_parts(code, guard)
        )

    # TODO(voz): Deduplicate w/ AOTAutograd dupe input guards
    def DUPLICATE_INPUT(self, guard, source_b):
        if self.serialization_mode == "save":
            raise RuntimeError("DUPLICATE_INPUT guard cannot be serialized yet.")
        ref_a = self.arg_ref(guard)
        ref_b = self.arg_ref(source_b.name())

        if is_from_optimizer_source(
            guard.originating_source
        ) or is_from_optimizer_source(source_b):
            return

        # Check that the guard has not been inserted already
        key = (ref_a, ref_b)
        if key in self._cached_duplicate_input_guards:
            return

        self._cached_duplicate_input_guards.add((ref_a, ref_b))
        self._cached_duplicate_input_guards.add((ref_b, ref_a))

        code = [f"{ref_b} is {ref_a}"]
        self._set_guard_export_info(guard, code)

        install_object_aliasing_guard(
            self.get_guard_manager(guard),
            self.get_guard_manager_from_source(source_b),
            get_verbose_code_parts(code, guard),
        )

    def WEAKREF_ALIVE(self, guard):
        if self.serialization_mode == "save":
            raise RuntimeError("WEAKREF_ALIVE guard cannot be serialized.")
        code = [f"{self.arg_ref(guard)} is not None"]

        self._set_guard_export_info(guard, code)
        self.get_guard_manager(guard).add_not_none_guard(
            get_verbose_code_parts(code, guard)
        )

    def MAPPING_KEYS_CHECK(self, guard):
        """Guard on the key order of types.MappingProxyType object"""
        ref = self.arg_ref(guard)
        value = self.get(guard.name)

        code = []
        code.append(f"list({ref}.keys()) == {list(value.keys())}")
        self._set_guard_export_info(guard, code)
        self.get_guard_manager(guard).add_mapping_keys_guard(value, code)

    def DICT_KEYS_MATCH(self, guard):
        """Insert guard to check that the keys of a dict are same"""
        ref = self.arg_ref(guard)
        value = self.get(guard.name)

        if value is torch.utils._pytree.SUPPORTED_NODES:
            # For SUPPORTED_NODES, we can guard on the dictionary version (PEP509).
            self.DICT_VERSION(guard)
            return

        self.SEQUENCE_LENGTH(guard)

        code = []
        # Ensure that we call dict.keys and not value.keys (which can call
        # overridden keys method). In the C++ guards, we relied on PyDict_Next
        # to traverse the dictionary, which uses the internal data structure and
        # does not call the overridden keys method.
        code.append(f"list(dict.keys({ref})) == {list(builtin_dict_keys(value))!r}")
        self._set_guard_export_info(guard, code)

        if self.requires_key_order_guarding(guard.originating_source):
            self.guard_on_dict_keys_and_order(value, guard)
        else:
            self.guard_on_dict_keys_and_ignore_order(value, guard)

    def EMPTY_NN_MODULE_HOOKS_DICT(self, guard):
        """Special guard to skip guards on empty hooks. This is controlled by skip_nnmodule_hook_guards"""
        if config.skip_nnmodule_hook_guards:
            # This is unsafe if you add/remove a hook on nn module variable
            return
        self.SEQUENCE_LENGTH(guard)

    def OBJECT_MUTATION(self, guard: Guard):
        mutation_guard.watch(self.get(guard.name), self.check_fn_manager)

    def GRAD_MODE(self, guard: Guard):
        pass  # we always guard on this via GlobalStateGuard()

    def DETERMINISTIC_ALGORITHMS(self, guard: Guard):
        pass  # we always guard on this via GlobalStateGuard()

    def TORCH_FUNCTION_STATE(self, guard: Guard):
        pass  # we always guard on this via GlobalStateGuard()

    def FSDP_TRAINING_STATE(self, guard: Guard):
        pass  # we always guard on this via GlobalStateGuard()

    def DEFAULT_DEVICE(self, guard: Guard):
        """Guard on CURRENT_DEVICE per torch.utils._device"""
        assert guard.source is GuardSource.GLOBAL
        import torch.utils._device as m

        code = [f"utils_device.CURRENT_DEVICE == {m.CURRENT_DEVICE!r}"]
        self._set_guard_export_info(guard, code)

        self.get_guard_manager(guard).add_default_device_guard(
            get_verbose_code_parts(code, guard)
        )

    def SHAPE_ENV(self, guard: Guard):
        # Let's handle ShapeEnv guards.  To do this, we will resolve
        # shape variables to sources from tracked_fakes.  This must happen after
        # tensor checks.
        assert guard.name == ""
        output_graph = self.check_fn_manager.output_graph
        # NB: self.output_graph can be None in the debug_nops tests
        fs = output_graph.tracked_fakes
        input_contexts = [a.symbolic_context for a in fs]

        def get_sources(t_id, dim):
            # Looks up base sources mapped to a tensor id and uses them to create
            # sources for the corresponding tensor dimension.
            return [
                TensorPropertySource(source, TensorProperty.SIZE, dim)
                for source in output_graph.tracked_fakes_id_to_source[t_id]
            ]

        if output_graph.export_constraints:
            names: dict[str, tuple[int, int]] = {}
            source_pairs: list[tuple[Source, Source]] = []
            derived_equalities: list[  # type: ignore[type-arg]
                tuple[Source, Union[Source, Symbol], Callable]
            ] = []
            phantom_symbols: dict[str, Symbol] = {}
            relaxed_sources: set[Source] = set()
            for constraint in output_graph.export_constraints:
                if constraint.t_id in output_graph.tracked_fakes_id_to_source:
                    torch.export.dynamic_shapes._process_equalities(
                        constraint,
                        get_sources,
                        output_graph.shape_env,
                        names,
                        source_pairs,
                        derived_equalities,
                        phantom_symbols,
                        relaxed_sources,
                    )
                else:
                    log.warning("Untracked tensor used in export constraints")
            equalities_inputs = EqualityConstraint(
                source_pairs=source_pairs,
                derived_equalities=derived_equalities,
                phantom_symbols=list(phantom_symbols.values()),
                relaxed_sources=relaxed_sources,
                warn_only=False,
            )
        else:
            equalities_inputs = None

        def _get_code_parts(langs):
            return output_graph.shape_env.produce_guards_verbose(
                [a.fake for a in fs],
                [a.source for a in fs],
                input_contexts=input_contexts,
                equalities_inputs=equalities_inputs,
                source_ref=self.source_ref,
                # Export keeps static.
                ignore_static=(not self.check_fn_manager.output_graph.export),
                langs=langs,
            )

        if config.enable_cpp_symbolic_shape_guards:
            try:
                # For exporting we need the python code parts
                python_code_parts, verbose_code_parts, cpp_code_parts = _get_code_parts(
                    ("python", "verbose_python", "cpp")
                )
                python_fallback = False
            except OverflowError:
                # Cannot use int64_t
                python_fallback = True
                python_code_parts, verbose_code_parts = _get_code_parts(
                    ("python", "verbose_python")
                )
        else:
            python_fallback = True
            python_code_parts, verbose_code_parts = _get_code_parts(
                ("python", "verbose_python")
            )

        # When exporting, we may work with the shape constraints some more in
        # postprocessing, so don't freeze yet
        if not self.check_fn_manager.output_graph.export:
            output_graph.shape_env.freeze()
        for code in python_code_parts.exprs:
            self._set_guard_export_info(guard, [code])

        # Make ShapeEnv guards available for testing.
        if compile_context := CompileContext.try_get():
            compile_context.shape_env_guards.extend(verbose_code_parts.exprs)

        int_source_to_symbol = []
        float_source_to_symbol = []

        if not python_fallback:
            assert cpp_code_parts  # type: ignore[possibly-undefined]
            code_parts, source_to_symbol = (
                cpp_code_parts.exprs,
                cpp_code_parts.source_to_symbol,
            )

            if not code_parts:
                return

            for source, symbol in source_to_symbol.items():
                if isinstance(source, ConstantSource):
                    python_fallback = True
                else:
                    example_value = self.get(
                        source.name(),
                        closure_vars={**SYMPY_INTERP, **_get_closure_vars()},
                    )
                    if isinstance(example_value, int):
                        int_source_to_symbol.append((source, symbol))
                    elif isinstance(example_value, float):
                        float_source_to_symbol.append((source, symbol))
                    else:
                        # SymInts/SymFloats go through python guard as we only support
                        # int64_t/double in C++ guards for now.
                        python_fallback = True

        if not python_fallback:
            import ctypes

            from torch._inductor.codecache import CppCodeCache

            assert cpp_code_parts  # type: ignore[possibly-undefined]
            code_parts, source_to_symbol = (
                cpp_code_parts.exprs,
                cpp_code_parts.source_to_symbol,
            )

            source_to_symbol = dict(int_source_to_symbol + float_source_to_symbol)
            try:
                guard_managers = [
                    self.get_guard_manager_from_source(IndexedSource(source, i))
                    for i, source in enumerate(source_to_symbol)
                ]

                int_symbols_str = ", ".join(
                    f"{symbol} = int_values[{i}]"
                    for i, (_, symbol) in enumerate(int_source_to_symbol)
                )
                float_symbols_str = ", ".join(
                    f"{symbol} = float_values[{i}]"
                    for i, (_, symbol) in enumerate(float_source_to_symbol)
                )

                if int_symbols_str:
                    int_symbols_str = f"int64_t {int_symbols_str};"
                if float_symbols_str:
                    float_symbols_str = f"double {float_symbols_str};"

                func_str = textwrap.dedent(
                    f"""
                #include <cstdint>
                #include <cmath>
                #include <c10/util/generic_math.h>

                #if defined(_MSC_VER)
                #  define EXTERN_DLL_EXPORT extern "C" __declspec(dllexport)
                #else
                #  define EXTERN_DLL_EXPORT extern "C"
                #endif

                EXTERN_DLL_EXPORT int8_t guard(int64_t *int_values, double *float_values) {{
                  {int_symbols_str}
                  {float_symbols_str}
                  return ({") && (".join(code_parts)});
                }}
                """
                )
                guards_log.debug(
                    "C++ shape guard function: %s %s",
                    func_str,
                    verbose_code_parts.exprs,
                )
                clib = CppCodeCache.load(func_str)
                cguard = ctypes.cast(clib.guard, ctypes.c_void_p).value
                assert cguard
            except torch._inductor.exc.InvalidCxxCompiler:
                # No valid C++ compiler to compile the shape guard
                pass
            else:
                install_symbolic_shape_guard(
                    guard_managers,
                    len(int_source_to_symbol),
                    len(float_source_to_symbol),
                    cguard,
                    clib,
                    verbose_code_parts.exprs,
                )
                return

        # Install all the symbolic guards in one python lambda guard. These are run
        # at the very end of the RootGuardManager via epilogue guards.
        # TODO(anijain2305,williamwen42) - Consider moving this to C++.
        if python_code_parts.exprs:
            self.add_python_lambda_leaf_guard_to_root(
                python_code_parts.exprs,
                verbose_code_parts.exprs,
                closure_vars={**SYMPY_INTERP, **_get_closure_vars()},
            )

    def TENSOR_MATCH(self, guard: Guard, value=None):
        if config._unsafe_skip_fsdp_module_guards and guard.is_fsdp_module():
            return
        # For tensors that are part of the Dynamo extracted Fx graph module, an
        # ID_MATCH suffices. Once we turn on inline_inbuilt_nn_modules, these
        # will be lifted as inputs and have a TENSOR_MATCH guard.
        if match_on_id_for_tensor(guard):
            self.ID_MATCH(guard)
        else:
            if isinstance(value, TensorWeakRef):
                value = value()

            value = value if value is not None else self.get(guard.name)

            pytype = type(value)
            dispatch_keys = torch._C._dispatch_keys(value)
            if isinstance(value, torch._subclasses.FakeTensor):
                if value.pytype is not None:
                    pytype = value.pytype
                if value.dispatch_keys is not None:
                    dispatch_keys = value.dispatch_keys

            assert isinstance(value, torch.Tensor)

            if config.log_compilation_metrics and isinstance(value, torch.nn.Parameter):
                metrics_context = get_metrics_context()
                metrics_context.increment("param_numel", value.numel())
                metrics_context.increment("param_bytes", value.nbytes)
                metrics_context.increment("param_count", 1)

            tensor_name = self.arg_ref(guard)
            # [Note - On Export Tensor Guards]
            #
            # In eager mode, tensor guards are evaluated through C++, in guards.cpp
            # see [Note - On Eager Tensor Guards] for more info.
            #
            # In export mode, we instead maintain parallel logic between C++ and python
            # here, with an exception of checking the dispatch key - with the idea that a dispatch key
            # is an entirely runtime notion that would make no sense to keep in an exported graph.
            #
            # Now, this idea is okay, but to paraphrase @ezyang, this mental model is sufficient for now, although
            # not entirely true.
            # For example, suppose one of the input tensors had the negative dispatch key.
            # You should end up with a graph that is specialized for tensors that have a negative dispatch key.
            # If you allow a Tensor that does NOT have this bit set, you will accidentally run it "as if" it were negated.
            # Now, negative key only shows up for complex numbers, and most likely, the exported to target doesn't
            # support this feature at all, but the point stands that :some: tensor state only shows up on dispatch key.
            # TODO(voz): Either populate a dispatch_key check into the guards, or error on users passing in an unsupported
            # subset of keys during export.
            #
            # The list of tensor fields and calls we care about can be found in `terms` below.
            # TODO(voz): We are missing storage offset in all our tensor guards?
            code: list[str] = []
            if self.check_fn_manager.output_graph.export:
                self.TYPE_MATCH(guard)
                terms = [
                    "dtype",
                    "device",
                    "requires_grad",
                    "ndimension()",
                ]

                for term in terms:
                    real_value = self.get(tensor_name + "." + term)
                    if istype(real_value, (torch.device, torch.dtype)):
                        # copy pasted from EQUALS_MATCH
                        code.append(f"str({tensor_name}.{term}) == {str(real_value)!r}")
                    else:
                        code.append(f"{tensor_name}.{term} == {real_value}")
            else:
                guard_manager = self.get_guard_manager(guard)

                # skip_no_tensor_aliasing_guards_on_parameters bring
                # unsoundness. If you compile a function with two different
                # parameters, but later on you pass on same tensor as two
                # different outputs (aliasing), Dynamo will not detect this.
                # But we deliberately take this soundness hit because this
                # usecase is quite rare and there is substantial reduction in
                # guard overhead.
                # For numpy tensors, since those are ephemeral, we don't have to
                # insert aliasing guards on them
                if not (
                    config.skip_no_tensor_aliasing_guards_on_parameters
                    and istype(value, torch.nn.Parameter)
                ) and not isinstance(guard.originating_source, NumpyTensorSource):
                    # Keep track of all the tensor guard managers to insert
                    # NoAliasing check at the end.
                    self.no_tensor_aliasing_names.append(tensor_name)
                    self.no_tensor_aliasing_guard_managers.append(guard_manager)

                output_graph = self.check_fn_manager.output_graph
                metadata = output_graph.input_source_to_sizes_strides[
                    guard.originating_source
                ]
                size = convert_to_concrete_values(metadata["size"])
                stride = convert_to_concrete_values(metadata["stride"])

                verbose_code_parts = get_verbose_code_parts(
                    get_tensor_guard_code_part(
                        value, tensor_name, size, stride, pytype, dispatch_keys
                    ),
                    guard,
                )
                guard_manager.add_tensor_match_guard(
                    value,
                    size,
                    stride,
                    tensor_name,
                    verbose_code_parts,
                    pytype,
                    dispatch_keys,
                )

                # We consider TENSOR_MATCH guard to be important enough to be
                # included in diff guard manager by default.
                if not isinstance(value, torch.nn.Parameter):
                    self.guard_manager.diff_guard_sources.add(guard.name)

            # A frame is valid for reuse with dynamic dimensions if the new
            # (user-requested) dynamic dimensions are a subset of the old
            # (already compiled) dynamic dimensions.
            #
            # It's a little non-obvious why you'd want this: in particular,
            # if an already compiled frame matches all of the guards, why
            # not just use it, why force a recompile?
            #
            # We force it for two reasons:
            #
            #   - The user *required* us to compile with a new dynamic dimension,
            #     we should not ignore that and serve up the old, specialized
            #     frame.  Listen to the user!
            #
            #   - In fact, we are obligated to *raise an error* if we fail to
            #     make the requested dimension dynamic.  If we don't
            #     recompile, we can't tell if that dimension can actually be
            #     made dynamic.
            #
            # If the new dynamic dims are a subset of the old, we already know
            # we can make them dynamic (since we made them dynamic in old).
            # This is slightly unsound, because maybe your input size is
            # [s0, s0, s1] and so you can do it dynamic if you say dynamic
            # dims {0, 1, 2} but you can't if you only do {0, 2} (because now
            # the second s0 is specialized).  But we're not entirely sure if
            # this is a good idea anyway lol... (if you want to try removing
            # this logic, be my guest!  -- ezyang 2024)
            #
            assert guard.source is not None
            static, _reason = tensor_always_has_static_shape(
                value, is_tensor=True, tensor_source=guard.originating_source
            )

            if not static:
                if hasattr(value, "_dynamo_dynamic_indices"):
                    dynamic_indices = value._dynamo_dynamic_indices
                    code_part = f"(({tensor_name}._dynamo_dynamic_indices.issubset({dynamic_indices})) if hasattr({tensor_name}, '_dynamo_dynamic_indices') else True)"  # noqa: B950
                    code.append(code_part)
                    self.get_guard_manager(guard).add_dynamic_indices_guard(
                        dynamic_indices, get_verbose_code_parts(code_part, guard)
                    )
                # In the case of us not having any dynamic dimension indices, we compiled the frame with no chance of
                # raising for this specific tensor - and any inputs with more dynamic user directives specified must be recompiled.
                else:
                    code_part = (
                        f"hasattr({tensor_name}, '_dynamo_dynamic_indices') == False"
                    )
                    code.append(code_part)
                    self.get_guard_manager(guard).add_no_hasattr_guard(
                        "_dynamo_dynamic_indices",
                        get_verbose_code_parts(code_part, guard),
                    )
            if len(code) > 0:
                self._set_guard_export_info(guard, code)

    # A util that in the case of export, adds data onto guards
    def _set_guard_export_info(self, guard, code_list, provided_guarded_object=None):
        # WARNING: It is important that cur_frame/caller do NOT stay in
        # the current frame, because they will keep things live longer
        # than they should.  See TestMisc.test_release_module_memory
        cur_frame = currentframe()
        assert cur_frame is not None
        caller = cur_frame.f_back
        del cur_frame
        assert caller is not None
        func_name = caller.f_code.co_name
        del caller
        # We use func_name for export, so might as well get a nice defensive check out of it
        assert func_name in self.__class__.__dict__, (
            f"_produce_guard_code must be called from inside GuardedCode. Called from {func_name}"
        )

        # Not all guards have names, some can be installed globally (see asserts on HAS_GRAD)
        if provided_guarded_object is None:
            name = guard.name
            guarded_object = None if not name else self.get(name)
        else:
            guarded_object = provided_guarded_object

        guarded_object_type = (
            weakref.ref(type(guarded_object)) if guarded_object is not None else None
        )
        obj_ref = None
        # Not necessary to have weakref for Enum type, but there is a bug that
        # makes hasattr(guarded_object.__class__, "__weakref__") return True.
        supports_weakref = (
            getattr(guarded_object.__class__, "__weakrefoffset__", 0) != 0
        )
        # See D64140537 for why we are checking for tuple.
        if supports_weakref and not isinstance(guarded_object, (enum.Enum, tuple)):
            obj_ref = weakref.ref(guarded_object)

        guard.set_export_info(
            func_name,
            guarded_object_type,
            code_list,
            obj_ref,
        )


# Common Sub-Expression Elimination for Python expressions.
#
# There are 2 steps to this pass:
#     1. Count the frequency of each sub-expression (i.e. inner
#        node in the AST tree)
#
#     2. Replace those that occur more than once by a fresh variable 'v'.
#        'v' will be defined in the 'preface' list (output argument to
#        'NodeTransformer')
#
# NB: the use of 'ast.unparse' while visiting the nodes makes this pass
# quadratic on the depth of the tree.
#
# NB: this pass creates a new variable for each AST node that is repeated
# more than 'USE_THRESHOLD'. e.g. if 'a.b.c.d' is used 10 times, 'a.b.c'
# and 'a.b' are also used 10 times. So, there will be a new variable for
# each of them.
class PyExprCSEPass:
    # Maximum number of times a given expression can be used without being
    # replaced by a fresh variable.
    USE_THRESHOLD = 1

    # Ad-Hoc: AST nodes this pass focuses on.
    ALLOWED_NODE_TYPES = (ast.Attribute, ast.Call, ast.Subscript)

    @dataclasses.dataclass
    class Config:
        expr_count: dict[str, int]
        expr_to_name: dict[str, str]

    class ExprCounter(ast.NodeVisitor):
        def __init__(self, config: PyExprCSEPass.Config) -> None:
            self._config = config

        def visit(self, node: ast.AST) -> Any:
            if isinstance(node, PyExprCSEPass.ALLOWED_NODE_TYPES):
                self._config.expr_count[_ast_unparse(node)] += 1
            super().visit(node)

    class Replacer(ast.NodeTransformer):
        def __init__(
            self,
            config: PyExprCSEPass.Config,
            gen_name: Callable[[], str],
        ) -> None:
            super().__init__()
            self._config = config
            self._gen_name = gen_name
            self.preface: list[str] = []

        def visit(self, node: ast.AST) -> Any:
            if isinstance(node, PyExprCSEPass.ALLOWED_NODE_TYPES):
                expr = _ast_unparse(node)

                # Replacement only occurs if a given expression is used more
                # than once.
                if self._config.expr_count[expr] > PyExprCSEPass.USE_THRESHOLD:
                    if expr not in self._config.expr_to_name:
                        # Parent 'visit' is called so that we CSE the inner expressions first.
                        #
                        # The resulting expression is used as right-hand-side of the variable
                        # assignment. i.e. we are CSE-ing the children before the parents.
                        #
                        # Indexing still uses the old 'node', since that's what was counted
                        # by the 'NodeVisitor'.
                        node_ = super().visit(node)
                        expr_ = _ast_unparse(node_)
                        var_name = self._gen_name()
                        self.preface.append(f"{var_name} = {expr_}")
                        self._config.expr_to_name[expr] = var_name
                    else:
                        var_name = self._config.expr_to_name[expr]
                    return ast.Name(var_name, ast.Load())

            return super().visit(node)

    def __init__(self) -> None:
        self._counter = 0
        self._config = self.Config(
            expr_count=collections.defaultdict(lambda: 0), expr_to_name={}
        )

    def _new_var(self, prefix: str = "_var") -> str:
        name = f"{prefix}{self._counter}"
        self._counter += 1
        return name

    def count(self, exprs: list[str]) -> None:
        counter = self.ExprCounter(self._config)
        for e in exprs:
            try:
                counter.visit(ast.parse(e))
            except SyntaxError as ex:
                log.exception("Failed to visit expr at line %s.\n%s", ex.lineno, e)
                raise

    def replace(self, expr: str) -> tuple[list[str], str]:
        replacer = self.Replacer(self._config, self._new_var)
        new_node = replacer.visit(ast.parse(expr))
        return replacer.preface, _ast_unparse(new_node)


def must_add_nn_module_guards(guard):
    # For config.guard_nn_modules=False, we can skip all the guards that
    # originate from inside of nn module except for a few categories.
    return (
        # Guard for defaults
        isinstance(guard.originating_source, DefaultsSource)
        # Guard using dict tags if the config flag is set
        or (
            config.guard_nn_modules_using_dict_tags
            and guard.create_fn is GuardBuilder.NN_MODULE
        )
    )


class DeletedGuardManagerWrapper(GuardManagerWrapper):
    def __init__(self, reason):
        super().__init__()
        self.invalidation_reason = reason

    def populate_diff_guard_manager(self):
        self.diff_guard_root = None


@dataclasses.dataclass
class GuardsState:
    output_graph: OutputGraphGuardsState
    # TODO SHAPE_ENV states here


class GuardsStatePickler(pickle.Pickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_mode = torch._subclasses.FakeTensorMode()
        self.tensor_converter = torch._subclasses.fake_tensor.FakeTensorConverter()

    @classmethod
    def _unpickle_module(cls, state):
        mod = torch.nn.Module()
        mod.__setstate__(state)
        return mod

    @classmethod
    def _unpickle_tensor(cls, meta_tensor, device, pytype, dispatch_keys_raw):
        fake_mode = torch._subclasses.FakeTensorMode()
        tensor_converter = torch._subclasses.fake_tensor.FakeTensorConverter()
        return tensor_converter.from_meta_and_device(
            fake_mode,
            meta_tensor,
            device,
            pytype,
            torch._C.DispatchKeySet.from_raw_repr(dispatch_keys_raw),
        )

    @classmethod
    def _unpickle_traceable_wrapper_subclass(
        cls, meta_tensor, device, pytype, dispatch_keys_raw, ctx, inner_data
    ):
        # Unpickle the inner tensor components. These could also be subclass instances.
        inner_tensors = {}
        for attr, unpickle_func, unpickle_func_args in inner_data:
            inner_tensors[attr] = unpickle_func(*unpickle_func_args)

        outer_size, outer_stride = meta_tensor.shape, meta_tensor.stride()
        out = type(meta_tensor).__tensor_unflatten__(
            inner_tensors, ctx, outer_size, outer_stride
        )
        out.pytype = pytype
        out.dispatch_keys = torch._C.DispatchKeySet.from_raw_repr(dispatch_keys_raw)
        return out

    @classmethod
    def _unpickle_python_module(cls, alias: str):
        return importlib.import_module(alias)

    @classmethod
    def _unpickle_dispatch_key_set(cls, raw_repr: int):
        return torch._C.DispatchKeySet.from_raw_repr(raw_repr)

    @classmethod
    def _unpickle_functorch_interpreter(cls, json: bytes):
        return torch._C._functorch.CInterpreter.deserialize(json)

    @classmethod
    def _unpickle_mapping_proxy(cls, d):
        return types.MappingProxyType(d)

    def reducer_override(self, obj):
        import sympy

        if isinstance(obj, torch.Tensor) and obj.device.type != "meta":
            from torch.utils._python_dispatch import is_traceable_wrapper_subclass

            if is_traceable_wrapper_subclass(obj):
                # inner_data is a list of tuples of:
                #   (inner attr name, unpickle func, tuple of func inputs)
                # This supports traceable wrapper subclass inner tensors.
                inner_data = []
                attrs, ctx = obj.__tensor_flatten__()
                # recursively call for inner tensor components
                for attr in attrs:
                    inner = getattr(obj, attr)
                    func, args_tuple = self.reducer_override(inner)
                    inner_data.append((attr, func, args_tuple))

                return type(self)._unpickle_traceable_wrapper_subclass, (
                    torch.empty_like(obj, device="meta"),
                    obj.device,
                    type(obj),
                    torch._C._dispatch_keys(obj).raw_repr(),
                    ctx,
                    inner_data,
                )

            return type(self)._unpickle_tensor, (
                torch.empty_like(obj, device="meta"),
                obj.device,
                type(obj),
                torch._C._dispatch_keys(obj).raw_repr(),
            )

        elif isinstance(obj, torch.nn.Module):
            if type(obj).__qualname__ == type(obj).__name__:
                return NotImplemented
            if obj.__class__.__getstate__ == torch.nn.Module.__getstate__:
                return type(self)._unpickle_module, (obj.__getstate__(),)

        elif inspect.ismodule(obj):
            return type(self)._unpickle_python_module, (obj.__name__,)

        elif isinstance(obj, torch._C.DispatchKeySet):
            return type(self)._unpickle_dispatch_key_set, (obj.raw_repr(),)

        elif isinstance(obj, torch._C._functorch.CInterpreter):
            return type(self)._unpickle_functorch_interpreter, (obj.serialize(),)

        elif (
            inspect.isclass(obj)
            and issubclass(obj, sympy.Function)
            and hasattr(obj, "_torch_handler_name")
        ):
            assert hasattr(obj, "_torch_unpickler")
            return obj._torch_unpickler, (obj._torch_handler_name,)

        elif isinstance(obj, torch.SymInt):
            raise RuntimeError(f"Cannot serialize SymInt {obj} (node: {obj.node})")

        elif isinstance(obj, types.MappingProxyType):
            return type(self)._unpickle_mapping_proxy, (obj.copy(),)

        if type(obj).__qualname__ != type(obj).__name__:
            raise RuntimeError(
                f"Type {type(obj)} for object {obj} cannot be saved "
                + "into torch.compile() package since it's defined in local scope. "
                + "Please define the class at global scope (top level of a module)."
            )

        return NotImplemented


def pickle_guards_state(state: GuardsState) -> bytes:
    buf = io.BytesIO()
    pickler = GuardsStatePickler(buf)
    pickler.dump(state)
    return buf.getvalue()


# NB: Naively, you'd expect this to only be a function that produces
# the callable that constitutes the guard.  However, there is some
# delicate handling for invalidating this check function when the
# locals/globals get invalidated, so there's some extra state
# we have to hold in this manager class.
class CheckFunctionManager:
    def __init__(
        self,
        f_code,
        output_graph=None,
        cache_entry=None,
        guard_fail_fn: Optional[Callable[[GuardFail], None]] = None,
        guard_filter_fn: Optional[
            Callable[[list[GuardFilterEntry]], list[bool]]
        ] = None,
        guards_serialization_mode: Optional[str] = None,
    ):
        guards = output_graph.guards if output_graph else None
        self._weakrefs: dict[int, ReferenceType[object]] = {}

        existing_diff_guard_sources = (
            update_diff_guard_managers_for_existing_cache_entries(cache_entry)
        )
        self.output_graph = output_graph

        # NB: Until we trace device contexts, we need to use the stack recorded at the beginning of tracing
        # in case a set default device call was made in the graph.
        self.torch_function_mode_stack = (
            output_graph.torch_function_mode_stack if output_graph else None
        )
        self.guards_serialization_mode = guards_serialization_mode

        if not justknobs_check("pytorch/compiler:guard_nn_modules"):
            log.warning("guard_nn_modules is turned off using justknobs killswitch")

        sorted_guards = sorted(guards or (), key=Guard.sort_key)
        builder, guard_manager = self.build_guards(
            sorted_guards,
            existing_diff_guard_sources,
            f_code,
            output_graph,
            None if guard_filter_fn else self.guards_serialization_mode,
        )

        if guard_filter_fn:

            def make_guard_filter_entry(guard):
                MISSING = object()
                name = strip_local_scope(guard.name)
                if name == "":
                    has_value = False
                    value = MISSING
                else:
                    has_value = True
                    value = builder.get(guard.name)
                is_global = get_global_source_name(guard.originating_source) is not None
                guard_fn = guard.create_fn
                if isinstance(guard_fn, functools.partial):
                    guard_fn = guard.create_fn.func
                return GuardFilterEntry(
                    name=name,
                    has_value=has_value,
                    value=value,
                    guard_type=guard_fn.__name__,
                    derived_guard_types=tuple(guard.guard_types)
                    if guard.guard_types
                    else (),
                    is_global=is_global,
                    orig_guard=guard,
                )

            filter_results = guard_filter_fn(
                [make_guard_filter_entry(guard) for guard in sorted_guards]
            )
            assert len(filter_results) == len(sorted_guards)
            assert all(type(x) == bool for x in filter_results)
            sorted_guards = [
                guard for i, guard in enumerate(sorted_guards) if filter_results[i]
            ]
            # Redo the guards because filtering relies on the results from the last guard builder.
            builder, guard_manager = self.build_guards(
                sorted_guards,
                existing_diff_guard_sources,
                f_code,
                output_graph,
                self.guards_serialization_mode,
            )

        self.guard_manager = guard_manager
        self.compile_check_fn(builder, sorted_guards, guard_fail_fn)

        # Keep track of weak references of objects with ID_MATCH guard. This
        # info is stored alongside optimized_code and guard_manager and is used to
        # limit the number of cache entries with same ID_MATCH'd object.
        # TODO(anijain2305) - Currently this information is stored as an attr on
        # the guard_manager itself to avoid changing CacheEntry data structure in
        # eval_frame.c. In future, we should probably replace guard_manager with a
        # queryable data structure such that this information is already present
        # in some form.
        self.guard_manager.id_matched_objs = builder.id_matched_objs

        guards_log.debug("%s", self.guard_manager)
        self.guard_manager.id_matched_objs = builder.id_matched_objs

        # Check that the guard returns True. False means that we will always
        # recompile.
        # TODO(anijain2305, ydwu4) - Skipping export because of following test
        # python -s test/dynamo/test_export.py -k test_export_with_symbool_inputs
        latency = 0.0
        if not output_graph.export and self.guards_serialization_mode != "load":
            if not self.guard_manager.check(output_graph.local_scope):
                reasons = get_guard_fail_reason_helper(
                    self.guard_manager,  # type: ignore[arg-type]
                    output_graph.local_scope,
                    CompileContext.current_compile_id(),
                )
                raise AssertionError(f"Guard check failed: {reasons}")

            if guard_manager_testing_hook_fn is not None:
                guard_manager_testing_hook_fn(
                    self.guard_manager, output_graph.local_scope
                )

            # NB for developers: n_iters is chosen to be 50 to achieve
            # statistical significance.  If you are working on a guard
            # optimization, it might be a good idea to increase this number for
            # more stabiilty during development.
            latency = profile_guard_manager(
                self.guard_manager.root, output_graph.local_scope, 50
            )
            guards_log.debug("Guard eval latency = %s us", f"{latency:.2f}")
            # Note: We use `increment_toplevel` instead of `compilation_metric`
            # here.  This is because, in scenarios where `torch._dynamo.reset`
            # is invoked, the same frame ID and compile ID may be reused during
            # a new compilation cycle.  This behavior causes issues with
            # `compilation_metric`, as it expects the metric field to be empty.
            # Ideally, we would overwrite the existing entry in such cases, but
            # we currently lack an API to support overwriting metrics.  However,
            # since these situations are rare and typically impractical to
            # account for, we simply increment at the toplevel instead.
            CompileEventLogger.increment_toplevel("guard_latency_us", int(latency))

        self.guards_state: Optional[bytes] = None
        if self.guards_serialization_mode == "save":
            output_graph_guards_state = self.output_graph.dump_guards_state()
            # Only serialize the global variables that are actually used in guards.
            used_global_vars = set()
            for guard in sorted_guards:
                if name := get_global_source_name(guard.originating_source):
                    assert isinstance(name, str)
                    used_global_vars.add(name)
            for source in self.output_graph.guard_on_key_order:
                if name := get_global_source_name(source):
                    assert isinstance(name, str)
                    used_global_vars.add(name)

            def normalize_create_fn(x):
                if isinstance(x, functools.partial):

                    def _ref(x):
                        if isinstance(x, (TensorWeakRef, weakref.ref)):
                            return x()
                        return x

                    new_args = tuple(_ref(a) for a in x.args)
                    new_keywords = {k: _ref(v) for k, v in x.keywords.items()}
                    return functools.partial(x.func, *new_args, **new_keywords)

                return x

            output_graph_guards_state = dataclasses.replace(
                output_graph_guards_state,
                global_scope={
                    k: v
                    for k, v in output_graph_guards_state.global_scope.items()
                    if k in used_global_vars
                },
                _guards=torch._guards.GuardsSet(
                    {
                        dataclasses.replace(
                            guard,
                            obj_weakref=None,
                            guarded_class_weakref=None,
                            create_fn=normalize_create_fn(guard.create_fn),
                        )
                        for guard in sorted_guards
                    }
                ),
            )
            guards_state = GuardsState(
                output_graph=output_graph_guards_state,
            )
            self.guards_state = pickle_guards_state(guards_state)

        # TODO: don't do the string rep, do something more structured here
        torch._logging.trace_structured(
            "dynamo_cpp_guards_str",
            payload_fn=lambda: f"{self.guard_manager}\nGuard latency = {latency:.2f} us",
        )
        # NB - We have to very careful of cleaning up here. Because of the
        # invalidate function, we can create a weakref finalizer that keeps
        # `self` alive for very long. Sometimes by mistake, we can run
        # invalidate for a type/object (check id_ref method) that Python can
        # leak by design, preventing us from calling the finalizer. In that
        # case, the `self` will be alive even though the cache entry will be
        # deleted (check invalidate method), which can cause a memory leak,
        # e.g., not setting output_graph = None can keep hold of nn_modules.
        self._weakrefs.clear()
        self.output_graph = None

    def build_guards(
        self,
        sorted_guards,
        existing_diff_guard_sources,
        f_code,
        output_graph,
        serialization_mode=None,
    ):
        guard_manager = GuardManagerWrapper()
        guard_manager.diff_guard_sources = existing_diff_guard_sources

        w_builder = None

        def source_ref(source):
            guard_source = source.guard_source()
            if guard_source is GuardSource.CONSTANT:
                # No need to track constants
                return source.name()
            assert w_builder
            r_builder = w_builder()
            assert r_builder is not None
            return r_builder.arg_ref(source.name())

        builder = GuardBuilder(
            f_code,
            self.id_ref,
            source_ref,
            self.lookup_weakrefs,
            output_graph.local_scope,
            output_graph.global_scope,
            guard_manager,
            self,
            serialization_mode,
        )

        # Break retain cycle. See test_release_scope_memory
        def cleanup_builder(weak_b):
            b = weak_b()
            if b:
                b.scope = None

        # Break retain cycle. See test_release_input_memory
        w_builder = weakref.ref(builder, cleanup_builder)

        guard_on_nn_modules = config.guard_nn_modules and justknobs_check(
            "pytorch/compiler:guard_nn_modules"
        )

        for guard in sorted_guards:
            if (
                not guard_on_nn_modules
                and guard.is_specialized_nn_module()
                # Default func args must be guarded on.
                # TODO: we could make use of 'DefaultsSource' and offer a .guard.is_defaults() API
                and "__defaults__" not in guard.name
                and "__kwdefaults__" not in guard.name
                and (config.skip_nnmodule_hook_guards or "hooks" not in guard.name)
            ):
                continue

            guard.create(builder)
        return builder, guard_manager

    def compile_check_fn(self, builder, guards_out, guard_fail_fn):
        # see parallel handling of ".0" / "___implicit0" in _eval_frame.c
        largs = builder.argnames
        largs += ["**___kwargs_ignored"]

        guards_log.debug("GUARDS:")

        code_parts = []
        verbose_code_parts = []
        structured_guard_fns: list[Callable[[], dict[str, Any]]] = []

        torch_function_mode_stack_check_fn = make_torch_function_mode_stack_guard(
            self.torch_function_mode_stack
        )

        # Insert the global_state guard
        self.guard_manager.root.add_global_state_guard(["___check_global_state()"])

        self.guard_manager.root.add_torch_function_mode_stack_guard(
            self.torch_function_mode_stack,
            ["___check_torch_function_mode_stack()"],
        )
        # Clear references to torch_function modes held in the list
        self.torch_function_mode_stack = None

        def add_code_part(code_part, guard, log_only=False):
            verbose_code_part = get_verbose_code_part(code_part, guard)
            guards_log.debug("%s", verbose_code_part)

            structured_guard_fns.append(
                lambda: {
                    "code": code_part,
                    "stack": (
                        structured.from_traceback(guard.stack.summary())
                        if guard and guard.stack
                        else None
                    ),
                    "user_stack": (
                        structured.from_traceback(guard.user_stack)
                        if guard and guard.user_stack
                        else None
                    ),
                }
            )

            if verbose_guards_log.isEnabledFor(logging.DEBUG):
                maybe_stack = ""
                maybe_user_stack = ""
                if guard is not None:
                    if guard.stack:
                        maybe_stack = f"\nStack:\n{''.join(guard.stack.format())}"
                    if guard.user_stack:
                        maybe_user_stack = (
                            f"\nUser stack:\n{''.join(guard.user_stack.format())}"
                        )
                verbose_guards_log.debug(
                    "Guard: %s%s%s",
                    code_part,
                    maybe_stack,
                    maybe_user_stack,
                )

            if not log_only:
                code_parts.append(code_part)
                verbose_code_parts.append(verbose_code_part)

        seen = set()
        for gcl in builder.code:
            for code in gcl.code_list:
                if code not in seen:
                    # If Cpp guard manager is enabled, we don't need to add to
                    # code_parts.
                    add_code_part(code, gcl.guard, True)
                    seen.add(code)

        no_tensor_aliasing_names = builder.no_tensor_aliasing_names
        check_tensors_fn = None
        check_tensors_verbose_fn = None

        if len(no_tensor_aliasing_names) > 1:
            # Install tensor aliasing guard. TENSOR_MATCH guards are already
            # installed for cpp guard manager.
            install_no_tensor_aliasing_guard(
                builder.no_tensor_aliasing_guard_managers,
                no_tensor_aliasing_names,
                ["check_no_aliasing(" + ", ".join(no_tensor_aliasing_names) + ")"],
            )

        aotautograd_guards: list[GuardEnvExpr] = (
            self.output_graph.aotautograd_guards if self.output_graph else []
        )

        # TODO(anijain2305) - There is a duplicate logic in Dynamo to find
        # aliased input tensors. So most probably we don't need this here.
        # Revisit.
        for guard in aotautograd_guards:
            if isinstance(guard, DuplicateInputs):
                source_a = guard.input_source_a
                source_b = guard.input_source_b
                code_part = f"{source_a.name()} is {source_b.name()}"
                install_object_aliasing_guard(
                    builder.get_guard_manager_from_source(source_a),
                    builder.get_guard_manager_from_source(source_b),
                    [code_part],
                )
                add_code_part(code_part, None, True)
            elif isinstance(guard, StorageOverlap):
                overlapping_guard_managers = [
                    builder.get_guard_manager_from_source(s)
                    for s in guard.overlapping_sources
                ]
                non_overlapping_guard_managers = [
                    builder.get_guard_manager_from_source(s)
                    for s in guard.non_overlapping_sources
                ]
                code_part = (
                    """check_overlapping("""
                    f"""overlapping=[{", ".join(s.name() for s in guard.overlapping_sources)}], """
                    f"""non_overlapping=[{", ".join(s.name() for s in guard.non_overlapping_sources)}])"""
                )
                install_storage_overlapping_guard(
                    overlapping_guard_managers,
                    non_overlapping_guard_managers,
                    [code_part],
                )
                add_code_part(code_part, None, True)
            else:
                raise RuntimeError(f"Unknown GuardEnvExpr: {guard}")

        # TODO: the "guard" here is actually just the top level SHAPE_ENV
        # which is useless.  Get ShapeEnv to pass in more provenance.
        for gcl in builder.shape_env_code:
            for code in gcl.code_list:
                # Shape env guards are already added for CPP guard manager in
                # SHAPE_ENV implementation.
                add_code_part(code, gcl.guard, True)

        # OK, all done generating guards
        if structured_guard_fns:
            torch._logging.trace_structured(
                "dynamo_guards", payload_fn=lambda: [f() for f in structured_guard_fns]
            )

        global_state = convert_frame.initial_global_state
        if global_state is None:
            # we should only hit this case in NopTests()
            global_state = convert_frame.GlobalStateGuard()
        closure_vars = {
            "___check_tensors": check_tensors_fn,
            "___check_tensors_verbose": check_tensors_verbose_fn,
            "___check_global_state": global_state.check,
            "___check_torch_function_mode_stack": torch_function_mode_stack_check_fn,
            **SYMPY_INTERP,
            **_get_closure_vars(),
        }

        self.guard_manager.finalize()

        globals_for_guard_fn = {"G": builder.scope["G"]}
        # Guard manager construction is complete. Ensure we did not miss to
        # insert a guard in cpp guard manager.
        assert len(code_parts) == 0

        self.guard_manager.closure_vars = closure_vars
        self.guard_manager.args = largs
        self.guard_manager.populate_code_parts_for_debugging()
        self.guard_manager.verbose_code_parts = verbose_code_parts
        # Grab only G, but preserve "G" because guards access it as "G"
        self.guard_manager.global_scope = globals_for_guard_fn
        self.guard_manager.guard_fail_fn = guard_fail_fn
        # will be populated by a non-owning reference to CacheEntry/ExtraState
        # when the CacheEntry is constructed
        self.guard_manager.cache_entry = None
        self.guard_manager.extra_state = None
        self.guard_manager.no_tensor_aliasing_sources = no_tensor_aliasing_names

    def invalidate(self, obj_str):
        # Some tests reveal that CheckFunctionManager has no attribute
        # guard_manager, but this case should not be of any concern.
        # This case doesn't seem easy to repro.
        if (
            hasattr(self, "guard_manager")
            and not isinstance(self.guard_manager, DeletedGuardManagerWrapper)
            and (cache_entry := self.guard_manager.cache_entry) is not None
            and (extra_state := self.guard_manager.extra_state) is not None
        ):
            assert isinstance(cache_entry, CacheEntry)
            assert isinstance(extra_state, ExtraState)
            reason = f"Cache line invalidated because {obj_str} got deallocated"
            deleted_guard_manager = DeletedGuardManagerWrapper(reason)
            extra_state.invalidate(cache_entry, deleted_guard_manager)
            self.guard_manager = deleted_guard_manager

    def id_ref(self, obj, obj_str):
        """add a weakref, return the id"""
        try:
            if id(obj) not in self._weakrefs:
                # We will clear the _weakrefs dict at the end of __init__
                # function, which will delete the callbacks as well. Therefore,
                # we are using a finalizer which is kept alive.
                self._weakrefs[id(obj)] = weakref.ref(obj)
                weakref.finalize(
                    obj, functools.partial(self.invalidate, obj_str=obj_str)
                )
        except TypeError:
            pass  # cannot weakref bool object
        return id(obj)

    def lookup_weakrefs(self, obj):
        """Lookup the _weakrefs created in id_ref function for ID_MATCH'd objects"""
        if id(obj) in self._weakrefs:
            return self._weakrefs[id(obj)]
        return None


def build_guard_function(code_parts, closure_args) -> tuple[str, str]:
    from torch._inductor.utils import IndentedBuffer

    csepass = PyExprCSEPass()
    csepass.count(code_parts)

    def replace(expr: str) -> tuple[list[str], str]:
        return csepass.replace(expr)

    # Generate the inner body of the guard function.
    # i.e. if-chain of the guard expressions.
    guard_body = IndentedBuffer()
    for expr in code_parts:
        preface, expr = replace(expr)
        guard_body.writelines(preface)
        guard_body.writeline(f"if not ({expr}):")
        with guard_body.indent():
            guard_body.writeline("return False")

    # Wrap the inner body into the actual guard function.
    guard = IndentedBuffer()
    guard.writeline("def guard(L):")
    with guard.indent():
        guard.splice(guard_body)
        guard.writeline("return True")

    # Wrap the whole guard function into another function
    # with the closure variables.
    make_guard_fn = IndentedBuffer()
    make_guard_fn.writeline(f"def ___make_guard_fn({closure_args}):")
    with make_guard_fn.indent():
        make_guard_fn.splice(guard)
        make_guard_fn.writeline("return guard")

    return guard_body.getvalue(), make_guard_fn.getvalue()


def is_recompiles_enabled():
    return torch._logging._internal.log_state.is_artifact_enabled("recompiles")


def is_recompiles_verbose_enabled():
    return torch._logging._internal.log_state.is_artifact_enabled("recompiles_verbose")


# this will only be used if cpp guards are disabled
def make_torch_function_mode_stack_guard(initial_stack):
    types = [type(x) for x in initial_stack]

    def check_torch_function_mode_stack():
        cur_stack = get_torch_function_mode_stack()

        if len(cur_stack) != len(types):
            return False

        for ty, mode in zip(types, cur_stack):
            if ty != type(mode):
                return False

        return True

    return check_torch_function_mode_stack


def recompilation_reason_for_no_tensor_aliasing_guard(guard_manager, scope):
    global_scope = dict(guard_manager.global_scope)
    ids_to_source = collections.defaultdict(list)
    for tensor_source in guard_manager.no_tensor_aliasing_sources:  # type: ignore[attr-defined]
        global_scope["__compile_source__"] = tensor_source
        tensor_id = id(eval(tensor_source, global_scope, scope))
        ids_to_source[tensor_id].append(tensor_source)

    duplicate_tensors = [
        f"{ids_to_source[key]}" for key in ids_to_source if len(ids_to_source[key]) > 1
    ]

    reason = ", ".join(duplicate_tensors)
    return [f"Duplicate tensors found: {reason}"]


def strip_local_scope(s: str) -> str:
    """
    Replace occurrences of L[...] with just the inner content.
    Handles both single and double quotes.

    This is to generate user friendly recompilation messages.
    """
    import re

    pattern = r"L\[\s*['\"](.*?)['\"]\s*\]"
    return re.sub(pattern, r"\1", s)


def get_guard_fail_reason_helper(
    guard_manager: GuardFn,
    f_locals: dict[str, object],
    compile_id: CompileId,
) -> str:
    """
    Return the reason why `guard_manager` failed.
    Updates `guard_failures` with the generated reason.
    Only the first failed check of guard_manager is reported.
    """
    scope = {"L": f_locals, "G": guard_manager.global_scope["G"]}
    scope.update(guard_manager.closure_vars)
    reasons: list[str] = []

    no_tensor_aliasing_check_failed = False

    verbose_code_parts: list[str] = []
    guard_debug_info = guard_manager.check_verbose(f_locals)  # type: ignore[attr-defined]
    # For test_export_with_map_cond, the check_verbose fail even without the
    # C++ guard manager. We need to fix the issue to remove the comment.
    # assert not guard_debug_info.result
    if not guard_debug_info.result:
        verbose_code_parts = guard_debug_info.verbose_code_parts
        # verbose_code_parts is either the actual reason (e.g. in case of
        # TENSOR_MATCH) or it could be a list of verbose_code_part that we
        # passed to the leaf guard at construction time. If its a list, we
        # walk through this list and find the guard that failed. This is
        # very important for symbolic shape guards which are currently
        # installed as a lambda guard and can encompass a long list of code_parts.

        if len(verbose_code_parts) == 1:
            if "Duplicate tensor found" in verbose_code_parts[0]:
                no_tensor_aliasing_check_failed = True
            else:
                reasons = verbose_code_parts
                verbose_code_parts = []

    if no_tensor_aliasing_check_failed:
        reasons = recompilation_reason_for_no_tensor_aliasing_guard(
            guard_manager, scope
        )
    else:
        for part in verbose_code_parts:
            global_scope = dict(guard_manager.global_scope)
            global_scope["__compile_source__"] = part
            with report_compile_source_on_error():
                try:
                    fail_reason = eval(part, global_scope, scope)
                except Exception:
                    if is_recompiles_verbose_enabled():
                        continue
                    else:
                        raise
            # Only ___check_tensors knows how to return a fancy fail reason;
            # for everything else we just report the code that failed

            if isinstance(fail_reason, bool) and not fail_reason:
                fail_reason = part
            if isinstance(fail_reason, str):
                reasons.append(fail_reason)
                if not is_recompiles_verbose_enabled():
                    break

    reason_str = f"{compile_id}: " + "; ".join(reasons)
    return strip_local_scope(reason_str)


def get_guard_fail_reason(
    guard_manager: GuardFn,
    code: types.CodeType,
    f_locals: dict[str, object],
    compile_id: CompileId,
) -> str:
    if isinstance(guard_manager, DeletedGuardManagerWrapper):
        return f"{compile_id}: {guard_manager.invalidation_reason}"
    reason_str = get_guard_fail_reason_helper(guard_manager, f_locals, compile_id)
    guard_failures[orig_code_map[code]].append(reason_str)

    try:
        if guard_manager.guard_fail_fn is not None:
            guard_manager.guard_fail_fn(
                GuardFail(reason_str or "unknown reason", orig_code_map[code])
            )
    except Exception:
        log.exception(
            "Failure in guard_fail_fn callback - raising here will cause a NULL Error on guard eval",
        )

    return reason_str


def get_and_maybe_log_recompilation_reasons(
    cache_entry, frame: DynamoFrameType
) -> list[str]:
    """
    Return the list of guard failure reasons using cache_entry.
    Logs the recompilation reason if `recompiles` logging is enabled.
    Raises a RecompileError if `config.error_on_recompile` is enabled.
    """
    reasons = []
    while cache_entry is not None:
        reason = get_guard_fail_reason(
            cache_entry.guard_manager,
            cache_entry.code,
            frame.f_locals,
            cache_entry.compile_id,
        )
        if reason:
            reasons.append(reason)
        cache_entry = cache_entry.next

    code = frame.f_code

    # at least one of "recompiles" or "recompiles_verbose" is enabled
    do_recompiles_log = is_recompiles_enabled() or is_recompiles_verbose_enabled()

    if do_recompiles_log or config.error_on_recompile:
        if is_recompiles_verbose_enabled():
            failures = "\n\n".join(
                f"guard {i} failures:\n" + textwrap.indent(reason, "- ")
                for i, reason in enumerate(reasons)
            )
        else:
            failures = textwrap.indent("\n".join(reasons), "- ")
        guard_failure_details = (
            f"triggered by the following guard failure(s):\n{failures}"
        )
        message = (
            f"Recompiling function {code.co_name} in {code.co_filename}:{code.co_firstlineno}\n"
            f"{textwrap.indent(guard_failure_details, '    ')}"
        )
        if do_recompiles_log:
            if is_recompiles_verbose_enabled():
                recompiles_verbose_log.debug(message)
            else:
                recompiles_log.debug(message)
        if config.error_on_recompile:
            raise exc.RecompileError(message)

    torch._logging.trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "recompile_reasons",
            "encoding": "json",
        },
        payload_fn=lambda: reasons,
    )

    return reasons


def update_diff_guard_managers_for_existing_cache_entries(cache_entry):
    first_cache_entry = cache_entry

    # On the first pass, go through the cache entries and accumulate the diff
    # guard sources. Different guard managers can fail with different sources.
    # So, we collect all of them first.
    acc_diff_guard_sources = set()
    while cache_entry is not None:
        acc_diff_guard_sources.update(
            cache_entry.guard_manager.collect_diff_guard_sources()
        )
        cache_entry = cache_entry.next

    # On the second pass, set the diff_guard_sources for each cache line to the
    # accumulated value. And the re-populate the diff guard manager.
    cache_entry = first_cache_entry
    while cache_entry is not None:
        cache_entry.guard_manager.diff_guard_sources = acc_diff_guard_sources
        cache_entry.guard_manager.populate_diff_guard_manager()
        cache_entry = cache_entry.next

    # return the accumulated sources to set up the new cache line.
    return acc_diff_guard_sources


def guard_error_hook(
    guard_manager: GuardFn,
    code: types.CodeType,
    f_locals: dict[str, object],
    index: int,
    last: bool,
):
    print(
        f"ERROR RUNNING GUARDS {code.co_name} {code.co_filename}:{code.co_firstlineno}"
    )
    print("lambda " + ", ".join(guard_manager.args) + ":")
    print(" ", " and\n  ".join(guard_manager.code_parts))

    print(guard_manager)

    local_scope = {"L": f_locals, **guard_manager.closure_vars}
    for guard in guard_manager.code_parts:
        try:
            eval(guard, guard_manager.global_scope, local_scope)
        except:  # noqa: B001,E722
            print(f"Malformed guard:\n{guard}")


set_guard_error_hook(guard_error_hook)


def unique(seq):
    seen = set()
    for x in seq:
        if x not in seen:
            yield x
            seen.add(x)


def make_dupe_guard(obj_source, dupe_source):
    # Note - we may end up in a situation where we invoke something like
    # def fn(x, y)
    # with fn(x, x)
    # Prior to the addition of tracking to all relevant objects, we would handle this just fine by
    # eagerly re-entering VB and rewrapping inputs, correctly creating graphargs and placeholders. However,
    # with tracking on inputs, duplicate inputs or aliased relationships may end up getting erased here -
    # In the fn(x, x) example call above look like a graph with a single input.
    # In order to ensure that we do not reuse fn(x, x) for fn(x, y), we create a duplicate input guard.

    # Note - we may not have a source, that is fine, it just means we had an object that is safe to have
    # leave unsourced - like a local list created and discharged entirely within a local scope.
    if dupe_source and dupe_source != obj_source:
        ser_source_is_local = is_from_local_source(dupe_source)
        source_is_local = is_from_local_source(obj_source)
        if is_from_flatten_script_object_source(
            dupe_source
        ) or is_from_flatten_script_object_source(obj_source):
            raise exc.UnsafeScriptObjectError(
                f"{obj_source.name()} is alising {dupe_source.name()}. This is not supported."
                f" Please do a clone for corresponding input."
            )

        # Note - both must be local, or global, or we will run afoul of a lack of merging in how we currently
        # reconcile guards builder scopes in compile_check_fn. This technically means we miss a guard here,
        # so maybe we should do this refactor before we land this...
        # TODO(voz): Combine local and global guard builders.
        if ser_source_is_local == source_is_local:
            # Note - this is a little aggressive - these being duplicate input does not always matter.
            # However, this should always be a sound guard to add here.
            return functools.partial(GuardBuilder.DUPLICATE_INPUT, source_b=dupe_source)
    return None


def install_guard(*guards, skip=0):
    """
    Add dynamo guards to the current tracing context.

    Args:
        guards: guard(s) to add
        skip: number of stack frames to ignore for debug stack trace
    """
    from torch._guards import TracingContext

    collect_debug_stack = guards_log.isEnabledFor(
        logging.DEBUG
    ) or verbose_guards_log.isEnabledFor(logging.DEBUG)
    add = TracingContext.get().guards_context.dynamo_guards.add
    for guard in guards:
        assert isinstance(guard, Guard)
        add(guard, collect_debug_stack=collect_debug_stack, skip=skip + 1)
