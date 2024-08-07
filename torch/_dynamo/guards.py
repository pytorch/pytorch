# mypy: allow-untyped-defs
from __future__ import annotations

import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import logging
import math
import os
import re
import sys
import textwrap
import types
import weakref
from contextlib import contextmanager
from copy import deepcopy
from inspect import currentframe, getframeinfo
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)
from weakref import ReferenceType

import torch
import torch.utils._device
from torch._C._dynamo.guards import (
    check_obj_id,
    check_type_id,
    dict_version,
    DictGuardManager,
    install_no_tensor_aliasing_guard,
    install_object_aliasing_guard,
    RootGuardManager,
    TensorGuards,
)
from torch._dynamo.source import (
    is_from_flatten_script_object_source,
    is_from_local_source,
    is_from_optimizer_source,
    TensorProperty,
    TensorPropertySource,
)
from torch._guards import (
    CompileContext,
    CompileId,
    DuplicateInputs,
    Guard,
    GuardBuilderBase,
    GuardEnvExpr,
    GuardSource,
    Source,
)
from torch._logging import structured
from torch.fx.experimental.symbolic_shapes import (
    EqualityConstraint,
    is_symbolic,
    SYMPY_INTERP,
)
from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef

from . import config, convert_frame, exc, mutation_guard
from .eval_frame import set_guard_error_hook
from .source import (
    AttrSource,
    ChainedSource,
    ConstDictKeySource,
    DefaultsSource,
    FlattenScriptObjectSource,
    FSDPNNModuleSource,
    GetItemSource,
    GlobalSource,
    GlobalStateSource,
    GlobalWeakRefSource,
    GradSource,
    LocalSource,
    NNModuleSource,
    NumpyTensorSource,
    ODictGetItemSource,
    OptimizerSource,
    ScriptObjectQualifiedNameSource,
    ShapeEnvSource,
    SubclassAttrListSource,
    TupleIteratorGetItemSource,
    TypeSource,
    UnspecializedBuiltinNNModuleSource,
    UnspecializedNNModuleSource,
    UnspecializedParamBufferSource,
    WeakRefCallSource,
)
from .types import CacheEntry, ExtraState, GuardedCode, GuardFail, GuardFn  # noqa: F401
from .utils import (
    common_constant_types,
    dict_keys_repr,
    get_custom_getattr,
    guard_failures,
    istype,
    key_is_id,
    key_to_id,
    orig_code_map,
    tensor_always_has_static_shape,
    tuple_iterator_getitem,
    tuple_iterator_len,
    unpatched_nn_module_getattr,
    verify_guard_fn_signature,
)


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from sympy import Symbol


log = logging.getLogger(__name__)
guards_log = torch._logging.getArtifactLogger(__name__, "guards")
recompiles_log = torch._logging.getArtifactLogger(__name__, "recompiles")
recompiles_verbose_log = torch._logging.getArtifactLogger(
    __name__, "recompiles_verbose"
)
verbose_guards_log = torch._logging.getArtifactLogger(__name__, "verbose_guards")


class GuardManager:
    """
    A helper class that contains the root guard manager. An instance of this
    class is stored in the Dynamo cache entry, so that the cache entry can
    access the RootGuardManager stored in the "root" attribute and directly call
    the check_nopybind from C++.
    """

    def __init__(self):
        self.root = RootGuardManager()

        self.closure_vars = None
        self.args = None
        self.code_parts = None
        self.verbose_code_parts = None
        self.global_scope = None
        self.guard_fail_fn = None
        self.cache_entry = None
        self.extra_state = None
        self.id_matched_objs = None
        self.no_tensor_aliasing_sources = []

        self.print_no_tensor_aliasing_guard = True

    @contextmanager
    def _preserve_print_no_tensor_aliasing_flag(self):
        self.print_no_tensor_aliasing_guard = True
        try:
            yield
        finally:
            self.print_no_tensor_aliasing_guard = True

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
                if isinstance(guard, torch._C._dynamo.guards.NO_TENSOR_ALIASING):  # type: ignore[attr-defined]
                    if self.print_no_tensor_aliasing_guard:
                        self.print_no_tensor_aliasing_guard = False
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

        with self._preserve_print_no_tensor_aliasing_flag():
            body = IndentedBufferWithPrefix()
            body.tabwidth = 1
            body.writeline("", skip_prefix=True)
            body.writeline("TREE_GUARD_MANAGER:", skip_prefix=True)
            body.writeline("RootGuardManager")
            self.construct_manager_string(self.root, body)
            for guard in self.root.get_epilogue_lambda_guards():
                body.writelines(self.get_guard_lines(guard))
            return body.getvalue()

    def check(self, x):
        # Only needed for debugging purposes.
        return self.root.check(x)

    def check_verbose(self, x):
        # Only needed for debugging purposes.
        return self.root.check_verbose(x)


def from_numpy(a):
    # If not numpy array, piggy back on e.g. tensor guards to check type
    return torch.as_tensor(a) if isinstance(a, (np.generic, np.ndarray)) else a


# For user stack printing
@functools.lru_cache(None)
def uninteresting_files():
    import torch._dynamo.external_utils

    mods = [
        torch._dynamo.external_utils,
    ]
    return {inspect.getfile(m) for m in mods}


CLOSURE_VARS = {
    "___check_type_id": check_type_id,
    "___check_obj_id": check_obj_id,
    "___odict_getitem": collections.OrderedDict.__getitem__,
    "___key_to_id": key_to_id,
    "___dict_version": dict_version,
    "___dict_contains": lambda a, b: a in b,
    "___tuple_iterator_len": tuple_iterator_len,
    "___tuple_iterator_getitem": tuple_iterator_getitem,
    "__math_isnan": math.isnan,
    "__numpy_isnan": None if np is None else np.isnan,
    "inf": float("inf"),
    "__load_module": importlib.import_module,
    "utils_device": torch.utils._device,
    "device": torch.device,
    "___from_numpy": from_numpy,
    "___as_tensor": torch.as_tensor,
    "torch": torch,
    "inspect": inspect,
}

if sys.version_info[:2] <= (3, 8):
    # [Note: Python Version <= 3.8]
    # This branch should be dropped when we drop support for Python 3.8.
    # Reason: 'ast.unparse' function was introduced in Python 3.9.

    try:
        import astunparse  # type: ignore[import]

        def _ast_unparse(node: ast.AST) -> str:
            return astunparse.unparse(node).replace("\n", "")

        HAS_UNPARSE_FUNCTIONS = True
    except ImportError:
        HAS_UNPARSE_FUNCTIONS = False
else:
    HAS_UNPARSE_FUNCTIONS = True

    def _ast_unparse(node: ast.AST) -> str:
        return ast.unparse(node).replace("\n", "")


def strip_function_call(name):
    """
    "___odict_getitem(a, 1)" => "a"
    "a.layers[slice(2)][0]._xyz" ==> "a"
    "getattr(a.layers[slice(2)][0]._abc, '0')" ==> "a"
    "getattr(getattr(a.x[3], '0'), '3')" ==> "a"
    "a.layers[slice(None, -1, None)][0]._xyz" ==> "a"
    """
    # recursively find valid object name in function
    valid_name = re.compile("[A-Za-z_].*")
    curr = ""
    for char in name:
        if char in " (":
            curr = ""
        elif char in "),[]":
            if curr and curr != "None" and valid_name.match(curr):
                return strip_function_call(curr)
        else:
            curr += char

    return strip_getattr_getitem(name)


def strip_getattr_getitem(name):
    """
    "a[1]" => "a"
    "a.foo" => "a"
    """
    return re.split(r"[.\[]", name)[0]


def get_verbose_code_part(code_part: str, guard: Guard) -> str:
    extra = ""
    if guard.user_stack:
        for fs in reversed(guard.user_stack):
            if fs.filename not in uninteresting_files():
                extra = f"  # {format_frame(fs, line=True)}"
                break
    elif guard.stack:
        extra = f"  # {format_frame(guard.stack.summary()[-1])}"

    return f"{code_part:<60}{extra}"


def get_verbose_code_parts(
    code_parts: Union[str | List[str]], guard: Guard
) -> List[str]:
    if not isinstance(code_parts, list):
        code_parts = [code_parts]
    return [get_verbose_code_part(code_part, guard) for code_part in code_parts]


def convert_to_concrete_values(size_or_stride):
    converted: List[Optional[int]] = []
    for dim in size_or_stride:
        if not is_symbolic(dim):
            converted.append(dim)
        else:
            assert isinstance(dim, torch.SymInt)
            converted.append(dim.node.maybe_as_int())
    return converted


def get_tensor_guard_code_part(value, name, sizes, strides):
    pytype = type(value)
    dispatch_key = (
        torch._C._dispatch_keys(value) | torch._C._dispatch_tls_local_include_set()
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
    return list(dct.keys()).index(key)


def get_key_index_source(source, index):
    return f"list({source}.keys())[{index}]"


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
    source_name = source.name()
    if isinstance(source.index, ConstDictKeySource):
        index = source.index.index
    else:
        assert isinstance(base_example_value, dict)
        index = get_key_index(base_example_value, source.index)

    key_source = get_key_index_source(base_source_name, index)
    key_example_value = list(base_example_value.keys())[index]
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
    return source.is_dict_key() and not isinstance(source, GradSource)


# The ready to eval generated code (possibly multiple parts) for a guard, plus
# the original guard object that created it for provenance
@dataclasses.dataclass
class GuardCodeList:
    code_list: List[str]
    guard: Guard


class GuardManagerType(enum.Enum):
    GUARD_MANAGER = 1
    DICT_GUARD_MANAGER = 2
    DICT_SUBCLASS_GUARD_MANAGER = 3


class GuardBuilder(GuardBuilderBase):
    def __init__(
        self,
        id_ref: Callable[[Any], str],
        source_ref: Callable[[Source], str],
        lookup_weakrefs: Callable[[object], ReferenceType[object]],
        local_scope: Dict[str, object],
        global_scope: Dict[str, object],
        guard_manager: Optional[GuardManager],
        check_fn_manager: CheckFunctionManager,
    ):
        self.id_ref = id_ref
        self.source_ref = source_ref
        self.lookup_weakrefs = lookup_weakrefs
        self.scope: Dict[str, Dict[str, object]] = {"L": local_scope, "G": global_scope}
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

        self.argnames: List[str] = []
        # Code is python expression strings generated for each guard
        self.code: List[GuardCodeList] = []
        # shape_env_code is only used by builder and is used for
        # shape env code.  This exists only because we need to make sure
        # shape env guards get run after tensor match guards (since the
        # tensor match guards make sure we actually have tensors)
        self.shape_env_code: List[GuardCodeList] = []

        # [Note - On Eager Tensor Guards]
        # Most of the time, we generate Python code in a guard to directly
        # check various properties.  However, tensors are a bit special;
        # it is too slow to check their properties one-by-one in Python.
        # Instead, there is a C++ function TensorGuards.check which takes
        # all of the tensor arguments and checks them all against compile-time
        # examples entirely in C++.  Thus, every time we process a
        # TENSOR_MATCH guard, we just add another entry to
        # tensor_check_names/tensor_check_examples, saying "for this local,
        # check it against this example", and it all ends up getting
        # swept up into a single call to ___check_tensors.  Invariant:
        # len(tensor_check_names) == len(tensor_check_examples).
        # TODO: something here
        self.tensor_check_names: List[str] = []
        self.tensor_check_examples: List[torch.Tensor] = []
        self.tensor_check_guards: List[Guard] = []
        self.tensor_check_guard_managers: List[GuardManager] = []

        self.check_fn_manager: CheckFunctionManager = check_fn_manager

        # Collect the ids of dicts which need key order guarding. source_name is
        # not sufficient because for nn modules, we can have different sources
        # to access the same object - self._module["param"] is same as
        # self.param.
        self.key_order_guarded_dict_ids = set()
        for source_name in self.check_fn_manager.output_graph.guard_on_key_order:
            self.key_order_guarded_dict_ids.add(id(self.get(source_name)))

        # Keep track of weak references of objects with ID_MATCH guard. This
        # info is stored alongside optimized_code and check_fn and is used to
        # limit the number of cache entries with same ID_MATCH'd object.
        self.id_matched_objs: Dict[str, ReferenceType[object]] = {}

        # Save the guard managers to avoid repeatedly traversing sources.
        self._cached_guard_managers: Dict[
            str, torch._C._dynamo.guards.GuardManager
        ] = {}

    def guard_on_dict_keys_and_ignore_order(self, example_value, guard):
        dict_mgr = self.get_guard_manager(guard)
        if isinstance(dict_mgr, DictGuardManager):
            raise NotImplementedError(
                "Not expecting a DictGuardManager. Seems like Dynamo incorrectly "
                f"added the dict to tx.output.guard_on_key_order for {guard.name}"
            )

        # Iterate over the dicts and install a dict_getitem_manager.
        dict_source = guard.originating_source.name()
        for key in example_value.keys():
            value = example_value[key]
            value_source = GetItemSource(guard.originating_source, index=key)
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

        for idx, key in enumerate(value.keys()):
            key_source = get_key_index_source(guard.name, idx)
            key_manager = dict_mgr.get_key_manager(
                index=idx,
                source=key_source,
                example_value=key,
                guard_manager_enum=GuardManagerType.GUARD_MANAGER,
            )
            if key_is_id(key):
                # Install ID_MATCH guard
                id_val = self.id_ref(key)
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
                key_source = f"list({source_name}.keys())[{index!r}]"
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

        all_class_attribute_names: Set[str] = set()
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
                example_value=mod_dict,
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
            assert isinstance(example_value, dict)
            # If keys method is not overriden, we can use PyDict_Next to get key
            # orderings. Read more in guards.cpp
            if type(example_value).keys is type({}).keys:
                guard_manager_enum = GuardManagerType.DICT_GUARD_MANAGER
            else:
                guard_manager_enum = GuardManagerType.DICT_SUBCLASS_GUARD_MANAGER
        return guard_manager_enum

    def manager_guards_on_keys(self, mgr_enum):
        return (
            mgr_enum == GuardManagerType.DICT_GUARD_MANAGER
            or mgr_enum == GuardManagerType.DICT_SUBCLASS_GUARD_MANAGER
        )

    def get_global_guard_manager(self):
        assert self.guard_manager  # to make mypy happy
        return self.guard_manager.root.globals_dict_manager(
            f_globals=self.scope["G"],
            source="G",
            example_value=self.scope["G"],
            guard_manager_enum=GuardManagerType.GUARD_MANAGER,
        )

    def get_guard_manager_from_source(self, source):
        assert self.guard_manager  # to make mypy happy
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
            # RootGuardManager accepts a dict but still its not a
            # DictGuardManager because we will eventually move to
            # fastlocals.
            out = root_guard_manager.dict_getitem_manager(
                key=source.local_name,
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
        elif istype(source, GradSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.grad_manager(
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
        elif istype(source, GetItemSource):
            assert base_guard_manager  # to make mypy happy
            if isinstance(base_example_value, (dict, collections.OrderedDict)):
                # TODO(anijain2305) - Consider isolating GetItemSource and
                # DictGetItemSource (or maybe use ODictGetItemSource for
                # dicts) so that GetItemSource is only for non dict objects.
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
            elif isinstance(base_example_value, list) and not source.index_is_slice:
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
        elif istype(source, ODictGetItemSource):
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
                assert base_guard_manager  # to make mypy happy
                out = base_guard_manager.dict_getitem_manager(
                    key=source.index,
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
        elif isinstance(source, WeakRefCallSource):
            assert base_guard_manager  # to make mypy happy
            out = base_guard_manager.weakref_call_manager(
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
        closure_vars=CLOSURE_VARS,
        is_epilogue=True,
    ):
        # Adds a lambda leaf guard to the root guard manager. It wraps the
        # code_parts in a function object which is then passed on to the leaf
        # guard.
        make_guard_fn_args = ", ".join(closure_vars.keys())
        guard_body, pycode = build_guard_function(code_parts, make_guard_fn_args)
        out: Dict[str, Any] = {}
        globals_for_guard_fn = {"G": self.scope["G"]}
        exec(pycode, globals_for_guard_fn, out)
        guard_fn = out["___make_guard_fn"](*closure_vars.values())
        assert self.guard_manager  # to make mypy happy
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
    def get(self, name: str) -> Any:
        return eval(name, self.scope, CLOSURE_VARS)

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
        base = strip_getattr_getitem(strip_function_call(name))
        if base not in self.argnames:
            if re.match(r"[a-zA-Z0-9_]+", base):
                if re.match(r"^\d+$", base):
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

        if config.enable_cpp_guard_manager:
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
                base_manager.add_no_hasattr_guard(
                    attr, get_verbose_code_parts(code, guard)
                )
        else:
            self._produce_guard_code(guard, [code])

    def NOT_PRESENT_IN_GENERIC_DICT(self, guard: Guard, attr=None) -> None:
        assert attr is not None
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        assert isinstance(val, torch.nn.Module)

        base_manager = self.get_guard_manager(guard)

        mod_dict_source = f"{guard.name}.__dict__"
        mod_generic_dict_manager = base_manager.get_generic_dict_manager(
            source=mod_dict_source,
            example_value=val.__dict__,
            guard_manager_enum=GuardManagerType.GUARD_MANAGER,
        )

        code = f"not ___dict_contains({attr!r}, {ref}.__dict__)"
        mod_generic_dict_manager.add_dict_contains_guard(
            False, attr, get_verbose_code_parts(code, guard)
        )

    def TYPE_MATCH(self, guard: Guard) -> None:
        # ___check_type_id is same as `id(type(x)) == y`
        t = type(self.get(guard.name))
        obj_id = self.id_ref(t)
        code = f"___check_type_id({self.arg_ref(guard)}, {obj_id})"
        self._set_guard_export_info(guard, [code])

        if config.enable_cpp_guard_manager:
            self.get_guard_manager(guard).add_type_match_guard(
                obj_id, get_verbose_code_parts(code, guard)
            )
        else:
            self._produce_guard_code(guard, [code])

    def DICT_VERSION(self, guard: Guard):
        # ___check_dict_version is same as `dict_version(x) == y`
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        version = dict_version(self.get(guard.name))
        code = f"___dict_version({ref}) == {version}"
        self._set_guard_export_info(guard, [code])

        if config.enable_cpp_guard_manager:
            # TODO(anijain2305) - Delete this when DictGuardManager uses tags
            # for dicts.
            self.get_guard_manager(guard).add_dict_version_guard(
                val, get_verbose_code_parts(code, guard)
            )
        else:
            self._produce_guard_code(guard, [code])

    def DICT_CONTAINS(self, guard: Guard, key: str, invert: bool):
        dict_ref = self.arg_ref(guard)

        maybe_not = "not " if invert else ""
        code = f"{maybe_not}___dict_contains({key!r}, {dict_ref})"
        self._set_guard_export_info(guard, [code])

        if config.enable_cpp_guard_manager:
            self.get_guard_manager(guard).add_dict_contains_guard(
                not invert, key, get_verbose_code_parts(code, guard)
            )
        else:
            self._produce_guard_code(guard, [code])

    def ID_MATCH(self, guard: Guard):
        # ___check_obj_id is same as `id(x) == y`
        if isinstance(guard.originating_source, TypeSource):
            # optional optimization to produce cleaner/faster guard code
            return self.TYPE_MATCH(
                Guard(guard.originating_source.base, GuardBuilder.TYPE_MATCH)  # type: ignore[arg-type]
            )

        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        id_val = self.id_ref(val)
        code = f"___check_obj_id({ref}, {id_val})"
        self._set_guard_export_info(guard, [code])

        if config.enable_cpp_guard_manager:
            self.get_guard_manager(guard).add_id_match_guard(
                id_val, get_verbose_code_parts(code, guard)
            )
        else:
            self._produce_guard_code(guard, [code])

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

        if config.enable_cpp_guard_manager:
            self.get_guard_manager(guard).add_not_none_guard(
                get_verbose_code_parts(code, guard)
            )
        else:
            self._produce_guard_code(guard, [code])

    def NAME_MATCH(self, guard: Guard):
        self._guard_on_attribute(guard, "__name__", GuardBuilder.EQUALS_MATCH)

    def DATA_PTR_MATCH(self, guard: Guard):
        # Add a type check. C++ guard has the type check internally, so only
        # enable it for Python guards.
        if not config.enable_cpp_guard_manager:
            self.TYPE_MATCH(guard)

        obj = self.get(guard.name)
        code = f"{self.arg_ref(guard)}.data_ptr() == {obj.data_ptr()}"
        self._set_guard_export_info(guard, [code])

        if config.enable_cpp_guard_manager:
            self.get_guard_manager(guard).add_data_ptr_guard(
                obj, get_verbose_code_parts(code, guard)
            )
        else:
            self._produce_guard_code(guard, [code])

    def DUAL_LEVEL(self, guard: Guard):
        # Invalidate dual level if current dual level is different than the one
        # in the fx graph
        dual_level = torch.autograd.forward_ad._current_level
        code = [f"torch.autograd.forward_ad._current_level == {dual_level}"]
        self._set_guard_export_info(guard, [code])
        if config.enable_cpp_guard_manager:
            # TODO(anijain2305) - Consider this moving this guard to C++
            forward_ad = torch.autograd.forward_ad

            def fn(x):
                return forward_ad._current_level == dual_level

            assert self.guard_manager  # to make mypy happy
            self.guard_manager.root.add_lambda_guard(
                fn, get_verbose_code_parts(code, guard)
            )
        else:
            self._produce_guard_code(guard, code)

    def FUNCTORCH_STACK_MATCH(self, guard: Guard):
        # Invalidate functorch code if current level is different than
        # the one when FX graph was generated
        cis = torch._functorch.pyfunctorch.retrieve_all_functorch_interpreters()
        states = [ci.get_state() for ci in cis]
        code = [f"torch._functorch.pyfunctorch.compare_functorch_state({states})"]
        self._set_guard_export_info(guard, code)

        if config.enable_cpp_guard_manager:
            # TODO(anijain2305) - Consider this moving this guard to C++
            compare_fn = torch._functorch.pyfunctorch.compare_functorch_state

            def fn(x):
                return compare_fn(states)

            assert self.guard_manager  # to make mypy happy
            self.guard_manager.root.add_lambda_guard(
                fn, get_verbose_code_parts(code, guard)
            )
        else:
            self._produce_guard_code(guard, code)

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
        if config.enable_cpp_guard_manager:
            self.get_guard_manager(guard).add_lambda_guard(
                metadata_checker, get_verbose_code_parts(global_name, guard)
            )
        else:
            global_scope = self.get("G")
            global_scope[global_name] = metadata_checker
            code = [f"{global_name}({self.get(guard.name)})"]
            self._produce_guard_code(guard, code)

    def EQUALS_MATCH(self, guard: Guard):
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        t = type(val)
        if np:
            np_types: Tuple[Type[Any], ...] = (
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
        ok_types = tuple(
            common_constant_types
            | {
                type,
                list,
                tuple,
                set,
                frozenset,
                slice,
                range,
                torch.Size,
                *np_types,
            }
        )
        if istype(val, dict):
            assert all(
                istype(x, ok_types) for x in itertools.chain(val.keys(), val.values())
            )
        else:
            assert istype(
                val,
                ok_types,
            ), f"Unexpected type {type(val)}, not in {ok_types}"

        # Special case for nan because float("nan") == float("nan") evaluates to False
        if istype(val, float) and math.isnan(val):
            self.TYPE_MATCH(guard)
            code = []
            code.append(f"__math_isnan({ref})")
            self._set_guard_export_info(guard, code)

            if config.enable_cpp_guard_manager:
                self.get_guard_manager(guard).add_lambda_guard(
                    CLOSURE_VARS["__math_isnan"], get_verbose_code_parts(code, guard)
                )
            else:
                self._produce_guard_code(guard, code)
            return

        # Python math library doesn't support complex nan, so we need to use numpy
        if istype(val, complex) and np.isnan(val):
            self.TYPE_MATCH(guard)
            code = []
            code.append(f"__numpy_isnan({ref})")
            self._set_guard_export_info(guard, code)

            if config.enable_cpp_guard_manager:
                self.get_guard_manager(guard).add_lambda_guard(
                    CLOSURE_VARS["__numpy_isnan"], get_verbose_code_parts(code, guard)
                )
            else:
                self._produce_guard_code(guard, code)
            return

        if config.enable_cpp_guard_manager:
            # Construct a debug string to put into the c++ equals match guard.
            code = [f"{ref} == {val!r}"]
            self.get_guard_manager(guard).add_equals_match_guard(
                val, get_verbose_code_parts(code, guard)
            )
            self._set_guard_export_info(guard, code)
            return

        code = []

        # If matching equality against list/tuple, we must also check that
        # the internal types match.  (TODO: what about nested lists?)
        if istype(val, (list, tuple)):
            # NB: SEQUENCE_LENGTH takes care of the outer __check_type_id test
            self.SEQUENCE_LENGTH(guard)

            for idx, elem in enumerate(val):
                code.append(
                    f"___check_type_id({ref}[{idx}], {self.id_ref(type(elem))})"
                )
        else:
            # Add type check to prevent equality check between tensor and non-tensor.
            self.TYPE_MATCH(guard)

        if istype(val, torch.Size):
            val = tuple(val)

        # Code object can not be compared against their string representation
        # I.e `eval(f"{compile('2+2','','exec')!r}")` raises SyntaxError
        assert not istype(val, types.CodeType)

        # TODO: It feels like it would be better to just implement our own
        # equality test in C that handles all of the necessary type checking
        # and NaN tests
        code.append(f"{ref} == {val!r}")
        self._produce_guard_code(guard, code)
        self._set_guard_export_info(guard, code)

    def CONSTANT_MATCH(self, guard: Guard):
        val = self.get(guard.name)
        if istype(val, (bool, type(None), types.CodeType)):
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
            exc.unimplemented(f"Guard setup for uninitialized class {type(val)}")

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
        # This guard is used to check lenght of PySequence objects like list,
        # tuple, collections.deque etc
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)

        if not (config.enable_cpp_guard_manager and isinstance(value, dict)):
            # C++ DICT_LENGTH checks for type
            self.TYPE_MATCH(guard)

        code = []
        if len(value) == 0:
            code.append(f"not {ref}")
        else:
            code.append(f"len({ref}) == {len(value)}")

        self._set_guard_export_info(guard, code)
        if config.enable_cpp_guard_manager:
            if isinstance(value, dict):
                self.get_guard_manager(guard).add_dict_length_check_guard(
                    len(value), get_verbose_code_parts(code, guard)
                )
            else:
                self.get_guard_manager(guard).add_length_check_guard(
                    len(value), get_verbose_code_parts(code, guard)
                )
        else:
            self._produce_guard_code(guard, code)

    def TUPLE_ITERATOR_LEN(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)

        if not config.enable_cpp_guard_manager:
            # C++ guard already checks the type
            self.TYPE_MATCH(guard)

        code = []
        code.append(f"___tuple_iterator_len({ref}) == {tuple_iterator_len(value)}")
        self._set_guard_export_info(guard, code)

        if config.enable_cpp_guard_manager:
            t = type(value)
            obj_id = self.id_ref(t)

            self.get_guard_manager(guard).add_tuple_iterator_length_guard(
                tuple_iterator_len(value), obj_id, get_verbose_code_parts(code, guard)
            )
        else:
            self._produce_guard_code(guard, code)

    # TODO(voz): Deduplicate w/ AOTAutograd dupe input guards
    def DUPLICATE_INPUT(self, guard, source_b):
        ref_a = self.arg_ref(guard)
        ref_b = self.arg_ref(source_b.name())

        if is_from_optimizer_source(
            guard.originating_source
        ) or is_from_optimizer_source(source_b):
            return

        code = [f"{ref_b} is {ref_a}"]
        self._set_guard_export_info(guard, code)

        if config.enable_cpp_guard_manager:
            install_object_aliasing_guard(
                self.get_guard_manager(guard),
                self.get_guard_manager_from_source(source_b),
                get_verbose_code_parts(code, guard),
            )
        else:
            self._produce_guard_code(guard, code)

    def DICT_KEYS(self, guard):
        # Guard on the keys and their order
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)

        self.TYPE_MATCH(guard)
        code = []
        any_key_is_id = any(key_is_id(k) for k in value.keys())
        const_keys_repr = dict_keys_repr(
            key_to_id(value),
            local=is_from_local_source(guard.originating_source),
        )
        if any_key_is_id:
            code.append(f"___key_to_id({ref}) == {const_keys_repr}")
        else:
            code.append(f"list({ref}.keys()) == {const_keys_repr}")

        self._set_guard_export_info(guard, code)
        if config.enable_cpp_guard_manager:
            if self.requires_key_order_guarding(guard.originating_source):
                self.guard_on_dict_keys_and_order(value, guard)
            else:
                self.guard_on_dict_keys_and_ignore_order(value, guard)
        else:
            self._produce_guard_code(guard, code)

    def WEAKREF_ALIVE(self, guard):
        code = [f"{self.arg_ref(guard)} is not None"]

        self._set_guard_export_info(guard, code)
        if config.enable_cpp_guard_manager:
            self.get_guard_manager(guard).add_not_none_guard(
                get_verbose_code_parts(code, guard)
            )
        else:
            self._produce_guard_code(guard, code)

    def DICT_CONST_KEYS(self, guard):
        """Constant keys match"""
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)

        if not config.enable_cpp_guard_manager:
            # DictGuardManager supports TYPE_MATCH internally
            self.TYPE_MATCH(guard)

        code = []
        code.append(f"list({ref}.keys()) == {list(value.keys())!r}")
        self._set_guard_export_info(guard, code)

        if config.enable_cpp_guard_manager:
            if self.requires_key_order_guarding(guard.originating_source):
                self.guard_on_dict_keys_and_order(value, guard)
            else:
                self.guard_on_dict_keys_and_ignore_order(value, guard)
        else:
            self._produce_guard_code(guard, code)

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

        if config.enable_cpp_guard_manager:
            self.get_guard_manager(guard).add_default_device_guard(
                get_verbose_code_parts(code, guard)
            )
        else:
            self._produce_guard_code(guard, code)

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
            source_pairs: List[Tuple[Source, Source]] = []
            derived_equalities: List[  # type: ignore[type-arg]
                Tuple[Source, Union[Source, Symbol], Callable]
            ] = []
            phantom_symbols: Dict[str, Symbol] = {}
            for constraint in output_graph.export_constraints:
                if constraint.t_id in output_graph.tracked_fakes_id_to_source:
                    torch.export.dynamic_shapes._process_equalities(
                        constraint,
                        get_sources,
                        output_graph.shape_env,
                        source_pairs,
                        derived_equalities,
                        phantom_symbols,
                    )
                else:
                    log.warning("Untracked tensor used in export constraints")
            equalities_inputs = EqualityConstraint(
                source_pairs=source_pairs,
                derived_equalities=derived_equalities,
                phantom_symbols=list(phantom_symbols.values()),
                warn_only=False,
            )
        else:
            equalities_inputs = None
        guards = output_graph.shape_env.produce_guards(
            [a.fake for a in fs],
            [a.source for a in fs],
            input_contexts=input_contexts,
            equalities_inputs=equalities_inputs,
            source_ref=self.source_ref,
            # Export keeps static.
            ignore_static=(not self.check_fn_manager.output_graph.export),
        )
        # When exporting, we may work with the shape constraints some more in
        # postprocessing, so don't freeze yet
        if not self.check_fn_manager.output_graph.export:
            output_graph.shape_env.freeze()

        for shape_guard in guards:
            self._set_guard_export_info(guard, [shape_guard])

        if config.enable_cpp_guard_manager:
            # Install all the symbolic guards in one lambda guard. These are run
            # at the very end of the RootGuardManager via epilogue guards.
            # TODO(anijain2305,williamwen42) - Consider moving this to C++.
            code_parts = guards
            self.add_python_lambda_leaf_guard_to_root(
                code_parts,
                get_verbose_code_parts(code_parts, guard),
                closure_vars={**SYMPY_INTERP, **CLOSURE_VARS},
            )
        else:
            for shape_guard in guards:
                self._produce_guard_code(guard, [shape_guard], shape_env=True)

    def TENSOR_MATCH(self, guard: Guard, value=None):
        # For FSDP modules, we can skip guards on nn module tensors because FSDP
        # eager assumes that the params are unchanged once the model is wrapped.
        if guard.is_fsdp_module():
            return

        # For tensors that are part of the Dynamo extracted Fx graph module, an
        # ID_MATCH suffices. Once we turn on inline_inbuilt_nn_modules, these
        # will be lifted as inputs and have a TENSOR_MATCH guard.
        # For numpy tensors, always use TENSOR_MATCH because __from_numpy leads
        # to a new tensor everytime and therefore id differs.
        if (
            guard.is_specialized_nn_module()
            and not isinstance(guard.originating_source, NumpyTensorSource)
        ) or match_on_id_for_tensor(guard):
            self.ID_MATCH(guard)
        else:
            if isinstance(value, TensorWeakRef):
                value = value()

            value = value if value is not None else self.get(guard.name)
            assert isinstance(value, torch.Tensor)

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
            code: List[str] = []
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
                self.tensor_check_examples.append(value)
                self.tensor_check_names.append(tensor_name)
                self.tensor_check_guards.append(guard)

                if config.enable_cpp_guard_manager:
                    guard_manager = self.get_guard_manager(guard)
                    # Keep track of all the tensor guard managers to insert
                    # NoAliasing check at the end.
                    self.tensor_check_guard_managers.append(guard_manager)

                    output_graph = self.check_fn_manager.output_graph
                    metadata = output_graph.input_source_to_sizes_strides[
                        guard.originating_source
                    ]
                    size = convert_to_concrete_values(metadata["size"])
                    stride = convert_to_concrete_values(metadata["stride"])

                    verbose_code_parts = get_verbose_code_parts(
                        get_tensor_guard_code_part(value, tensor_name, size, stride),
                        guard,
                    )
                    guard_manager.add_tensor_match_guard(
                        value,
                        size,
                        stride,
                        tensor_name,
                        verbose_code_parts,
                    )

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
            static, reason = tensor_always_has_static_shape(
                value, is_tensor=True, tensor_source=guard.originating_source
            )

            if not static:
                if hasattr(value, "_dynamo_dynamic_indices"):
                    dynamic_indices = value._dynamo_dynamic_indices
                    code_part = f"(({tensor_name}._dynamo_dynamic_indices.issubset({dynamic_indices})) if hasattr({tensor_name}, '_dynamo_dynamic_indices') else True)"  # noqa: B950
                    code.append(code_part)
                    if config.enable_cpp_guard_manager:
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
                    if config.enable_cpp_guard_manager:
                        self.get_guard_manager(guard).add_no_hasattr_guard(
                            "_dynamo_dynamic_indices",
                            get_verbose_code_parts(code_part, guard),
                        )
            if len(code) > 0:
                self._set_guard_export_info(guard, code)
                if not config.enable_cpp_guard_manager:
                    self._produce_guard_code(guard, code)

    # A util that appends guarded code
    def _produce_guard_code(self, guard, code_list, shape_env=False):
        assert not config.enable_cpp_guard_manager
        if shape_env:
            self.shape_env_code.append(GuardCodeList(code_list, guard))
        else:
            self.code.append(GuardCodeList(code_list, guard))

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
        func_name = getframeinfo(caller)[2]
        del caller
        # We use func_name for export, so might as well get a nice defensive check out of it
        assert func_name in dir(
            self.__class__
        ), f"_produce_guard_code must be called from inside GuardedCode. Called from {func_name}"

        # Not all guards have names, some can be installed globally (see asserts on HAS_GRAD)
        if provided_guarded_object is None:
            name_valid = guard.name is not None and guard.name != ""

            guarded_object = self.get(guard.name) if name_valid else None
        else:
            guarded_object = provided_guarded_object

        guarded_object_type = (
            weakref.ref(type(guarded_object)) if guarded_object is not None else None
        )
        obj_ref = None
        # Not necessary to have weakref for Enum type, but there is a bug that
        # makes hasattr(guarded_object.__class__, "__weakref__") return True.
        if hasattr(guarded_object.__class__, "__weakref__") and not isinstance(
            guarded_object, enum.Enum
        ):
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
        expr_count: Dict[str, int]
        expr_to_name: Dict[str, str]

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
            self.preface: List[str] = []

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

    def count(self, exprs: List[str]) -> None:
        counter = self.ExprCounter(self._config)
        for e in exprs:
            try:
                counter.visit(ast.parse(e))
            except SyntaxError as ex:
                log.exception("Failed to visit expr at line %s.\n%s", ex.lineno, e)
                raise

    def replace(self, expr: str) -> Tuple[List[str], str]:
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


class DeletedGuardFn:
    pass


# NB: Naively, you'd expect this to only be a function that produces
# the callable that constitutes the guard.  However, there is some
# delicate handling for invalidating this check function when the
# locals/globals get invalidated, so there's some extra state
# we have to hold in this manager class.
class CheckFunctionManager:
    def __init__(
        self,
        output_graph=None,
        guard_fail_fn: Optional[Callable[[GuardFail], None]] = None,
    ):
        guards = output_graph.guards if output_graph else None
        self._weakrefs: Dict[int, ReferenceType[object]] = {}
        self.guard_manager = None
        if config.enable_cpp_guard_manager:
            self.guard_manager = GuardManager()
        self.output_graph = output_graph
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
            self.id_ref,
            source_ref,
            self.lookup_weakrefs,
            output_graph.local_scope,
            output_graph.global_scope,
            self.guard_manager,
            self,
        )

        # Break retain cycle. See test_release_scope_memory
        def cleanup_builder(weak_b):
            b = weak_b()
            if b:
                b.scope = None

        # Break retain cycle. See test_release_input_memory
        w_builder = weakref.ref(builder, cleanup_builder)

        for guard in sorted(guards or [], key=Guard.sort_key):
            if (
                not config.guard_nn_modules
                and guard.is_specialized_nn_module()
                # Default func args must be guarded on.
                # TODO: we could make use of 'DefaultsSource' and offer a .guard.is_defaults() API
                and "__defaults__" not in guard.name
                and "__kwdefaults__" not in guard.name
                and (config.skip_nnmodule_hook_guards or "hooks" not in guard.name)
            ):
                continue

            guard.create(builder)

        self.check_fn = self.compile_check_fn(builder, guards, guard_fail_fn)

        # Keep track of weak references of objects with ID_MATCH guard. This
        # info is stored alongside optimized_code and check_fn and is used to
        # limit the number of cache entries with same ID_MATCH'd object.
        # TODO(anijain2305) - Currently this information is stored as an attr on
        # the check_fn itself to avoid changing CacehEntry datastructure in
        # eval_frame.c. In future, we should probably replace check_fn with a
        # queryable data structure such that this information is already present
        # in some form.
        self.check_fn.id_matched_objs = builder.id_matched_objs

        if config.enable_cpp_guard_manager:
            # TODO: don't do the string rep, do something more structured here
            torch._logging.trace_structured(
                "dynamo_cpp_guards_str", payload_fn=lambda: str(self.guard_manager)
            )
            guards_log.debug("%s", self.guard_manager)
            assert self.guard_manager  # to make mypy happy
            self.guard_manager.id_matched_objs = builder.id_matched_objs
            self.check_fn = self.guard_manager

            # Check that the guard returns True. False means that we will always
            # recompile.
            # TODO(anijain2305, ydwu4) - Skipping export because of following test
            # python -s test/dynamo/test_export.py -k test_export_with_symbool_inputs
            if not output_graph.export:
                if not self.guard_manager.check(output_graph.local_scope):
                    reasons = get_guard_fail_reason_helper(
                        self.guard_manager,  # type: ignore[arg-type]
                        output_graph.local_scope,
                        CompileContext.current_compile_id(),
                    )
                    raise AssertionError(f"Guard check failed: {reasons}")

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

    def compile_check_fn(self, builder, guards_out, guard_fail_fn):
        # see parallel handling of ".0" / "___implicit0" in _eval_frame.c
        largs = builder.argnames
        largs += ["**___kwargs_ignored"]

        guards_log.debug("GUARDS:")

        code_parts = []
        verbose_code_parts = []
        structured_guard_fns = []

        if config.enable_cpp_guard_manager:
            # Insert the global_state guard
            assert self.guard_manager  # to make mypy happy
            self.guard_manager.root.add_global_state_guard(["___check_global_state()"])
        else:
            # Don't report this guard, it's always the same, useless!
            global_guard = "___check_global_state()"
            code_parts.append(global_guard)
            verbose_code_parts.append(global_guard)

        def add_code_part(code_part, guard, log_only=False):
            verbose_code_part = get_verbose_code_part(code_part, guard)
            guards_log.debug("%s", verbose_code_part)

            structured_guard_fns.append(
                lambda: {
                    "code": code_part,
                    "stack": structured.from_traceback(guard.stack.summary())
                    if guard.stack
                    else None,
                    "user_stack": structured.from_traceback(guard.user_stack)
                    if guard.user_stack
                    else None,
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
                    add_code_part(code, gcl.guard, config.enable_cpp_guard_manager)
                    seen.add(code)

        tensor_check_names = builder.tensor_check_names
        check_tensors_fn = None
        check_tensors_verbose_fn = None
        if tensor_check_names and not config.enable_cpp_guard_manager:
            tensor_check_guards = builder.tensor_check_guards
            assert (
                not self.output_graph.export
            ), "Illegal to set tensor_check_names in export."
            tensor_check_examples = builder.tensor_check_examples

            dynamic_dims_sizes = []
            dynamic_dims_strides = []
            for t, g in zip(tensor_check_examples, tensor_check_guards):
                metadata = self.output_graph.input_source_to_sizes_strides[
                    g.originating_source
                ]
                dynamic_dims_sizes.append(convert_to_concrete_values(metadata["size"]))
                dynamic_dims_strides.append(
                    convert_to_concrete_values(metadata["stride"])
                )

            tensor_guards = TensorGuards(
                *tensor_check_examples,
                dynamic_dims_sizes=dynamic_dims_sizes,
                dynamic_dims_strides=dynamic_dims_strides,
            )
            check_tensors_fn = tensor_guards.check
            check_tensors_verbose_fn = tensor_guards.check_verbose
            tensor_check_args = ", ".join(
                tensor_check_names + ["tensor_check_names=tensor_check_names"]
            )
            # Do this manually, to un-stagger the guards in log message
            code_parts.append(f"___check_tensors({tensor_check_args})")
            verbose_code_parts.append(f"___check_tensors({tensor_check_args})")

            for i, name in enumerate(tensor_check_names):
                # This is a copy of what guards.cpp checks against
                # Keep this in sync with TensorCheck constructor
                t = tensor_check_examples[i]
                sizes = dynamic_dims_sizes[i]
                strides = dynamic_dims_strides[i]
                code_part = get_tensor_guard_code_part(t, name, sizes, strides)
                add_code_part(code_part, tensor_check_guards[i], log_only=True)

        if len(tensor_check_names) > 1 and config.enable_cpp_guard_manager:
            # Install tensor aliasing guard. TENSOR_MATCH guards are already
            # installed for cpp guard manager.
            install_no_tensor_aliasing_guard(
                builder.tensor_check_guard_managers,
                tensor_check_names,
                ["check_no_aliasing(" + ", ".join(tensor_check_names) + ")"],
            )

        aotautograd_guards: List[GuardEnvExpr] = (
            self.output_graph.tracing_context.guards_context.aotautograd_guards
            if self.output_graph
            else []
        )

        # TODO(anijain2305) - There is a duplicate logic in Dynamo to find
        # aliased input tensors. So most probably we don't need this here.
        # Revisit.
        for guard in aotautograd_guards:
            if isinstance(guard, DuplicateInputs):
                source_a = guard.input_source_a
                source_b = guard.input_source_b
                code_part = f"{source_a.name()} is {source_b.name()}"
                if config.enable_cpp_guard_manager:
                    install_object_aliasing_guard(
                        builder.get_guard_manager_from_source(source_a),
                        builder.get_guard_manager_from_source(source_b),
                        [code_part],
                    )
                add_code_part(code_part, None, config.enable_cpp_guard_manager)
            else:
                raise RuntimeError(f"Unknown GuardEnvExpr: {guard}")

        # TODO: the "guard" here is actually just the top level SHAPE_ENV
        # which is useless.  Get ShapeEnv to pass in more provenance.
        for gcl in builder.shape_env_code:
            for code in gcl.code_list:
                # Shape env guards are already added for CPP guard manager in
                # SHAPE_ENV implementation.
                add_code_part(code, gcl.guard, config.enable_cpp_guard_manager)

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
            "tensor_check_names": tensor_check_names,
            **SYMPY_INTERP,
            **CLOSURE_VARS,
        }

        globals_for_guard_fn = {"G": builder.scope["G"]}
        if config.enable_cpp_guard_manager:
            # Guard manager construction is complete
            assert self.guard_manager  # to make mypy happy
            # TODO (anijain2305) - When enable_cpp_guard_manager is ON by
            # default, change the guard_fn name to be guard_manager everywhere
            # to avoid confusion.
            guard_fn = self.guard_manager
            # Ensure we did not miss to insert a guard in cpp guard manager.
            assert len(code_parts) == 0
        else:
            unique_code_parts = list(unique(code_parts))
            make_guard_fn_args = ", ".join(closure_vars.keys())
            guard_body, pycode = build_guard_function(
                unique_code_parts, make_guard_fn_args
            )

            if os.environ.get("TORCHDYNAMO_PRINT_GUARDS", None) == "1":
                print("GUARDS\n", guard_body)

            out: Dict[str, Any] = {}

            # We don't put builder.scope as the globals in exec call because
            # guard_fn.__globals__ becomes equal to builder.scope. This causes
            # guard_fn to hold a referece to f_locals sitting in builder.scope["L"]
            try:
                exec(pycode, globals_for_guard_fn, out)
            except SyntaxError as ex:
                log.exception("Failed to exec guard at line %s.\n%s", ex.lineno, pycode)
                raise
            guard_fn = out["___make_guard_fn"](*closure_vars.values())

        guard_fn.closure_vars = closure_vars
        # TODO(whc) maybe '.code_parts' was only kept around for the guard callback? so we don't need both
        guard_fn.args = largs
        guard_fn.code_parts = code_parts
        guard_fn.verbose_code_parts = verbose_code_parts
        # Grab only G, but preserve "G" because guards access it as "G"
        guard_fn.global_scope = globals_for_guard_fn
        guard_fn.guard_fail_fn = guard_fail_fn
        # will be populated by a non-owning reference to CacheEntry/ExtraState
        # when the CacheEntry is constructed
        guard_fn.cache_entry = None
        guard_fn.extra_state = None
        guard_fn.no_tensor_aliasing_sources = tensor_check_names
        return guard_fn

    def invalidate(self):
        # Some tests reveal that CheckFunctionManager has no attribute
        # check_fn, but this case should not be of any concern.
        # This case doesn't seem easy to repro.
        if (
            hasattr(self, "check_fn")
            and self.check_fn is not DeletedGuardFn
            and (cache_entry := self.check_fn.cache_entry) is not None
            and (extra_state := self.check_fn.extra_state) is not None
        ):
            assert isinstance(cache_entry, CacheEntry)
            assert isinstance(extra_state, ExtraState)
            extra_state.invalidate(cache_entry)
            self.check_fn.cache_entry = None
            self.check_fn.extra_state = None
            self.check_fn = DeletedGuardFn

    def id_ref(self, obj):
        """add a weakref, return the id"""
        try:
            if id(obj) not in self._weakrefs:
                # We will clear the _weakrefs dict at the end of __init__
                # function, which will delete the callbacks as well. Therefore,
                # we are using a finalizer which is kept alive.
                self._weakrefs[id(obj)] = weakref.ref(obj)
                weakref.finalize(obj, self.invalidate)
        except TypeError:
            pass  # cannot weakref bool object
        return id(obj)

    def lookup_weakrefs(self, obj):
        """Lookup the _weakrefs created in id_ref function for ID_MATCH'd objects"""
        if id(obj) in self._weakrefs:
            return self._weakrefs[id(obj)]
        return None


def build_guard_function(code_parts, closure_args) -> Tuple[str, str]:
    from torch._inductor.utils import IndentedBuffer

    if HAS_UNPARSE_FUNCTIONS:
        csepass = PyExprCSEPass()
        csepass.count(code_parts)

        def replace(expr: str) -> Tuple[List[str], str]:
            return csepass.replace(expr)

    else:

        def replace(expr: str) -> Tuple[List[str], str]:
            return [], expr

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


def recompilation_reason_for_no_tensor_aliasing_guard(guard_manager, scope):
    duplicate_tensors = []
    global_scope = dict(guard_manager.global_scope)
    ids_to_source = collections.defaultdict(list)
    for tensor_source in guard_manager.no_tensor_aliasing_sources:  # type: ignore[attr-defined]
        global_scope["__compile_source__"] = tensor_source
        tensor_id = id(eval(tensor_source, global_scope, scope))
        ids_to_source[tensor_id].append(tensor_source)

    for key in ids_to_source:
        if len(ids_to_source[key]) > 1:
            duplicate_tensors.append(f"{ids_to_source[key]}")

    reason = ", ".join(duplicate_tensors)
    return [f"Duplicate tensors found: {reason}"]


def get_guard_fail_reason_helper(
    guard_fn: GuardFn,
    f_locals: Dict[str, object],
    compile_id: CompileId,
) -> str:
    """
    Return the reason why `guard_fn` failed.
    Updates `guard_failures` with the generated reason.
    Only the first failed check of guard_fn is reported.
    """
    scope = {"L": f_locals, "G": guard_fn.global_scope["G"]}
    scope.update(guard_fn.closure_vars)
    reasons: List[str] = []

    no_tensor_aliasing_check_failed = False

    verbose_code_parts: List[str] = []
    if config.enable_cpp_guard_manager:
        guard_manager = guard_fn
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
    else:
        verbose_code_parts = guard_fn.verbose_code_parts
        # This is not needed for CPP guard because the verbose check is already
        # run in C++.
        scope["___check_tensors"] = scope["___check_tensors_verbose"]

    if no_tensor_aliasing_check_failed:
        reasons = recompilation_reason_for_no_tensor_aliasing_guard(guard_fn, scope)
    else:
        for part in verbose_code_parts:
            global_scope = dict(guard_fn.global_scope)
            global_scope["__compile_source__"] = part
            with report_compile_source_on_error():
                try:
                    fail_reason = eval(part, global_scope, scope)
                except Exception as e:
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
    return reason_str


def get_guard_fail_reason(
    guard_fn: GuardFn,
    code: types.CodeType,
    f_locals: Dict[str, object],
    compile_id: CompileId,
) -> str:
    reason_str = get_guard_fail_reason_helper(guard_fn, f_locals, compile_id)
    guard_failures[orig_code_map[code]].append(reason_str)

    try:
        if guard_fn.guard_fail_fn is not None:
            guard_fn.guard_fail_fn(
                GuardFail(reason_str or "unknown reason", orig_code_map[code])
            )
    except Exception as e:
        log.exception(
            "Failure in guard_fail_fn callback - raising here will cause a NULL Error on guard eval",
        )

    return reason_str


def get_and_maybe_log_recompilation_reason(
    cache_entry, frame: types.FrameType
) -> List[str]:
    """
    Return the list of guard failure reasons using cache_entry.
    Logs the recompilation reason if `recompiles` logging is enabled.
    Raises a RecompileError if `config.error_on_recompile` is enabled.
    """
    reasons = []
    while cache_entry is not None:
        reason = get_guard_fail_reason(
            cache_entry.check_fn,
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


def guard_error_hook(
    guard_fn: GuardFn,
    code: types.CodeType,
    f_locals: Dict[str, object],
    index: int,
    last: bool,
):
    print(
        f"ERROR RUNNING GUARDS {code.co_name} {code.co_filename}:{code.co_firstlineno}"
    )
    print("lambda " + ", ".join(guard_fn.args) + ":")
    print(" ", " and\n  ".join(guard_fn.code_parts))

    if config.enable_cpp_guard_manager:
        print(guard_fn)

    local_scope = {"L": f_locals, **guard_fn.closure_vars}
    for guard in guard_fn.code_parts:
        try:
            eval(guard, guard_fn.global_scope, local_scope)
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
