import enum
import traceback
from collections.abc import Callable
from typing import Any, Optional, TypeAlias

import torch

# TODO: We should move the `GuardManagerType`
# defined in `guards.py` here and update other
# imports
GuardManagerType: TypeAlias = enum.Enum

class GlobalStateGuard:
    def check(self) -> bool: ...
    def reason(self) -> str: ...

class LeafGuard:
    def verbose_code_parts(self) -> list[str]: ...

class RelationalGuard: ...

class GuardDebugInfo:
    verbose_code_parts: list[str]
    result: bool
    num_guards_executed: int
    user_stack: Optional[traceback.StackSummary]

class GuardManager:
    def check(self, value: Any) -> bool: ...
    def check_verbose(self, value: Any) -> GuardDebugInfo: ...

    # Accessors
    def globals_dict_manager(
        self,
        f_globals: dict[str, Any],
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def framelocals_manager(
        self,
        key: tuple[str, int],
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def dict_getitem_manager(
        self,
        key: Any,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def grad_manager(
        self,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def generic_getattr_manager(
        self,
        attr: str,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def getitem_manager(
        self,
        key: Any,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def get_generic_dict_manager(
        self,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def list_getitem_manager(
        self,
        key: Any,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def tuple_getitem_manager(
        self,
        key: Any,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def set_getitem_manager(
        self,
        index: Any,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def func_defaults_manager(
        self,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def func_kwdefaults_manager(
        self,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def tuple_iterator_getitem_manager(
        self,
        index: Any,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def weakref_call_manager(
        self,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def call_function_no_args_manager(
        self,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def global_weakref_manager(
        self,
        global_name: str,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def type_manager(
        self,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def getattr_manager(
        self,
        attr: str,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def tensor_property_size_manager(
        self,
        idx: int,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def tensor_property_shape_manager(
        self,
        idx: int,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def tensor_property_storage_offset_manager(
        self,
        idx: int,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def indexed_manager(
        self,
        idx: int,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def lambda_manager(
        self,
        python_lambda: Callable[..., Any],
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def get_root(self) -> RootGuardManager: ...
    def get_source(self) -> str: ...
    def fail_count(self) -> int: ...
    def get_child_managers(self) -> list[GuardManager]: ...
    def repr(self) -> str: ...
    def type_of_guarded_value(self) -> str: ...
    def get_leaf_guards(self) -> list[LeafGuard]: ...
    def get_accessors(self) -> list[GuardManager]: ...
    def is_guarded_value_immutable(self) -> bool: ...
    def is_tag_safe(self) -> bool: ...
    def is_tag_safe_root(self) -> bool: ...
    def has_no_accessors(self) -> bool: ...
    def has_object_aliasing_guard(self) -> bool: ...
    def get_type_of_guarded_value(self) -> type: ...
    def type_dict_manager(
        self,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def type_mro_manager(
        self,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def code_manager(
        self,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def closure_manager(
        self,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    # Leaf guards
    def add_lambda_guard(
        self,
        user_lambda: Callable[..., Any],
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_id_match_guard(
        self,
        id_val: int,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_equals_match_guard(
        self,
        equals_val: Any,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_global_state_guard(
        self,
        initial_state: Any,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_torch_function_mode_stack_guard(
        self,
        initial_stack: list[Any],
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_mapping_keys_guard(
        self,
        value: Any,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_dict_length_check_guard(
        self,
        value: int,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_length_check_guard(
        self,
        value: int,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_true_match_guard(
        self,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_false_match_guard(
        self,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_none_match_guard(
        self,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_not_none_guard(
        self,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_dispatch_key_set_guard(
        self,
        dispatch_key: Any,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_tensor_match_guard(
        self,
        value: Any,
        sizes: list[int],
        strides: list[int],
        tensor_name: str,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
        ptype: Any,
        dispatch_keys: Any,
    ) -> None: ...
    def add_dynamic_indices_guard(
        self,
        value: set[Any],
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_no_hasattr_guard(
        self,
        attr_name: str,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_dict_contains_guard(
        self,
        contains: bool,
        key: Any,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_type_match_guard(
        self,
        value: int,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_dict_version_guard(
        self,
        value: Any,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_set_contains_guard(
        self,
        contains: bool,
        item: Any,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_dual_level_match_guard(
        self,
        level: int,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_float_is_nan_guard(
        self,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_complex_is_nan_guard(
        self,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_tuple_iterator_length_guard(
        self,
        length: int,
        type_id: int,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_range_iterator_match_guard(
        self,
        start: int,
        stop: int,
        step: int,
        type_id: int,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def add_default_device_guard(
        self,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def mark_tag_safe(self) -> None: ...
    def mark_tag_safe_root(self) -> None: ...

class RootGuardManager(GuardManager):
    def get_epilogue_lambda_guards(self) -> list[LeafGuard]: ...
    def add_epilogue_lambda_guard(
        self,
        guard: LeafGuard,
        verbose_code_parts: list[str],
        user_stack: Optional[traceback.StackSummary],
    ) -> None: ...
    def clone_manager(
        self, clone_filter_fn: Callable[[GuardManager], bool]
    ) -> RootGuardManager: ...
    def attach_compile_id(self, compile_id: str) -> None: ...

class DictGuardManager(GuardManager):
    def get_key_manager(
        self,
        index: int,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def get_value_manager(
        self,
        index: int,
        source: str,
        example_value: Any,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager: ...
    def get_key_value_managers(
        self,
    ) -> dict[int, tuple[GuardManager, GuardManager]]: ...

# Guard accessor stubs
class GuardAccessor: ...
class DictGetItemGuardAccessor(GuardAccessor): ...
class GetGenericDictGuardAccessor(GuardAccessor): ...
class TypeDictGuardAccessor(GuardAccessor): ...
class TypeMROGuardAccessor(GuardAccessor): ...
class ClosureGuardAccessor(GuardAccessor): ...
class TupleGetItemGuardAccessor(GuardAccessor): ...
class TypeGuardAccessor(GuardAccessor): ...
class CodeGuardAccessor(GuardAccessor): ...
class FuncDefaultsGuardAccessor(GuardAccessor): ...
class FuncKwDefaultsGuardAccessor(GuardAccessor): ...

class GetAttrGuardAccessor(GuardAccessor):
    def get_attr_name(self) -> str: ...

def install_object_aliasing_guard(
    x: GuardManager,
    y: GuardManager,
    verbose_code_parts: list[str],
    user_stack: Optional[traceback.StackSummary],
) -> None: ...
def install_no_tensor_aliasing_guard(
    guard_managers: list[GuardManager],
    tensor_names: list[str],
    verbose_code_parts: list[str],
    user_stack: Optional[traceback.StackSummary],
) -> None: ...
def install_storage_overlapping_guard(
    overlapping_guard_managers: list[GuardManager],
    non_overlapping_guard_managers: list[GuardManager],
    verbose_code_parts: list[str],
    user_stack: Optional[traceback.StackSummary],
) -> None: ...
def install_symbolic_shape_guard(
    guard_managers: list[GuardManager],
    nargs_int: int,
    nargs_float: int,
    py_addr: int,
    py_addr_keep_alive: Any,
    verbose_code_parts: list[str],
    user_stack: Optional[traceback.StackSummary],
) -> None: ...
def profile_guard_manager(
    guard_manager: GuardManager,
    f_locals: dict[str, Any],
    n_iters: int,
) -> float: ...

class TensorGuards:
    def __init__(
        self,
        *,
        dynamic_dims_sizes: list[torch.SymInt | None] | None = None,
        dynamic_dims_strides: list[torch.SymInt | None] | None = None,
    ) -> None: ...
    def check(self, *args: Any) -> bool: ...
    def check_verbose(
        self, *args: Any, tensor_check_names: Optional[list[str]] = None
    ) -> bool | str: ...

def assert_size_stride(
    item: torch.Tensor,
    size: torch.types._size,
    stride: torch.types._size,
    op_name: str | None = None,
) -> None: ...
def assert_alignment(
    item: torch.Tensor,
    alignment: int,
    op_name: str | None = None,
) -> None: ...
def check_obj_id(obj: object, expected: int) -> bool: ...
def check_type_id(obj: object, expected: int) -> bool: ...
def dict_version(d: dict[Any, Any]) -> int: ...
def compute_overlapping_tensors(
    tensors: list[torch.Tensor], symbolic: bool = True
) -> set[int]: ...
def set_is_in_mode_without_ignore_compile_internals(value: bool) -> None: ...
