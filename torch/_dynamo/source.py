"""
This module provides Source classes that track the origins of values in PyTorch Dynamo.
Sources represent where values come from (e.g. local variables, globals, attributes) and
are used for guard generation and code reconstruction during compilation.

The module includes specialized sources for:
- Local variables and synthetic locals
- Global variables and constants
- Object attributes and method calls
- NN module specialization (specialized vs unspecialized)
- Random values and tensor properties
- Default argument handling
- FSDP (Fully Sharded Data Parallel) modules

Sources play a key role in Dynamo's guard system by tracking value origins for
guard generation, and in code reconstruction by providing methods to rebuild
the code needed to recreate values.
"""

import dataclasses
import enum
import functools
from collections.abc import Callable
from typing import Any, Optional, TYPE_CHECKING, Union

from torch import device as device_type
from torch._guards import (
    ChainedSource,
    dataclass_with_cached_hash,
    Guard,
    GuardSource,
    Source,
)

from . import utils
from .bytecode_transformation import (
    create_binary_subscr,
    create_build_tuple,
    create_call_function,
)


if TYPE_CHECKING:
    from .codegen import PyCodegen

# It shouldn't be supported to construct an NNModuleVariable inside an FSDP module,
# so those cases are omitted intentionally

# represents nn.Modules tracked with NNModuleVariable (specialized is implicit in the variable name)
_GUARD_SOURCE_SPECIALIZED_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
    # Just to ensure that guard_source() works
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}

# represents nn.Modules tracked with UnspecializedNNModuleVariable
_GUARD_SOURCE_UNSPECIALIZED_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    # this happens for an UnspecializedNNModule submodule on a NNModuleVariable
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    # Just to ensure that guard_source() works
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}

# represents nn.Modules tracked with UnspecializedBuiltinNNModuleVariable
_GUARD_SOURCE_UNSPECIALIZED_BUILTIN_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    # Just to ensure that guard_source() works
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}

_GUARD_SOURCE_FSDP_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}


def is_constant_source(source: Source) -> bool:
    if isinstance(source, ConstantSource):
        return True
    try:
        if source.guard_source == GuardSource.CONSTANT:
            return True
    except NotImplementedError:
        pass

    return False


def _get_source_debug_name(source: Optional[Source]) -> str:
    if source is None:
        return "<unknown source>"
    else:
        try:
            return source.name
        except NotImplementedError:
            return "<unknown source>"


def _esc_str(s: Any, apply_repr: bool = False) -> str:
    """
    Escapes curly brackets for format strings.
    e.g. "frozenset({0})" becomes "frozenset({{0}})".
    This is used by _name_template for example, because it's
    expected to return a format string, but we may wish to include
    strings that should not be accidentally formatted.
    """
    if apply_repr:
        s = repr(s)
    else:
        s = str(s)
    return s.replace("{", "{{").replace("}", "}}")


@dataclass_with_cached_hash(frozen=True)
class LocalSource(Source):
    local_name: str

    # Whether this local is an input to the root frame.
    is_input: bool = False

    # Whether we know this input is dynamic (based on example_inputs)
    # For non tensors, we simply look at the first index of the tuple
    dynamism: Optional[frozenset[str]] = None

    # Whether the item at this source is the _content_ of a cell that is
    # dereferenced from the root frame, i.e., it's a part of the `co_cellvars`
    # or `co_freevars`.
    is_derefed_cell_contents: bool = False

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self.is_derefed_cell_contents:
            codegen.load_deref(self.local_name)
        else:
            codegen.append_output(codegen.create_load(self.local_name))

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.LOCAL

    @functools.cached_property
    def _name_template(self) -> str:
        return f"L[{_esc_str(self.local_name, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class TempLocalSource(Source):
    # like LocalSource, but cannot be guarded on
    local_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.append_output(codegen.create_load(self.local_name))

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.TEMP_LOCAL

    @property
    def _name_template(self) -> str:
        raise NotImplementedError(
            "Cannot create guard on TempLocalSource - this is an internal Dynamo bug. Please file an issue on GitHub."
        )


@dataclass_with_cached_hash(frozen=True)
class SyntheticLocalSource(Source):
    local_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.append_output(codegen.create_load(self.local_name))

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.SYNTHETIC_LOCAL

    @functools.cached_property
    def _name_template(self) -> str:
        return f"SYNTHETIC_LOCAL[{_esc_str(self.local_name, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class RandomValueSource(Source):
    random_call_index: int

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.RANDOM_VALUE

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.append_output(codegen.create_load(codegen.tx.output.random_values_var))
        codegen.append_output(codegen.create_load_const(self.random_call_index))
        codegen.append_output(create_binary_subscr())

    @functools.cached_property
    def _name_template(self) -> str:
        return f"random_value_{_esc_str(self.random_call_index)}"


@dataclass_with_cached_hash(frozen=True)
class GlobalSource(Source):
    global_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.append_output(codegen.create_load_global(self.global_name, add=True))

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.GLOBAL

    @functools.cached_property
    def _name_template(self) -> str:
        return f"G[{_esc_str(self.global_name, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class GlobalWeakRefSource(Source):
    global_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.append_output(
                codegen.create_load_global(self.global_name, add=True)
            )
        )
        codegen.extend_output(create_call_function(0, False))

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.GLOBAL

    @functools.cached_property
    def _name_template(self) -> str:
        return f"G[{_esc_str(self.global_name, apply_repr=True)}]()"


@dataclass_with_cached_hash(frozen=True)
class WeakRefCallSource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen(self.base))
        codegen.extend_output(create_call_function(0, False))

    @property
    def _name_template(self) -> str:
        return "{0}()"


@dataclass_with_cached_hash(frozen=True)
class CallFunctionNoArgsSource(WeakRefCallSource):
    pass


@dataclass_with_cached_hash(frozen=True)
class AttrSource(ChainedSource):
    member: str

    def __post_init__(self) -> None:
        assert self.base, "Can't construct an AttrSource without a valid base source"
        if "." in self.member:
            member_parts = self.member.split(".")
            object.__setattr__(
                self, "base", AttrSource(self.base, ".".join(member_parts[:-1]))
            )
            object.__setattr__(self, "member", member_parts[-1])

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs(self.member))

    @functools.cached_property
    def _name_template(self) -> str:
        if not self.member.isidentifier():
            return f"getattr({{0}}, {_esc_str(self.member, apply_repr=True)})"
        return f"{{0}}.{_esc_str(self.member)}"


@dataclass_with_cached_hash(frozen=True)
class CellContentsSource(AttrSource):
    """
    Source for closure cell contents that also stores the freevar name.
    This allows guard failure messages to show which variable the closure cell refers to.
    """

    freevar_name: str = dataclasses.field(default="")

    def __post_init__(self) -> None:
        assert self.base, (
            "Can't construct a CellContentsSource without a valid base source"
        )
        assert self.member == "cell_contents", (
            "CellContentsSource should only be used for cell_contents"
        )


@dataclass_with_cached_hash(frozen=True)
class GenericAttrSource(ChainedSource):
    member: str

    def __post_init__(self) -> None:
        assert self.base, "Can't construct an AttrSource without a valid base source"
        if "." in self.member:
            member_parts = self.member.split(".")
            object.__setattr__(
                self, "base", AttrSource(self.base, ".".join(member_parts[:-1]))
            )
            object.__setattr__(self, "member", member_parts[-1])

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs(self.member))

    @functools.cached_property
    def _name_template(self) -> str:
        return (
            f"object.__getattribute__({{0}}, {_esc_str(self.member, apply_repr=True)})"
        )


# Represents obj.__dict__ where obj is a type object
@dataclass_with_cached_hash(frozen=True)
class TypeDictSource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs("__dict__"))

    @property
    def _name_template(self) -> str:
        # type(ob).__dict__ can return a proxy of the dict. But in the C++
        # guard accessor, we are use type->tp_dict which is a dict. So,
        # forcefully pass a dict object to ensure that the GuardManager
        # registers that its working on a dict object.
        return "dict({0}.__dict__)"


# Represents obj.__mro__ where object is type object
@dataclass_with_cached_hash(frozen=True)
class TypeMROSource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs("__mro__"))

    @property
    def _name_template(self) -> str:
        return "{0}.__mro__"


@dataclass_with_cached_hash(frozen=True)
class LocalCellSource(Source):
    """
    Conceptually, this class is `LocalSource` for cell objects implicitly
    generated by Python (e.g., captured variables).
    """

    local_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Although `LOAD_FAST` and `LOAD_CLOSURE` have the same semantics,
        # Dynamo's bytecode transformation differentiates them slightly, so we
        # always emit `LOAD_CLOSURE` here.
        codegen.append_output(codegen.create_load_closure(self.local_name))

    # All the other methods are intentionally unimplemented because e.g., a
    # local cell object should never be used for guards.


# Represents obj.__code__ where object is type object
@dataclass_with_cached_hash(frozen=True)
class CodeSource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs("__code__"))

    @property
    def _name_template(self) -> str:
        return "{0}.__code__"


# Represents obj.__closure__ where object is type object
@dataclass_with_cached_hash(frozen=True)
class ClosureSource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs("__closure__"))

    @property
    def _name_template(self) -> str:
        return "{0}.__closure__"


# Represents tensor.grad source. It could be represented by AttrSource as well.
# But, we could access grad field on tensor directly in C++ without going
# through the Python bytecodes. Therefore, we use a separate source for grad
# field.
@dataclass_with_cached_hash(frozen=True)
class GradSource(ChainedSource):
    member: str = "grad"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs(self.member))

    @functools.cached_property
    def _name_template(self) -> str:
        return f"{{0}}.{_esc_str(self.member)}"


@dataclass_with_cached_hash(frozen=True)
class ParamBufferSource(AttrSource):
    @functools.cached_property
    def guard_source(self) -> GuardSource:
        return _GUARD_SOURCE_SPECIALIZED_NN_MODULE[self.base.guard_source]


# Special AttrSource to differentiate module._buffers or module._parameters
@dataclass_with_cached_hash(frozen=True)
class UnspecializedParamBufferSource(AttrSource):
    pass


# This source is intended to be used in places where a source is needed but it is expected
# that the symbol will be simplified out later on. Symbols with ephemeral sources are
# prioritized to be simplified out when e.g. compared against a symbol without an ephemeral
# source. Guarding on this source is an error.
#
# Example: During subclass view fake-ification, any close-over ViewFunc state should be
# symbolicized / fake-ified to avoid invalid specialization during view replay. This source
# is useful for symbols utilized in the middle of the view chain that are not expected to be
# present within the final view shape metadata.
@dataclass_with_cached_hash(frozen=True)
class EphemeralSource(Source):
    desc: Optional[str] = None

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.EPHEMERAL

    @functools.cached_property
    def _name_template(self) -> str:
        desc = ": " + self.desc if self.desc is not None else ""
        return f"<ephemeral{_esc_str(desc)}>"

    def make_guard(self, fn: Callable[..., Any]) -> Guard:
        raise NotImplementedError

    def is_ephemeral(self) -> bool:
        return True


@dataclass_with_cached_hash(frozen=True)
class SkipGuardSource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        self.base.reconstruct(codegen)

    @property
    def _name_template(self) -> str:
        return "{0}"


class TensorProperty(enum.Enum):
    SIZE = 0
    STRIDE = 1
    STORAGE_OFFSET = 2

    def method_name(self) -> str:
        if self is TensorProperty.SIZE:
            return "size"
        elif self is TensorProperty.STRIDE:
            return "stride"
        elif self is TensorProperty.STORAGE_OFFSET:
            return "storage_offset"
        else:
            raise AssertionError(f"unhandled {_esc_str(self)}")


@dataclass_with_cached_hash(frozen=True)
class TensorPropertySource(ChainedSource):
    prop: TensorProperty
    idx: Optional[int] = None  # None for STORAGE_OFFSET

    def __post_init__(self) -> None:
        assert self.base is not None
        if self.prop is TensorProperty.STORAGE_OFFSET:
            assert self.idx is None
        else:
            assert self.idx is not None

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from(
                utils.__name__, f"call_{_esc_str(self.prop.method_name())}"
            )
        )
        codegen(self.base)

        if self.idx is not None:
            codegen.append_output(codegen.create_load_const(self.idx))
        codegen.extend_output(
            create_call_function(2 if self.idx is not None else 1, False)
        )

    @functools.cached_property
    def _name_template(self) -> str:
        if self.prop is TensorProperty.SIZE:
            return f"{{0}}.size()[{_esc_str(self.idx)}]"
        elif self.prop is TensorProperty.STRIDE:
            return f"{{0}}.stride()[{_esc_str(self.idx)}]"
        elif self.prop is TensorProperty.STORAGE_OFFSET:
            assert self.idx is None
            return "{0}.storage_offset()"
        else:
            raise AssertionError(f"unhandled {_esc_str(self.prop)}")


@dataclass_with_cached_hash(frozen=True)
class IndexedSource(ChainedSource):
    idx: int

    def __post_init__(self) -> None:
        assert self.base is not None

    def reconstruct(self, codegen: "PyCodegen") -> None:
        raise NotImplementedError

    @functools.cached_property
    def _name_template(self) -> str:
        return f"({_esc_str(self.idx)}, {{0}})"


@dataclass_with_cached_hash(frozen=True)
class NegateSource(ChainedSource):
    def __post_init__(self) -> None:
        assert self.base is not None

    def reconstruct(self, codegen: "PyCodegen") -> None:
        raise NotImplementedError

    @property
    def _name_template(self) -> str:
        # NB: use method call so that function stripping regexes work
        return "{0}.__neg__()"


@dataclass_with_cached_hash(frozen=True)
class ConvertIntSource(ChainedSource):
    def __post_init__(self) -> None:
        assert self.base is not None

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)

    @property
    def _name_template(self) -> str:
        return "cast_symbool_to_symint_guardless({0})"


@dataclass_with_cached_hash(frozen=True)
class DynamicScalarSource(ChainedSource):
    is_int: bool

    def __post_init__(self) -> None:
        assert self.base is not None

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Integer casting at reconstruction helps reduce the amount of DynamicInts returned
        # to the user, in favor of plain ints.
        # For example, a compiled region that only does int arithmetic could return a
        # DynamicInt without the casting here.
        codegen.add_push_null(lambda: codegen.load_import_from("builtins", "int"))
        codegen(self.base)
        codegen.extend_output(create_call_function(1, False))

    @property
    def _name_template(self) -> str:
        return "int({0})"


@dataclass_with_cached_hash(frozen=True)
class FlattenScriptObjectSource(ChainedSource):
    def __post_init__(self) -> None:
        assert self.base is not None

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)

    @property
    def _name_template(self) -> str:
        return "{0}.__obj_flatten__()"


@dataclass_with_cached_hash(frozen=True)
class ScriptObjectQualifiedNameSource(ChainedSource):
    def __post_init__(self) -> None:
        assert self.base is not None

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)

    @property
    def _name_template(self) -> str:
        return "{0}._type().qualified_name()"


class AttrProxySource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)

    @property
    def _name_template(self) -> str:
        return "{0}.get_base()"


@dataclass_with_cached_hash(frozen=True)
class DefaultsSource(ChainedSource):
    idx_key: Union[int, str]
    is_kw: bool = False
    field: str = dataclasses.field(init=False, repr=False, compare=False)
    _name: str = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        assert self.base, (
            "Base must be a valid source in order to properly track and guard this Defaults to its origin."
        )
        if self.is_kw:
            assert isinstance(self.idx_key, str)
            object.__setattr__(self, "field", "__kwdefaults__")
            object.__setattr__(
                self,
                "_name",
                f"{{0}}.{_esc_str(self.field)}['{_esc_str(self.idx_key)}']",
            )
        else:
            assert isinstance(self.idx_key, int)
            object.__setattr__(self, "field", "__defaults__")
            object.__setattr__(
                self, "_name", f"{{0}}.{_esc_str(self.field)}[{_esc_str(self.idx_key)}]"
            )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs(self.field))
        codegen.append_output(codegen.create_load_const(self.idx_key))
        codegen.append_output(create_binary_subscr())

    @functools.cached_property
    def _name_template(self) -> str:
        return self._name


@dataclass_with_cached_hash(frozen=True)
class GetItemSource(ChainedSource):
    index: Any
    index_is_slice: bool = False

    def __post_init__(self) -> None:
        assert self.base is not None
        if isinstance(self.index, slice):
            # store the hashable version of the slice so the whole GetItemSource is hashable
            super().__setattr__("index", self.index.__reduce__())
            super().__setattr__("index_is_slice", True)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)
        if self.index_is_slice:
            codegen.append_output(codegen.create_load_const(self.unpack_slice()))
        else:
            codegen.append_output(codegen.create_load_const(self.index))
        codegen.append_output(create_binary_subscr())

    def unpack_slice(self) -> slice:
        assert self.index_is_slice
        slice_class, slice_args = self.index
        return slice_class(*slice_args)

    @functools.cached_property
    def _name_template(self) -> str:
        # Index can be of following types
        # 1) index is a slice - example 1:4
        # 2) index is a constant - example string, integer
        assert not isinstance(self.index, Source)
        if self.index_is_slice:
            return f"{{0}}[{_esc_str(self.unpack_slice(), apply_repr=True)}]"
        else:
            return f"{{0}}[{_esc_str(self.index, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class ConstDictKeySource(ChainedSource):
    index: Any

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "dict_keys_getitem")
        )
        codegen(self.base)
        codegen.append_output(codegen.create_load_const(self.index))
        codegen.extend_output(create_call_function(2, False))

    @functools.cached_property
    def _name_template(self) -> str:
        # The list creation will be CSE'd by PyExprCSEPass
        return f"list(dict.keys({{0}}))[{_esc_str(self.index, apply_repr=True)}]"

    def is_dict_key(self) -> bool:
        return True


@dataclass_with_cached_hash(frozen=True)
class NonSerializableSetGetItemSource(ChainedSource):
    index: int

    def __post_init__(self) -> None:
        from .variables import ConstantVariable

        assert ConstantVariable.is_literal(self.index)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "set_getitem")
        )
        codegen(self.base)
        codegen.append_output(codegen.create_load_const(self.index))
        codegen.extend_output(create_call_function(2, False))

    @functools.cached_property
    def _name_template(self) -> str:
        # set ordering might not be stable
        return f"list({{0}})[{_esc_str(self.index, apply_repr=True)}]"

    def is_dict_key(self) -> bool:
        return False


# Used to access an item from the dictionary
@dataclass_with_cached_hash(frozen=True)
class DictGetItemSource(ChainedSource):
    # Key to access in the dictionary. It can be one of the following types
    # 1) ConstDictKeySource
    # 2) constant - like string, integer
    index: Any

    def __post_init__(self) -> None:
        from .variables import ConstantVariable

        assert isinstance(
            self.index, ConstDictKeySource
        ) or ConstantVariable.is_literal(self.index)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Load dict
        codegen(self.base)

        # Load key
        if isinstance(self.index, Source):
            codegen(self.index)
        else:
            codegen.append_output(codegen.create_load_const(self.index))
        codegen.append_output(create_binary_subscr())

    @functools.cached_property
    def _name_template(self) -> str:
        if isinstance(self.index, ConstDictKeySource):
            return f"{{0}}[{_esc_str(self.index.name)}]"
        else:
            return f"{{0}}[{_esc_str(self.index, apply_repr=True)}]"


# Same as DictGetItemSource but used for dict.__getitem__ calls to ensure that
# torch.compile does not run the overridden __getitem__ method
@dataclass_with_cached_hash(frozen=True)
class DictSubclassGetItemSource(ChainedSource):
    # Key to access in the dictionary. It can be one of the following types
    # 1) ConstDictKeySource
    # 2) constant - like string, integer
    index: Any

    def __post_init__(self) -> None:
        from .variables import ConstantVariable

        assert isinstance(
            self.index, ConstDictKeySource
        ) or ConstantVariable.is_literal(self.index)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # reconstruct dict.__getitem__(dct, key)

        # Load dict.__getitem__
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "dict_getitem")
        )

        # Load dict
        codegen(self.base)

        # Load key
        if isinstance(self.index, Source):
            codegen(self.index)
        else:
            codegen.append_output(codegen.create_load_const(self.index))

        codegen.extend_output(create_call_function(2, False))

    @functools.cached_property
    def _name_template(self) -> str:
        if isinstance(self.index, ConstDictKeySource):
            return f"dict.__getitem__({{0}}, {_esc_str(self.index.name)})"
        else:
            return f"{{0}}[{_esc_str(self.index, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class ListGetItemSource(GetItemSource):
    """
    Same as GetItemSource with reconstruct and name overridden to be list specific.
    """

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Reconstruct list.__getitem__(lst, index) to avoid any side effects
        # from possibly overridden __getitem__.

        # Load list.__getitem__
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "list_getitem")
        )

        # Load the list
        codegen(self.base)

        # Load the index
        if self.index_is_slice:
            raise RuntimeError(
                "List[slice] is a temporary object and should not have a source"
            )
        else:
            codegen.append_output(codegen.create_load_const(self.index))

        codegen.extend_output(create_call_function(2, False))

    @functools.cached_property
    def _name_template(self) -> str:
        # Index can be of following types
        # 1) index is a slice - example 1:4
        # 2) index is a constant - example string, integer
        assert not isinstance(self.index, Source)
        if self.index_is_slice:
            raise RuntimeError(
                "List[slice] is a temporary object and should not have a source"
            )
        else:
            return f"list.__getitem__({{0}}, {_esc_str(self.index, apply_repr=True)})"


@dataclass_with_cached_hash(frozen=True)
class TupleIteratorGetItemSource(GetItemSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "tuple_iterator_getitem")
        )
        codegen(self.base)
        codegen.append_output(codegen.create_load_const(self.index))
        codegen.extend_output(create_call_function(2, False))

    @functools.cached_property
    def _name_template(self) -> str:
        return (
            f"___tuple_iterator_getitem({{0}}, {_esc_str(self.index, apply_repr=True)})"
        )


@dataclass_with_cached_hash(frozen=True)
class NamedTupleFieldsSource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs("_fields"))

    @property
    def _name_template(self) -> str:
        return "___namedtuple_fields({0})"


@dataclass_with_cached_hash(frozen=True)
class DataclassFieldsSource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "dataclass_fields")
        )
        codegen(self.base)
        codegen.extend_output(create_call_function(1, False))

    @property
    def _name_template(self) -> str:
        return "___dataclass_fields({0})"


@dataclass_with_cached_hash(frozen=True)
class TypeSource(ChainedSource):
    def __post_init__(self) -> None:
        assert self.base is not None

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen.load_import_from("builtins", "type"))
        codegen(self.base)
        codegen.extend_output(create_call_function(1, False))

    @property
    def _name_template(self) -> str:
        return "type({0})"


@dataclass_with_cached_hash(frozen=True)
class OptimizerSource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)

    @property
    def _name_template(self) -> str:
        return "{0}"


@dataclass_with_cached_hash(frozen=True)
class NNModuleSource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)

    @functools.cached_property
    def guard_source(self) -> GuardSource:
        return _GUARD_SOURCE_SPECIALIZED_NN_MODULE[self.base.guard_source]

    @property
    def _name_template(self) -> str:
        return "{0}"


@dataclass_with_cached_hash(frozen=True)
class UnspecializedNNModuleSource(NNModuleSource):
    @functools.cached_property
    def guard_source(self) -> GuardSource:
        return _GUARD_SOURCE_UNSPECIALIZED_NN_MODULE[self.base.guard_source]


@dataclass_with_cached_hash(frozen=True)
class UnspecializedBuiltinNNModuleSource(UnspecializedNNModuleSource):
    @functools.cached_property
    def guard_source(self) -> GuardSource:
        return _GUARD_SOURCE_UNSPECIALIZED_BUILTIN_NN_MODULE[self.base.guard_source]


@dataclass_with_cached_hash(frozen=True)
class FSDPNNModuleSource(NNModuleSource):
    @functools.cached_property
    def guard_source(self) -> GuardSource:
        return _GUARD_SOURCE_FSDP_MODULE[self.base.guard_source]


@dataclass_with_cached_hash(frozen=True)
class GlobalStateSource(Source):
    @property
    def _name_template(self) -> str:
        return ""

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.GLOBAL


@dataclass_with_cached_hash(frozen=True)
class TorchSource(Source):
    """Points to the actual `torch` module - used instead of GlobalSource
    in case the user has overridden `torch` in their local namespace"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from .guards import GuardBuilder, install_guard

        install_guard(self.make_guard(GuardBuilder.ID_MATCH))

    @property
    def _name_template(self) -> str:
        return "__import__('torch')"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.extend_output(
            [
                codegen.create_load_const(0),  # level
                create_build_tuple(0),  # fromlist
                codegen.create_import_name("torch"),
            ]
        )

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.GLOBAL


@dataclass_with_cached_hash(frozen=True)
class CollectionsSource(Source):
    """Points to the actual `collections` module - used instead of GlobalSource
    in case the user has overridden `collections` in their local namespace"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from .guards import GuardBuilder, install_guard

        install_guard(self.make_guard(GuardBuilder.ID_MATCH))

    @property
    def _name_template(self) -> str:
        return "__import__('collections')"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.extend_output(
            [
                codegen.create_load_const(0),  # level
                create_build_tuple(0),  # fromlist
                codegen.create_import_name("collections"),
            ]
        )

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.GLOBAL


@dataclass_with_cached_hash(frozen=True)
class TorchFunctionModeStackSource(Source):
    ind: int

    @functools.cached_property
    def _name_template(self) -> str:
        return f"___get_torch_function_mode_stack_at({_esc_str(self._get_index())})"

    def _get_index(self) -> int:
        from .variables.torch_function import TorchFunctionModeStackVariable

        return TorchFunctionModeStackVariable.get_mode_index(self.ind)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from(
                utils.__name__, "get_torch_function_mode_stack_at"
            )
        )
        codegen.extend_output([codegen.create_load_const(self._get_index())])
        codegen.extend_output(create_call_function(1, False))

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.GLOBAL


@dataclass_with_cached_hash(frozen=True)
class ConstantSource(Source):
    source_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.append_output(codegen.create_load_global(self.source_name, add=False))

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.CONSTANT

    @functools.cached_property
    def _name_template(self) -> str:
        return self.source_name

    def make_guard(self, fn: Any) -> Any:
        raise NotImplementedError


@dataclass_with_cached_hash(frozen=True)
class NumpyTensorSource(ChainedSource):
    @property
    def _name_template(self) -> str:
        return "___from_numpy({0})"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen.load_import_from("torch", "as_tensor"))
        codegen(self.base)
        codegen.extend_output(create_call_function(1, False))


@dataclass_with_cached_hash(frozen=True)
class SubclassAttrListSource(ChainedSource):
    @property
    def _name_template(self) -> str:
        return "{0}.__tensor_flatten__()[0]"


# NB: We don't expect you to actually ever generate guards against this
# source, it is ephemeral
@dataclass_with_cached_hash(frozen=True)
class FloatTensorSource(ChainedSource):
    @property
    def _name_template(self) -> str:
        return "___as_tensor({0})"


@dataclass_with_cached_hash(frozen=True)
class CallMethodItemSource(ChainedSource):
    @property
    def _name_template(self) -> str:
        return "{0}.item()"


# This is a synthetic source that is associated with the singleton
# shape env guard we always register for all frames.  We get the actual
# guard contents from the ambient ShapeEnv
@dataclass_with_cached_hash(frozen=True)
class ShapeEnvSource(Source):
    @property
    def _name_template(self) -> str:
        return ""

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.SHAPE_ENV


@dataclass_with_cached_hash(frozen=True)
class CurrentStreamSource(Source):
    device: device_type

    @functools.cached_property
    def _name_template(self) -> str:
        return f"___get_current_stream(torch.device('{_esc_str(self.device.type)}', {_esc_str(self.device.index)}))"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        num_args = 1
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "get_current_stream")
        )
        codegen.add_push_null(lambda: codegen.load_import_from("torch", "device"))
        codegen.extend_output([codegen.create_load_const(self.device.type)])
        if self.device.index is not None:
            num_args += 1
            codegen.extend_output([codegen.create_load_const(self.device.index)])
        codegen.extend_output(create_call_function(num_args, False))
        codegen.extend_output(create_call_function(1, False))

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.GLOBAL


@dataclass_with_cached_hash(frozen=True)
class BackwardStateSource(Source):
    @property
    def _name_template(self) -> str:
        return ""

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.BACKWARD_STATE


def get_local_source_name(
    source: Source, *, only_allow_input: bool = False
) -> Optional[str]:
    if isinstance(source, ChainedSource):
        return get_local_source_name(source.base, only_allow_input=only_allow_input)
    if not isinstance(source, LocalSource):
        return None
    if only_allow_input and not source.is_input:
        return None
    return source.local_name


def is_from_local_source(source: Source, *, only_allow_input: bool = False) -> bool:
    return get_local_source_name(source, only_allow_input=only_allow_input) is not None


def is_from_global_source(source: Source) -> bool:
    return get_global_source_name(source) is not None


def get_global_source_name(source: Source | None) -> str | None:
    if isinstance(source, ChainedSource):
        return get_global_source_name(source.base)
    if not isinstance(source, GlobalSource):
        return None
    return source.global_name


def is_from_nonlocal_source(source: Source) -> bool:
    if isinstance(source, ChainedSource):
        return is_from_nonlocal_source(source.base)
    return (
        isinstance(source, LocalSource)
        and source.is_derefed_cell_contents
        and not source.is_input
    )


def is_from_closure_source(source: Source) -> bool:
    if isinstance(source, ClosureSource):
        return True
    if isinstance(source, ChainedSource):
        return is_from_closure_source(source.base)
    return False


def is_from_source(source: Source, target: Source) -> bool:
    if isinstance(source, ChainedSource):
        return is_from_source(source.base, target)
    return source == target


@functools.lru_cache
def is_from_unspecialized_nn_module_source(source: Source) -> bool:
    if isinstance(source, UnspecializedNNModuleSource):
        return True
    if isinstance(source, ChainedSource):
        return is_from_unspecialized_nn_module_source(source.base)
    return False


@functools.lru_cache
def is_from_unspecialized_builtin_nn_module_source(source: Source) -> bool:
    if isinstance(source, UnspecializedBuiltinNNModuleSource):
        return True
    if isinstance(source, ChainedSource):
        return is_from_unspecialized_builtin_nn_module_source(source.base)
    return False


@functools.lru_cache
def is_from_unspecialized_param_buffer_source(source: Source) -> bool:
    if isinstance(source, UnspecializedParamBufferSource):
        return True
    if isinstance(source, ChainedSource):
        return is_from_unspecialized_param_buffer_source(source.base)
    return False


@functools.lru_cache
def is_from_flatten_script_object_source(source: Source) -> bool:
    if isinstance(source, FlattenScriptObjectSource):
        return True
    elif isinstance(source, ChainedSource):
        return is_from_flatten_script_object_source(source.base)
    return False


@functools.lru_cache
def is_from_optimizer_source(source: Source) -> bool:
    if isinstance(source, OptimizerSource):
        return True
    if isinstance(source, ChainedSource):
        return is_from_optimizer_source(source.base)
    return False


# TODO: can probably write a generic "test this on everything in the chain"
# helper
@functools.lru_cache
def is_from_defaults(source: Source) -> bool:
    if isinstance(source, DefaultsSource):
        return True

    # Accessed with func.__kwdefaults__["foo"]
    if (
        isinstance(source, DictGetItemSource)
        and isinstance(source.base, AttrSource)
        and source.base.member == "__kwdefaults__"
    ):
        return True

    # Accessed with func.__defaults__[0]
    if (
        isinstance(source, GetItemSource)
        and isinstance(source.base, AttrSource)
        and source.base.member == "__defaults__"
    ):
        return True

    if isinstance(source, ChainedSource):
        return is_from_defaults(source.base)
    return False


@functools.lru_cache
def is_from_skip_guard_source(source: Source) -> bool:
    if isinstance(source, SkipGuardSource):
        return True

    if isinstance(source, ChainedSource):
        return is_from_skip_guard_source(source.base)

    return False
