import collections
import dataclasses
import enum
from typing import Any, Optional, Union

from torch._guards import GuardSource, Source

from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr

# It shouldn't be supported to construct an NNModuleVariable inside an FSDP module,
# so those cases are omitted intentionally
_GUARD_SOURCE_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_NN_MODULE,
    GuardSource.LOCAL_NN_MODULE: GuardSource.LOCAL_NN_MODULE,
    GuardSource.GLOBAL_NN_MODULE: GuardSource.GLOBAL_NN_MODULE,
}

_GUARD_SOURCE_FSDP_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_NN_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_NN_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}

_GUARD_SOURCE_NOT_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL,
    GuardSource.GLOBAL: GuardSource.GLOBAL,
    GuardSource.LOCAL_NN_MODULE: GuardSource.LOCAL,
    GuardSource.GLOBAL_NN_MODULE: GuardSource.GLOBAL,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL,
}


def is_constant_source(source):
    if isinstance(source, ConstantSource):
        return True
    try:
        if source.guard_source() == GuardSource.CONSTANT:
            return True
    except NotImplementedError:
        pass

    return False


def is_input_source(source):
    return source.guard_source() in [
        GuardSource.LOCAL,
        GuardSource.GLOBAL,
        GuardSource.LOCAL_NN_MODULE,
        GuardSource.GLOBAL_NN_MODULE,
        GuardSource.LOCAL_FSDP_MODULE,
        GuardSource.GLOBAL_FSDP_MODULE,
    ]


@dataclasses.dataclass(frozen=True)
class LocalSource(Source):
    local_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load(self.local_name)]

    def guard_source(self):
        return GuardSource.LOCAL

    def name(self):
        return f"L[{repr(self.local_name)}]"


@dataclasses.dataclass(frozen=True)
class RandomValueSource(Source):
    random_call_index: int

    def guard_source(self):
        return GuardSource.RANDOM_VALUE

    def reconstruct(self, codegen):
        return [
            codegen.create_load(codegen.tx.output.random_values_var),
            codegen.create_load_const(self.random_call_index),
            create_instruction("BINARY_SUBSCR"),
        ]

    def name(self):
        return f"random_value_{self.random_call_index}"


@dataclasses.dataclass(frozen=True)
class GeneratorStateSource(Source):
    device: str
    initial_seed: int

    def guard_source(self):
        return GuardSource.RANDOM_VALUE

    def reconstruct(self, codegen):
        # generator state is a torch.ByteTensor, so we reuse TensorVariable reconstruction in codegen.py
        raise NotImplementedError()

    def name(self):
        name = f"generator_state_{self.device}_{self.initial_seed}"
        return f"L[{name}]"


@dataclasses.dataclass(frozen=True)
class GlobalSource(Source):
    global_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load_global(self.global_name, False, add=True)]

    def guard_source(self):
        return GuardSource.GLOBAL

    def name(self):
        return f"G[{repr(self.global_name)}]"


@dataclasses.dataclass(frozen=True)
class GlobalWeakRefSource(Source):
    global_name: str

    def reconstruct(self, codegen):
        return [
            codegen.create_load_global(self.global_name, True, add=True),
            *create_call_function(0, False),
        ]

    def guard_source(self):
        return GuardSource.GLOBAL

    def name(self):
        return f"G[{repr(self.global_name)}]()"


@dataclasses.dataclass(frozen=True)
class AttrSource(Source):
    base: Source
    member: str

    def __post_init__(self):
        assert self.base, "Can't construct an AttrSource without a valid base source"
        if "." in self.member:
            member_parts = self.member.split(".")
            object.__setattr__(
                self, "base", AttrSource(self.base, ".".join(member_parts[:-1]))
            )
            object.__setattr__(self, "member", member_parts[-1])

    def reconstruct(self, codegen):
        return self.base.reconstruct(codegen) + codegen.create_load_attrs(self.member)

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        if not self.member.isidentifier():
            return f"getattr({self.base.name()}, {self.member!r})"
        return f"{self.base.name()}.{self.member}"


@dataclasses.dataclass(frozen=True)
class ParamBufferSource(AttrSource):
    def guard_source(self):
        return _GUARD_SOURCE_NN_MODULE[self.base.guard_source()]


class TensorProperty(enum.Enum):
    SIZE = 0
    STRIDE = 1
    STORAGE_OFFSET = 2

    def method_name(self):
        if self is TensorProperty.SIZE:
            return "size"
        elif self is TensorProperty.STRIDE:
            return "stride"
        elif self is TensorProperty.STORAGE_OFFSET:
            return "storage_offset"


@dataclasses.dataclass(frozen=True)
class TensorPropertySource(Source):
    base: Source
    prop: TensorProperty
    idx: Optional[int] = None  # None for STORAGE_OFFSET

    def __post_init__(self):
        assert self.base is not None
        if self.prop is TensorProperty.STORAGE_OFFSET:
            assert self.idx is None
        else:
            assert self.idx is not None

    def reconstruct(self, codegen):
        instructions = [
            *self.base.reconstruct(codegen),
            codegen.create_load_attr(self.prop.method_name()),
        ]
        if self.idx is not None:
            instructions.append(codegen.create_load_const(self.idx))
        instructions.extend(
            create_call_function(1 if self.idx is not None else 0, True)
        )
        return instructions

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        if self.prop is TensorProperty.SIZE:
            return f"{self.base.name()}.size()[{self.idx}]"
        elif self.prop is TensorProperty.STRIDE:
            return f"{self.base.name()}.stride()[{self.idx}]"
        elif self.prop is TensorProperty.STORAGE_OFFSET:
            assert self.idx is None
            return f"{self.base.name()}.storage_offset()"
        else:
            raise AssertionError(f"unhandled {self.prop}")


@dataclasses.dataclass(frozen=True)
class NegateSource(Source):
    base: Source

    def __post_init__(self):
        assert self.base is not None

    def reconstruct(self, codegen):
        raise NotImplementedError()

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        # NB: use method call so that function stripping regexes work
        return f"{self.base.name()}.__neg__()"


@dataclasses.dataclass(frozen=True)
class DefaultsSource(Source):
    base: Source
    idx_key: Union[int, str]
    is_kw: bool = False
    field: str = dataclasses.field(init=False, repr=False, compare=False)
    _name: str = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self):
        assert (
            self.base
        ), "Base must be a valid source in order to properly track and guard this Defaults to its origin."
        if self.is_kw:
            assert isinstance(self.idx_key, str)
            object.__setattr__(self, "field", "__kwdefaults__")
            object.__setattr__(
                self, "_name", f"{self.base.name()}.{self.field}['{self.idx_key}']"
            )
        else:
            assert isinstance(self.idx_key, int)
            object.__setattr__(self, "field", "__defaults__")
            object.__setattr__(
                self, "_name", f"{self.base.name()}.{self.field}[{self.idx_key}]"
            )

    def reconstruct(self, codegen):
        instrs = self.base.reconstruct(codegen)
        instrs.extend(codegen.create_load_attrs(self.field))
        instrs.extend(
            [
                codegen.create_load_const(self.idx_key),
                create_instruction("BINARY_SUBSCR"),
            ]
        )
        return instrs

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        return self._name


@dataclasses.dataclass(frozen=True)
class GetItemSource(Source):
    base: Source
    index: Any
    index_is_slice: bool = False

    def __post_init__(self):
        assert self.base is not None
        if isinstance(self.index, slice):
            # store the hashable version of the slice so the whole GetItemSource is hashable
            super().__setattr__("index", self.index.__reduce__())
            super().__setattr__("index_is_slice", True)

    def reconstruct(self, codegen):
        instrs = self.base.reconstruct(codegen)

        if isinstance(self.index, Source):
            instrs.extend(self.index.reconstruct(codegen))
        else:
            if self.index_is_slice:
                instrs.append(codegen.create_load_const(self.unpack_slice()))
            else:
                instrs.append(codegen.create_load_const(self.index))
        instrs.append(create_instruction("BINARY_SUBSCR"))

        return instrs

    def guard_source(self):
        return self.base.guard_source()

    def unpack_slice(self):
        assert self.index_is_slice
        slice_class, slice_args = self.index
        return slice_class(*slice_args)

    def name(self):
        if isinstance(self.index, Source):
            return f"{self.base.name()}[{self.index.name()}]"
        else:
            if self.index_is_slice:
                return f"{self.base.name()}[{self.unpack_slice()!r}]"
            elif isinstance(self.index, enum.Enum):
                return f"{self.base.name()}[{enum_repr(self.index, self.guard_source().is_local())}]"
            else:
                return f"{self.base.name()}[{self.index!r}]"


@dataclasses.dataclass(frozen=True)
class TupleIteratorGetItemSource(GetItemSource):
    def reconstruct(self, codegen):
        codegen.load_import_from(utils.__name__, "tuple_iterator_getitem")
        return [
            *self.base.reconstruct(codegen),
            codegen.create_load_const(self.index),
            *create_call_function(2, True),
        ]

    def name(self):
        return f"___tuple_iterator_getitem({self.base.name()}, {self.index!r})"


@dataclasses.dataclass(frozen=True)
class TypeSource(Source):
    base: Source

    def __post_init__(self):
        assert self.base is not None

    def reconstruct(self, codegen):
        codegen.load_import_from("builtins", "type")
        return self.base.reconstruct(codegen) + create_call_function(1, True)

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        return f"type({self.base.name()})"


@dataclasses.dataclass(frozen=True)
class SuperSource(Source):
    type: Source
    obj: Source

    def __post_init__(self):
        assert self.type is not None
        assert self.obj is not None

    def reconstruct(self, codegen):
        codegen.load_import_from("builtins", "super")
        return (
            self.type.reconstruct(codegen)
            + self.obj.reconstruct(codegen)
            + create_call_function(2, True)
        )

    def guard_source(self):
        return self.obj.guard_source()

    def name(self):
        return f"super({self.type.name()}, {self.obj.name()})"


@dataclasses.dataclass(frozen=True)
class ODictGetItemSource(Source):
    base: Source
    index: Any

    def __post_init__(self):
        assert self.base is not None

    def reconstruct(self, codegen):
        return [
            codegen._create_load_const(collections.OrderedDict.__getitem__),
            *self.base.reconstruct(codegen),
            codegen.create_load_const(self.index),
            *create_call_function(2, True),
        ]

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        if isinstance(self.index, type):
            rep = f'__load_module("{self.index.__module__}").{self.index.__qualname__}'
            return f"___odict_getitem({self.base.name()}, {rep})"
        else:
            return f"___odict_getitem({self.base.name()}, {self.index!r})"


@dataclasses.dataclass(frozen=True)
class NNModuleSource(Source):
    inner: Source

    def reconstruct(self, codegen):
        return self.inner.reconstruct(codegen)

    def guard_source(self):
        return _GUARD_SOURCE_NN_MODULE[self.inner.guard_source()]

    def name(self):
        return self.inner.name()


@dataclasses.dataclass(frozen=True)
class NotNNModuleSource(NNModuleSource):
    def guard_source(self):
        return _GUARD_SOURCE_NOT_NN_MODULE[self.inner.guard_source()]


@dataclasses.dataclass(frozen=True)
class FSDPNNModuleSource(NNModuleSource):
    def guard_source(self):
        return _GUARD_SOURCE_FSDP_MODULE[self.inner.guard_source()]


@dataclasses.dataclass(frozen=True)
class DeterministicAlgorithmsSource(Source):
    def name(self):
        return ""

    def guard_source(self):
        return GuardSource.GLOBAL


@dataclasses.dataclass(frozen=True)
class GradModeSource(Source):
    def name(self):
        return ""

    def guard_source(self):
        return GuardSource.GLOBAL


@dataclasses.dataclass(frozen=True)
class DefaultDeviceSource(Source):
    def name(self):
        return ""

    def guard_source(self):
        return GuardSource.GLOBAL


@dataclasses.dataclass(frozen=True)
class ConstantSource(Source):
    source_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load_global(self.source_name, False, add=False)]

    def guard_source(self):
        return GuardSource.CONSTANT

    def name(self):
        return self.source_name

    def make_guard(self, fn, is_volatile=False):
        raise NotImplementedError()


# This is a synthetic source that is associated with the singleton
# shape env guard we always register for all frames.  We get the actual
# guard contents from the ambient ShapeEnv
@dataclasses.dataclass(frozen=True)
class ShapeEnvSource(Source):
    def name(self):
        return ""

    def guard_source(self):
        return GuardSource.SHAPE_ENV
