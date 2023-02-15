import collections
import dataclasses
import enum
from typing import Any, Optional, Union

from torch._guards import GuardSource, Source

from . import utils
from .bytecode_transformation import create_instruction
from .utils import enum_repr, rename_implicit

_GUARD_SOURCE_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_NN_MODULE,
    GuardSource.LOCAL_NN_MODULE: GuardSource.LOCAL_NN_MODULE,
    GuardSource.GLOBAL_NN_MODULE: GuardSource.GLOBAL_NN_MODULE,
}

_GUARD_SOURCE_NOT_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL,
    GuardSource.GLOBAL: GuardSource.GLOBAL,
    GuardSource.LOCAL_NN_MODULE: GuardSource.LOCAL,
    GuardSource.GLOBAL_NN_MODULE: GuardSource.GLOBAL,
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
    ]


@dataclasses.dataclass
class LocalSource(Source):
    local_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load(self.local_name)]

    def guard_source(self):
        return GuardSource.LOCAL

    def name(self):
        return rename_implicit(self.local_name)


@dataclasses.dataclass
class LocalInputSource(LocalSource):
    pos: int


@dataclasses.dataclass
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
        return rename_implicit(f"random_value_{self.random_call_index}")


@dataclasses.dataclass
class GlobalSource(Source):
    global_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load_global(self.global_name, add=True)]

    def guard_source(self):
        return GuardSource.GLOBAL

    def name(self):
        return self.global_name


@dataclasses.dataclass
class GlobalWeakRefSource(Source):
    global_name: str

    def reconstruct(self, codegen):
        return [
            codegen.create_load_global(self.global_name, add=True),
            create_instruction("CALL_FUNCTION", 0),
        ]

    def guard_source(self):
        return GuardSource.GLOBAL

    def name(self):
        return f"{self.global_name}()"


@dataclasses.dataclass
class AttrSource(Source):
    base: Source
    member: str

    def __init__(self, base, member):
        super().__init__()
        assert base, "Can't construct an AttrSource without a valid base source"
        if "." in member:
            member_parts = member.split(".")
            self.base = AttrSource(base, ".".join(member_parts[:-1]))
            self.member = member_parts[-1]
        else:
            self.base = base
            self.member = member
        assert self.base is not None

    def reconstruct(self, codegen):
        return self.base.reconstruct(codegen) + codegen.create_load_attrs(self.member)

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        if self.member.isnumeric():
            return f"getattr({self.base.name()}, {self.member!r})"
        return f"{self.base.name()}.{self.member}"


class TensorProperty(enum.Enum):
    SIZE = 0
    STRIDE = 1
    STORAGE_OFFSET = 2


@dataclasses.dataclass
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
        raise NotImplementedError()

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


@dataclasses.dataclass
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


@dataclasses.dataclass
class DefaultsSource(Source):
    base: Source
    idx_key: Union[int, str]
    is_kw: bool
    field: str

    def __init__(self, base, idx_key, is_kw=False):
        super().__init__()
        assert (
            base
        ), "Base must be a valid source in order to properly track and guard this Defaults to its origin."
        self.base = base
        self.idx_key = idx_key
        self.is_kw = is_kw
        if self.is_kw:
            assert isinstance(idx_key, str)
            self.field = "__kwdefaults__"
            self._name = f"{self.base.name()}.{self.field}['{self.idx_key}']"
        else:
            assert isinstance(idx_key, int)
            self.field = "__defaults__"
            self._name = f"{self.base.name()}.{self.field}[{self.idx_key}]"

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


@dataclasses.dataclass
class GetItemSource(Source):
    base: Source
    index: Any

    def __post_init__(self):
        assert self.base is not None

    def reconstruct(self, codegen):
        instrs = self.base.reconstruct(codegen)

        if isinstance(self.index, Source):
            instrs.extend(self.index.reconstruct(codegen))
        else:
            instrs.append(codegen.create_load_const(self.index))
        instrs.append(create_instruction("BINARY_SUBSCR"))

        return instrs

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        if isinstance(self.index, Source):
            return f"{self.base.name()}[{self.index.name()}]"
        else:
            if isinstance(self.index, enum.Enum):
                return f"{self.base.name()}[{enum_repr(self.index)}]"
            else:
                return f"{self.base.name()}[{self.index!r}]"


@dataclasses.dataclass
class TupleIteratorGetItemSource(GetItemSource):
    def reconstruct(self, codegen):
        codegen.load_import_from(utils.__name__, "tuple_iterator_getitem")
        return self.base.reconstruct(codegen) + [
            codegen.create_load_const(self.index),
            create_instruction("CALL_FUNCTION", 2),
        ]

    def name(self):
        return f"___tuple_iterator_getitem({self.base.name()}, {self.index!r})"


@dataclasses.dataclass
class TypeSource(Source):
    base: Source

    def __post_init__(self):
        assert self.base is not None

    def reconstruct(self, codegen):
        codegen.load_import_from("builtins", "type")
        return self.base.reconstruct(codegen) + [create_instruction("CALL_FUNCTION", 1)]

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        return f"type({self.base.name()})"


@dataclasses.dataclass
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
            + [create_instruction("CALL_FUNCTION", 2)]
        )

    def guard_source(self):
        return self.obj.guard_source()

    def name(self):
        return f"super({self.type.name()}, {self.obj.name()})"


@dataclasses.dataclass
class ODictGetItemSource(Source):
    base: Source
    index: Any

    def __post_init__(self):
        assert self.base is not None

    def reconstruct(self, codegen):
        return (
            [codegen._create_load_const(collections.OrderedDict.__getitem__)]
            + self.base.reconstruct(codegen)
            + [
                codegen.create_load_const(self.index),
                create_instruction("CALL_FUNCTION", 2),
            ]
        )

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        return f"___odict_getitem({self.base.name()}, {self.index!r})"


@dataclasses.dataclass
class NNModuleSource(Source):
    inner: Source

    def reconstruct(self, codegen):
        return self.inner.reconstruct(codegen)

    def guard_source(self):
        return _GUARD_SOURCE_NN_MODULE[self.inner.guard_source()]

    def name(self):
        return self.inner.name()


class NotNNModuleSource(NNModuleSource):
    def guard_source(self):
        return _GUARD_SOURCE_NOT_NN_MODULE[self.inner.guard_source()]


@dataclasses.dataclass
class ConstantSource(Source):
    source_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load_global(self.source_name, add=False)]

    def guard_source(self):
        return GuardSource.CONSTANT

    def name(self):
        return self.source_name

    def make_guard(self, fn, is_volatile=False):
        raise NotImplementedError()


# This is a synthetic source that is associated with the singleton
# shape env guard we always register for all frames.  We get the actual
# guard contents from the ambient ShapeEnv
@dataclasses.dataclass
class ShapeEnvSource(Source):
    def name(self):
        return ""

    def guard_source(self):
        return GuardSource.SHAPE_ENV
