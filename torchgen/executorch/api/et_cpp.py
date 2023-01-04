from typing import List, Optional, Sequence, Set, Union

from torchgen import local
from torchgen.api.types import (
    ArgName,
    ArrayCType,
    BaseCType,
    Binding,
    ConstRefCType,
    CType,
    MutRefCType,
    NamedCType,
    SpecialArgName,
    TupleCType,
    VectorCType,
    voidT,
)
from torchgen.model import (
    Argument,
    Arguments,
    BaseTy,
    BaseType,
    ListType,
    NativeFunction,
    OptionalType,
    Return,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from torchgen.utils import assert_never
from .types import (
    ArrayRefCType,
    BaseTypeToCppMapping,
    OptionalCType,
    scalarT,
    tensorListT,
    tensorT,
)

"""
This file describes the translation of JIT schema to the public C++ API, which is what people use when they call
functions like at::add. It also serves as a native function API, which is the signature of kernels,
since in Executorch CppSignature is the same as NativeSignature.

Difference between this file and torchgen.api.cpp.py:

  - Executorch doesn't support TensorOptions, however in this file we still keep the logic here to be compatible with
    torchgen.api.cpp, so that we can do stuff like ATen mode (running ATen kernels in Executorch).

  - Executorch doesn't support Dimname.

  - Executorch runtime doesn't support SymInt, will treat it as int.
"""


# Translation of "value types" in JIT schema to C++ API type.  Value
# types look the same no matter if they are argument types or return
# types.  Returns None if the type in question is not a value type.
def valuetype_type(
    t: Type,
    *,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
) -> Optional[NamedCType]:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor or t.name == BaseTy.Scalar:
            return None
        # For SymInt we simply treat it as int.
        elif str(t) == "SymInt":
            return NamedCType(binds, BaseCType(BaseTypeToCppMapping[BaseTy.int]))
        if remove_non_owning_ref_types:
            if t.name == BaseTy.str:
                raise AssertionError(
                    "string ref->value conversion: not implemented yet"
                )
        # All other BaseType currently map directly to BaseCppTypes.
        return NamedCType(binds, BaseCType(BaseTypeToCppMapping[t.name]))
    elif isinstance(t, OptionalType):
        elem = valuetype_type(t.elem, binds=binds)
        if elem is None:
            return None
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if str(t.elem) == "bool":
            assert t.size is not None
            return NamedCType(
                binds, ArrayCType(BaseCType(BaseTypeToCppMapping[BaseTy.bool]), t.size)
            )
        else:
            return None
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# Translation of types occuring in JIT arguments to a C++ argument type.
# If remove_non_owning_ref_types is set, we'll guarantee that the outputed CType is not a non-owning reference type.
# For example, we'll return std::vector<int> instead of IntArrayRef.
# See Note [translation from C++ reference to value types]
def argumenttype_type(
    t: Type,
    *,
    mutable: bool,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
) -> NamedCType:
    # If it's a value type, do the value type translation
    r = valuetype_type(
        t,
        binds=binds,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
    )
    if r is not None:
        return r
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable and not local.use_const_ref_for_mutable_tensors():
                return NamedCType(binds, MutRefCType(BaseCType(tensorT)))
            else:
                return NamedCType(binds, ConstRefCType(BaseCType(tensorT)))
        elif t.name == BaseTy.Scalar:
            return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
        else:
            raise AssertionError(f"base type should have been value type {t}")
    elif isinstance(t, OptionalType):
        if str(t.elem) == "Tensor":
            if mutable and not local.use_const_ref_for_mutable_tensors():
                return NamedCType(
                    binds, MutRefCType(BaseCType(tensorT))
                )  # TODO: fix this discrepancy
            else:
                return NamedCType(
                    binds, ConstRefCType(OptionalCType(BaseCType(tensorT)))
                )
        elif str(t.elem) == "Scalar":
            return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(scalarT))))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        # TODO: keeping these special cases for Tensor[] and Tensor?[] so that we can hookup with ATen kernels.
        if str(t.elem) == "Tensor":
            return NamedCType(binds, BaseCType(tensorListT))
        elif str(t.elem) == "Dimname":
            raise NotImplementedError("Executorch doesn't support Dimname")
        elif str(t.elem) == "Tensor?":
            return NamedCType(binds, ArrayRefCType(OptionalCType(BaseCType(tensorT))))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, ArrayRefCType(elem.type))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# Translate a JIT argument into its C++ type
def argument_type(a: Argument, *, binds: ArgName) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds)


# Translation of a (non-multi) return type from JIT to C++
# N.B: returntype_type returns a CType, not a NamedCType.
# This is mostly because of the mismatch between return types and return names.
# e.g. a function with a return type of 'void' has 0 return names,
# and a function with a return type of 'std::tuple' has >1 return name.
def returntype_type(t: Type, *, mutable: bool) -> CType:
    # placeholder is ignored
    r = valuetype_type(t, binds="__placeholder__")
    if r is not None:
        return r.type

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                if local.use_const_ref_for_mutable_tensors():
                    return ConstRefCType(BaseCType(tensorT))
                else:
                    return MutRefCType(BaseCType(tensorT))
            else:
                # Note [Tensor Copy Returns]
                # Currently, we use "Argument.is_write" to determine
                # whether or not Tensor return types should be copies or references.
                # If that ever changes, take a look at other locations of this note!
                return BaseCType(tensorT)
        elif t.name == BaseTy.Scalar:
            return BaseCType(scalarT)
    elif isinstance(t, ListType):
        assert (
            not mutable
        ), "Native functions should never return a mutable tensor list. They should return void."
        elem = returntype_type(t.elem, mutable=False)
        assert t.size is None, f"fixed size list returns not supported: {t}"
        return VectorCType(elem)

    raise AssertionError(f"unrecognized return type {t}")


# Translation of a single return to its C++ type
def return_type(r: Return) -> CType:
    return returntype_type(r.type, mutable=r.is_write)


# Translation of a full (possibly multi) return from JIT to its C++ type
def returns_type(rs: Sequence[Return]) -> CType:
    if len(rs) == 0:
        return BaseCType(voidT)
    elif len(rs) == 1:
        return return_type(rs[0])
    else:
        return TupleCType([return_type(r) for r in rs])


def return_names(f: NativeFunction, *, fallback_name: str = "result") -> Sequence[str]:
    returns: List[str] = []
    for i, r in enumerate(f.func.returns):
        # If we have an inplace function, the return argument is
        # implicitly named self.
        # TODO: Consider incorporating this into the data model
        if f.func.name.name.inplace:
            assert i == 0, "illegal inplace function with multiple returns"
            name = "self"
        # If we are out function, the name is the name of the
        # corresponding output function (r.name will get recorded
        # in field_name later.)
        elif f.func.is_out_fn():
            name = f.func.arguments.out[i].name
        # If the return argument is explicitly named...
        elif r.name:
            name_conflict = any(
                r.name == a.name for a in f.func.schema_order_arguments()
            )
            if name_conflict and not f.func.is_out_fn():
                name = f"{r.name}_return"
            else:
                name = r.name
        # If there is no explicit name and no fallback name was passed in, we just name the output result,
        # unless it's a multi-return, in which case it's result0,
        # result1, etc (zero-indexed)
        else:
            name = fallback_name if len(f.func.returns) == 1 else f"{fallback_name}{i}"
        returns.append(name)
    return returns


JIT_TO_CPP_DEFAULT = {
    "False": "false",
    "True": "true",
    "None": "torch::executorch::nullopt",  # UGH this one is type directed
    "[]": "{}",
    "contiguous_format": "torch::executorch::MemoryFormat::Contiguous",
    "long": "torch::executorch::kLong",
}


# Convert a JIT default into C++ expression representing the default
def default_expr(d: str, t: Type) -> str:
    if d == "None" and str(t) == "Tensor?":
        return "{}"
    if isinstance(t, BaseType) and t.name is BaseTy.str:
        # Schema allows single quotes but C++ needs double
        if len(d) >= 2 and d[0] == "'" and d[-1] == "'":
            s = ""
            i = 1
            while i + 1 < len(d):
                if d[i] != "\\":
                    if d[i] == '"':
                        s += '\\"'
                    else:
                        s += d[i]
                    i += 1
                else:
                    if d[i + 1] == "'":
                        s += "'"
                    else:
                        s += d[i : i + 2]
                    i += 2

            return f'"{s}"'

    if isinstance(t, OptionalType):
        if d == "None":
            return "torch::executor::nullopt"

        return default_expr(d, t.elem)

    if isinstance(t, ListType):
        if d.startswith("[") and d.endswith("]"):
            return "{" + d[1:-1] + "}"
        elif t.size is None:
            # NOTE: Sized lists can have scalar defaults
            raise ValueError(f"Expected a list default '[...]' but found: '{d}'")

    return JIT_TO_CPP_DEFAULT.get(d, d)


# Convert an argument into its C++ API form


def argument(
    a: Union[Argument, TensorOptionsArguments, SelfArgument],
    *,
    cpp_no_default_args: Set[str],
    method: bool,
    faithful: bool,
    has_tensor_options: bool,
) -> List[Binding]:
    def sub_argument(
        a: Union[Argument, TensorOptionsArguments, SelfArgument]
    ) -> List[Binding]:
        return argument(
            a,
            cpp_no_default_args=cpp_no_default_args,
            method=method,
            faithful=faithful,
            has_tensor_options=has_tensor_options,
        )

    if isinstance(a, Argument):
        binds: ArgName
        if a.name == "memory_format" and has_tensor_options:
            binds = SpecialArgName.possibly_redundant_memory_format
        else:
            binds = a.name
        default: Optional[str] = None
        if a.name not in cpp_no_default_args and a.default is not None:
            default = default_expr(a.default, a.type)
        return [
            Binding(
                nctype=argument_type(a, binds=binds),
                name=a.name,
                default=default,
                argument=a,
            )
        ]
    elif isinstance(a, TensorOptionsArguments):
        raise NotImplementedError("Need to implement type resolution for TensorOptions")
    elif isinstance(a, SelfArgument):
        if method:
            # Caller is responsible for installing implicit this in context!
            return []
        else:
            return sub_argument(a.argument)
    else:
        assert_never(a)


def arguments(
    arguments: Arguments,
    *,
    faithful: bool,
    method: bool,
    cpp_no_default_args: Set[str],
) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    if faithful:
        args.extend(arguments.non_out)
        args.extend(arguments.out)
    else:
        args.extend(arguments.out)
        args.extend(arguments.non_out)
    return [
        r.no_default() if faithful else r
        for a in args
        for r in argument(
            a,
            faithful=faithful,
            method=method,
            has_tensor_options=arguments.tensor_options is not None,
            cpp_no_default_args=cpp_no_default_args,
        )
    ]
