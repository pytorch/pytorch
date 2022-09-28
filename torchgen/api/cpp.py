from typing import List, Optional, Sequence, Set, Union

from torchgen import local
from torchgen.api.types import (
    ArgName,
    ArrayCType,
    ArrayRefCType,
    BaseCType,
    BaseTypeToCppMapping,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    dimnameListT,
    intArrayRefT,
    iTensorListRefT,
    ListCType,
    longT,
    MutRefCType,
    NamedCType,
    OptionalCType,
    optionalIntArrayRefT,
    scalarT,
    SpecialArgName,
    symIntArrayRefT,
    SymIntT,
    tensorListT,
    tensorOptionsT,
    tensorT,
    TupleCType,
    VectorCType,
    voidT,
)
from torchgen.model import (
    Argument,
    Arguments,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    OptionalType,
    Return,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from torchgen.utils import assert_never

# This file describes the translation of JIT schema to the public C++
# API, which is what people use when they call functions like at::add.
#
# Prominent characteristics of the C++ API:
#
#   - dtype, layout, device and pin_memory are collected into
#     a single C++ type TensorOptions  (the native functions API
#     also has this, but tensor options is really most relevant
#     for the C++ API; it makes calling kwarg factory functions
#     pleasant)
#
#   - defaulting lives here (in fact, the dispatcher is completely
#     oblivious of defaults!)
#
# BTW: policy on name collisions: we try not to have types with
# collisions, but functions are fair game to collide


def name(
    func: FunctionSchema,
    *,
    faithful_name_for_out_overloads: bool = False,
    symint_overload: bool = False,
) -> str:
    name = str(func.name.name)
    if symint_overload:
        name += "_symint"
    if func.is_out_fn():
        if faithful_name_for_out_overloads:
            name += "_outf"
        else:
            name += "_out"

    return name


# Translation of "value types" in JIT schema to C++ API type.  Value
# types look the same no matter if they are argument types or return
# types.  Returns None if the type in question is not a value type.
def valuetype_type(
    t: Type,
    *,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
    symint: bool = False,
) -> Optional[NamedCType]:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor or t.name == BaseTy.Scalar:
            return None
        elif str(t) == "SymInt":
            if symint:
                return NamedCType(binds, BaseCType(SymIntT))
            else:
                return NamedCType(binds, BaseCType(longT))
        if remove_non_owning_ref_types:
            if t.name == BaseTy.str:
                raise AssertionError(
                    "string ref->value conversion: not implemented yet"
                )
        # All other BaseType currently map directly to BaseCppTypes.
        return NamedCType(binds, BaseCType(BaseTypeToCppMapping[t.name]))
    elif isinstance(t, OptionalType):
        if str(t.elem) == "SymInt" and symint:
            # clang7 appears to mis-compile c10::optional<c10::SymInt>
            # so instead we wrap optional SymInt arguments as const references.
            # This manifested as an ASAN error under a clang7 compilation:
            #
            # test_view_inplace (__main__.TestFunctionalization) ... =================================================================  # noqa: B950
            #  6234
            #   ==792==ERROR: AddressSanitizer: stack-use-after-return on address 0x7f02e9ddb460 at pc 0x7f02be9d2380 bp 0x7ffde3af28b0 sp 0x7ffde3af28a8  # noqa: B950
            # 6235
            #   READ of size 8 at 0x7f02e9ddb460 thread T0
            # 6236
            #       #0 0x7f02be9d237f in std::decay<c10::guts::infer_function_traits<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor const& (c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>), &(at::functionalization::as_strided_(c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>))>, at::Tensor const&, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt> > > >::type::return_type>::type c10::impl::call_functor_with_args_from_stack_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor const& (c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>), &(at::functionalization::as_strided_(c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>))>, at::Tensor const&, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt> > >, false, 0ul, 1ul, 2ul, 3ul, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt> >(c10::OperatorKernel*, c10::DispatchKeySet, std::vector<c10::IValue, std::allocator<std::vector> >*, std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul>, c10::guts::typelist::typelist<at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt> >*) (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so+0x1200837f)  # noqa: B950
            # 6237
            #       #1 0x7f02be9d1b36 in auto c10::impl::make_boxed_from_unboxed_functor<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor const& (c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>), &(at::functionalization::as_strided_(c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>))>, at::Tensor const&, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt> > >, false>::call(c10::OperatorKernel*, c10::OperatorHandle const&, c10::DispatchKeySet, std::vector<c10::IValue, std::allocator<c10::IValue> >*)::'lambda'(auto)::operator()<c10::guts::detail::_identity>(auto) const (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so+0x12007b36)  # noqa: B950
            # 6238
            #       #2 0x7f02be9d1904 in c10::impl::make_boxed_from_unboxed_functor<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor const& (c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>), &(at::functionalization::as_strided_(c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>))>, at::Tensor const&, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt> > >, false>::call(c10::OperatorKernel*, c10::OperatorHandle const&, c10::DispatchKeySet, std::vector<c10::IValue, std::allocator<c10::IValue> >*) (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so+0x12007904)  # noqa: B950
            # 6239
            #       #3 0x7f02b9e013f1 in c10::Dispatcher::callBoxed(c10::OperatorHandle const&, std::vector<c10::IValue, std::allocator<c10::IValue> >*) const (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so+0xd4373f1)  # noqa: B950
            # 6240
            #       #4 0x7f02ba4a8b3c in at::functorch::FunctionalizeInterpreterPtr::processImpl(c10::OperatorHandle const&, std::vector<c10::IValue, std::allocator<c10::IValue> >*) (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so+0  # noqa: B950xdadeb3c)
            # 6241
            #       #5 0x7f02ba4aa8a0 in at::functorch::Interpreter::process(c10::OperatorHandle const&, std::vector<c10::IValue, std::allocator<c10::IValue> >*) (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so+0xdae08a0)  # noqa: B950
            # 6242
            #       #6 0x7f02ba49f6b9 in at::functorch::dynamicLayerFrontFallback(c10::OperatorHandle const&, std::vector<c10::IValue, std::allocator<c10::IValue> >*) (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so+0xdad56b9)  # noqa: B950
            # 6243
            #       #7 0x7f02bbfbc222 in c10::impl::BoxedKernelWrapper<at::Tensor const& (at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>), void>::call(c10::BoxedKernel const&, c10::OperatorHandle const&  # noqa: B950, c10::DispatchKeySet, at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>) (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so+0xf5f2222)
            # 6244
            #       #8 0x7f02bbd78b42 in at::_ops::as_strided_::call(at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>) (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so+0xf3aeb42)  # noqa: B950
            # 6245
            #       #9 0x7f02e4d453e1 in at::Tensor::as_strided_(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::optional<long>) const (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_python.so+0x2d8b3e1)  # noqa: B950
            # 6246
            #       #10 0x7f02e4d3e2de in torch::functorch::impl::_propagate_functional_input_mutation(at::Tensor const&, at::Tensor const&) (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_python.so+0x2d842de)  # noqa: B950
            # 6247
            #       #11 0x7f02e4d5417c in void pybind11::cpp_function::initialize<void (*&)(at::Tensor const&, at::Tensor const&), void, at::Tensor const&, at::Tensor const&, pybind11::name, pybind11::scope, pybind11::sibling, char [3  # noqa: B9507]>(void (*&)(at::Tensor const&, at::Tensor const&), void (*)(at::Tensor const&, at::Tensor const&), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&, char const (&) [37])::'lambda'(pybind11::detail::function_call&)::operator()(pybind11::detail::function_call&) const (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_python.so+0x2d9a17c)
            # 6248
            #       #12 0x7f02e426183d in pybind11::cpp_function::dispatcher(_object*, _object*, _object*) (/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_python.so+0x22a783d)  # noqa: B950
            return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(SymIntT))))
        elem = valuetype_type(t.elem, binds=binds, symint=symint)
        if elem is None:
            return None
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if str(t.elem) == "bool":
            assert t.size is not None
            return NamedCType(binds, ArrayCType(BaseCType(boolT), t.size))
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
    symint: bool = False,
) -> NamedCType:
    # If it's a value type, do the value type translation
    r = valuetype_type(
        t,
        binds=binds,
        symint=symint,
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
        elif isinstance(t.elem, ListType) and str(t.elem.elem) == "int":
            return NamedCType(binds, BaseCType(optionalIntArrayRefT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds, symint=symint)
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        # TODO: remove these special cases, ArrayRef fallthrough works fine
        if str(t.elem) == "int":
            if remove_non_owning_ref_types:
                return NamedCType(binds, VectorCType(BaseCType(longT)))
            else:
                return NamedCType(binds, BaseCType(intArrayRefT))
        if str(t.elem) == "SymInt":
            if remove_non_owning_ref_types:
                if symint:
                    return NamedCType(binds, VectorCType(BaseCType(SymIntT)))
                else:
                    return NamedCType(binds, VectorCType(BaseCType(longT)))
            else:
                if symint:
                    return NamedCType(binds, BaseCType(symIntArrayRefT))
                else:
                    return NamedCType(binds, BaseCType(intArrayRefT))
        if str(t.elem) == "Tensor":
            if local.use_ilistref_for_tensor_lists():
                return NamedCType(binds, ConstRefCType(BaseCType(iTensorListRefT)))
            else:
                return NamedCType(binds, BaseCType(tensorListT))
        elif str(t.elem) == "Scalar":
            return NamedCType(binds, ArrayRefCType(BaseCType(scalarT)))
        elif str(t.elem) == "Dimname":
            return NamedCType(binds, BaseCType(dimnameListT))
        elif str(t.elem) == "Tensor?":
            return NamedCType(
                binds, ConstRefCType(ListCType(OptionalCType(BaseCType(tensorT))))
            )
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds, symint=symint)
        return NamedCType(binds, ArrayRefCType(elem.type))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# Translate a JIT argument into its C++ type
def argument_type(a: Argument, *, binds: ArgName, symint: bool = False) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, symint=symint, binds=binds)


# Translation of a (non-multi) return type from JIT to C++
# N.B: returntype_type returns a CType, not a NamedCType.
# This is mostly because of the mismatch between return types and return names.
# e.g. a function with a return type of 'void' has 0 return names,
# and a function with a return type of 'std::tuple' has >1 return name.
def returntype_type(t: Type, *, mutable: bool, symint: bool = False) -> CType:
    # placeholder is ignored
    r = valuetype_type(t, binds="__placeholder__", symint=symint)
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
        elem = returntype_type(t.elem, mutable=False, symint=symint)
        assert t.size is None, f"fixed size list returns not supported: {t}"
        return VectorCType(elem)

    raise AssertionError(f"unrecognized return type {t}")


# Translation of a single return to its C++ type
def return_type(r: Return, *, symint: bool = False) -> CType:
    return returntype_type(r.type, mutable=r.is_write, symint=symint)


# Translation of a full (possibly multi) return from JIT to its C++ type
def returns_type(rs: Sequence[Return], *, symint: bool = False) -> CType:
    if len(rs) == 0:
        return BaseCType(voidT)
    elif len(rs) == 1:
        return return_type(rs[0], symint=symint)
    else:
        return TupleCType([return_type(r, symint=symint) for r in rs])


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
    "None": "c10::nullopt",  # UGH this one is type directed
    "Mean": "at::Reduction::Mean",
    "[]": "{}",
    "contiguous_format": "MemoryFormat::Contiguous",
    "long": "at::kLong",
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
            return "c10::nullopt"

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
    symint: bool = False,
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
            symint=symint,
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
                nctype=argument_type(a, binds=binds, symint=symint),
                name=a.name,
                default=default,
                argument=a,
            )
        ]
    elif isinstance(a, TensorOptionsArguments):
        if faithful:
            return (
                sub_argument(a.dtype)
                + sub_argument(a.layout)
                + sub_argument(a.device)
                + sub_argument(a.pin_memory)
            )
        else:
            default = None
            # Enforced by NativeFunction.__post_init__
            assert "options" not in cpp_no_default_args
            if all(x.default == "None" for x in a.all()):
                default = "{}"
            elif a.dtype.default == "long":
                default = "at::kLong"  # TODO: this is wrong
            return [
                Binding(
                    nctype=NamedCType("options", BaseCType(tensorOptionsT)),
                    name="options",
                    default=default,
                    argument=a,
                )
            ]
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
    symint: bool = False,
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
            symint=symint,
            method=method,
            has_tensor_options=arguments.tensor_options is not None,
            cpp_no_default_args=cpp_no_default_args,
        )
    ]
