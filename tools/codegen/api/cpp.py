from tools.codegen.model import *
from tools.codegen.api.types import *
from typing import Optional, Sequence, Union, List, Set

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

def name(func: FunctionSchema, *, faithful_name_for_out_overloads: bool = False) -> str:
    name = str(func.name.name)
    if func.is_out_fn():
        if faithful_name_for_out_overloads:
            name += '_outf'
        else:
            name += '_out'

    return name

# Translation of "value types" in JIT schema to C++ API type.  Value
# types look the same no matter if they are argument types or return
# types.  Returns None if the type in question is not a value type.
def valuetype_type(t: Type, *, binds: ArgName) -> Optional[CType]:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor or t.name == BaseTy.Scalar:
            return None
        elif t.name == BaseTy.int:
            return BaseCType('int64_t', binds)
        elif t.name == BaseTy.float:
            return BaseCType('double', binds)
        elif t.name == BaseTy.str:
            return BaseCType('std::string', binds)
        elif t.name in [BaseTy.bool, BaseTy.QScheme, BaseTy.Scalar,
                        BaseTy.ScalarType, BaseTy.Generator, BaseTy.Storage,
                        BaseTy.Layout, BaseTy.Device, BaseTy.MemoryFormat,
                        BaseTy.Dimname, BaseTy.Stream, BaseTy.ConstQuantizerPtr]:
            # These C++ names line up with their schema names
            return BaseCType(t.name.name, binds)
        else:
            raise AssertionError(f"unsupported type: {t}")
    elif isinstance(t, OptionalType):
        elem = valuetype_type(t.elem, binds=binds)
        if elem is None:
            return None
        return OptionalCType(elem)
    elif isinstance(t, ListType):
        if str(t.elem) == 'bool':
            assert t.size is not None
            return BaseCType(f"std::array<bool,{t.size}>", binds)
        else:
            return None
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")

# Translation of types occuring in JIT arguments to a C++ argument type.
def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName) -> CType:
    # If it's a value type, do the value type translation
    r = valuetype_type(t, binds=binds)
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                return MutRefCType(BaseCType('Tensor', binds))
            else:
                return ConstRefCType(BaseCType('Tensor', binds))
        elif t.name == BaseTy.Scalar:
            return ConstRefCType(BaseCType('Scalar', binds))
        else:
            raise AssertionError(f"base type should have been value type {t}")
    elif isinstance(t, OptionalType):
        if str(t.elem) == 'Tensor':
            if mutable:
                return MutRefCType(BaseCType('Tensor', binds))  # TODO: fix this discrepancy
            else:
                return ConstRefCType(OptionalCType(BaseCType('Tensor', binds)))
        elif str(t.elem) == 'Scalar':
            return ConstRefCType(OptionalCType(BaseCType('Scalar', binds)))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return OptionalCType(elem)
    elif isinstance(t, ListType):
        # TODO: remove these special cases, ArrayRef fallthrough works fine
        # NB: CType throws away ArrayRef structure because it is not currently
        # relevant in translation.  When it becomes relevant, need to add back
        if str(t.elem) == 'int':
            return BaseCType("IntArrayRef", binds)
        elif str(t.elem) == 'Tensor':
            return BaseCType("TensorList", binds)
        elif str(t.elem) == 'Scalar':
            return BaseCType("ArrayRef<Scalar>", binds)
        elif str(t.elem) == 'Dimname':
            return BaseCType("DimnameList", binds)
        elif str(t.elem) == 'Tensor?':
            return ConstRefCType(BaseCType("c10::List<c10::optional<Tensor>>", binds))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        # TODO: explicitly qualify namespace here
        return BaseCType(f"ArrayRef<{elem.cpp_type()}>", binds)
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")

# Translate a JIT argument into its C++ type
def argument_type(a: Argument, *, binds: ArgName) -> CType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds)

# Translation of a (non-multi) return type from JIT to C++
# NB: if need translations on return types, make this return CType too.  Need to
# take care; ArgName is misnomer now, and inputs are permitted to conflict with outputs
# so need to make sure you don't have trouble
def returntype_type(t: Type, *, mutable: bool) -> str:
    # placeholder is ignored
    r = valuetype_type(t, binds="__placeholder__")
    if r is not None:
        return r.cpp_type()

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                return 'Tensor &'
            else:
                return 'Tensor'
        elif t.name == BaseTy.Scalar:
            return 'Scalar'
    elif isinstance(t, ListType):
        elem = returntype_type(t.elem, mutable=mutable)
        assert t.size is None, f"fixed size list returns not supported: {t}"
        return f"std::vector<{elem}>"

    raise AssertionError(f"unrecognized return type {t}")

# Translation of a single return to its C++ type
def return_type(r: Return) -> str:
    return returntype_type(r.type, mutable=r.is_write)

# Translation of a full (possibly multi) return from JIT to its C++ type
def returns_type(rs: Sequence[Return]) -> str:
    if len(rs) == 0:
        return 'void'
    elif len(rs) == 1:
        return return_type(rs[0])
    else:
        args = ','.join(map(return_type, rs))
        return f'std::tuple<{args}>'

def return_names(f: NativeFunction) -> Sequence[str]:
    returns: List[str] = []
    for i, r in enumerate(f.func.returns):
        # If we have an inplace function, the return argument is
        # implicitly named self.
        # TODO: Consider incorporating this into the data model
        if f.func.name.name.inplace:
            assert i == 0, "illegal inplace function with multiple returns"
            name = 'self'
        # If we are out function, the name is the name of the
        # corresponding output function (r.name will get recorded
        # in field_name later.)
        elif f.func.is_out_fn():
            name = f.func.arguments.out[i].name
        # If the return argument is explicitly named...
        elif r.name:
            name_conflict = any(r.name == a.name for a in f.func.schema_order_arguments())
            if name_conflict and not f.func.is_out_fn():
                name = f'{r.name}_return'
            else:
                name = r.name
        # If there is no explicit name, we just name the output result,
        # unless it's a multi-return, in which case it's result0,
        # result1, etc (zero-indexed)
        else:
            name = 'result' if len(f.func.returns) == 1 else f'result{i}'
        returns.append(name)
    return returns

JIT_TO_CPP_DEFAULT = {
    'False': 'false',
    'True': 'true',
    'None': 'c10::nullopt',  # UGH this one is type directed
    'Mean': 'at::Reduction::Mean',
    '[]': '{}',
    'contiguous_format': 'MemoryFormat::Contiguous',
    'long': 'at::kLong',
}

# Convert a JIT default into C++ expression representing the default
def default_expr(d: str, t: Type) -> str:
    if d == 'None' and str(t) == 'Tensor?':
        return '{}'
    if isinstance(t, BaseType) and t.name is BaseTy.str:
        # Schema allows single quotes but C++ needs double
        if len(d) >= 2 and d[0] == "'" and d[-1] == "'":
            s = ''
            i = 1
            while i + 1 < len(d):
                if d[i] != '\\':
                    if d[i] == '"':
                        s += '\\"'
                    else:
                        s += d[i]
                    i += 1
                else:
                    if d[i + 1] == "'":
                        s += "'"
                    else:
                        s += d[i:i + 2]
                    i += 2

            return f'"{s}"'

    if isinstance(t, OptionalType):
        if d == 'None':
            return 'c10::nullopt'

        return default_expr(d, t.elem)

    if isinstance(t, ListType):
        if (d.startswith('[') and d.endswith(']')):
            return '{' + d[1:-1] + '}'
        elif t.size is None:
            # NOTE: Sized lists can have scalar defaults
            raise ValueError(f"Expected a list default '[...]' but found: '{d}'")

    return JIT_TO_CPP_DEFAULT.get(d, d)

# Convert an argument into its C++ API form

def argument(
    a: Union[Argument, TensorOptionsArguments, SelfArgument],
    *, cpp_no_default_args: Set[str], method: bool, faithful: bool,
    has_tensor_options: bool
) -> List[Binding]:
    def sub_argument(a: Union[Argument, TensorOptionsArguments, SelfArgument]) -> List[Binding]:
        return argument(
            a, cpp_no_default_args=cpp_no_default_args, method=method, faithful=faithful,
            has_tensor_options=has_tensor_options)

    if isinstance(a, Argument):
        binds: ArgName
        if a.name == "memory_format" and has_tensor_options:
            binds = SpecialArgName.possibly_redundant_memory_format
        else:
            binds = a.name
        default: Optional[str] = None
        if a.name not in cpp_no_default_args and a.default is not None:
            default = default_expr(a.default, a.type)
        return [Binding(
            ctype=argument_type(a, binds=binds),
            name=a.name,
            default=default,
            argument=a,
        )]
    elif isinstance(a, TensorOptionsArguments):
        if faithful:
            return sub_argument(a.dtype) + sub_argument(a.layout) + \
                sub_argument(a.device) + sub_argument(a.pin_memory)
        else:
            default = None
            # Enforced by NativeFunction.__post_init__
            assert 'options' not in cpp_no_default_args
            if all(x.default == "None" for x in a.all()):
                default = '{}'
            elif a.dtype.default == "long":
                default = 'at::kLong'  # TODO: this is wrong
            return [Binding(
                ctype=BaseCType('TensorOptions', 'options'),
                name='options',
                default=default,
                argument=a,
            )]
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
    *, faithful: bool, method: bool, cpp_no_default_args: Set[str]
) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    if faithful:
        args.extend(arguments.non_out)
        args.extend(arguments.out)
    else:
        args.extend(arguments.out)
        args.extend(arguments.non_out)
    return [
        r.no_default() if faithful else r for a in args
        for r in argument(
            a, faithful=faithful, method=method,
            has_tensor_options=arguments.tensor_options is not None,
            cpp_no_default_args=cpp_no_default_args)
    ]
