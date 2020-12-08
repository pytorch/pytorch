from tools.codegen.model import *
from tools.codegen.api.types import *
import tools.codegen.local as local
from typing import Optional, Sequence, Union, List

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
#   - for 'use_c10_dispatcher: full' functions, optional tensors are
#     represented explicitly using c10::optional
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
def valuetype_type(t: Type) -> Optional[str]:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            return None
        elif t.name == BaseTy.int:
            return 'int64_t'
        elif t.name == BaseTy.float:
            return 'double'
        elif t.name == BaseTy.str:
            return 'std::string'
        elif t.name in [BaseTy.bool, BaseTy.QScheme, BaseTy.Scalar,
                        BaseTy.ScalarType, BaseTy.Generator, BaseTy.Storage,
                        BaseTy.Layout, BaseTy.Device, BaseTy.MemoryFormat,
                        BaseTy.Dimname, BaseTy.Stream, BaseTy.ConstQuantizerPtr]:
            # These C++ names line up with their schema names
            return t.name.name
        else:
            raise AssertionError(f"unsupported type: {t}")
    elif isinstance(t, OptionalType):
        elem = valuetype_type(t.elem)
        if elem is None:
            return None
        return f"c10::optional<{elem}>"
    elif isinstance(t, ListType):
        if str(t.elem) == 'bool':
            assert t.size is not None
            return f"std::array<bool,{t.size}>"
        else:
            return None
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")

# Translation of types occuring in JIT arguments to a C++ argument type.
def argumenttype_type(t: Type, *, mutable: bool) -> str:
    # If it's a value type, do the value type translation
    r = valuetype_type(t)
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                return 'Tensor &'
            else:
                return 'const Tensor &'
        else:
            raise AssertionError(f"base type should have been value type {t}")
    elif isinstance(t, OptionalType):
        if str(t.elem) == 'Tensor':
            if mutable:
                return 'Tensor &'  # TODO: fix this discrepancy
            else:
                if local.use_c10_dispatcher().dispatcher_uses_new_style():
                    return 'const c10::optional<Tensor>&'
                else:
                    return 'const Tensor &'
        elem = argumenttype_type(t.elem, mutable=mutable)
        return f"c10::optional<{elem}>"
    elif isinstance(t, ListType):
        # TODO: remove these special cases, ArrayRef fallthrough works fine
        if str(t.elem) == 'int':
            return "IntArrayRef"
        elif str(t.elem) == 'Tensor':
            return "TensorList"
        elif str(t.elem) == 'Dimname':
            return "DimnameList"
        # TODO: do something reasonable about lists of optional tensors
        elif (not local.use_c10_dispatcher().dispatcher_uses_new_style()) and str(t.elem) == 'Tensor?':
            return "TensorList"
        elem = argumenttype_type(t.elem, mutable=mutable)
        # TODO: explicitly qualify namespace here
        return f"ArrayRef<{elem}>"
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")

# Translate a JIT argument into its C++ type
def argument_type(a: Argument) -> str:
    return argumenttype_type(a.type, mutable=a.is_write)

# Translation of a (non-multi) return type from JIT to C++
def returntype_type(t: Type, *, mutable: bool) -> str:
    r = valuetype_type(t)
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                return 'Tensor &'
            else:
                return 'Tensor'
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

def argument_not_this(
    a: Union[Argument, TensorOptionsArguments],
) -> CppArgument:
    if isinstance(a, Argument):
        return CppArgument(
            type=argument_type(a),
            name=a.name,
            default=default_expr(a.default, a.type) if a.default is not None else None,
            argument=a,
        )
    elif isinstance(a, TensorOptionsArguments):
        default = None
        if all(x.default == "None" for x in a.all()):
            default = '{}'
        elif a.dtype.default == "long":
            default = 'at::kLong'  # TODO: this is wrong
        return CppArgument(
            type='const TensorOptions &',
            name='options',
            default=default,
            argument=a,
        )
    else:
        assert_never(a)

def argument(
    a: Union[Argument, TensorOptionsArguments, SelfArgument],
    *,
    method: bool,
) -> Union[CppSingleArgumentPack, CppThisArgumentPack]:
    if isinstance(a, SelfArgument):
        if method:
            return CppThisArgumentPack(argument=a, type=argument_type(a.argument))
        else:
            return CppSingleArgumentPack(argument_not_this(a.argument))
    else:
        return CppSingleArgumentPack(argument_not_this(a))

def argument_faithful(
    a: Union[Argument, TensorOptionsArguments, SelfArgument],
    *,
    method: bool,
) -> CppArgumentPack:
    if isinstance(a, TensorOptionsArguments):
        return CppTensorOptionsArgumentPack(
            argument=a,
            dtype=argument_not_this(a.dtype),
            layout=argument_not_this(a.layout),
            device=argument_not_this(a.device),
            pin_memory=argument_not_this(a.pin_memory),
        )
    else:
        return argument(a, method=method)
