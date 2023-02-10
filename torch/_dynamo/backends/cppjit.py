import ctypes
import logging
import os
import torch
from torch._functorch.aot_autograd import make_boxed_compiler
import torch.fx
import torch.nn._reduction as reduction
import torchgen
import traceback

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from torch._inductor.codecache import CppCodeCache
from torchgen.api.types.signatures import CppSignature, CppSignatureGroup, DispatcherSignature
from torchgen.api.types.types import (
    ArrayRefCType,
    OptionalCType,
    boolT,
    intArrayRefT,
    iOptTensorListRefT,
    iTensorListRefT,
    layoutT,
    longT,
    memoryFormatT,
    optionalIntArrayRefT,
    optionalScalarRefT,
    optionalSymIntArrayRefT,
    optionalTensorRefT,
    scalarT,
    scalarTypeT,
    symIntArrayRefT,
    tensorListT,
    tensorT,
)
from torchgen.api.types.types_base import ArrayCType, BaseCType, CType
from torchgen.gen import (
    ParsedYaml,
    parse_native_yaml,
    DispatchKey
)
from torchgen.model import (
    Argument,
    BackendIndex,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    OptionalType,
    Type
)
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
    Union
)

from .common import aot_autograd
from .registry import register_backend

log = logging.getLogger(__name__)
Signature = Union[CppSignature, DispatcherSignature]

ELEM_TYPE_FOR = {
    intArrayRefT: BaseCType(longT),
    iOptTensorListRefT: OptionalCType(BaseCType(tensorT)),
    iTensorListRefT: BaseCType(tensorT),
    optionalIntArrayRefT: BaseCType(intArrayRefT),
    optionalScalarRefT: ArrayRefCType(BaseCType(scalarT)),
    optionalSymIntArrayRefT: BaseCType(intArrayRefT),
    optionalTensorRefT: BaseCType(tensorListT),
    symIntArrayRefT: BaseCType(longT),
    tensorListT: BaseCType(tensorT),
}

LAYOUT_CPPSTR = {
    torch.strided: f"{layoutT}::Strided",
    torch.sparse: f"{layoutT}::Sparse",
    torch.sparse_csr: f"{layoutT}::SparseCsr",
    # torch.mkldnn: f"{layoutT}::Mkldnn",
    torch.sparse_csc: f"{layoutT}::SparseCsc",
    torch.sparse_bsr: f"{layoutT}::SparseBsr",
    torch.sparse_bsc: f"{layoutT}::SparseBsc",
}

MEMORYFORMAT_CPPSTR = {
    torch.contiguous_format: f"{memoryFormatT}::Contiguous",
    torch.preserve_format: f"{memoryFormatT}::Preserve",
    torch.channels_last: f"{memoryFormatT}::ChannelsLast",
    torch.channels_last_3d: f"{memoryFormatT}::ChannelsLast3d",
}

SCALARTYPE_CPPSTR = {
    torch.uint8: f"{scalarTypeT}::Byte",
    torch.int8: f"{scalarTypeT}::Char",

    torch.int16: f"{scalarTypeT}::Short",
    torch.short: f"{scalarTypeT}::Short",

    torch.int32: f"{scalarTypeT}::Int",
    torch.int: f"{scalarTypeT}::Int",

    torch.int64: f"{scalarTypeT}::Long",
    torch.long: f"{scalarTypeT}::Long",

    torch.float16: f"{scalarTypeT}::Half",
    torch.half: f"{scalarTypeT}::Half",

    torch.float32: f"{scalarTypeT}::Float",
    torch.float: f"{scalarTypeT}::Float",

    torch.float64: f"{scalarTypeT}::Double",
    torch.double: f"{scalarTypeT}::Double",

    torch.complex32: f"{scalarTypeT}::ComplexHalf",
    # torch.chalf: f"{scalarTypeT}::ComplexHalf",

    torch.complex64: f"{scalarTypeT}::ComplexFloat",
    torch.cfloat: f"{scalarTypeT}::ComplexFloat",

    torch.complex128: f"{scalarTypeT}::ComplexDouble",
    torch.cdouble: f"{scalarTypeT}::ComplexDouble",

    torch.bool: f"{scalarTypeT}::Bool",
    torch.qint8: f"{scalarTypeT}::QInt8",
    torch.quint8: f"{scalarTypeT}::QUInt8",
    torch.qint32: f"{scalarTypeT}::QInt32",
    torch.bfloat16: f"{scalarTypeT}::BFloat16",
    torch.quint2x4: f"{scalarTypeT}::QUInt4x2",
    torch.quint4x2: f"{scalarTypeT}::QUInt2x4",
}

ENUM_CPPSTR_DISPATCH = {
    layoutT: LAYOUT_CPPSTR,
    memoryFormatT: MEMORYFORMAT_CPPSTR,
    scalarTypeT: SCALARTYPE_CPPSTR,
}


@dataclass(frozen=True)
class ExceptionGroup(Exception):
    message: str
    exceptions: List[Exception]

    def __str__(self) -> str:
        # First, print the message.
        # Then, print the stacktrace of all the inner exceptions.
        return "\n".join([
            self.message,
            ""  # Empty line
        ] + [
            "".join(traceback.format_exception(type(e), e, e.__traceback__)) for e in self.exceptions
        ])


@dataclass(frozen=True)
class AlignedArg:
    param: Argument
    value: Any
    default: bool = False

    def with_value(self, value: Any) -> "AlignedArg":
        return replace(self, value=value)


@dataclass(frozen=True)
class CppTypedAlignedArg:
    alignedarg: AlignedArg
    cpptype: CType


@dataclass(frozen=True)
class OverloadInfo:
    f: NativeFunction

    @property
    def arguments(self) -> int:
        return len(self.f.func.arguments.flat_all)

    @property
    def default_arguments(self) -> int:
        return sum(arg.default is not None for arg in self.f.func.arguments.flat_all)

    @property
    def needed_arguments(self) -> int:
        return self.arguments - self.default_arguments


@dataclass(frozen=True)
class Kernel(ABC):
    f: NativeFunction

    DISPATCH_KEY_PRIORITY_LIST = [
        DispatchKey.CPU,
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.CompositeImplicitAutograd
    ]

    @classmethod
    def from_function_and_indices(
            cls,
            f: NativeFunction,
            indices: Dict[DispatchKey, BackendIndex]
    ) -> "Kernel":
        for key in cls.DISPATCH_KEY_PRIORITY_LIST:
            index = indices[key]
            if index.has_kernel(f) or f.structured_delegate:
                return DeviceKernel(f, key.name.lower())
        return DispatchKernel(f)

    @abstractmethod
    def namespace(self) -> str: ...

    @abstractmethod
    def sig(self) -> Signature: ...

    @abstractmethod
    def incl(self) -> str: ...

    @abstractmethod
    def name(self) -> str: ...


@dataclass(frozen=True)
class DeviceKernel(Kernel):
    dev: str

    def namespace(self) -> str:
        return f"at::{self.dev}"

    def sig(self) -> Signature:
        return CppSignatureGroup.from_native_function(
            self.f, method=False, fallback_binding=False
        ).most_faithful_signature()

    def incl(self) -> str:
        return f"{self.f.root_name}_{self.dev}_dispatch"

    def name(self) -> str:
        return self.sig().name()


@dataclass(frozen=True)
class DispatchKernel(Kernel):
    def namespace(self) -> str:
        return f"at::_ops::{self.f.func.name.unambiguous_name()}"

    def sig(self) -> Signature:
        return DispatcherSignature.from_schema(self.f.func)

    def incl(self) -> str:
        return f"{self.f.root_name}_ops"

    def name(self) -> str:
        return "call"


def groupby(keyfn, collection) -> Dict:
    groups = defaultdict(list)
    for item in collection:
        groups[keyfn(item)].append(item)
    return groups


def is_reduction_str(s: str) -> bool:
    try:
        reduction.get_enum(s)
        return True
    except:
        return False


def str_to_reduction(s: str) -> int:
    return reduction.get_enum(s)


def native_function_key(f: NativeFunction) -> str:
    return str(f.func.name.name)


def parse_native_functions_yaml() -> ParsedYaml:
    # Torchgen base file.
    torchgen_init = torchgen.__file__
    torchgen_dir = os.path.dirname(torchgen_init)

    # Packaged files directory.
    packaged_dir = os.path.join(torchgen_dir, "packaged", "ATen", "native")

    # Path to YAML files.
    native_functions_yaml_path = os.path.join(packaged_dir, "native_functions.yaml")
    tags_yaml_path = os.path.join(packaged_dir, "tags.yaml")

    return parse_native_yaml(native_functions_yaml_path, tags_yaml_path)


def group_native_functions_overloads(
        native_functions: List[NativeFunction]
) -> Dict[str, List[OverloadInfo]]:
    map_by_name = defaultdict(list)
    for f in native_functions:
        map_by_name[native_function_key(f)].append(OverloadInfo(f))
    return map_by_name


NATIVE_FUNCTIONS, BACKEND_INDICES = parse_native_functions_yaml()
NATIVE_FUNCTIONS_OVERLOAD_MAP = group_native_functions_overloads(NATIVE_FUNCTIONS)


def align_arguments(
        parameters: Sequence[Argument],
        args: Sequence[Any],
        kwargs: Dict[str, Any]
) -> List[AlignedArg]:
    """Aligns the formal parameters with the given arguments.

    Tries to align each formal parameter with its corresponding argument.
    This function may fail if:

        - It didn't find a corresponding positional or keyword argument for
          a parameter without a default value.

        - It found both a positional and keyword argument for the same formal
          parameter.

    Thus, if successfull, this function guarantees that there's at least 1
    positional or keyword argument that corresponds to each formal parameter.
    """
    def align_to_parameter(i: int, param: Argument) -> AlignedArg:
        # The i-th parameter may be found as:
        if i < len(args):
            # - A positional argument.
            #     - Can't have multiple arguments for a parameter.
            if param.name in kwargs:
                raise ValueError(
                    f"positional argument {param.name} also passed as keyword-argument."
                )

            return AlignedArg(param, args[i])

        elif param.name in kwargs:
            # - A keyword argument.
            return AlignedArg(param, kwargs[param.name])
        elif param.default is not None:
            # - Not really found, but its default value is used.
            return AlignedArg(param, param.default, default=True)
        else:
            # Otherwise, it's missing.
            raise ValueError(f"missing {param.type} parameter: {param.name}")

    return [align_to_parameter(i, param) for i, param in enumerate(parameters)]


def type_aligned_arguments(aligned_arguments: Sequence[AlignedArg], sig: Signature) -> List[CppTypedAlignedArg]:
    param_to_bindings = groupby(lambda b: b.argument, sig.arguments())

    for param, bindings in param_to_bindings.items():
        assert len(bindings) == 1, f"unsupported multi-binding parameter {param} with bindings: {bindings}"

    return [
        CppTypedAlignedArg(a, cpptype=param_to_bindings[a.param][0].nctype.type.remove_const_ref())
        for a in aligned_arguments
    ]


def py_to_cppstr(thing: Any, ty: CType) -> str:
    """Parses the a Python value of a given type into a C++ string.
    """
    if isinstance(thing, torch.fx.Node):
        return thing.name

    if ty == BaseCType(boolT):
        return str(thing).lower()

    if isinstance(ty, BaseCType) and ty.type in (layoutT, memoryFormatT, scalarTypeT):
        return ENUM_CPPSTR_DISPATCH[ty.type][thing]

    if isinstance(ty, ArrayCType) and isinstance(ty.elem, BaseCType):
        cpptype = ELEM_TYPE_FOR[ty.type] if isinstance(ty, BaseCType) else ty.elem
        thing_str = ", ".join(py_to_cppstr(x, cpptype) for x in thing)
        return f"std::array<{cpptype.cpp_type()}, {ty.size}>({{{thing_str}}})"

    if (
            (
                isinstance(ty, BaseCType)
                and ty.type in (intArrayRefT, iTensorListRefT, iOptTensorListRefT, symIntArrayRefT, tensorListT)
            )
            or (isinstance(ty, ArrayRefCType) and isinstance(ty.elem, BaseCType))
    ):
        cpptype = ELEM_TYPE_FOR[ty.type] if isinstance(ty, BaseCType) else ty.elem
        thing_str = ", ".join(py_to_cppstr(x, cpptype) for x in thing)
        return f"at::ArrayRef<{cpptype.cpp_type()}>({{{thing_str}}})"

    if (
            (
                isinstance(ty, BaseCType)
                and ty.type in (optionalIntArrayRefT, optionalScalarRefT, optionalSymIntArrayRefT, optionalTensorRefT)
            )
            or isinstance(ty, OptionalCType)
    ):
        if thing is None:
            return "c10::nullopt"
        else:
            elem_ty = ELEM_TYPE_FOR[ty.type] if isinstance(ty, BaseCType) else ty.elem
            return py_to_cppstr(thing, elem_ty)

    if isinstance(ty, BaseCType):
        return str(thing)

    raise ValueError(f"can't create C++ with type {repr(ty)} from object: {thing}")


def str_to_py(thing: str, ty: Type) -> Any:
    """Parses the default string into a Python value.
    """
    if ty in (BaseType(BaseTy.int), BaseType(BaseTy.SymInt)):
        # Special case: at::Reduction.
        # Defer translation to Function.
        if isinstance(thing, str) and is_reduction_str(thing):
            return str_to_reduction(thing)

        # Otherwise, we try to parse it into an int.
        return int(thing)

    elif ty == BaseType(BaseTy.float):
        return float(thing)

    elif ty == BaseType(BaseTy.str):
        if thing[0] == thing[-1] == "'" or thing[0] == thing[-1] == '"':
            return thing[1:-1]

    elif ty == BaseType(BaseTy.bool):
        assert thing == "True" or thing == "False"
        return thing == "True"

    elif ty == BaseType(BaseTy.Scalar):
        for convert in (int, float, complex):
            try:
                return convert(thing)
            except ValueError:
                pass

    elif ty == BaseType(BaseTy.MemoryFormat):
        if thing == "contiguous_format":
            return torch.contiguous_format

    elif isinstance(ty, OptionalType):
        if thing == "None":
            return None
        else:
            return str_to_py(thing, ty.elem)

    elif isinstance(ty, ListType):
        if ty.elem == BaseType(BaseTy.int):
            if len(thing) > 2 and thing[0] == "[" and thing[-1] == "]":
                return [int(x) for x in thing[1:-1].split(",")]
            else:
                return [int(thing)]

    raise ValueError(f"can't build {ty} from str: {thing}")


def check_type(thing: Any, ty: Type) -> bool:
    if ty == BaseType(BaseTy.Tensor):
        return isinstance(thing, torch.Tensor)

    elif ty == BaseType(BaseTy.int):
        # Special case: at::Reduction.
        if isinstance(thing, str):
            return is_reduction_str(thing)
        # Otherwise, we just check if it is an integer.
        return isinstance(thing, int)

    elif ty == BaseType(BaseTy.Dimname):
        return isinstance(thing, str)

    elif ty == BaseType(BaseTy.float):
        return isinstance(thing, float)

    elif ty == BaseType(BaseTy.str):
        return isinstance(thing, str)

    elif ty == BaseType(BaseTy.bool):
        return isinstance(thing, bool)

    elif ty == BaseType(BaseTy.ScalarType):
        return isinstance(thing, torch.dtype)

    elif ty == BaseType(BaseTy.Scalar):
        return (
            check_type(thing, BaseType(BaseTy.int))
            or check_type(thing, BaseType(BaseTy.float))
            or isinstance(thing, complex)
        )

    elif ty == BaseType(BaseTy.MemoryFormat):
        return isinstance(thing, torch.memory_format)

    elif isinstance(ty, OptionalType):
        return (
            thing is None
            or check_type(thing, ty.elem)
        )

    elif isinstance(ty, ListType):
        return (
            isinstance(thing, (list, tuple))
            and all(check_type(x, ty.elem) for x in thing)
        )

    raise ValueError(f"couldn't check instance for type: {ty}")


def check_schema_match(
        func: FunctionSchema,
        args: Sequence[Any],
        kwargs: Dict[str, Any]
) -> None:
    parameters = func.arguments.flat_all

    # Check whether we have enough arguments.
    aligned_arguments = align_arguments(parameters, args, kwargs)

    # Check whether we have too many arguments. Either:
    #   - There are more arguments than formal parameters.
    if len(parameters) < len(args) + len(kwargs):
        raise ValueError(
            "invalid number of parameters. "
            f"Expected {len(parameters)}. Got: {len(args) + len(kwargs)}."
        )

    #   - There are some extra keyword arguments not being used.
    if len(set(kwargs.keys()) - set([param.name for param in parameters])) != 0:
        raise ValueError(
            "unexpected keyword arguments: "
            f"{set(kwargs.keys()) - set([param.name for param in parameters])}"
        )

    # Check whether each parameter type matches the argument type.
    for param, arg in zip(parameters, aligned_arguments):
        # If we are using this parameter default value, we don't have
        # to check for its type.
        if arg.default:
            continue

        if not (isinstance(arg.value, torch.fx.Node) or check_type(arg.value, param.type)):
            raise ValueError(
                f"argument value not instance of {param.type}: {arg.value} ({type(arg.value)})"
            )


def build_debug_target_name(target: Union[str, Callable]) -> str:
    if isinstance(target, str):
        return target
    return f"{target.__module__}.{target.__name__}"


def find_native_function(
        op_name: Union[str, Tuple[str, str]],
        args: Sequence[Any],
        kwargs: Dict[str, Any]
) -> NativeFunction:
    if isinstance(op_name, tuple):
        name, overload = op_name
        for ovl in NATIVE_FUNCTIONS_OVERLOAD_MAP[name]:
            if ovl.f.func.name.overload_name == overload:
                return ovl.f
        raise ValueError(f"could not find operation: {name} (overload: {overload})")

    if op_name not in NATIVE_FUNCTIONS_OVERLOAD_MAP:
        raise ValueError(f"operation not in 'native_functions.yaml': {op_name}")

    exceptions = []

    for ovl in NATIVE_FUNCTIONS_OVERLOAD_MAP[op_name]:
        try:
            check_schema_match(ovl.f.func, args, kwargs)
            return ovl.f
        except Exception as e:
            exceptions.append(e)

    raise ExceptionGroup(f"could not find matching function overload for: {op_name}", exceptions)


def resolve_target_name(node: torch.fx.Node) -> Union[str, Tuple[str, str]]:
    if isinstance(node.target, str):
        return node.target

    module, name = node.target.__module__, node.target.__name__

    if module == "torch":
        return name
    elif module == "torch._ops.aten":
        terms = name.split(".")

        if len(terms) == 2:
            name, overload = terms
            if overload == "default":
                return name, ""
            else:
                return name, overload
        elif len(terms) == 1 and name == "sym_size":
            return "tensor", "size"
        else:
            raise ValueError(f"unsupported operation: {module} (module) {name} (name)")

    elif module == "_operator":
        if name == "truediv":
            return "div"
        elif name == "getitem":
            assert (
                len(node.args) == 2
                and isinstance(node.args[0], (tuple, torch.fx.Node))
                and isinstance(node.args[1], int)
            ), (
                f"unexpected 'getitem' args: {node.args}"
            )
            return "tuple", "getitem"
        else:
            return name
    else:
        raise ValueError(f"couldn't resolve target: {module} -> {name}")


def replace_nodes_by_name(thing: Any) -> Any:
    if isinstance(thing, torch.fx.Node):
        return thing.name
    elif isinstance(thing, (list, tuple)):
        return type(thing)(replace_nodes_by_name(t) for t in thing)
    elif isinstance(thing, dict):
        return {k: replace_nodes_by_name(v) for k, v in thing.items()}
    else:
        return thing


def typed_alignedarg_into_cppstr(a: CppTypedAlignedArg) -> str:
    py_value = a.alignedarg.value \
        if not isinstance(a.alignedarg.value, str) \
        else str_to_py(a.alignedarg.value, a.alignedarg.param.type)
    return py_to_cppstr(py_value, a.cpptype)

@make_boxed_compiler
def cppjit(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    g = gm.graph

    body = []
    input_length = 0
    output_length = 0

    body.append("auto ___gil = PyGILState_Ensure();")

    for node in g.nodes:
        print(node, node.type)
        if node.op == "placeholder":
            body.append(f"const auto& {node.name} = THPVariable_Unpack(inputs[{input_length}]);")
            input_length += 1

        elif node.op in ("call_function", "call_method"):
            is_method = node.op == "call_method"
            op_name = resolve_target_name(node)

            if len(op_name) == 2 and op_name == ("tuple", "getitem"):
                body.append(f"auto {node.name} = std::get<{node.args[1]}>({node.args[0].name});")
                continue
            elif len(op_name) == 2 and op_name == ("tensor", "size"):
                body.append(f"auto {node.name} = {node.args[0].name}.size({node.args[1]});")
                continue

            f = find_native_function(op_name, node.args, node.kwargs)
            kernel = Kernel.from_function_and_indices(f, BACKEND_INDICES)

            log.debug(f"[{node.op}] found function for {build_debug_target_name(node.target)}:")
            log.debug(f"""{" " * 4}{f.func}""")

            aligned_arguments = align_arguments(f.func.arguments.flat_all, node.args, node.kwargs)
            typed_aligned_arguments = type_aligned_arguments(aligned_arguments, kernel.sig())

            self_arg = f.func.arguments.self_arg.argument \
                if f.func.arguments.self_arg is not None \
                else None
            arg_values = [
                typed_alignedarg_into_cppstr(a)
                for a in typed_aligned_arguments
                if not is_method or self_arg != a.alignedarg.param
            ]
            arg_values_str = ", ".join(str(a) for a in arg_values)

            if node.op == "call_function":
                prefix = f"{kernel.namespace()}::{kernel.name()}"
            else:
                assert node.op == "call_method"
                assert self_arg is not None
                self_arg_value = next(filter(lambda a: a.param == self_arg, aligned_arguments))

                assert isinstance(self_arg_value, torch.fx.Node)
                prefix = f"{self_arg_value.name}."

            body.append(f"""const auto& {node.name} = {prefix}({arg_values_str});""")

        elif node.op == "output":
            assert isinstance(node.args[0], (list, tuple)), f"unexpected type: {type(node.args[0])}"
            outputs = list(node.args[0])
            output_length = len(outputs)

            for i, o in enumerate(outputs):
                assert o is None or isinstance(o, torch.fx.Node)
                expr = "Py_None" if o is None else f"THPVariable_Wrap({o.name})"
                body.append(f"outputs[{i}] = {expr};")

        else:
            raise ValueError(f"invalid fx.Node operation: {node.op}")


    body.append("PyGILState_Release(___gil);")
    body.append("return 0;")
    body_str = "\n".join([f"""{" " * 4}{line}""" for line in body])

    cpp_code = f"""
#include <torch/python.h>
#include <ATen/CompositeExplicitAutogradFunctions.h>
#include <ATen/CPUFunctions.h>
#include <ATen/Operators.h>

extern "C" int function(PyObject** inputs, PyObject** outputs) {{
{body_str}
}}
"""
    try:
        lib = CppCodeCache.load(cpp_code)

        def wrapper(*args):
            c_inputs = (ctypes.c_void_p * len(args))(*[id(a) for a in args])
            c_outputs = (ctypes.c_void_p * output_length)()
            lib.function(c_inputs, c_outputs)
            return [ctypes.cast(o, ctypes.py_object).value for o in c_outputs]
        return wrapper
    except:
        log.error(f"failed to compile: {cpp_code}")
        # return gm.forward
        raise

aot_cppjit = aot_autograd(fw_compiler=cppjit)
register_backend(name="cppjit", compiler_fn=aot_cppjit)
