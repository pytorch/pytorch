import ctypes
import logging
import math
import os
import traceback

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torchgen
from torchgen.api.types.signatures import (
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
)
from torchgen.api.types.types import (
    ArrayRefCType,
    boolT,
    deviceT,
    doubleT,
    floatT,
    intArrayRefT,
    iOptTensorListRefT,
    iTensorListRefT,
    layoutT,
    ListCType,
    longT,
    memoryFormatT,
    OptionalCType,
    optionalIntArrayRefT,
    optionalScalarRefT,
    optionalSymIntArrayRefT,
    optionalTensorRefT,
    scalarT,
    scalarTypeT,
    stringT,
    symIntArrayRefT,
    tensorListT,
    tensorT,
)
from torchgen.api.types.types_base import (
    ArrayCType,
    BaseCType,
    ConstRefCType,
    CType,
    TupleCType,
    VectorCType,
)
from torchgen.context import native_function_manager
from torchgen.gen import DispatchKey, parse_native_yaml, ParsedYaml
from torchgen.model import (
    Argument,
    BackendIndex,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    OptionalType,
    Type,
)

import torch
import torch.fx
import torch.nn._reduction as reduction
from torch._functorch.aot_autograd import make_boxed_compiler
from torch._inductor.codecache import CppCodeCache

from .common import aot_autograd
from .registry import register_backend

log = logging.getLogger(__name__)
Signature = Union[CppSignature, DispatcherSignature]

SCALAR_TO_TENSOR_NATIVEFUNCTION = {
    "add.Scalar": ("add", "Tensor"),
    "sub.Scalar": ("sub", "Tensor"),
    "mul.Scalar": ("mul", "Tensor"),
    "div.Scalar": ("div", "Tensor"),
}

PYTYPE_TO_CTYPE = {
    int: BaseCType(longT),
    float: BaseCType(doubleT),
    bool: BaseCType(boolT),
}

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
        return "\n".join(
            [self.message, ""]  # Empty line
            + [
                "".join(traceback.format_exception(type(e), e, e.__traceback__))
                for e in self.exceptions
            ]
        )


@dataclass(frozen=True)
class AlignedArg:
    param: Argument
    value: Any
    default: bool = False

    def with_value(self, value: Any) -> "AlignedArg":
        return replace(self, value=value)


@dataclass(frozen=True)
class CTypedAlignedArg:
    alignedarg: AlignedArg
    ctype: CType


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
        DispatchKey.CompositeImplicitAutograd,
    ]

    @classmethod
    def from_function_and_indices(
        cls, f: NativeFunction, indices: Dict[DispatchKey, BackendIndex]
    ) -> "Kernel":
        for key in cls.DISPATCH_KEY_PRIORITY_LIST:
            index = indices[key]
            if index.has_kernel(f) or f.structured_delegate:
                return DeviceKernel(f, key.name.lower())
        return DispatchKernel(f)

    @abstractmethod
    def namespace(self) -> str:
        ...

    @abstractmethod
    def sig(self) -> Signature:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


@dataclass(frozen=True)
class DeviceKernel(Kernel):
    dev: str

    def namespace(self) -> str:
        return f"at::{self.dev}"

    def sig(self) -> Signature:
        return CppSignatureGroup.from_native_function(
            self.f, method=False, fallback_binding=False
        ).most_faithful_signature()

    def name(self) -> str:
        return self.sig().name()


@dataclass(frozen=True)
class DispatchKernel(Kernel):
    def namespace(self) -> str:
        return f"at::_ops::{self.f.func.name.unambiguous_name()}"

    def sig(self) -> Signature:
        return DispatcherSignature.from_schema(self.f.func)

    def name(self) -> str:
        return "call"


@dataclass(frozen=True)
class NodeInfo:
    type: Union[Type, Tuple[Type, ...]]
    ctype: CType

    def cpp_type(self) -> str:
        return self.ctype.cpp_type()

    def remove_const_ref(self) -> "NodeInfo":
        return NodeInfo(self.type, self.ctype.remove_const_ref())

    def __getitem__(self, index: int) -> "NodeInfo":
        if isinstance(self.type, tuple):
            assert isinstance(self.ctype, TupleCType)
            return NodeInfo(self.type[index], self.ctype.elems[index])

        if self.type == BaseType(BaseTy.Tensor):
            return NodeInfo(self.type, self.ctype)

        assert isinstance(
            self.type, ListType
        ), f"unsupported 'getitem' on type: {self.type}"

        parent_ctype = self.ctype.remove_const_ref()
        if isinstance(parent_ctype, BaseCType) and parent_ctype.type in ELEM_TYPE_FOR:
            ctype = ELEM_TYPE_FOR[parent_ctype.type]
        else:
            assert isinstance(
                parent_ctype, (ArrayCType, ArrayRefCType, ListCType, VectorCType)
            ), f"unsupported 'getitem' on C++ type: {parent_ctype}"
            ctype = parent_ctype.elem

        return NodeInfo(self.type.elem, ctype)

    @staticmethod
    def from_signature(sig: Signature) -> "NodeInfo":
        rtypes = tuple(r.type for r in sig.func.returns)
        return NodeInfo(rtypes if len(rtypes) > 1 else rtypes[0], sig.returns_type())


def groupby(keyfn, collection) -> Dict:
    groups = defaultdict(list)
    for item in collection:
        groups[keyfn(item)].append(item)
    return groups


def is_reduction_str(s: str) -> bool:
    try:
        reduction.get_enum(s.lower())
        return True
    except ValueError:
        return False


def str_to_reduction(s: str) -> int:
    return reduction.get_enum(s.lower())


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
    native_functions: List[NativeFunction],
) -> Dict[str, List[OverloadInfo]]:
    map_by_name = defaultdict(list)
    for f in native_functions:
        map_by_name[native_function_key(f)].append(OverloadInfo(f))
    return map_by_name


NATIVE_FUNCTIONS, BACKEND_INDICES = parse_native_functions_yaml()
NATIVE_FUNCTIONS_OVERLOAD_MAP = group_native_functions_overloads(NATIVE_FUNCTIONS)


def align_arguments(
    parameters: Sequence[Argument], args: Sequence[Any], kwargs: Dict[str, Any]
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


def type_aligned_arguments(
    aligned_arguments: Sequence[AlignedArg], sig: Signature
) -> List[CTypedAlignedArg]:
    """Associates the corresponding parameter CType to each argument.

    This function assumes each argument (FunctionSchema parameter) has
    only one corresponding Binding, which is the source of the CType.
    """
    param_to_bindings = groupby(lambda b: b.argument, sig.arguments())

    for param, bindings in param_to_bindings.items():
        assert (
            len(bindings) == 1
        ), f"unsupported multi-binding parameter {param} with bindings: {bindings}"

    return [
        CTypedAlignedArg(a, ctype=param_to_bindings[a.param][0].nctype.type)
        for a in aligned_arguments
    ]


def py_to_cppstr(thing: Any, ty: CType) -> str:
    """Parses the a Python value of a given type into a C++ string."""
    if isinstance(thing, torch.fx.Node):
        return thing.name

    if (
            ty == BaseCType(boolT)
            or (isinstance(thing, bool) and ty == BaseCType(scalarT))
    ):
        return str(thing).lower()

    if ty == BaseCType(stringT):
        return f'"{thing}"'

    if ty == BaseCType(deviceT):
        return f"""at::Device("{thing}")"""

    if isinstance(ty, BaseCType) and ty.type in (layoutT, memoryFormatT, scalarTypeT):
        return ENUM_CPPSTR_DISPATCH[ty.type][thing]

    if isinstance(ty, (ArrayCType, ListCType)):
        assert isinstance(thing, list)
        thing_str = ", ".join(py_to_cppstr(x, ty.elem) for x in thing)
        return f"{ty.cpp_type()}({{{thing_str}}})"

    if (
        isinstance(ty, BaseCType)
        and ty.type
        in (
            intArrayRefT,
            iTensorListRefT,
            iOptTensorListRefT,
            symIntArrayRefT,
            tensorListT,
        )
    ) or (isinstance(ty, ArrayRefCType) and isinstance(ty.elem, BaseCType)):
        assert isinstance(thing, list)
        cpptype = ELEM_TYPE_FOR[ty.type] if isinstance(ty, BaseCType) else ty.elem
        thing_str = ", ".join(py_to_cppstr(x, cpptype) for x in thing)
        return f"at::ArrayRef<{cpptype.cpp_type()}>({{{thing_str}}})"

    if (
        isinstance(ty, BaseCType)
        and ty.type
        in (
            optionalIntArrayRefT,
            optionalScalarRefT,
            optionalSymIntArrayRefT,
            optionalTensorRefT,
        )
    ) or isinstance(ty, OptionalCType):
        if thing is None:
            return "c10::nullopt"

        elem_ty = ELEM_TYPE_FOR[ty.type] if isinstance(ty, BaseCType) else ty.elem
        return py_to_cppstr(thing, elem_ty)

    if isinstance(ty, BaseCType):
        if ty == BaseCType(tensorT):
            assert isinstance(thing, (bool, int, float)), f"unsupported scalar value: {thing} ({type(thing)})"
            return f"THPVariable_Unpack(THPVariable_Wrap(at::native::wrapped_scalar_tensor({py_to_cppstr(thing, PYTYPE_TO_CTYPE[type(thing)])})))"

        if isinstance(thing, float) and math.isinf(thing):
            assert ty.type in (doubleT, floatT, scalarT), f"unsupported infinity for type: {ty}"
            numeric_type = doubleT if ty.type == scalarT else ty.type
            return f"std::numeric_limits<{numeric_type}>::infinity()"

        return str(thing)

    raise ValueError(f"can't create C++ with type {repr(ty)} from object: {thing}")


def str_to_py(thing: str, ty: Type) -> Any:
    """Parses the default string into a Python value."""
    if ty in (BaseType(BaseTy.int), BaseType(BaseTy.SymInt)):
        # Special case: at::Reduction.
        if isinstance(thing, str) and is_reduction_str(thing.lower()):
            return str_to_reduction(thing.lower())

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
            if thing[0] == "[" and thing[-1] == "]":
                if len(thing) == 2:
                    return []
                else:
                    return [int(x) for x in thing[1:-1].split(",")]
            else:
                return [int(thing)]

    raise ValueError(f"can't build {ty} from str: {thing}")


# Helper class for finding the correct NativeFunction overload.
#
# This class groups the 3 functions used for matching a function
# call to its correct overload.
#
# It does so by taking into consideration:
#   - The number of given arguments
#   - Whether they are keyword arguments or not
#   - Their types
#
# In order for checking their types, this class consults the
# NodeInfo associated with each argument, which holds typing
# information for Node. Otherwise, we check whether the given
# argument is of the expected type.
@dataclass(frozen=True)
class NativeFunctionFinder:
    nodeinfo: Dict[torch.fx.Node, NodeInfo]

    def torch_isinstance(self, thing: Any, ty: Type) -> bool:
        """Checks whether thing is of type ty."""
        CHECK_BASETY_ISINSTANCE_PYTYPE = {
            BaseTy.ScalarType: torch.dtype,
            BaseTy.Tensor: torch.Tensor,
            BaseTy.Dimname: str,
            BaseTy.float: float,
            BaseTy.str: str,
            BaseTy.bool: bool,
            BaseTy.Layout: torch.layout,
            BaseTy.Device: torch.device,
            BaseTy.MemoryFormat: torch.memory_format,
        }

        if isinstance(thing, torch.fx.Node):
            if thing not in self.nodeinfo:
                return True
            return self.nodeinfo[thing].type == ty or (
                ty == OptionalType(BaseType(BaseTy.Tensor))
                and self.nodeinfo[thing].type == BaseType(BaseTy.Tensor)
            )

        if isinstance(ty, BaseType) and ty.name in CHECK_BASETY_ISINSTANCE_PYTYPE:
            return isinstance(thing, CHECK_BASETY_ISINSTANCE_PYTYPE[ty.name])

        elif ty in (BaseType(BaseTy.int), BaseType(BaseTy.SymInt)):
            # Special case: at::Reduction.
            if isinstance(thing, str):
                return is_reduction_str(thing)
            # Otherwise, we just check if it is an integer.
            return isinstance(thing, int)

        elif ty == BaseType(BaseTy.Scalar):
            return (
                self.torch_isinstance(thing, BaseType(BaseTy.int))
                or self.torch_isinstance(thing, BaseType(BaseTy.float))
                or isinstance(thing, complex)
            )

        elif isinstance(ty, OptionalType):
            return thing is None or self.torch_isinstance(thing, ty.elem)

        elif isinstance(ty, ListType):
            return isinstance(thing, (list, tuple)) and all(
                self.torch_isinstance(x, ty.elem) for x in thing
            )

        raise ValueError(f"couldn't check instance for type: {ty}")

    def check_schema_match(
        self, func: FunctionSchema, args: Sequence[Any], kwargs: Dict[str, Any]
    ) -> None:
        """Checks whether the FunctionSchema matches the arguments and
        keyword-arguments combination.

        On matching failure, raises an Exception with the corresponding reason.
        """
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

            if not self.torch_isinstance(arg.value, param.type):
                arg_type = (
                    self.nodeinfo[arg.value].type
                    if arg.value in self.nodeinfo
                    else type(arg.value)
                )
                raise ValueError(
                    f"argument value not instance of {param.type}: {arg.value} ({arg_type})"
                )

    def find(
        self,
        op_name: Union[str, Tuple[str, str]],
        args: Sequence[Any],
        kwargs: Dict[str, Any],
    ) -> NativeFunction:
        """Looks for a matching overload for op_name.

        If this function fails to find a matching overload for the given op_name,
        arguments, and keyword-arguments, it raises an ExceptionGroup, which contains
        the reasons why it failed matching with each overload.
        """
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
                self.check_schema_match(ovl.f.func, args, kwargs)
                return ovl.f
            except Exception as e:
                exceptions.append(e)

        raise ExceptionGroup(
            f"could not find matching function overload for: {op_name}", exceptions
        )


def build_debug_target_name(target: Union[str, Callable]) -> str:
    if isinstance(target, str):
        return target
    return f"{target.__module__}.{target.__name__}"


def resolve_target_name(node: torch.fx.Node) -> Union[str, Tuple[str, str]]:
    """Computes what is the function identifier a node should execute.

    - Removes the previously recorded overload
    - Identifies special functions (e.g. tensor.size(), tuple.__getitem__)
    """
    if isinstance(node.target, str):
        return node.target

    module, name = node.target.__module__, node.target.__name__

    if module == "torch":
        return name
    elif module == "torch._ops.aten":
        terms = name.split(".")

        if len(terms) == 2:
            return terms[0]
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
                and isinstance(node.args[0], torch.fx.Node)
                and isinstance(node.args[1], int)
            ), f"unexpected 'getitem' args on: {type(node.args[0])}"
            return "collection", "getitem"
        else:
            return name
    else:
        raise ValueError(f"couldn't resolve target: {module} -> {name}")


def typed_alignedarg_into_cppstr(a: CTypedAlignedArg, info: Optional[NodeInfo]) -> str:
    """Generates code for a given argument in a function call.

    In summary, this function transforms the argument into a Python value
    (if it is a string representing the default value), and then transforms
    that into a valid C++ string.

    In the end, we check whether a const_cast is needed. There are a few cases
    where the function expects a "T&", but the argument is of type "const T&".
    """
    py_value = (
        a.alignedarg.value
        if not isinstance(a.alignedarg.value, str)
        else str_to_py(a.alignedarg.value, a.alignedarg.param.type)
    )
    cppstr = py_to_cppstr(py_value, a.ctype.remove_const_ref())

    if info is not None and info.ctype != a.ctype:
        # When the actual type and the parameter type are different, we might end
        # up in a situation where we are trying to make 'const Tensor&' into 'Tensor&'.
        # That is clearly undesirable, but may happen (there are still some functions
        # that use 'Tensor&' as parameter).

        if a.ctype.remove_const_ref() == info.ctype.remove_const_ref() and isinstance(
            info.ctype, ConstRefCType
        ):
            assert info.ctype.elem == BaseCType(tensorT), (
                "the only allowed 'const T&' to 'T&' conversion is when T = Tensor. "
                f"Got: {info.ctype.elem.cpp_type()}"
            )
            return f"const_cast<{a.ctype.cpp_type()}>({cppstr})"

    return cppstr


def getitem(node: torch.fx.Node, ctype: CType) -> str:
    """Emits code for the __getitem__ operation."""
    collection = node.args[0].name  # type: ignore[attr-defined]
    index = node.args[1]

    if isinstance(ctype, VectorCType):
        return f"auto {node.name} = {collection}[{index}];"
    elif isinstance(ctype, TupleCType):
        return f"auto {node.name} = std::get<{index}>({collection});"

    raise ValueError(f"unsupported 'getitem' type: {ctype}")


def indent(body: List[str], size: int = 4) -> List[str]:
    return [f"""{" " * size}{line}""" for line in body]


@make_boxed_compiler
def cppjit(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    """Generates C++ code, skipping the dispatcher, for each node."""
    g = gm.graph

    body = []
    input_length = 0
    output_length = 0
    nodeinfo = {}

    body.append("at::AutoDispatchBelowADInplaceOrView __guard;")
    body.append("py::gil_scoped_acquire guard;")

    for node in g.nodes:
        if node.op == "placeholder":
            nodeinfo[node] = NodeInfo(
                BaseType(BaseTy.Tensor), ConstRefCType(BaseCType(tensorT))
            )
            body.append(
                f"{nodeinfo[node].cpp_type()} {node.name} = THPVariable_Unpack(inputs[{input_length}]);"
            )
            input_length += 1

        elif node.op == "get_attr":
            nodeinfo[node] = NodeInfo(
                BaseType(BaseTy.Tensor), ConstRefCType(BaseCType(tensorT))
            )
            objptr = f"{node.name}_obj"

            body.append(f"PyObject* {objptr};")
            body.append("{")
            body.extend(indent([
                f'{objptr} = PyObject_GetAttrString(self_obj, "{node.target}");'
            ]))
            body.append("}")
            body.append(
                f"{nodeinfo[node].cpp_type()} {node.name} = THPVariable_Unpack({objptr});"
            )

        elif node.op in ("call_function", "call_method"):
            is_method = node.op == "call_method"

            # Function call code generation has 7 steps:

            # 1. Compute the actual function to be called
            op_name = resolve_target_name(node)

            # 2. Check for special case functions
            #    Generate code for each of them accordingly
            if len(op_name) == 2 and op_name == ("collection", "getitem"):
                # Generates code for 'getitem', depending on node.args[0] type.
                #   - std::vector => node[index]
                #   - std::tuple  => std::get<index>(node)
                args_0_info = nodeinfo[node.args[0]]
                body.append(getitem(node, args_0_info.ctype.remove_const_ref()))
                nodeinfo[node] = args_0_info[node.args[1]]
                continue
            elif len(op_name) == 2 and op_name == ("tensor", "size"):
                body.append(
                    f"auto {node.name} = {node.args[0].name}.size({node.args[1]});"
                )
                nodeinfo[node] = NodeInfo(BaseType(BaseTy.int), BaseCType(longT))
                continue

            # 3. Find the matching NativeFunction.
            finder = NativeFunctionFinder(nodeinfo)
            f = finder.find(op_name, node.args, node.kwargs)

            if str(f.func.name) in SCALAR_TO_TENSOR_NATIVEFUNCTION:
                f = finder.find(SCALAR_TO_TENSOR_NATIVEFUNCTION[str(f.func.name)], node.args, node.kwargs)

            with native_function_manager(f):
                kernel = Kernel.from_function_and_indices(f, BACKEND_INDICES)

                log.debug(
                    f"[{node.op}] found function for {build_debug_target_name(node.target)}:"
                )
                log.debug(f"""{" " * 4}{f.func}""")

                # 4. Align the given arguments with the function parameters
                aligned_arguments = align_arguments(
                    f.func.arguments.flat_all, node.args, node.kwargs
                )
                # 5. Match each argument with its desired C++ type
                #    The types depend on the Signature used (e.g. CppSignature, DispatcherSignature)
                typed_aligned_arguments = type_aligned_arguments(
                    aligned_arguments, kernel.sig()
                )

                # 6. Retrieve the self argument, and generate code for the rest.
                self_arg = (
                    f.func.arguments.self_arg.argument
                    if f.func.arguments.self_arg is not None
                    else None
                )
                arg_values = [
                    typed_alignedarg_into_cppstr(
                        a, nodeinfo.get(a.alignedarg.value, None)
                    )
                    for a in typed_aligned_arguments
                    if not is_method or self_arg != a.alignedarg.param
                ]
                arg_values_str = ", ".join(str(a) for a in arg_values)

                # 7. Actually generate the function call.
                if node.op == "call_function":
                    prefix = f"{kernel.namespace()}::{kernel.name()}"
                else:
                    assert node.op == "call_method"
                    assert self_arg is not None
                    self_arg_value = next(
                        filter(lambda a: a.param == self_arg, aligned_arguments)
                    )

                    assert isinstance(self_arg_value, torch.fx.Node)
                    prefix = f"{self_arg_value.name}."

                nodeinfo[node] = NodeInfo.from_signature(kernel.sig())
                body.append(
                    f"""{nodeinfo[node].cpp_type()} {node.name} = {prefix}({arg_values_str});"""
                )

        elif node.op == "output":
            assert isinstance(
                node.args[0], (list, tuple)
            ), f"unexpected type: {type(node.args[0])}"
            outputs = list(node.args[0])
            output_length = len(outputs)

            inner_body = []
            for i, o in enumerate(outputs):
                assert o is None or isinstance(o, torch.fx.Node)
                expr = "Py_None" if o is None else f"THPVariable_Wrap({o.name})"
                inner_body.append(f"outputs[{i}] = {expr};")

            body.append("{")
            body.extend(indent(inner_body))
            body.append("}")

        else:
            raise ValueError(f"invalid fx.Node operation: {node.op}")

    body.append("return 0;")
    body_str = "\n".join(indent(body))

    cpp_code = f"""
#include <torch/python.h>
#include <ATen/CompositeExplicitAutogradFunctions.h>
#include <ATen/CPUFunctions.h>
#include <ATen/Operators.h>
#include <limits>

extern "C" int __cppjit_function(PyObject* self_obj, PyObject** inputs, PyObject** outputs) noexcept {{
{body_str}
}}
"""

    try:
        lib = CppCodeCache.load(cpp_code)

        def wrapper(*args):
            c_self = ctypes.c_void_p(id(gm))
            c_inputs = (ctypes.c_void_p * len(args))(*[id(a) for a in args])
            c_outputs = (ctypes.c_void_p * output_length)()
            lib.__cppjit_function(c_self, c_inputs, c_outputs)
            return [ctypes.cast(o, ctypes.py_object).value for o in c_outputs]

        return wrapper
    except Exception:
        log.error(f"failed to compile: {cpp_code}")
        raise


aot_cppjit = aot_autograd(fw_compiler=cppjit)
register_backend(name="aot_cppjit", compiler_fn=aot_cppjit)  # type: ignore[arg-type]
