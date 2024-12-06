# mypy: allow-untyped-defs
import contextlib
import dataclasses
import enum
import functools
import itertools
import logging
import math
import operator
import re
from enum import auto, Enum
from itertools import chain
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import sympy

import torch
import torch.fx
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.printers import PythonPrinter as _PythonPrinter
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT
from torch.utils._sympy.value_ranges import bound_sympy, ValueRangeAnalysis, ValueRanges

from .. import config, metrics
from ..utils import (
    boolean_ops,
    DeferredLineBase,
    generate_assert,
    IndentedBuffer,
    ir_dataclass,
    sympy_dot,
    sympy_subs,
    unique,
)
from ..virtualized import ops, OpsHandler, OpsValue, ReductionType, StoreMode, V


schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
log = logging.getLogger(__name__)


def data_type_logger(msg):
    if schedule_log.isEnabledFor(logging.DEBUG):
        schedule_log.debug("Data type propagation: %s", msg)


class WorkspaceZeroMode(enum.Enum):
    UNINITIALIZED = 0
    ZERO_ON_CALL = 1  # kernel may leave workspace dirty
    ZERO_PER_GRAPH = 2  # must be re-zeroed by kernel

    @staticmethod
    def combine(a, b):
        if a == b or b == WorkspaceZeroMode.UNINITIALIZED:
            return a
        if a == WorkspaceZeroMode.UNINITIALIZED:
            return b
        raise NotImplementedError(f"WorkspaceZeroMode.combine({a!r}, {b!r})")

    @staticmethod
    def from_bool(zero_fill):
        if zero_fill:
            return WorkspaceZeroMode.ZERO_ON_CALL
        return WorkspaceZeroMode.UNINITIALIZED


@ir_dataclass(frozen=True)
class WorkspaceArg:
    """A temporary buffer used for a single kernel, then discarded.

    Not registered as a traditional buffer since there are no users,
    so it would be dead code eliminated.

    Args:
        nbytes: The size of the buffer in bytes.
        zero_fill: Whether the buffer should be initialized to zero.

    """

    count: sympy.Expr
    zero_mode: WorkspaceZeroMode
    device: torch.device
    outer_name: str
    inner_name: str = "ws_ptr"
    dtype: torch.dtype = torch.uint8

    @staticmethod
    def unique_name(prefix="workspace_"):
        return f"{prefix}{next(V.graph.workspace_id)}"

    @staticmethod
    def can_join(a, b) -> bool:
        return (
            a.inner_name == b.inner_name and a.dtype == b.dtype and a.device == b.device
        )

    @staticmethod
    def join(a, b):
        return WorkspaceArg(
            count=a.count + b.count,
            zero_mode=WorkspaceZeroMode.combine(a.zero_mode, b.zero_mode),
            dtype=a.dtype,
            device=a.device,
            inner_name=a.inner_name,
            outer_name=a.outer_name,
        )

    @staticmethod
    def maximum(a, b):
        assert (
            a.dtype == b.dtype and a.device == b.device and a.inner_name == b.inner_name
        )
        return WorkspaceArg(
            count=sympy.Max(a.count, b.count),
            zero_mode=WorkspaceZeroMode.combine(a.zero_mode, b.zero_mode),
            dtype=a.dtype,
            device=a.device,
            inner_name=a.inner_name,
            outer_name=a.outer_name,
        )

    # These methods let WorkspaceArg pretend it is a buffer to reuse allocation code
    def get_device(self):
        return self.device

    get_device_or_error = get_device

    def get_dtype(self):
        return self.dtype

    def get_layout(self):
        from ..ir import FixedLayout

        return FixedLayout(
            device=self.device,
            dtype=self.dtype,
            size=[self.count],
            stride=[1],
        )

    @property
    def layout(self):
        return self.get_layout()

    get_output_spec = get_layout
    maybe_get_output_spec = get_layout
    maybe_get_layout = get_layout

    def get_size(self):
        return [self.count]

    def get_stride(self):
        return [1]

    def get_name(self):
        return self.outer_name

    def get_inputs_that_alias_output(self):
        return []


@dataclasses.dataclass
class TensorArg:
    name: str
    buffer: str
    dtype: torch.dtype
    offset: sympy.Expr = sympy.S.Zero  # c++ only
    alias_of: Optional[str] = None  # halide only


@dataclasses.dataclass
class SizeArg:
    name: str
    expr: sympy.Expr

    @property
    def alias_of(self):
        return None


@dataclasses.dataclass
class TMADescriptorArg:
    name: str


@dataclasses.dataclass
class DeviceCodegen:
    scheduling: Any
    wrapper_codegen: type
    cpp_wrapper_codegen: type = type(None)


KernelArgType = Union[WorkspaceArg, TensorArg, SizeArg, TMADescriptorArg]

device_codegens: Dict[str, DeviceCodegen] = {}


class DeviceOpOverrides:
    def import_get_raw_stream_as(self, name):
        raise NotImplementedError

    def set_device(self, device_idx):
        raise NotImplementedError

    def synchronize(self):
        raise NotImplementedError

    def device_guard(self, device_idx):
        raise NotImplementedError

    def cpp_device_guard(self):
        raise NotImplementedError

    def cpp_aoti_device_guard(self):
        raise NotImplementedError

    def cpp_stream_guard(self):
        raise NotImplementedError

    def cpp_aoti_stream_guard(self):
        raise NotImplementedError

    def cpp_getStreamFromExternal(self):
        raise NotImplementedError

    def kernel_header(self):
        raise NotImplementedError

    def kernel_driver(self):
        raise NotImplementedError

    def abi_compatible_header(self):
        raise NotImplementedError

    def cpp_stream_type(self):
        raise NotImplementedError

    def aoti_get_stream(self):
        raise NotImplementedError

    def cpp_kernel_type(self):
        raise NotImplementedError

    def cpp_device_ptr(self):
        raise NotImplementedError

    def tma_descriptor_helpers(self):
        raise NotImplementedError


device_op_overrides_dict: Dict[str, DeviceOpOverrides] = {}


# The code generated by Inductor consists of two main parts: kernel code and wrapper code.
# For any new backend looking to integrate with Inductor, customization of these two main
# parts are necessary to generate its specific code.
#
# Kernel code generation is determined by different Scheduling. Consequently, a new
# backend needs to provide a custom Scheduling for its unique kernel code generation. Currently,
# CppScheduling and TritonScheduling serve the C++/OpenMP and Triton backends, respectively.
#
# For the Wrapper, Inductor provides a PythonWrapperCodegen class to generate the Python wrapper code
# that bridges kernels. This allows out-of-tree backends to inherit from PythonWrapperCodegen,
# and override specific member functions to create backend-specific Python wrapper code.
#
# Other classes, such as CppKernel and TritonKernel, used for code generation, typically form part
# of the logic for either Scheduling or PythonWrapperCodegen. So the Scheduling and PythonWrapperCodegen interfaces
# provide flexibility to the backend. A backend can choose to implement these classes from scratch,
# or reuse them by extending and overriding as necessary. And Inductor provides the registration API,
# register_backend_for_device, to equip a new backend at runtime.
#
# Intel has developed a new backend on top of Triton to support Intel GPUs, leveraging these interfaces.
# This backend can be used as a reference:
# https://github.com/intel/intel-extension-for-pytorch/blob/5dcc9d57e5422cf295e1a1ee97896d6b6a554a85/intel_extension_for_pytorch/_inductor/__init__.py#L9
def register_backend_for_device(
    device: str,
    device_scheduling: Any,
    device_wrapper_codegen: type,
    device_cpp_wrapper_codegen: type = type(None),
):
    device_codegens[device] = DeviceCodegen(
        device_scheduling, device_wrapper_codegen, device_cpp_wrapper_codegen
    )


class BackendFeature(Enum):
    FOREACH = auto()
    BUCKETIZE = auto()
    INPLACE_BUFFERS = auto()
    MASKED_SCATTER_WITH_INDEX = auto()
    SCAN = auto()
    SORT = auto()
    TUPLE_REDUCTION = auto()
    PREFER_STORE_LOOP_ORDER = auto()
    TRITON_TEMPLATES = auto()
    REDUCE_TO_SINGLE_ELEMENT = auto()


def get_backend_features(device: Union[torch.device, str, None]):
    if device is None:
        return {}
    init_backend_registration()
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        assert isinstance(device, str)
        device_type = device
        device = torch.device(device_type)
    scheduling = get_scheduling_for_device(device_type)
    return scheduling(None).get_backend_features(device)


def has_backend_feature(device, feature):
    """See also V.graph.has_feature"""
    assert isinstance(feature, BackendFeature)
    return feature in get_backend_features(device)


def get_scheduling_for_device(device: str):
    return device_codegens[device].scheduling if device in device_codegens else None


def get_wrapper_codegen_for_device(device: str, cpp_wrapper: bool = False):
    if device in device_codegens:
        wrapper_codegen_obj: DeviceCodegen = device_codegens[device]
        return (
            wrapper_codegen_obj.cpp_wrapper_codegen
            if cpp_wrapper
            else wrapper_codegen_obj.wrapper_codegen
        )
    return None


@functools.lru_cache(None)
def init_backend_registration():
    from .cpp import CppScheduling
    from .cpp_wrapper_cpu import CppWrapperCpu
    from .cpp_wrapper_cpu_array_ref import CppWrapperCpuArrayRef
    from .cpp_wrapper_gpu import CppWrapperGpu
    from .cuda_combined_scheduling import CUDACombinedScheduling
    from .halide import HalideScheduling
    from .triton import TritonScheduling
    from .wrapper import PythonWrapperCodegen

    if get_scheduling_for_device("cpu") is None:
        cpu_backends = {
            "cpp": CppScheduling,
            "halide": HalideScheduling,
            "triton": TritonScheduling,
        }
        register_backend_for_device(
            "cpu",
            lambda *args, **kwargs: cpu_backends[config.cpu_backend](*args, **kwargs),
            PythonWrapperCodegen,
            CppWrapperCpuArrayRef
            if config.aot_inductor.allow_stack_allocation
            else CppWrapperCpu,
        )

    if get_scheduling_for_device("cuda") is None:
        # CUDACombinedScheduling combines Triton and CUDA C++ scheduling for CUDA devices via delegation
        cuda_backends = {"triton": CUDACombinedScheduling, "halide": HalideScheduling}
        register_backend_for_device(
            "cuda",
            lambda *args, **kwargs: cuda_backends[config.cuda_backend](*args, **kwargs),
            PythonWrapperCodegen,
            CppWrapperGpu,
        )

    if get_scheduling_for_device("xpu") is None:
        register_backend_for_device(
            "xpu",
            TritonScheduling,
            PythonWrapperCodegen,
            CppWrapperGpu,
        )

    private_backend = torch._C._get_privateuse1_backend_name()
    if (
        private_backend != "privateuseone"
        and get_scheduling_for_device(private_backend) is None
    ):
        from torch.utils.backend_registration import _get_custom_mod_func

        try:
            device_scheduling = _get_custom_mod_func("Scheduling")
            wrapper_codegen = _get_custom_mod_func("PythonWrapperCodegen")
            cpp_wrapper_codegen = _get_custom_mod_func("CppWrapperCodegen")
            if device_scheduling and wrapper_codegen and cpp_wrapper_codegen:
                register_backend_for_device(
                    private_backend,
                    device_scheduling,
                    wrapper_codegen,
                    cpp_wrapper_codegen,
                )
        except RuntimeError:
            pass


def index_prevent_reordering(index: List[sympy.Expr], index_vars, sizes):
    from ..ir import FlexibleLayout

    # added contiguous index prevents reordering
    return [*index, sympy_dot(index_vars, FlexibleLayout.contiguous_strides(sizes))]


def register_device_op_overrides(device: str, device_op_overrides: DeviceOpOverrides):
    device_op_overrides_dict[device] = device_op_overrides


def get_device_op_overrides(device: str):
    assert isinstance(device, str)

    if not device_op_overrides_dict.keys():
        from . import cpu_device_op_overrides  # noqa: F401
        from .cuda import device_op_overrides  # noqa: F401
        from .xpu import device_op_overrides as xpu_op_overrides  # noqa: F401

    if device in device_op_overrides_dict.keys():
        return device_op_overrides_dict[device]


DTYPE_TO_COMPUTATION_DTYPE = {
    torch.bfloat16: torch.float,
    torch.float16: torch.float,
    **{
        dtype: dtype
        for dtype in [
            torch.bool,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        ]
    },
}


def deduce_output_dtype_by_name(
    op_name: str,
    *args,
    **kwargs,
) -> Optional[torch.dtype]:
    """
    Given op name and a list of input dtypes, deduce the output dtype
    """
    if op_name in boolean_ops():
        return torch.bool
    elif op_name in (
        "to_dtype",
        "index_expr",
    ):
        return kwargs["dtype"] if "dtype" in kwargs else args[-1]
    elif op_name in (
        "rand",
        "randn",
    ):
        return torch.float
    elif op_name in (
        "get_index",
        "randint64",
        "load_seed",
    ):
        return torch.int64
    elif op_name == "reduction":
        return kwargs["dtype"] if "dtype" in kwargs else args[1]
    elif op_name == "constant":
        dtype = kwargs["dtype"] if "dtype" in kwargs else args[-1]
        return DTYPE_TO_COMPUTATION_DTYPE[dtype]  # type: ignore[index]
    elif op_name in (
        "load",
        "store",
        "store_reduction",
    ):
        buf_name = args[1]
        return V.graph.get_dtype(buf_name)  # type: ignore[arg-type]
    elif op_name == "to_dtype_bitcast":
        return kwargs["dtype"] if "dtype" in kwargs else args[-2]
    return None


class DataTypePropagation:
    def __init__(self, body) -> None:
        self.body = body
        self.graphs: Dict[Union[Callable[..., Any], str], Any] = {
            "root": body.root_block.graph
        }
        for k, v in body.subblocks.items():
            self.graphs[k] = v.graph

    def deduce_node_dtype_by_inputs(self, node: torch.fx.Node):
        inputs = node.all_input_nodes
        input_nodes = [
            n for n in inputs if isinstance(n, torch.fx.Node) and n.op != "placeholder"
        ]
        if len(input_nodes) == 0:
            return None

        all_input_nodes_propagated = all(
            OptimizationContext.key in n.meta
            and n.meta[OptimizationContext.key].dtype is not None
            for n in input_nodes
        )
        if not all_input_nodes_propagated:
            return None

        return functools.reduce(
            torch.promote_types,
            [n.meta[OptimizationContext.key].dtype for n in input_nodes],
        )

    def deduce_node_dtype_by_subgraph(self, node: torch.fx.Node):
        sub_graph = self.graphs[node.target]
        dtype = self.propagate_graph(sub_graph)
        assert dtype
        return dtype

    def deduce_node_dtype(self, node: torch.fx.Node):
        if node.op == "placeholder":
            return None

        if node.target == "output" and len(node.args) != 1:
            # we can infer output node if it only have 1 arg
            return None

        if node.target == operator.getitem:
            return self.deduce_node_dtype(node.args[0])  # type: ignore[arg-type]

        assert isinstance(node.target, str)

        if node.target.startswith("masked_subblock"):
            return self.deduce_node_dtype_by_subgraph(node)

        if (
            output_dtype := deduce_output_dtype_by_name(
                node.target,
                *node.args,
                **node.kwargs,
            )
        ) is not None:
            return output_dtype

        return self.deduce_node_dtype_by_inputs(node)

    def propagate_graph(self, graph: torch.fx.Graph):
        assert graph.nodes
        graph_dtype = None
        # For masked_subblock, we use output's dtype to represent
        # the dtype of this subgraph. For other cases, graph_dtype
        # might be None
        for node in graph.nodes:
            if OptimizationContext.key in node.meta:
                opt_ctx = node.meta[OptimizationContext.key]
            else:
                opt_ctx = OptimizationContext()

            opt_ctx.dtype = self.deduce_node_dtype(node)
            node.meta[OptimizationContext.key] = opt_ctx
            if node.target == "output":
                graph_dtype = opt_ctx.dtype
        return graph_dtype

    def propagate(self):
        self.propagate_graph(self.graphs["root"])

    @classmethod
    def propagate_loopbody(cls, body):
        return cls(body).propagate()

    @classmethod
    def propagate_scheduler_node(cls, node):
        from ..loop_body import LoopBody
        from ..scheduler import SchedulerNode

        assert isinstance(node, SchedulerNode)
        assert isinstance(node._body, LoopBody)
        DataTypePropagation.propagate_loopbody(node._body)


class PythonPrinter(_PythonPrinter):
    def doprint(self, expr, *, simplify: bool = True, p=True):
        # TODO: why are people passing strings to the printer here :think:
        if simplify and isinstance(expr, sympy.Expr) and hasattr(V.graph, "sizevars"):
            expr = V.graph.sizevars.simplify(expr)
        return super().doprint(expr)


class OpOverrides:
    def __init__(self, parent):
        super().__init__()
        self._parent = parent

    @staticmethod
    def paren(string: str) -> str:
        def all_in_parens(string: str) -> bool:
            if string[0] != "(" or len(string) < 2:
                return False
            count = 1
            for i, char in enumerate(string[1:]):
                if char == "(":
                    count += 1
                elif char == ")":
                    count -= 1
                if count == 0 and i != len(string) - 2:
                    return False
            assert count == 0
            return True

        if (
            isinstance(string, CSEVariable)
            or re.match(r"^[a-z0-9_.]+$", string, re.IGNORECASE)
            or re.match(r"^\([^)]*\)$", string, re.IGNORECASE)
            or string == ""
        ):
            return string
        # don't put extra parens for strings that are already wrapped in parens
        if all_in_parens(string):
            return string
        return f"({string})"

    def __getattr__(self, item):
        return getattr(self._parent, item)

    @staticmethod
    def identity(value):
        # used to trigger cse
        return value

    @staticmethod
    def constant(value, dtype):
        return repr(value)

    @staticmethod
    def reciprocal(x):
        return ops.truediv(ops.constant(1, torch.int32), x)

    @staticmethod
    def square(x):
        return ops.mul(x, x)

    @staticmethod
    def erfc(x):
        return ops.sub(ops.constant(1, torch.float32), ops.erf(x))

    @staticmethod
    def erfcx(x):
        return ops.mul(ops.exp(ops.square(x)), ops.erfc(x))

    @staticmethod
    def expm1(x):
        return ops.sub(ops.exp(x), ops.constant(1, torch.float32))

    @staticmethod
    def log10(x):
        return ops.mul(ops.log(x), ops.constant(1 / math.log(10), torch.float32))

    @staticmethod
    def log2(x):
        return ops.mul(ops.log(x), ops.constant(1 / math.log(2), torch.float32))

    @staticmethod
    def exp2(x):
        return ops.exp(ops.mul(x, ops.constant(math.log(2), torch.float32)))

    @staticmethod
    def log1p(x):
        return ops.log(ops.add(x, ops.constant(1, torch.int32)))

    @staticmethod
    def sigmoid(x):
        one = ops.constant(1, torch.int32)
        return ops.truediv(one, ops.add(one, ops.exp(ops.neg(x))))

    @staticmethod
    def libdevice_sigmoid(x):
        one = ops.constant(1, torch.int32)
        return ops.truediv(one, ops.add(one, ops.libdevice_exp(ops.neg(x))))

    @staticmethod
    def relu(x):
        return ops.maximum(x, ops.constant(0, torch.int32))

    @staticmethod
    def libdevice_abs(x):
        return ops.abs(x)

    @staticmethod
    def libdevice_sqrt(x):
        return ops.sqrt(x)

    @staticmethod
    def libdevice_cos(x):
        return ops.cos(x)

    @staticmethod
    def libdevice_sin(x):
        return ops.sin(x)

    @staticmethod
    def libdevice_log(x):
        return ops.log(x)

    @staticmethod
    def libdevice_exp(x):
        return ops.exp(x)

    @staticmethod
    def bitwise_not(x):
        return f"~{OpOverrides.paren(x)}"

    @staticmethod
    def logical_not(a):
        return f"{OpOverrides.paren(a)} == 0"

    @staticmethod
    def bitwise_and(x, y):
        return f"{OpOverrides.paren(x)} & {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_or(x, y):
        return f"{OpOverrides.paren(x)} | {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_xor(x, y):
        return f"{OpOverrides.paren(x)} ^ {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_left_shift(x, y):
        return f"{OpOverrides.paren(x)} << {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_right_shift(x, y):
        return f"{OpOverrides.paren(x)} >> {OpOverrides.paren(y)}"

    @staticmethod
    def remainder(a, b):
        r = ops.mod(a, b)
        cond = ops.and_(
            ops.ne(r, ops.constant(0, torch.int32)),
            ops.ne(ops.signbit(r), ops.signbit(b)),
        )
        return ops.where(cond, ops.add(r, b), r)

    @staticmethod
    def fma(x, y, z):
        # for backends that don't override this (halide)
        return ops.add(ops.mul(x, y), z)

    @staticmethod
    def trunc_to_int(a, dtype):
        return ops.to_dtype(ops.trunc(a), dtype)

    @staticmethod
    def floor_to_int(a, dtype):
        return ops.to_dtype(ops.floor(a), dtype)

    @staticmethod
    def ceil_to_int(a, dtype):
        return ops.to_dtype(ops.ceil(a), dtype)

    @staticmethod
    def round_to_int(a, dtype):
        return ops.to_dtype(ops.round(a), dtype)

    @staticmethod
    def int_truediv(a, b):
        # TODO: this is wrong
        # TODO: an easy bandaid is to generate runtime asserts that it's
        # <= 2**53, which is when this equation is correct
        return ops.truediv(a, b)

    @staticmethod
    def load_seed(name, offset):
        return ops.load(name, sympy.Integer(offset))

    @classmethod
    def _initialize_pointwise_overrides(cls, target):
        assert target in {"triton", "cpp", "cppvec"}, target

        for funcname, data in pointwise_overrides_data.items():
            impl = getattr(data, target)
            if impl is None:
                continue
            setattr(cls, funcname, staticmethod(impl))


@dataclasses.dataclass
class OverridesData:
    name: str
    cpp: Callable[..., str]
    # None when not impl in libdevice/triton
    triton: Optional[Callable[..., str]] = None
    # None when not impl in aten/.../vec
    cppvec: Optional[Callable[..., str]] = None
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND = (
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


# NB: if you add a new special function, don't forget to update
# torch._inductor.ops_handler too
pointwise_overrides_data: Dict[str, OverridesData] = dict(
    airy_ai=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"airy_ai_forward({x})",
        name="special_airy_ai",
    ),
    bessel_j0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_j0_forward({x})",
        triton=lambda x: f"libdevice.j0({x})",
        name="special_bessel_j0",
    ),
    bessel_j1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_j1_forward({x})",
        triton=lambda x: f"libdevice.j1({x})",
        name="special_bessel_j1",
    ),
    bessel_y0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_y0_forward({x})",
        triton=lambda x: f"libdevice.y0({x})",
        name="special_bessel_y0",
    ),
    bessel_y1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_y1_forward({x})",
        triton=lambda x: f"libdevice.y1({x})",
        name="special_bessel_y1",
    ),
    digamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_digamma({x})",
        cppvec=lambda x: f"{x}.digamma()",
        name="digamma",
    ),
    # no cpp nor triton implementation for entr, it is defined as decomposition
    # erf, erfc
    erfcx=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_erfcx({x})",
        triton=lambda x: f"libdevice.erfcx({x})",
        name="special_erfcx",
    ),
    fma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y, z: f"std::fma({x}, {y}, {z})",
        cppvec=lambda x, y, z: f"fmadd({x}, {y}, {z})",
        triton=lambda x, y, z: f"libdevice.fma({x}, {y}, {z})",
        name="fma",
    ),
    # erfinv, exp2, expit, gammaln
    igamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igamma({x}, {y})",
        name="igamma",
    ),
    igammac=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igammac({x}, {y})",
        name="igammac",
    ),
    gammainc=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igamma({x}, {y})",
        name="special_gammainc",
    ),
    gammaincc=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igammac({x}, {y})",
        name="special_gammaincc",
    ),
    i0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i0({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i0({x})",
        cppvec=lambda x: f"{x}.i0()",
        name="i0",
    ),
    i0e=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i0e({x})",
        cppvec=lambda x: f"{x}.i0e()",
        name="special_i0e",
    ),
    i1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i1({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i1({x})",
        name="special_i1",
    ),
    i1e=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i1e({x})",
        name="special_i1e",
    ),
    log_ndtr=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_log_ndtr({x})",
        name="special_log_ndtr",
    ),
    # logit
    modified_bessel_i0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_i0_forward({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i0({x})",
        name="special_modified_bessel_i0",
    ),
    modified_bessel_i1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_i1_forward({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i1({x})",
        name="special_modified_bessel_i1",
    ),
    modified_bessel_k0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_k0_forward({x})",
        name="special_modified_bessel_k0",
    ),
    modified_bessel_k1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_k1_forward({x})",
        name="special_modified_bessel_k1",
    ),
    # multigamma
    ndtr=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_ndtr({x})",
        name="special_ndtr",
    ),
    ndtri=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_ndtri({x})",
        name="special_ndtri",
    ),
    polygamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_polygamma({y}, {x})",
        name="polygamma",
    ),
    # psi - alias to digamma
    # round
    scaled_modified_bessel_k0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"scaled_modified_bessel_k0_forward({x})",
        name="special_scaled_modified_bessel_k0",
    ),
    scaled_modified_bessel_k1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"scaled_modified_bessel_k1_forward({x})",
        name="special_scaled_modified_bessel_k1",
    ),
    # sinc
    spherical_bessel_j0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"spherical_bessel_j0_forward({x})",
        name="special_spherical_bessel_j0",
    ),
    zeta=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"zeta({x}, {y})",
        name="special_zeta",
    ),
    chebyshev_polynomial_t=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_t_forward({x}, {y})",
        name="special_chebyshev_polynomial_t",
    ),
    chebyshev_polynomial_u=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_u_forward({x}, {y})",
        name="special_chebyshev_polynomial_u",
    ),
    chebyshev_polynomial_v=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_v_forward({x}, {y})",
        name="special_chebyshev_polynomial_v",
    ),
    chebyshev_polynomial_w=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_w_forward({x}, {y})",
        name="special_chebyshev_polynomial_w",
    ),
    legendre_polynomial_p=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"legendre_polynomial_p_forward({x}, {y})",
        name="special_legendre_polynomial_p",
    ),
    shifted_chebyshev_polynomial_t=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_t_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_t",
    ),
    shifted_chebyshev_polynomial_u=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_u_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_u",
    ),
    shifted_chebyshev_polynomial_v=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_v_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_v",
    ),
    shifted_chebyshev_polynomial_w=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_w_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_w",
    ),
    hermite_polynomial_h=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_h_forward({x}, {y})",
        name="special_hermite_polynomial_h",
    ),
    hermite_polynomial_he=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_he_forward({x}, {y})",
        name="special_hermite_polynomial_he",
    ),
    laguerre_polynomial_l=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"laguerre_polynomial_l_forward({x}, {y})",
        name="special_laguerre_polynomial_l",
    ),
)


# Use mypy to check protocol implemented correctly
def _typecheck_OpOverrides(h: OpOverrides) -> OpsHandler[str]:
    return h


class DeferredLine(DeferredLineBase):
    """A line that can be 'unwritten' by adding name to V.graph.removed_buffers"""

    def __init__(self, name, line):
        super().__init__(line)
        self.name = name
        assert not isinstance(line, DeferredLineBase)

    def __call__(self):
        if all(
            self.name not in x
            for x in (
                V.graph.removed_buffers,
                V.kernel.removed_buffers,
                V.graph.inplaced_to_remove,
                V.kernel.inplaced_to_remove,
            )
        ):
            return self.line
        return None

    def _new_line(self, line):
        return DeferredLine(self.name, line)


class BracesBuffer(IndentedBuffer):
    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                self._indent -= 1
                self.writeline("}")
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
                self._indent -= 1
                self.writeline("}")

        return ctx()


class InplacedBuffer(NamedTuple):
    inner_name: str
    other_names: List[str]


class KernelArgs:
    @staticmethod
    def _lookup(prefix, odict, name):
        assert isinstance(name, (str, sympy.Symbol))
        if name not in odict:
            odict[name] = f"{prefix}{len(odict)}"
        return odict[name]

    def __init__(self, sizevars=None):
        self.input_buffers = {}
        self.output_buffers = {}
        self.inplace_buffers = {}
        self.sizevars = sizevars or {}
        self.workspace_args = []

    def __repr__(self):
        return "KernelArgs({})".format(
            ", ".join(
                map(
                    repr,
                    [
                        self.input_buffers,
                        self.output_buffers,
                        self.inplace_buffers,
                        self.sizevars,
                    ],
                )
            )
        )

    def _buffer_is_marked_removed(self, name):
        return isinstance(name, str) and name.startswith("REMOVED")

    def input(self, name):
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        assert name not in V.graph.removed_buffers, name
        if name in self.output_buffers:
            return self.output_buffers[name]
        if name in self.inplace_buffers:
            return self.inplace_buffers[name].inner_name
        if name.startswith("seed"):
            return self._lookup("seed", self.input_buffers, name)
        return self._lookup("in_ptr", self.input_buffers, name)

    def output(self, name):
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        assert name not in V.graph.removed_buffers, name
        if name in self.inplace_buffers:
            return self.inplace_buffers[name].inner_name
        return self._lookup("out_ptr", self.output_buffers, name)

    def make_inplace(self, input_name, output_name):
        assert output_name not in self.inplace_buffers
        if input_name in self.inplace_buffers:
            buf = self.inplace_buffers[input_name]
            buf.other_names.append(output_name)
            self.inplace_buffers[output_name] = buf
        else:
            buf = InplacedBuffer(
                f"in_out_ptr{len(unique(self.inplace_buffers.values()))}",
                [input_name, output_name],
            )
            self.inplace_buffers[input_name] = buf
            self.inplace_buffers[output_name] = buf

    def workspace(self, nbytes: sympy.Expr, zero_fill: bool):
        """
        Allocate or extend a workspace buffer of nbytes bytes.

        This function manages the allocation of a workspace buffer. It either creates
        a new WorkspaceArg or extends an existing one.

        Note:
        - Calling this function will in-place mutate the args by adding or updating
        a WorkspaceArg.
        - The codegen for generating the Python argdefs and call_defs will check
        this field and allocate the buffer accordingly.
        - A new argument "ws_ptr" will be present in the generated code.

        Args:
            nbytes (sympy.Expr): The number of bytes to allocate.
            zero_fill (bool): Whether to initialize the buffer to zero.

        Returns:
            Tuple[str, int]: A tuple containing:
                - "ws_ptr": A string identifier for the workspace pointer.
                - offset: An integer representing the byte offset in the workspace.
        """
        arg = WorkspaceArg(
            count=nbytes,
            zero_mode=WorkspaceZeroMode.from_bool(zero_fill),
            device=V.graph.get_current_device_or_throw(),
            outer_name=WorkspaceArg.unique_name(),
        )
        for i, existing_arg in enumerate(self.workspace_args):
            if WorkspaceArg.can_join(existing_arg, arg):
                offset = existing_arg.count
                self.workspace_args[i] = WorkspaceArg.join(existing_arg, arg)
                return existing_arg.inner_name, offset
            assert (
                existing_arg.inner_name != arg.inner_name
                and existing_arg.outer_name != arg.outer_name
            )
        self.workspace_args.append(arg)
        return arg.inner_name, 0

    def semaphores(self, min_size: sympy.Expr):
        """
        Lazily allocate a graph-wide semaphores buffer with at least min_size.  This is a single buffer shared by
        all kernels and zero initialized once at graph start.  Each kernel must leave the buffer zeroed on exit.

        Warning: multiple calls to this function will return the same buffer.

        Args:
            min_size: the number of int32 semaphores required

        Returns:
            name of the semaphores buffer
        """
        current_device = V.graph.get_current_device_or_throw()
        arg = WorkspaceArg(
            count=min_size,
            zero_mode=WorkspaceZeroMode.ZERO_PER_GRAPH,
            dtype=torch.uint32,
            inner_name="sem_ptr",
            outer_name=f"semaphores_{current_device.type}_{current_device.index}",
            device=current_device,
        )
        for existing_arg in self.workspace_args:
            if existing_arg.inner_name == arg.inner_name:
                assert arg == existing_arg
        self.workspace_args.append(arg)
        return arg.inner_name

    def seed_offset(self, name, value):
        if value in self.sizevars:
            return self.sizevars[value]
        if name in self.sizevars.values():
            name = (
                f"{name}{sum(1 for v in self.sizevars.values() if v.startswith(name))}"
            )
        self.sizevars[value] = name
        return name

    def size(self, name):
        if str(name) == "seed":
            self.sizevars["seed"] = "seed"
            return "seed"
        return self._lookup("ks", self.sizevars, name)

    def call_names(self):
        return chain(
            self.input_buffers.keys(), self.output_buffers.keys(), self.sizevars.keys()
        )

    def wrap_ptr_arg(self, buf, dtype):
        return buf

    def wrap_size_arg(self, size):
        return str(size)

    def cpp_argdefs(self):
        from .cpp_utils import DTYPE_TO_CPP, INDEX_TYPE

        call_args = []
        arg_defs = []
        arg_types = []
        for inplaced in unique(self.inplace_buffers.values()):
            if self._buffer_is_marked_removed(inplaced):
                continue
            outer = inplaced.other_names[-1]
            inner = inplaced.inner_name
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = DTYPE_TO_CPP[dtype]
            arg_defs.append(f"{cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.input_buffers.items():
            if outer in self.inplace_buffers:
                continue
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = DTYPE_TO_CPP[dtype]
            arg_defs.append(f"const {cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"const {cpp_dtype}*")
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers or self._buffer_is_marked_removed(inner):
                continue
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = DTYPE_TO_CPP[dtype]
            arg_defs.append(f"{cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.sizevars.items():
            arg_defs.append(f"const {INDEX_TYPE} {inner}")
            call_args.append(self.wrap_size_arg(outer))
            arg_types.append(f"const {INDEX_TYPE}")
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)
        assert not self.workspace_args, "Workspace not supported on CPU "
        return arg_defs, call_args, arg_types

    def python_argdefs(self):
        arg_defs: List[str] = []
        call_args: List[str] = []
        arg_types: List[torch.dtype] = []
        precompile_args: List[Union[TensorArg, SizeArg, WorkspaceArg]] = []
        for inplaced in unique(self.inplace_buffers.values()):
            if self._buffer_is_marked_removed(inplaced):
                continue
            arg_defs.append(inplaced.inner_name)
            call_args.append(inplaced.other_names[-1])
            arg_types.append(V.graph.get_dtype(inplaced.other_names[-1]))
            precompile_args.append(
                TensorArg(
                    name=inplaced.inner_name,
                    buffer=inplaced.other_names[-1],
                    dtype=V.graph.get_dtype(inplaced.other_names[-1]),
                )
            )
        for outer, inner in chain(
            self.input_buffers.items(), self.output_buffers.items()
        ):
            if outer in self.inplace_buffers or self._buffer_is_marked_removed(inner):
                continue
            arg_defs.append(inner)
            call_args.append(outer)
            arg_types.append(V.graph.get_dtype(outer))
            precompile_args.append(
                TensorArg(
                    name=inner,
                    buffer=outer,
                    dtype=V.graph.get_dtype(outer),
                )
            )
        for outer, inner in self.sizevars.items():
            arg_defs.append(inner)
            call_args.append(outer)
            arg_types.append(type(outer))  # type: ignore[arg-type]
            precompile_args.append(SizeArg(inner, outer))
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)
        for arg in self.workspace_args:
            arg_defs.append(arg.inner_name)
            call_args.append(arg.outer_name)
            precompile_args.append(arg)
            arg_types.append(arg.dtype)
        return arg_defs, call_args, precompile_args, arg_types

    def aliases(self):
        for inplaced in unique(self.inplace_buffers.values()):
            if self._buffer_is_marked_removed(inplaced):
                continue
            for other in inplaced.other_names:
                if (
                    other in V.graph.inplaced_to_remove
                    or other in V.kernel.inplaced_to_remove
                ):
                    continue
                if other in self.input_buffers:
                    yield self.input_buffers[other], inplaced.inner_name
                if other in self.output_buffers:
                    yield self.output_buffers[other], inplaced.inner_name

    def is_removed(self, name):
        def _is_removed(name, buffers):
            return name not in buffers or self._buffer_is_marked_removed(buffers[name])

        return _is_removed(name, self.output_buffers) and _is_removed(
            name, self.inplace_buffers
        )

    # Includes inplace buffers, excludes removed buffers.  Essentially,
    # after you do a call into this kernel, which buffers actually contain
    # updated data?  Modeled off of python_argdefs.
    def live_output_buffers(self):
        live_outs = OrderedSet()  # type: ignore[var-annotated]
        for inplaced in unique(self.inplace_buffers.values()):
            if self._buffer_is_marked_removed(inplaced):
                continue
            live_outs.add(inplaced.other_names[-1])
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers or self._buffer_is_marked_removed(inner):
                continue
            live_outs.add(outer)
        return live_outs


class CSEVariable:
    """A CSEVariable is just a name for an expression but it is useful to be able to annotate them on a backend dependent basis.
    To do so, the backends can simply overload `Kernel.create_cse_var`
    The "CSEVariable.update_on_args" method gives you a hook for annotations
    See example of TritonCSEVariable in triton.py
    """

    def __init__(
        self,
        name,
        bounds: ValueRanges[Any],
        dtype: Optional[torch.dtype] = None,
    ):
        assert isinstance(bounds, ValueRanges)
        self.name = name
        self.bounds = bounds
        self.use_count = 1  # track how many times this expression is used
        self.dtype = dtype

    def __str__(self):
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return type(other) == type(self) and other.name == self.name

    def update_on_args(self, name, args, kwargs):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"


class CppWrapperKernelArgs(KernelArgs):
    def wrap_size_arg(self, size):
        return f"{size}"


class CSE:
    """Common subexpression elimination"""

    def __init__(
        self,
        prefix="",
        suffix="",
        name_prefix="tmp",
        iter_buffers=None,
        store_cache=None,
        reduction_cache=None,
        varname_map=None,
    ):
        self.prefix = prefix
        self.suffix = suffix
        self._cache = {}
        self.name_prefix = name_prefix
        self.store_cache = store_cache or {}
        self.reduction_cache = reduction_cache or {}
        self.iter_buffer_ids = iter_buffers or itertools.count()
        self.invalidated_stores = OrderedSet()  # type: ignore[var-annotated]
        self.varname_map = varname_map or {}

    def invalidate(self, keep_vars: OrderedSet[str]):
        for name, tmp in list(self.store_cache.items()):
            if tmp not in keep_vars:
                del self.store_cache[name]
                self.invalidated_stores.add(name)
        self._cache = {k: v for k, v in self._cache.items() if v in keep_vars}

    def clone(self):
        # Note(fdrocha): reduction_cache is not being cloned, not sure if this is intentional
        return type(self)(
            prefix=self.prefix,
            suffix=self.suffix,
            name_prefix=self.name_prefix,
            iter_buffers=self.iter_buffer_ids,
            store_cache=self.store_cache,
            varname_map=self.varname_map,
        )

    def augment_key(self, cache_key: object) -> object:
        "Override this method to augment cache key with backend specifics"
        return cache_key

    def put(self, cache_key: object, val: CSEVariable) -> None:
        self._cache[self.augment_key(cache_key)] = val

    def contains(self, cache_key) -> bool:
        return self.augment_key(cache_key) in self._cache

    def try_get(self, cache_key: object) -> Optional[CSEVariable]:
        return self._cache.get(self.augment_key(cache_key), None)

    def get(self, cache_key: object) -> CSEVariable:
        return self._cache[self.augment_key(cache_key)]

    def generate(
        self,
        buffer: IndentedBuffer,
        expr: Union[str, CSEVariable, OpsValue, IndentedBuffer, DeferredLineBase],
        *,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        write=True,
        assignment=True,
        dtype: Optional[torch.dtype] = None,
    ) -> CSEVariable:
        if isinstance(expr, OpsValue):
            expr = expr.value

        assert write or assignment
        if isinstance(expr, CSEVariable):
            # If the expressions were always created with all the information, we could
            # assert expr.bounds == bounds, but sometimes the expression is created
            # with the loose ValueRanges.unknown(), so we need to tighten the bounds
            expr.bounds = expr.bounds.tighten(bounds)
            expr.use_count += 1
            return expr
        elif isinstance(expr, IndentedBuffer):
            cache_key = expr.getvalue()
        elif isinstance(expr, DeferredLineBase):
            cache_key = expr.line
        else:
            assert isinstance(expr, str)
            cache_key = expr
        var = self.try_get(cache_key)
        if not var:
            var = self.newvar(bounds, dtype)
            self.put(cache_key, var)
            if write:
                if V.kernel.current_node:
                    V.kernel.current_node.codegen_originating_info(
                        buffer, only_once=True
                    )
                if isinstance(expr, IndentedBuffer):
                    if assignment:
                        buffer.writeline(f"{self.prefix}{var} =")
                    buffer.splice(expr)
                    buffer.writeline(self.suffix)
                elif isinstance(expr, DeferredLineBase):
                    assert assignment
                    buffer.writeline(
                        expr._new_line(f"{self.prefix}{var} = {expr.line}{self.suffix}")
                    )
                else:
                    if assignment:
                        line = f"{self.prefix}{var} = {expr}{self.suffix}"
                    else:
                        line = f"{expr}{self.suffix}"
                    buffer.writeline(line)
        else:
            var.bounds = var.bounds.tighten(bounds)
            var.use_count += 1

        return var

    def newvar(
        self,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        dtype: Optional[torch.dtype] = None,
    ) -> CSEVariable:
        var_name = f"{self.name_prefix}{next(self.iter_buffer_ids)}"
        var = V.kernel.create_cse_var(var_name, bounds, dtype)
        self.varname_map[var_name] = var
        return var

    def namedvar(
        self,
        name: str,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        dtype: Optional[torch.dtype] = None,
    ) -> CSEVariable:
        torch._check_value(
            name not in self.varname_map, lambda: f"duplicate name: {name}"
        )
        var = V.kernel.create_cse_var(name, bounds, dtype)
        self.varname_map[name] = var
        return var


class CodeGen:
    def __init__(self) -> None:
        super().__init__()
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        self.exit_stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)


class ScopedDict:
    def __init__(self, original_dict):
        self.original_dict = original_dict
        self.new_items = {}

    def __getitem__(self, key):
        if key in self.new_items:
            return self.new_items[key]
        return self.original_dict[key]

    def __setitem__(self, key, value):
        self.new_items[key] = value

    def __contains__(self, key):
        return key in self.new_items or key in self.original_dict

    def get(self, key, default=None):
        if key in self.new_items:
            return self.new_items[key]
        return self.original_dict.get(key, default)


class Kernel(CodeGen):
    newvar_prefix = ""
    suffix = ""
    overrides: Optional[Callable[[OpsHandler[Any]], OpsHandler[Any]]] = None
    # TODO: these look dead, but with all the getattr it's hard to tell...
    load_format: None = None
    store_format: None = None

    def __init__(self, args=None, increase_kernel_count=True):
        super().__init__()
        if increase_kernel_count:
            metrics.generated_kernel_count += 1
        self.args = args or KernelArgs()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()

        self.num_load = 0
        self.num_reduction = 0

        self.cse: CSE = CSE(self.newvar_prefix, self.suffix)
        self.must_keep_buffers = OrderedSet()  # type: ignore[var-annotated]
        self.store_buffer_names = OrderedSet()  # type: ignore[var-annotated]
        self._load_mask = None
        self._load_other = None
        # OrderedSet in set_current_node
        self.current_node = None
        self.node_to_bounds: Optional[Dict[torch.fx.Node, ValueRanges[Any]]] = None

        self.removed_buffers = OrderedSet()  # type: ignore[var-annotated]
        self.inplaced_to_remove = OrderedSet()  # type: ignore[var-annotated]

        # key: the buffer to write
        # value: the buffer to read and whose memory can be reused for
        #   the buffer specified by key
        self.inplace_update_buffers = {}
        # Set minimum number of elements processed per thread.
        self.min_elem_per_thread = 1
        self.kernel_name = None

    @contextlib.contextmanager
    def set_current_node(self, node):
        prior = self.current_node
        self.current_node = node
        self.node_to_bounds = node._body.bounds().get_bounds()
        try:
            yield
        finally:
            self.current_node = prior

    @contextlib.contextmanager
    def swap_buffers(self, lb, cb=None, sb=None):
        def scope_cse(cse):
            new_cse = cse.clone()
            new_cse._cache = ScopedDict(cse._cache)
            new_cse.reduction_cache = ScopedDict(cse.reduction_cache)
            new_cse.store_cache = ScopedDict(cse.store_cache)
            return new_cse

        if cb is None:
            cb = lb
        loads = self.loads
        compute = self.compute
        stores = self.stores
        cse = self.cse
        self.loads = lb
        self.compute = cb
        self.stores = sb
        self.cse = scope_cse(cse)
        try:
            yield
        finally:
            self.loads = loads
            self.compute = compute
            self.stores = stores
            self.cse = cse

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        raise NotImplementedError

    def indirect_load(self, name: str, index: sympy.Expr):
        """A load the depends on an index we have read"""
        prior = self.loads
        try:
            # put the load in the compute section as it might have deps
            self.loads = self.compute
            return self.load(name, index)
        finally:
            self.loads = prior

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable):
        raise NotImplementedError

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        raise NotImplementedError

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
        raise NotImplementedError

    def scan(
        self,
        dtypes: Tuple[torch.dtype, ...],
        combine_fn: Callable[
            [Tuple[CSEVariable, ...], Tuple[CSEVariable, ...]], Tuple[CSEVariable, ...]
        ],
        values: Tuple[CSEVariable, ...],
    ) -> Tuple[CSEVariable, ...]:
        raise NotImplementedError

    def sort(
        self,
        dtypes: Tuple[torch.dtype, ...],
        values: Tuple[CSEVariable, ...],
        stable: bool,
        descending: bool,
    ) -> Tuple[CSEVariable, ...]:
        raise NotImplementedError

    def var_ranges(self):
        raise NotImplementedError

    def bucketize(
        self,
        values: CSEVariable,
        boundaries: Tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: CSEVariable,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: Optional[Tuple[str, sympy.Expr]] = None,
        sorter_indices: Optional[CSEVariable] = None,
    ) -> CSEVariable:
        """
        See [Note: Inductor bucketize op]
        """
        raise NotImplementedError

    @property
    def assert_function(self) -> str:
        raise NotImplementedError

    def indirect_assert(
        self,
        var: Union[CSEVariable, str],
        lower: Optional[str],
        upper: Optional[str],
        mask: Optional[Union[CSEVariable, str]] = None,
    ) -> str:
        if isinstance(var, CSEVariable):
            var = str(var)
        assert isinstance(var, str)
        assert lower is None or isinstance(lower, str)
        assert upper is None or isinstance(upper, str)
        if lower and upper:
            # The conditions need to be in parens because of Python's operator precedence.
            # It'd be less error-prone to use and/or/not, which is suported by triton
            cond = f"({lower} <= {var}) & ({var} < {upper})"
            cond_print = f"{lower} <= {var} < {upper}"
        elif lower:
            cond = f"{lower} <= {var}"
            cond_print = cond
        else:
            assert upper
            cond = f"{var} < {upper}"
            cond_print = cond

        if mask:
            cond = f"({cond}) | ~({mask})"

        return f'{self.assert_function}({cond}, "index out of bounds: {cond_print}")'

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ):
        raise NotImplementedError

    def index_to_str(self, index: sympy.Expr) -> str:
        raise NotImplementedError

    def __enter__(self):
        # TODO: hoist this to top level
        class CSEProxy:
            self.name = "CSEProxy"
            vr_analysis = ValueRangeAnalysis()

            @staticmethod
            def __getattr__(name: str) -> Callable[..., CSEVariable]:  # type: ignore[misc]
                def inner(*args, **kwargs):
                    bounds = CSEProxy._bound_variable(name, *args, **kwargs)

                    value = getattr(parent_handler, name)(*args, **kwargs)  # type: ignore[has-type]
                    dtype_handler = DtypePropagationOpsHandler()

                    output_idx = 0

                    def do_cse(v):
                        # cpp backend doesnt set current device - TODO: fix
                        if V.graph.current_device is not None:
                            device_str = V.graph.get_current_device_or_throw().type
                            triton_backend = (
                                config.cpu_backend == "triton"
                                if device_str == "cpu"
                                else config.cuda_backend == "triton"
                            )
                        else:
                            triton_backend = False

                        # only triton backend tracks dtype currently
                        if triton_backend:
                            if name == "masked":
                                output_dtype = value.dtype
                            else:
                                output_dtype = getattr(
                                    dtype_handler,
                                    name,
                                )(*args, **kwargs)
                        else:
                            # cpp backend doesnt track dtype yet
                            output_dtype = None

                        csevar = V.kernel.cse.generate(
                            V.kernel.compute,
                            v,
                            bounds=bounds,
                            dtype=output_dtype,
                        )

                        nonlocal output_idx
                        if (
                            config.test_configs.runtime_triton_dtype_assert
                            and triton_backend
                        ):
                            from torch._inductor.codegen.triton import triton_type

                            # we tree_map over the output, so we need to fetch corresponding dtype
                            if isinstance(output_dtype, (list, tuple)):
                                output_dtype = output_dtype[output_idx]

                            V.kernel.compute.writeline(
                                f"tl.static_assert({csevar}.dtype == {triton_type(output_dtype)})"
                            )
                        output_idx += 1

                        csevar.update_on_args(name, args, kwargs)

                        return csevar

                    return pytree.tree_map(do_cse, value)

                return inner

            @staticmethod
            def _bound_variable(name, *args, **kwargs):
                """
                If the variable comes from an FX node, we forward the bound we have already computed
                Else, if the variable when codegen'ing another op, we try to compute its bounds
                """
                from ..select_algorithm import TritonTemplateKernel

                if isinstance(V.kernel, TritonTemplateKernel):
                    return ValueRanges.unknown()

                fx_node = V.interpreter.current_node
                if fx_node.target == name and self.node_to_bounds is not None:
                    assert isinstance(self.node_to_bounds, dict)
                    return self.node_to_bounds.get(fx_node, ValueRanges.unknown())
                elif config.compute_all_bounds and hasattr(ValueRangeAnalysis, name):
                    # These create lots of inner strings. We would need to compute the bounds at the ops
                    # We will also likely not get much from computing VRs on these nodes
                    if any(
                        s in fx_node.target
                        for s in ("set_indirect", "reduction", "scan")
                    ):
                        return ValueRanges.unknown()

                    # We assume that the inputs come from `ops.` and are not strings. If you want to generate
                    # intermediary strings, wrap them in CSE variables with properly initialised bounds.

                    # If there is no FX bound but we know how to compute one we do so
                    assert not kwargs

                    def arg_to_bound(x):
                        if isinstance(x, CSEVariable):
                            return x.bounds
                        elif isinstance(x, sympy.Expr):
                            return bound_sympy(x)
                        else:
                            return x

                    arg_bounds = list(map(arg_to_bound, args))
                    return getattr(CSEProxy.vr_analysis, name)(*arg_bounds)
                return ValueRanges.unknown()

            @staticmethod
            def indirect_indexing(
                var: CSEVariable,
                size: Union[sympy.Expr, int],
                check: bool = True,
                wrap_neg=True,
            ):
                if isinstance(size, int):
                    size = sympy.Integer(size)
                assert isinstance(size, sympy.Expr), size
                # Skip CSE since this doesn't return an expression

                if var.bounds.lower < 0:  # type: ignore[operator]
                    if wrap_neg:
                        stm = ops.add(var, ops.index_expr(size, torch.long))
                        # Mixed negative and non-negative
                        if var.bounds.upper >= 0:  # type: ignore[operator]
                            lt = ops.lt(var, 0)
                            stm = ops.where(lt, stm, var)
                    else:
                        stm = var

                    # Propagate bounds as we know how to compute them properly
                    new_bounds = ValueRanges.unknown()
                    if var.bounds != ValueRanges.unknown() and isinstance(
                        size, sympy.Number
                    ):
                        # Take the negative part of the bound and add size to it
                        # Then take union of that and the positive part
                        # This is a tighter bound than that of a generic ops.where, as we have info on the cond
                        neg_bounds = var.bounds & ValueRanges(-int_oo, -1)
                        new_bounds = ValueRanges(
                            neg_bounds.lower + size, neg_bounds.upper + size
                        )
                        # We don't have a good way of representing the empty range
                        if var.bounds.upper >= 0:  # type: ignore[operator]
                            pos = var.bounds & ValueRanges(0, int_oo)
                            new_bounds = new_bounds | pos

                    var = self.cse.generate(self.compute, stm, bounds=new_bounds)

                sympy_var = parent_handler.indirect_indexing(var, size, check)
                if generate_assert(check):
                    assert_lower = not (var.bounds.lower >= 0)
                    # value ranges cannot x < s when x and s are symbols
                    assert_upper = not isinstance(size, sympy.Number) or not (
                        var.bounds.upper < size
                    )
                    self.check_bounds(sympy_var, size, assert_lower, assert_upper)
                return sympy_var

            @staticmethod
            def check_bounds(
                expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
            ):
                return self.check_bounds(expr, size, lower, upper)

            @staticmethod
            def load(name: str, index: sympy.Expr) -> CSEVariable:
                if name in self.cse.invalidated_stores:
                    # A load from an invalidated store requires us to
                    # keep the actual buffer around
                    V.kernel.must_keep_buffers.add(name)
                if free_symbol_is_type(index, SymT.TMP):
                    return self.indirect_load(name, index)
                store_cache = self.cse.store_cache
                if name in store_cache:
                    return store_cache[name]
                out = self.load(name, index)
                # count load that is not in the store_cache, and also not in the
                # cse cache.
                if out.use_count == 1:
                    self.num_load += 1
                return out

            @staticmethod
            def _update_store_cache(name: str, value: CSEVariable):
                self.cse.store_cache[name] = value
                if self.current_node and name in V.graph.name_to_buffer:
                    buf = self.current_node.get_output(name)
                    for other_name in buf.get_mutations():
                        self.cse.store_cache[other_name] = value

            @staticmethod
            def store(
                name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
            ) -> None:
                self.store_buffer_names.add(name)
                if mode is None:
                    CSEProxy._update_store_cache(name, value)
                if name not in V.graph.removed_buffers:
                    return self.store(name, index, value, mode=mode)
                return None  # type: ignore[return-value]

            @staticmethod
            def store_reduction(name: str, index: sympy.Expr, value: CSEVariable):
                self.store_buffer_names.add(name)
                CSEProxy._update_store_cache(name, value)

                if name not in V.graph.removed_buffers:
                    return self.store_reduction(name, index, value)

            @staticmethod
            def reduction(
                dtype: torch.dtype,
                src_dtype: torch.dtype,
                reduction_type: ReductionType,
                value: Union[CSEVariable, Tuple[CSEVariable, ...]],
            ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
                self.num_reduction += 1
                return self.reduction(dtype, src_dtype, reduction_type, value)

            @staticmethod
            def scan(
                dtypes: Tuple[torch.dtype, ...],
                combine_fn: Callable[
                    [Tuple[CSEVariable, ...], Tuple[CSEVariable, ...]],
                    Tuple[CSEVariable, ...],
                ],
                values: Tuple[CSEVariable, ...],
            ) -> Tuple[CSEVariable, ...]:
                return self.scan(dtypes, combine_fn, values)

            @staticmethod
            def sort(
                dtypes: Tuple[torch.dtype, ...],
                values: Tuple[CSEVariable, ...],
                stable: bool,
                descending: bool,
            ) -> Tuple[CSEVariable, ...]:
                return self.sort(dtypes, values, stable, descending)

            @staticmethod
            def bucketize(
                values: CSEVariable,
                boundaries: Tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
                boundary_indices: CSEVariable,
                indexing_dtype: torch.dtype,
                right: bool,
                sorter: Optional[Tuple[str, sympy.Expr]] = None,
                sorter_indices: Optional[CSEVariable] = None,
            ) -> CSEVariable:
                """
                [Note: Inductor bucketize op]

                Inputs:
                -------
                values: the values to be bucketized.
                boundaries: a tuple containing
                  (a) the name of the boundaries tensor (which must be sorted, unless
                  the sorting tensor is present),
                  (b) the length of the tensor in the last dimension (i.e. the length of
                  one set of boundaries),
                  (c) the number of elements in the underlying storage (i.e. the length
                  of the flattened tensor, ignoring striding), and
                  (d) the stride of the tensor in the last dimension.
                boundary_indices: indices into a flattened version of the boundaries
                tensor, of the same size and shape as "values".  Each index points to
                the first element in the set of boundaries to be used for the
                corresponding value.
                indexing_dtype: the dtype to use when indexing into the boundaries
                tensor.  This must be int64 or int32.  This additionally specifies the
                dtype of the return value.
                right: see "Details" below.
                sorter: an optional tuple containing
                  (a) the name of an optional sorting tensor, used to access unsorted
                  boundaries without reordering the boundaries tensor, and
                  (b) the stride of the tensor in the last dimension.
                The values in the sorting tensor are used as indices into the *last*
                dimension of the boundaries tensor, with all other indices matching.
                The size of the sorting and boundaries tensors must be equivalent.
                sorter_indices: must be present if the sorting array is present; see
                "boundary_indices" for the equivalent definition for the boundaries
                tensor.

                Output:
                -------
                The buckets each value belongs in, within a given set of boundaries.  0
                indicates a position before the first boundary, and len(boundaries_set)
                represents a position after the last boundary.

                Details:
                --------
                Given a value and a set of boundaries, calculate the bucket that each
                value belongs to.  This works differently in 1-D and N-D cases.

                for values [[-1, 0, 1, 2], [3, 4, 5, 9]], boundaries [0, 4, 4, 8], right=True
                return =   [[ 0, 1, 1, 1], [1, 3, 3, 4]].

                for values [[-1, 0, 1, 2], [3, 4, 5, 9]], boundaries [[0, 4], [4, 8]], right=True
                return =   [[ 0, 1, 1, 1], [0, 1, 1, 2]]

                Note that in the N-D boundaries case, the shape of "values" and
                "boundaries" must match in every dimension _except_ the last.

                When right == False, bucket i refers to range (boundaries[i], boundaries[i+1]].
                When right == True,  bucket i refers to range [boundaries[i], boundaries[i+1]).

                Boundaries must be non-decreasing, or a sorter must be provided which
                would re-index offsets in a non-decreasing order (e.g. the second output
                of torch.sort(offsets)).  Otherwise, the result is undefined.
                """
                return self.bucketize(
                    values,
                    boundaries,
                    boundary_indices,
                    indexing_dtype,
                    right,
                    sorter,
                    sorter_indices,
                )

        # Use mypy to check protocol implemented correctly
        def _typecheck_CSEProxy(h: CSEProxy) -> OpsHandler[CSEVariable]:
            return h

        super().__enter__()
        assert self.overrides
        parent_handler = self.overrides(V.get_ops_handler())
        self.exit_stack.enter_context(V.set_ops_handler(CSEProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_kernel_local_buffers()
        super().__exit__(exc_type, exc_val, exc_tb)

    def remove_kernel_local_buffers(self) -> None:
        """
        Any buffers that are both created and have a last use in the
        same kernel can be removed.

        Note that V.graph.scheduler can be None when codegening triton template
        kernels.
        """
        scheduler = V.graph.scheduler
        if not scheduler:
            return
        fused_node_names = OrderedSet(
            scheduler.name_to_buf[buf].defining_op.get_name()
            for buf in self.store_buffer_names
            if buf in scheduler.name_to_buf
        )
        names_to_remove: OrderedSet[str] = OrderedSet()
        for name in self.store_buffer_names:
            if (
                name not in self.must_keep_buffers
                and name not in self.args.input_buffers
                and scheduler.can_buffer_be_removed_through_fusion(
                    name, fused_node_names
                )
            ):
                names_to_remove.add(name)

        for name in names_to_remove:
            if name in self.args.inplace_buffers:
                buf = self.args.inplace_buffers[name]
                if isinstance(buf, str) and buf.startswith("REMOVED"):
                    continue
                remove = all(n in names_to_remove for n in buf.other_names)
                if remove:
                    self.remove_inplace_buffer(name)
                self.inplaced_to_remove.add(name)
            else:
                self.remove_buffer(name)

    def remove_buffer(self, name: str) -> None:
        # Assign a special value instead of deleting the entry
        # because we still rely on output_buffers's length to
        # generate unique arg name.
        log.debug("remove_buffer(%r)", name)
        self.args.output_buffers[name] = "REMOVED"
        self.removed_buffers.add(name)

    def remove_inplace_buffer(self, name: str) -> None:
        log.debug("removing_inplace_buffer(%r)", name)
        inner_name = self.args.inplace_buffers[name].inner_name
        self.args.inplace_buffers[name] = inner_name.replace("in_out_ptr", "REMOVED")
        self.removed_buffers.add(name)

    def rename_indexing(self, index) -> sympy.Expr:
        # adds the necessary kernel args for index expressions
        # and renames variables in index expressions to kernel arg names
        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]  # type: ignore[return-value]
        index = V.graph.sizevars.simplify(index)
        sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)
        replacements = {
            x: self.args.size(x)
            for x in sorted_symbols
            if symbol_is_type(
                x,
                (
                    SymT.UNBACKED_INT,
                    SymT.SIZE,
                    SymT.PRECOMPUTED_SIZE,
                ),
            )
        }
        return sympy_subs(index, replacements)

    def create_cse_var(self, *args, **kwargs):
        return CSEVariable(*args, **kwargs)


@dataclasses.dataclass
class OptimizationContext:
    key: ClassVar[str] = "opt_ctx"

    dtype: Optional[torch.dtype] = None
    ops_name: str = ""


@functools.lru_cache(None)
def jinja2_env():
    try:
        import jinja2

        return jinja2.Environment(
            undefined=jinja2.StrictUndefined,
        )
    except ImportError:
        return None


class KernelTemplate:
    """
    Base class for defining kernel templates.

    Children classes: TritonTemplate, CUDATemplate
    """

    @staticmethod
    def indent_except_first(source: str, num_indents: int, indents_spacing=4):
        lines = source.splitlines(True)
        if len(lines) > 1:
            lines[1:] = [
                (" " * indents_spacing * num_indents) + line for line in lines[1:]
            ]
        return "".join(lines)

    @staticmethod
    def _template_from_string(source):
        env = jinja2_env()
        if env is None:
            return None
        env.filters["indent_except_first"] = KernelTemplate.indent_except_first
        from jinja2 import TemplateSyntaxError

        class DetailedTemplateSyntaxError(TemplateSyntaxError):
            def __init__(self, original_error):
                super().__init__(
                    original_error.message,
                    original_error.lineno,
                    original_error.name,
                    original_error.filename,
                )
                self.original_error = original_error

            def __str__(self):
                error_info = f"Error in template at line {self.lineno}\n"
                error_info += f"Error message: {self.message}\n"
                if hasattr(self.original_error, "source"):
                    lines = self.original_error.source.split("\n")
                    error_info += "Context:\n"
                    start = max(0, self.lineno - 2)
                    end = min(len(lines), self.lineno + 2)
                    for i in range(start, end):
                        if i == self.lineno - 1:
                            error_info += f"{i + 1}: --> {lines[i]}\n"
                            if hasattr(self.original_error, "column"):
                                error_info += (
                                    "     "
                                    + " " * (self.original_error.column - 1)
                                    + "^\n"
                                )
                        else:
                            error_info += f"{i + 1}:     {lines[i]}\n"
                return error_info

        try:
            return env.from_string(source)
        except TemplateSyntaxError as e:
            raise DetailedTemplateSyntaxError(e) from e

    @staticmethod
    def _fake_get_dtype(fake_out):
        _get_dtype_real = V.graph.get_dtype

        def get_dtype(name):
            if name == fake_out.get_name():
                return fake_out.get_dtype()
            return _get_dtype_real(name)

        return get_dtype

    def __init__(self, name: str):
        self.name = name

    def maybe_append_choice(self, choices, **kwargs):
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.
        Returns None if success, otherwise returns the error.

        choices: A list of ChoiceCallers.
        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """

        try:
            choices.append(self.generate(**kwargs))
            return None
        except NotImplementedError as e:
            return e

    def generate(self, **kwargs) -> "torch._inductor.ir.ChoiceCaller":
        """
        Generates a ChoiceCaller instance from the given arguments.
        """

        raise NotImplementedError
