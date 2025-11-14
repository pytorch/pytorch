from __future__ import annotations

import atexit
import contextlib
import dataclasses
import enum
import functools
import itertools
import logging
import math
import operator
import os
import re
import tempfile
from abc import ABC, abstractmethod
from enum import auto, Enum
from itertools import chain
from typing import (
    Any,
    cast,
    ClassVar,
    Generic,
    NamedTuple,
    Optional,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import Self, TypeVar

import sympy

import torch
import torch.fx
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.utils import _pytree as pytree
from torch.utils._config_module import ConfigModule
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.printers import PythonPrinter as _PythonPrinter
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from .. import config, metrics
from ..dtype_propagation import DtypePropagationOpsHandler
from ..ops_handler import BasicMathOpsMixin, DefaultHandler
from ..shape_propagation import ShapePropagationOpsHandler
from ..utils import (
    boolean_ops,
    DeferredLineBase,
    generate_assert,
    get_current_backend,
    IndentedBuffer,
    ir_dataclass,
    ScopedDict,
    sympy_dot,
    sympy_index_symbol,
    sympy_subs,
    triton_type,
    unique,
)
from ..virtualized import (
    NullHandler,
    ops,
    OpsHandler,
    OpsValue,
    ReductionType,
    StoreMode,
    V,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, MutableMapping, Sequence

    from torch.fx import GraphModule

    from ..custom_graph_pass import CustomGraphModulePass
    from ..ir import Buffer, ChoiceCaller, FixedLayout, IRNode
    from ..loop_body import LoopBody
    from ..scheduler import BaseScheduling, Scheduler, SchedulerNode
    from ..shape_propagation import BlockShapeType
    from .wrapper import PythonWrapperCodegen

    _T = TypeVar("_T")
    SchedulingConstructor = Callable[[Optional[Scheduler]], BaseScheduling]
    WrapperConstructor = type[PythonWrapperCodegen]
    SymbolLike = Union[str, sympy.Symbol]

    # OpVarT should really be Union[CSEVariable, str], however this
    # causes typing errors in subclasses (defined in other files).
    OpVarT = str

schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
log = logging.getLogger(__name__)


def data_type_logger(msg: str) -> None:
    if schedule_log.isEnabledFor(logging.DEBUG):
        schedule_log.debug("Data type propagation: %s", msg)


@dataclasses.dataclass
class FileBackedGraphModule:
    """
    Output of FX wrapper codegen. Exposes the same methods as ModuleType, but these
    map back to a GraphModule instead of Python source.
    """

    gm: GraphModule
    compiled_fn: Callable[..., Any]

    def __post_init__(self) -> None:
        # Write the code to a file for compatibility with debugging utilities.
        # The file is deleted upon program termination.
        self.tempfile = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".py", delete=False
        )
        atexit.register(os.remove, self.tempfile.name)
        with self.tempfile as f:
            f.write(self.value)

    @property
    def __file__(self) -> str:
        return self.tempfile.name

    def call(self, args: list[Any]) -> Any:
        return self.compiled_fn(*args)

    @property
    def value(self) -> str:
        return self.gm.code


class WorkspaceZeroMode(enum.Enum):
    UNINITIALIZED = 0
    ZERO_ON_CALL = 1  # kernel may leave workspace dirty
    ZERO_PER_GRAPH = 2  # must be re-zeroed by kernel

    @staticmethod
    def combine(a: WorkspaceZeroMode, b: WorkspaceZeroMode) -> WorkspaceZeroMode:
        if a == b or b == WorkspaceZeroMode.UNINITIALIZED:
            return a
        if a == WorkspaceZeroMode.UNINITIALIZED:
            return b
        raise NotImplementedError(f"WorkspaceZeroMode.combine({a!r}, {b!r})")

    @staticmethod
    def from_bool(zero_fill: bool) -> WorkspaceZeroMode:
        if zero_fill:
            return WorkspaceZeroMode.ZERO_ON_CALL
        return WorkspaceZeroMode.UNINITIALIZED


class CodegenSymbol(ABC):
    """
    An IR object possibly corresponding to a variable in the wrapper code.
    """

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_example(self) -> Union[torch.Tensor, sympy.Symbol]:
        pass


@ir_dataclass(frozen=True)
class WorkspaceArg(CodegenSymbol):
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
    def unique_name(prefix: str = "workspace_") -> str:
        return f"{prefix}{next(V.graph.workspace_id)}"

    @staticmethod
    def can_join(a: WorkspaceArg, b: WorkspaceArg) -> bool:
        return (
            a.inner_name == b.inner_name and a.dtype == b.dtype and a.device == b.device
        )

    @staticmethod
    def join(a: WorkspaceArg, b: WorkspaceArg) -> WorkspaceArg:
        return WorkspaceArg(
            count=a.count + b.count,
            zero_mode=WorkspaceZeroMode.combine(a.zero_mode, b.zero_mode),
            dtype=a.dtype,
            device=a.device,
            inner_name=a.inner_name,
            outer_name=a.outer_name,
        )

    @staticmethod
    def maximum(a: WorkspaceArg, b: WorkspaceArg) -> WorkspaceArg:
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
    def get_device(self) -> torch.device:
        return self.device

    get_device_or_error = get_device

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def get_example(self) -> Union[torch.Tensor, sympy.Symbol]:
        return self.get_layout().get_example()

    def get_layout(self) -> FixedLayout:
        from ..ir import FixedLayout

        return FixedLayout(
            device=self.device,
            dtype=self.dtype,
            size=[self.count],
            stride=[1],
        )

    @property
    def layout(self) -> FixedLayout:
        return self.get_layout()

    get_output_spec = get_layout
    maybe_get_output_spec = get_layout
    maybe_get_layout = get_layout

    def get_offset(self) -> sympy.Expr:
        return sympy.S.Zero

    def get_size(self) -> list[sympy.Expr]:
        return [self.count]

    def get_stride(self) -> list[sympy.Expr]:
        return [sympy.S.One]

    def get_name(self) -> str:
        return self.outer_name

    def get_is_pinned(self) -> bool:
        return False

    def get_inputs_that_alias_output(self) -> list[str]:
        return []


class TritonScratchWorkspace:
    def __init__(self, size: int, generate_dtype_str: Callable[..., str]):
        self.size = size
        self._generate_dtype_str = generate_dtype_str

    def generate_dtype_str(self) -> str:
        return self._generate_dtype_str()


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
    def alias_of(self) -> Optional[str]:
        return None


@dataclasses.dataclass
class ConstexprArg:
    name: str


@dataclasses.dataclass
class TMADescriptorArg:
    name: str
    api_type: str  # "experimental" or "stable"
    block_shape: Optional[list[sympy.Expr]]  # only needed for "stable"
    dtype: Optional[torch.dtype]  # only needed for "stable"


@dataclasses.dataclass
class DeviceCodegen:
    scheduling: SchedulingConstructor
    wrapper_codegen: WrapperConstructor
    cpp_wrapper_codegen: Optional[WrapperConstructor] = None
    fx_wrapper_codegen: Optional[WrapperConstructor] = None


KernelArgType = Union[WorkspaceArg, TensorArg, SizeArg, TMADescriptorArg, ConstexprArg]

device_codegens: dict[str, DeviceCodegen] = {}


class DeviceOpOverrides:
    def import_get_raw_stream_as(self, name: str) -> str:
        raise NotImplementedError

    def set_device(self, device_idx: int) -> str:
        raise NotImplementedError

    def synchronize(self) -> str:
        raise NotImplementedError

    def device_guard(self, device_idx: int) -> str:
        raise NotImplementedError

    def cpp_device_guard(self) -> str:
        raise NotImplementedError

    def cpp_aoti_device_guard(self) -> str:
        raise NotImplementedError

    def cpp_stream_guard(self) -> str:
        raise NotImplementedError

    def cpp_aoti_stream_guard(self) -> str:
        raise NotImplementedError

    def cpp_getStreamFromExternal(self) -> str:
        raise NotImplementedError

    def kernel_header(self) -> str:
        raise NotImplementedError

    def kernel_driver(self) -> str:
        raise NotImplementedError

    def cpp_stream_type(self) -> str:
        raise NotImplementedError

    def aoti_get_stream(self) -> str:
        raise NotImplementedError

    def cpp_kernel_type(self) -> str:
        raise NotImplementedError

    def cpp_device_ptr(self) -> str:
        raise NotImplementedError

    def tma_descriptor_helpers(self) -> str:
        raise NotImplementedError

    def cpp_scratch(
        self, idx: int, workspace: TritonScratchWorkspace, prefix: Optional[str] = None
    ) -> Optional[tuple[list[str], str]]:
        # optionally return (scratch definition, arg name)
        raise NotImplementedError


device_op_overrides_dict: dict[str, DeviceOpOverrides] = {}
custom_backend_passes: dict[str, Optional[CustomGraphModulePass]] = {}
custom_backend_codegen_configs: dict[str, Optional[ConfigModule]] = {}


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
    device_scheduling: SchedulingConstructor,
    device_wrapper_codegen: WrapperConstructor,
    device_cpp_wrapper_codegen: Optional[WrapperConstructor] = None,
    device_fx_wrapper_codegen: Optional[WrapperConstructor] = None,
    device_custom_pass: Optional[CustomGraphModulePass] = None,
    device_custom_config: Optional[ConfigModule] = None,
) -> None:
    device_codegens[device] = DeviceCodegen(
        device_scheduling,
        device_wrapper_codegen,
        device_cpp_wrapper_codegen,
        device_fx_wrapper_codegen,
    )
    custom_backend_passes[device] = device_custom_pass
    if device_custom_config:
        assert (
            isinstance(device_custom_config, ConfigModule)
            and device_custom_config is not config
        ), (
            f"{device_custom_config=} cannot be the same as the default inductor config {config=}"
        )
    custom_backend_codegen_configs[device] = device_custom_config


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


def get_backend_features(
    device: Union[torch.device, str, None],
) -> OrderedSet[BackendFeature]:
    if device is None:
        return OrderedSet()
    init_backend_registration()
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        assert isinstance(device, str), type(device)
        device_type = device
        device = torch.device(device_type)
    scheduling_ctor = get_scheduling_for_device(device_type)
    assert scheduling_ctor
    scheduling = scheduling_ctor(None)
    return scheduling.get_backend_features(device)


def has_backend_feature(
    device: Union[torch.device, str, None], feature: BackendFeature
) -> bool:
    """See also V.graph.has_feature"""
    assert isinstance(feature, BackendFeature)
    return feature in get_backend_features(device)


def get_scheduling_for_device(device: str) -> Optional[SchedulingConstructor]:
    return device_codegens[device].scheduling if device in device_codegens else None


def get_wrapper_codegen_for_device(
    device: str, cpp_wrapper: bool = False, fx_wrapper: bool = False
) -> Optional[WrapperConstructor]:
    if device in device_codegens:
        wrapper_codegen_obj: DeviceCodegen = device_codegens[device]
        if fx_wrapper:
            return wrapper_codegen_obj.fx_wrapper_codegen
        elif cpp_wrapper:
            return wrapper_codegen_obj.cpp_wrapper_codegen
        else:
            return wrapper_codegen_obj.wrapper_codegen
    return None


def get_custom_backend_pass_for_device(device: str) -> Optional[CustomGraphModulePass]:
    return custom_backend_passes.get(device)


def get_custom_backend_config_for_device(device: str) -> Optional[ConfigModule]:
    return custom_backend_codegen_configs.get(device)


@functools.cache
def init_backend_registration() -> None:
    """
    Register the backend for different devices, including the scheduling
    for kernel code generation and the host side wrapper code generation.
    """
    from .cpp import CppScheduling
    from .cpp_wrapper_cpu import CppWrapperCpu
    from .cpp_wrapper_cpu_array_ref import CppWrapperCpuArrayRef
    from .cpp_wrapper_gpu import CppWrapperGpu
    from .cpp_wrapper_mps import CppWrapperMps
    from .cuda_combined_scheduling import CUDACombinedScheduling
    from .halide import HalideScheduling
    from .mps import MetalScheduling
    from .pallas import PallasScheduling
    from .python_wrapper_mtia import PythonWrapperMtia
    from .triton import TritonScheduling
    from .wrapper import PythonWrapperCodegen
    from .wrapper_fxir import WrapperFxCodegen

    if get_scheduling_for_device("cpu") is None:
        cpu_backends = {
            "cpp": CppScheduling,
            "halide": HalideScheduling,
            "triton": TritonScheduling,
            "pallas": PallasScheduling,
        }
        register_backend_for_device(
            "cpu",
            lambda scheduling: cpu_backends[config.cpu_backend](scheduling),
            PythonWrapperCodegen,
            CppWrapperCpuArrayRef
            if config.aot_inductor.allow_stack_allocation
            else CppWrapperCpu,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("cuda") is None:
        # CUDACombinedScheduling combines Triton and CUDA C++ scheduling for CUDA devices via delegation
        cuda_backends = {
            "triton": CUDACombinedScheduling,
            "halide": HalideScheduling,
            "pallas": PallasScheduling,
        }
        register_backend_for_device(
            "cuda",
            lambda scheduling: cuda_backends[config.cuda_backend](scheduling),
            PythonWrapperCodegen,
            CppWrapperGpu,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("xpu") is None:
        register_backend_for_device(
            "xpu",
            TritonScheduling,
            PythonWrapperCodegen,
            CppWrapperGpu,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("mps") is None:
        register_backend_for_device(
            "mps",
            MetalScheduling,
            PythonWrapperCodegen,
            CppWrapperMps,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("mtia") is None:
        register_backend_for_device(
            "mtia",
            TritonScheduling,
            PythonWrapperMtia,
            CppWrapperGpu,
            WrapperFxCodegen,
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
            fx_wrapper_codegen = _get_custom_mod_func("WrapperFxCodegen")
            if device_scheduling and wrapper_codegen and cpp_wrapper_codegen:
                register_backend_for_device(
                    private_backend,
                    device_scheduling,
                    wrapper_codegen,
                    cpp_wrapper_codegen,
                    fx_wrapper_codegen,
                )
        except RuntimeError:
            pass


def index_prevent_reordering(
    index: Sequence[sympy.Expr],
    index_vars: Sequence[sympy.Expr],
    sizes: Sequence[sympy.Expr],
) -> list[sympy.Expr]:
    from ..ir import FlexibleLayout

    # added contiguous index prevents reordering
    return [*index, sympy_dot(index_vars, FlexibleLayout.contiguous_strides(sizes))]


def register_device_op_overrides(
    device: str, device_op_overrides: DeviceOpOverrides
) -> None:
    device_op_overrides_dict[device] = device_op_overrides


def get_device_op_overrides(device: str) -> DeviceOpOverrides:
    assert isinstance(device, str), type(device)

    if not device_op_overrides_dict:
        from . import cpu_device_op_overrides, mps_device_op_overrides  # noqa: F401
        from .cuda import device_op_overrides  # noqa: F401
        from .mtia import device_op_overrides as mtia_op_overrides  # noqa: F401
        from .xpu import device_op_overrides as xpu_op_overrides  # noqa: F401

    return device_op_overrides_dict[device]


DTYPE_TO_COMPUTATION_DTYPE: dict[torch.dtype, torch.dtype] = {
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
    *args: Any,
    **kwargs: Any,
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
        return kwargs["dtype"] if "dtype" in kwargs else args[-1]
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


def check_dtype(
    buffer: IndentedBuffer, var: CSEVariableType, dtype: torch.dtype
) -> None:
    backend = get_current_backend()
    if config.test_configs.runtime_triton_dtype_assert and backend == "triton":
        buffer.writeline(f"tl.static_assert({var}.dtype == {triton_type(dtype)})")
    elif config.test_configs.static_cpp_dtype_assert and backend == "cpp":
        from .cpp_utils import CppCSEVariable, DTYPE_TO_CPP

        assert isinstance(var, CppCSEVariable), type(var)
        if dtype == torch.bool:
            if var.is_vec:
                is_same_dt = f"IsVecMaskType<decltype({var})>::value"
            else:
                # operator&(bool, bool) returns int and it can be used as boolean in C++
                is_same_dt = f"std::is_same_v<decltype({var}), bool> || std::is_same_v<decltype({var}), int>"
        else:
            c_var_type = f"decltype({var})"
            if var.is_vec:
                c_var_type = f"typename {c_var_type}::value_type"
            is_same_dt = f"std::is_same_v<{c_var_type}, {DTYPE_TO_CPP[dtype]}>"

        buffer.writeline(f"static_assert({is_same_dt});")


def check_shape(
    buffer: IndentedBuffer, var: CSEVariableType, shape: BlockShapeType
) -> None:
    backend = get_current_backend()
    assert shape is not None
    if config.test_configs.runtime_triton_shape_assert and backend == "triton":
        shape_str = (
            ", ".join(str(d) for d in shape) if len(shape) != 1 else f"{shape[0]},"
        )
        buffer.writeline(f"tl.static_assert({var}.shape == ({shape_str}))")


def check_nan(buffer: IndentedBuffer, var: CSEVariableType) -> None:
    backend = get_current_backend()
    if backend == "triton":
        msg = "NaN or Inf found"
        buffer.writeline(
            f"tl.device_assert(({var} == {var}) & ({var} != float('inf')) & ({var} != float('-inf')), '{msg}')"
        )


class DataTypePropagation:
    def __init__(self, body: LoopBody) -> None:
        self.body = body
        self.graphs: dict[Union[Callable[..., Any], str], Any] = {
            "root": body.root_block.graph
        }
        for k, v in body.subblocks.items():
            self.graphs[k] = v.graph

    def deduce_node_dtype_by_inputs(self, node: torch.fx.Node) -> Optional[torch.dtype]:
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

    def deduce_node_dtype_by_subgraph(self, node: torch.fx.Node) -> torch.dtype:
        sub_graph = self.graphs[node.target]
        dtype = self.propagate_graph(sub_graph)
        assert dtype
        return dtype

    def deduce_node_dtype(self, node: torch.fx.Node) -> Optional[torch.dtype]:
        if node.op == "placeholder":
            return None

        if node.target == "output" and len(node.args) != 1:
            # we can infer output node if it only have 1 arg
            return None

        if node.target is operator.getitem:
            node_arg = node.args[0]
            assert isinstance(node_arg, torch.fx.Node), type(node_arg)
            return self.deduce_node_dtype(node_arg)

        assert isinstance(node.target, str), type(node.target)

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

    def propagate_graph(self, graph: torch.fx.Graph) -> Optional[torch.dtype]:
        assert graph.nodes
        graph_dtype: Optional[torch.dtype] = None
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

    def propagate(self) -> Optional[torch.dtype]:
        return self.propagate_graph(self.graphs["root"])

    @classmethod
    def propagate_loopbody(cls, body: LoopBody) -> Optional[torch.dtype]:
        return cls(body).propagate()

    @classmethod
    def propagate_scheduler_node(cls, node: SchedulerNode) -> Optional[torch.dtype]:
        from ..loop_body import LoopBody
        from ..scheduler import SchedulerNode

        assert isinstance(node, SchedulerNode), type(node)
        assert isinstance(node._body, LoopBody), type(node._body)
        return DataTypePropagation.propagate_loopbody(node._body)


class PythonPrinter(_PythonPrinter):
    def doprint(
        self, expr: sympy.Expr, *, simplify: bool = True, p: bool = True
    ) -> str:
        # TODO: why are people passing strings to the printer here :think:
        if simplify and isinstance(expr, sympy.Expr) and hasattr(V.graph, "sizevars"):
            expr = V.graph.sizevars.simplify(expr)
        return super().doprint(expr)

    def parenthesize(self, item: sympy.Expr, level: int, strict: bool = False) -> str:
        if isinstance(item, sympy.Mod):
            # use parenthesis to enforce precedence.
            # in sympy 1.13.3, -2*Mod(x,y) becomes -2*x%y, which is wrong.
            return f"({self._print(item)})"
        else:
            return super().parenthesize(item, level, strict)


class OpDecompositions:
    """
    Decomposes inductor ops
    """

    @staticmethod
    def identity(value: OpVarT) -> OpVarT:
        # used to trigger cse
        return value

    @staticmethod
    def reciprocal(x: OpVarT) -> OpVarT:
        return ops.truediv(ops.constant(1, torch.int32), x)

    @staticmethod
    def square(x: OpVarT) -> OpVarT:
        return ops.mul(x, x)

    @staticmethod
    def erfc(x: OpVarT) -> OpVarT:
        return ops.sub(ops.constant(1, torch.float32), ops.erf(x))

    @staticmethod
    def erfcx(x: OpVarT) -> OpVarT:
        return ops.mul(ops.exp(ops.square(x)), ops.erfc(x))

    @staticmethod
    def expm1(x: OpVarT) -> OpVarT:
        return ops.sub(ops.exp(x), ops.constant(1, torch.float32))

    @staticmethod
    def log10(x: OpVarT) -> OpVarT:
        return ops.mul(ops.log(x), ops.constant(1 / math.log(10), torch.float32))

    @staticmethod
    def log2(x: OpVarT) -> OpVarT:
        return ops.mul(ops.log(x), ops.constant(1 / math.log(2), torch.float32))

    @staticmethod
    def exp2(x: OpVarT) -> OpVarT:
        return ops.exp(ops.mul(x, ops.constant(math.log(2), torch.float32)))

    @staticmethod
    def log1p(x: OpVarT) -> OpVarT:
        return ops.log(ops.add(x, ops.constant(1, torch.int32)))

    @staticmethod
    def sigmoid(x: OpVarT) -> OpVarT:
        one = ops.constant(1, torch.int32)
        return ops.truediv(one, ops.add(one, ops.exp(ops.neg(x))))

    @staticmethod
    def relu(x: OpVarT) -> OpVarT:
        return ops.maximum(x, ops.constant(0, torch.int32))

    @staticmethod
    def fma(x: OpVarT, y: OpVarT, z: OpVarT) -> OpVarT:
        # for backends that don't override this (halide)
        return ops.add(ops.mul(x, y), z)

    @staticmethod
    def floor_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        return ops.to_dtype(ops.floor(a), dtype)

    @staticmethod
    def ceil_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        return ops.to_dtype(ops.ceil(a), dtype)

    @staticmethod
    def trunc_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        return ops.to_dtype(ops.trunc(a), dtype)

    @staticmethod
    def remainder(a: OpVarT, b: OpVarT) -> OpVarT:
        r = ops.mod(a, b)
        cond = ops.and_(
            ops.ne(r, ops.constant(0, torch.int32)),
            ops.ne(ops.signbit(r), ops.signbit(b)),
        )
        return ops.where(cond, ops.add(r, b), r)

    @staticmethod
    def round_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        return ops.to_dtype(ops.round(a), dtype)


_RE_PAREN_NOT_NEEDED = re.compile(r"[a-z0-9_.]+|\([^)]*\)|", flags=re.IGNORECASE)


def _all_in_parens(string: str) -> bool:
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


class OpOverrides(BasicMathOpsMixin, OpDecompositions, OpsHandler[Any]):
    @staticmethod
    def paren(string: OpVarT) -> OpVarT:
        if (
            isinstance(string, CSEVariable)
            or _RE_PAREN_NOT_NEEDED.fullmatch(string)
            or _all_in_parens(string)
        ):
            # don't put extra parens for strings that are already wrapped in parens
            # pyrefly: ignore [bad-return]
            return string
        return f"({string})"

    @staticmethod
    def constant(value: Union[bool, float, int], dtype: torch.dtype) -> OpVarT:
        return repr(value)

    @staticmethod
    def bitwise_not(x: OpVarT) -> OpVarT:
        return f"~{OpOverrides.paren(x)}"

    @staticmethod
    def logical_not(a: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(a)} == 0"

    @staticmethod
    def bitwise_and(x: OpVarT, y: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(x)} & {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_or(x: OpVarT, y: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(x)} | {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_xor(x: OpVarT, y: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(x)} ^ {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_left_shift(x: OpVarT, y: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(x)} << {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_right_shift(x: OpVarT, y: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(x)} >> {OpOverrides.paren(y)}"

    @staticmethod
    def int_truediv(a: OpVarT, b: OpVarT) -> OpVarT:
        # TODO: this is wrong
        # TODO: an easy bandaid is to generate runtime asserts that it's
        # <= 2**53, which is when this equation is correct
        return ops.truediv(a, b)

    @staticmethod
    def load_seed(name: str, offset: OpVarT) -> OpVarT:
        return ops.load(name, sympy.Integer(offset))

    def indirect_indexing(
        self,
        var: OpVarT,
        size: Union[sympy.Expr, int],
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        return sympy_index_symbol(str(var))

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: check_bounds should be handled by CSEProxy"
        )

    def load(self, name: str, index: sympy.Expr) -> OpVarT:
        raise NotImplementedError(
            f"{type(self).__name__}: load should be handled by CSEProxy"
        )

    def store(
        self, name: str, index: sympy.Expr, value: OpVarT, mode: StoreMode = None
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: store should be handled by CSEProxy"
        )

    def device_assert_async(self, cond: CSEVariable, msg: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: device_assert_async should be handled by CSEProxy"
        )

    def store_reduction(self, name: str, index: sympy.Expr, value: OpVarT) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: store_reduction should be handled by CSEProxy"
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[OpVarT, tuple[OpVarT, ...]],
    ) -> Union[OpVarT, tuple[OpVarT, ...]]:
        raise NotImplementedError(
            f"{type(self).__name__}: reduction should be handled by CSEProxy"
        )

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[OpVarT, ...], tuple[OpVarT, ...]],
            tuple[OpVarT, ...],
        ],
        values: tuple[OpVarT, ...],
    ) -> tuple[OpVarT, ...]:
        raise NotImplementedError(
            f"{type(self).__name__}: scan should be handled by CSEProxy"
        )

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[OpVarT, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[OpVarT, ...]:
        raise NotImplementedError(
            f"{type(self).__name__}: sort should be handled by CSEProxy"
        )

    def bucketize(
        self,
        values: OpVarT,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: OpVarT,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: Optional[tuple[str, sympy.Expr]] = None,
        sorter_indices: Optional[OpVarT] = None,
    ) -> OpVarT:
        raise NotImplementedError(
            f"{type(self).__name__}: bucketize should be handled by CSEProxy"
        )

    def halide_clamp(self, value: OpVarT, size: sympy.Expr, check: bool) -> OpVarT:
        raise NotImplementedError(
            f"{type(self).__name__}: halide_clamp only implemented for Halide backend"
        )

    def dot(self, x: OpVarT, y: OpVarT) -> OpVarT:
        raise NotImplementedError(
            f"{type(self).__name__}: dot only implemented for Triton backend"
        )

    def inline_asm_elementwise(
        self,
        *inputs: OpVarT,
        asm: str,
        constraints: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        is_pure: bool = True,
        pack: int = 1,
    ) -> OpVarT:
        raise NotImplementedError(
            f"{type(self).__name__}: inline_asm_elementwise only implemented for Triton backend"
        )

    def output(self, *args: OpVarT) -> None:
        raise AssertionError(
            f"{type(self).__name__}: ops.output should not appear at codegen time"
        )

    def placeholder(self, index: int) -> OpVarT:
        raise AssertionError(
            f"{type(self).__name__}: ops.placeholder should not appear at codegen time"
        )

    @staticmethod
    def _unimplemented(name: str) -> Callable[..., OpVarT]:
        def unimplemented(self: OpOverrides, *args: Any, **kwargs: Any) -> OpVarT:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement ops.{name}"
            )

        unimplemented.__name__ = name
        unimplemented.is_unimplemented = True  # type: ignore[attr-defined]
        return unimplemented

    @classmethod
    def _is_unimplemented(cls, name: str) -> bool:
        fn = getattr(cls, name, None)
        default_fn = getattr(OpsHandler, name, None)
        return not fn or fn == default_fn or getattr(fn, "is_unimplemented", False)

    @classmethod
    def _initialize_pointwise_overrides(cls, target: str) -> None:
        assert target in ("triton", "cpp", "cppvec", "halide", "mps"), target

        for funcname, data in pointwise_overrides_data.items():
            impl = getattr(data, target)
            if impl is None:
                if cls._is_unimplemented(funcname):
                    setattr(cls, funcname, cls._unimplemented(funcname))
            else:
                assert funcname not in cls.__dict__, (
                    f"multiple definitions of {funcname} on {cls.__name__}"
                )
                impl.__name__ = funcname
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
    halide: Optional[Callable[..., str]] = None
    mps: Optional[Callable[..., str]] = None


# NB: if you add a new special function, don't forget to update
# torch._inductor.ops_handler too
pointwise_overrides_data: dict[str, OverridesData] = dict(
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
        cpp=lambda x,
        y: f"{x} == 0 ? calc_digamma({y}) : ({x} == 1 ? trigamma({y}) : calc_polygamma({y}, {x}))",
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


def is_buffer_removed(name: str) -> bool:
    return any(
        name in x
        for x in (
            V.graph.removed_buffers,
            V.kernel.removed_buffers,
            V.graph.inplaced_to_remove,
            V.kernel.inplaced_to_remove,
        )
    )


class DeferredLine(DeferredLineBase):
    """A line that can be 'unwritten' by adding name to V.graph.removed_buffers"""

    def __init__(self, name: str, line: str):
        super().__init__(line)
        self.name = name
        assert not isinstance(line, DeferredLineBase)

    def __call__(self) -> Optional[str]:
        if not is_buffer_removed(self.name):
            return self.line
        return None

    def _new_line(self, line: str) -> DeferredLine:
        return DeferredLine(self.name, line)


class BracesBuffer(IndentedBuffer):
    def indent(self, offset: int = 1) -> contextlib.AbstractContextManager[None]:
        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
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
    other_names: list[str]


@dataclasses.dataclass
class ArgName:
    name: str
    # is_constexpr=True is used to attach a " : tl.constexpr" into the argument list
    is_constexpr: bool = False

    def full_name(self) -> str:
        return f"{self.name}{' : tl.constexpr' if self.is_constexpr else ''}"


class RemovedArg:
    def __str__(self) -> str:
        return "REMOVED"


REMOVED = RemovedArg()


class KernelArgs:
    @staticmethod
    def _lookup(
        prefix: str,
        odict: Union[dict[_T, Union[str, RemovedArg]], dict[_T, str]],
        name: _T,
    ) -> str:
        result: Union[str, RemovedArg] = odict.get(name, REMOVED)
        if isinstance(result, RemovedArg):
            odict[name] = new_result = f"{prefix}{len(odict)}"
            return new_result
        return result

    def __init__(self) -> None:
        self.input_buffers: dict[str, str] = {}
        self.output_buffers: dict[str, Union[str, RemovedArg]] = {}
        self.inplace_buffers: dict[str, Union[InplacedBuffer, RemovedArg]] = {}
        self.sizevars: dict[sympy.Expr, str] = {}
        self.workspace_args: list[WorkspaceArg] = []

    def __repr__(self) -> str:
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

    @staticmethod
    def _buffer_is_marked_removed(name: Any) -> bool:
        # this function is needed by MTIA
        return isinstance(name, RemovedArg)

    def input(self, name: str) -> str:
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        assert name not in V.graph.removed_buffers, name
        if name in self.output_buffers:
            return cast(str, self.output_buffers[name])
        if name in self.inplace_buffers:
            return cast(InplacedBuffer, self.inplace_buffers[name]).inner_name
        if name.startswith("seed"):
            return self._lookup("seed", self.input_buffers, name)
        return self._lookup("in_ptr", self.input_buffers, name)

    def output(self, name: str) -> str:
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        assert name not in V.graph.removed_buffers, name
        if name in self.inplace_buffers:
            return cast(InplacedBuffer, self.inplace_buffers[name]).inner_name
        return self._lookup("out_ptr", self.output_buffers, name)

    def make_inplace(self, input_name: str, output_name: str) -> None:
        if input_name in V.graph.unaligned_buffers:
            V.graph.unaligned_buffers.add(output_name)
        assert output_name not in self.inplace_buffers, output_name
        if input_name in self.inplace_buffers:
            buf = self.inplace_buffers[input_name]
            assert not isinstance(buf, RemovedArg)
            buf.other_names.append(output_name)
            self.inplace_buffers[output_name] = buf
        else:
            alive_buffers = [
                val
                for val in self.inplace_buffers.values()
                if not isinstance(val, RemovedArg)
            ]
            removed_buffers = [
                val
                for val in self.inplace_buffers.values()
                if isinstance(val, RemovedArg)
            ]
            inplace_buffer_idx = len(unique(alive_buffers)) + len(removed_buffers)
            buf = InplacedBuffer(
                f"in_out_ptr{inplace_buffer_idx}",
                [input_name, output_name],
            )
            self.inplace_buffers[input_name] = buf
            self.inplace_buffers[output_name] = buf

    def workspace(
        self, nelem: sympy.Expr, zero_fill: bool, dtype: torch.dtype = torch.uint8
    ) -> tuple[str, str, int]:
        """
        Allocate or extend a workspace buffer of nelem elements.

        This function manages the allocation of a workspace buffer. It either creates
        a new WorkspaceArg or extends an existing one.

        Note:
        - Calling this function will in-place mutate the args by adding or updating
        a WorkspaceArg.
        - The codegen for generating the Python argdefs and call_defs will check
        this field and allocate the buffer accordingly.
        - A new argument "ws_ptr" will be present in the generated code.

        Args:
            nelem (sympy.Expr): The number of elements to allocate.
            zero_fill (bool): Whether to initialize the buffer to zero.
            dtype (torch.dtype): the dtype of the workspace tensor

        Returns:
            Tuple[str, str, int]: A tuple containing:
                - "ws_ptr": A string identifier for the workspace pointer.
                - "workspace_{i}": agraph level unique identifier for
                    the workspace tensor.
                - offset: An integer representing the item offset in the workspace.
        """
        arg = WorkspaceArg(
            count=nelem,
            zero_mode=WorkspaceZeroMode.from_bool(zero_fill),
            device=V.graph.get_current_device_or_throw(),
            outer_name=WorkspaceArg.unique_name(),
            dtype=dtype,
        )
        for i, existing_arg in enumerate(self.workspace_args):
            if WorkspaceArg.can_join(existing_arg, arg):
                offset = existing_arg.count
                self.workspace_args[i] = WorkspaceArg.join(existing_arg, arg)
                return existing_arg.inner_name, existing_arg.outer_name, offset
            assert (
                existing_arg.inner_name != arg.inner_name
                and existing_arg.outer_name != arg.outer_name
            ), existing_arg
        self.workspace_args.append(arg)
        return arg.inner_name, arg.outer_name, 0

    def semaphores(self, min_size: sympy.Expr) -> str:
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
                assert arg == existing_arg, (arg, existing_arg)
        self.workspace_args.append(arg)
        return arg.inner_name

    def seed_offset(self, name: str, value: int) -> str:
        assert isinstance(value, int), (type(value), value)
        # here we are lifting a constant integer into an arg to the kernel to try to get additional cache hits
        value = sympy.Integer(value)
        if value in self.sizevars:
            return self.sizevars[value]
        if name in self.sizevars.values():
            name = (
                f"{name}{sum(1 for v in self.sizevars.values() if v.startswith(name))}"
            )
        self.sizevars[value] = name
        return name

    def size(self, name: sympy.Symbol) -> str:
        assert isinstance(name, sympy.Symbol), (type(name), name)
        if name.name == "seed":
            self.sizevars[name] = "seed"  # don't manage the name of seeds
            return "seed"
        return self._lookup("ks", self.sizevars, name)

    def call_names(self) -> Iterator[str]:
        return chain(
            self.input_buffers.keys(), self.output_buffers.keys(), self.sizevars.keys()
        )

    def arg_name(self, name: str) -> Optional[str]:
        """
        Returns inner name of a given outer name.
        """
        inplaced = self.inplace_buffers.get(name, None)
        if inplaced is not None and not isinstance(inplaced, RemovedArg):
            return inplaced.inner_name
        output_name = self.output_buffers.get(name, None)
        if output_name is not None and not isinstance(output_name, RemovedArg):
            return output_name
        return self.input_buffers.get(name, None)

    def wrap_ptr_arg(self, buf: str, dtype: torch.dtype) -> str:
        return buf

    def wrap_size_arg(self, size: SymbolLike) -> str:
        return str(size)

    def cpp_argdefs(
        self, dtype_to_cpp_type: Optional[dict[torch.dtype, str]] = None
    ) -> tuple[list[str], list[str], list[str]]:
        from .cpp_utils import INDEX_TYPE

        if dtype_to_cpp_type is None:
            from .cpp_utils import DTYPE_TO_CPP

            dtype_to_cpp_type = DTYPE_TO_CPP

        call_args = []
        arg_defs = []
        arg_types = []
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            outer = inplaced.other_names[-1]
            inner = inplaced.inner_name
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"{cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.input_buffers.items():
            if outer in self.inplace_buffers:
                continue
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"const {cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"const {cpp_dtype}*")
        for outer, maybe_inner in self.output_buffers.items():
            if outer in self.inplace_buffers or isinstance(maybe_inner, RemovedArg):
                continue
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"{cpp_dtype}* {maybe_inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.sizevars.items():
            if isinstance(outer, sympy.Symbol) and symbol_is_type(
                outer, (SymT.UNBACKED_FLOAT)
            ):
                arg_defs.append(f"const float {inner}")
                arg_types.append("const float")
            else:
                arg_defs.append(f"const {INDEX_TYPE} {inner}")
                arg_types.append(f"const {INDEX_TYPE}")
            call_args.append(self.wrap_size_arg(outer))
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)
        assert not self.workspace_args, "Workspace not supported on CPU "
        return arg_defs, call_args, arg_types

    def python_argdefs(
        self,
    ) -> tuple[list[ArgName], list[str], list[KernelArgType], list[Any]]:
        arg_defs: list[ArgName] = []
        call_args: list[str] = []
        arg_types: list[Any] = []
        precompile_args: list[KernelArgType] = []
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            arg_defs.append(ArgName(inplaced.inner_name))
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
            self.input_buffers.items(),
            # pyrefly: ignore [bad-argument-type]
            self.output_buffers.items(),
        ):
            if outer in self.inplace_buffers or isinstance(inner, RemovedArg):
                continue
            arg_defs.append(ArgName(inner))
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
            arg_defs.append(ArgName(inner))
            call_args.append(outer)
            arg_types.append(type(outer))
            precompile_args.append(SizeArg(inner, outer))
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)
        for arg in self.workspace_args:
            arg_defs.append(ArgName(arg.inner_name))
            call_args.append(arg.outer_name)
            precompile_args.append(arg)
            arg_types.append(arg.dtype)
        return arg_defs, call_args, precompile_args, arg_types

    def aliases(self) -> Iterator[tuple[str, str]]:
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
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
                    yield cast(str, self.output_buffers[other]), inplaced.inner_name

    def is_removed(self, name: str) -> bool:
        return isinstance(
            self.output_buffers.get(name, REMOVED), RemovedArg
        ) and isinstance(self.inplace_buffers.get(name, REMOVED), RemovedArg)

    # Includes inplace buffers, excludes removed buffers.  Essentially,
    # after you do a call into this kernel, which buffers actually contain
    # updated data?  Modeled off of python_argdefs.
    def live_output_buffers(self) -> OrderedSet[str]:
        live_outs: OrderedSet[str] = OrderedSet()
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            live_outs.add(inplaced.other_names[-1])
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers or isinstance(inner, RemovedArg):
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
        name: str,
        bounds: ValueRanges[Any],
        dtype: Optional[torch.dtype] = None,
        shape: BlockShapeType = None,
    ):
        super().__init__()
        assert isinstance(bounds, ValueRanges), type(bounds)
        self.name = name
        self.bounds = bounds
        self.use_count = 1  # track how many times this expression is used
        self.dtype = dtype
        self.shape = shape

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CSEVariable) and other.name == self.name

    def update_on_args(self, name: str, args: Any, kwargs: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


AugmentedKeyT = TypeVar("AugmentedKeyT", default=str)
CSEVariableType = TypeVar("CSEVariableType", bound=CSEVariable, default=CSEVariable)

if TYPE_CHECKING:
    ReductionCacheKey = tuple[
        torch.dtype,
        ReductionType,
        Union[CSEVariable, tuple[CSEVariable, ...]],
    ]


class CSE(Generic[CSEVariableType, AugmentedKeyT]):
    """Common subexpression elimination"""

    def __init__(
        self,
        prefix: str = "",
        suffix: str = "",
        name_prefix: str = "tmp",
        iter_buffers: Optional[itertools.count[int]] = None,
        store_cache: Optional[MutableMapping[str, CSEVariableType]] = None,
        reduction_cache: Optional[
            MutableMapping[ReductionCacheKey, CSEVariableType]
        ] = None,
        varname_map: Optional[dict[str, CSEVariableType]] = None,
    ):
        self.prefix = prefix
        self.suffix = suffix
        self._cache: MutableMapping[AugmentedKeyT, CSEVariableType] = {}
        self.name_prefix = name_prefix
        self.store_cache: MutableMapping[str, CSEVariableType] = store_cache or {}
        self.reduction_cache: MutableMapping[ReductionCacheKey, CSEVariableType] = (
            reduction_cache or {}
        )
        self.iter_buffer_ids: itertools.count[int] = iter_buffers or itertools.count()
        self.invalidated_stores: OrderedSet[str] = OrderedSet()
        self.varname_map: dict[str, CSEVariableType] = varname_map or {}

    def invalidate(self, keep_vars: OrderedSet[CSEVariable]) -> None:
        for name, tmp in [*self.store_cache.items()]:
            if tmp not in keep_vars:
                del self.store_cache[name]
                self.invalidated_stores.add(name)
        if keep_vars:
            self._cache = {k: v for k, v in self._cache.items() if v in keep_vars}
        else:
            self._cache = {}

    def clone(self) -> Self:
        return type(self)(
            prefix=self.prefix,
            suffix=self.suffix,
            name_prefix=self.name_prefix,
            iter_buffers=self.iter_buffer_ids,
            store_cache=self.store_cache,
            varname_map=self.varname_map,
            reduction_cache=self.reduction_cache,
        )

    def scoped_copy(self) -> Self:
        """Return a copy of using ScopedDict so changes to *_cache aren't visible in self"""
        new_cse = self.clone()
        new_cse._cache = ScopedDict(self._cache)
        new_cse.reduction_cache = ScopedDict(self.reduction_cache)
        new_cse.store_cache = ScopedDict(self.store_cache)
        return new_cse

    def augment_key(self, cache_key: str) -> AugmentedKeyT:
        "Override this method to augment cache key with backend specifics"
        return cast(AugmentedKeyT, cache_key)

    def put(self, cache_key: str, val: CSEVariableType) -> None:
        self._cache[self.augment_key(cache_key)] = val

    def contains(self, cache_key: str) -> bool:
        return self.augment_key(cache_key) in self._cache

    def try_get(self, cache_key: str) -> Optional[CSEVariableType]:
        return self._cache.get(self.augment_key(cache_key), None)

    def get(self, cache_key: str) -> CSEVariableType:
        return self._cache[self.augment_key(cache_key)]

    def generate(
        self,
        buffer: IndentedBuffer,
        expr: Union[str, CSEVariable, OpsValue, IndentedBuffer, DeferredLineBase],
        *,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        write: bool = True,
        assignment: bool = True,
        dtype: Optional[torch.dtype] = None,
        shape: BlockShapeType = None,
    ) -> CSEVariableType:
        if isinstance(expr, OpsValue):
            expr = expr.value

        assert write or assignment
        if isinstance(expr, CSEVariable):
            # If the expressions were always created with all the information, we could
            # assert expr.bounds == bounds, but sometimes the expression is created
            # with the loose ValueRanges.unknown(), so we need to tighten the bounds
            expr.bounds = expr.bounds.tighten(bounds)
            expr.use_count += 1
            return cast(CSEVariableType, expr)
        elif isinstance(expr, IndentedBuffer):
            cache_key = expr.getvalue()
        elif isinstance(expr, DeferredLineBase):
            cache_key = expr.line
        else:
            assert isinstance(expr, str)
            cache_key = expr
        var = self.try_get(cache_key)
        if shape is None and not assignment:
            # since there's no assignment to a variable, use any shape here
            # other than None to avoid the unknown shape failures
            shape = ()
        if not var:
            var = self.newvar(bounds, dtype, shape)
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

                    # cpp backend cannot determine is_vec at this point
                    if (
                        assignment
                        and (
                            config.test_configs.runtime_triton_dtype_assert
                            or config.test_configs.static_cpp_dtype_assert
                        )
                        and dtype is not None
                        and get_current_backend() != "cpp"
                    ):
                        check_dtype(buffer, var, dtype)

        else:
            var.bounds = var.bounds.tighten(bounds)
            var.use_count += 1

        return var

    def newvar(
        self,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        dtype: Optional[torch.dtype] = None,
        shape: BlockShapeType = None,
    ) -> CSEVariableType:
        var_name = f"{self.name_prefix}{next(self.iter_buffer_ids)}"
        var = V.kernel.create_cse_var(var_name, bounds, dtype, shape)
        self.varname_map[var_name] = var
        return var

    def namedvar(
        self,
        name: str,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        dtype: Optional[torch.dtype] = None,
        shape: BlockShapeType = None,
    ) -> CSEVariableType:
        torch._check_value(
            name not in self.varname_map, lambda: f"duplicate name: {name}"
        )
        var = V.kernel.create_cse_var(name, bounds, dtype, shape)
        self.varname_map[name] = var
        return var


class CodeGen:
    def __init__(self) -> None:
        super().__init__()
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self) -> Self:
        self.exit_stack.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)


class Kernel(CodeGen, Generic[CSEVariableType]):
    newvar_prefix: str = ""
    suffix: str = ""
    overrides: Optional[Callable[[], OpsHandler[Any]]] = None

    def __init__(
        self, args: Optional[KernelArgs] = None, increase_kernel_count: bool = True
    ) -> None:
        super().__init__()
        if increase_kernel_count:
            # pyrefly: ignore [bad-assignment]
            metrics.generated_kernel_count += 1
        self.args = args or KernelArgs()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()

        self.atomic_add_found = False
        self.num_load = 0
        self.num_store = 0
        self.num_reduction = 0

        self.cse: CSE[CSEVariableType, Any] = CSE(self.newvar_prefix, self.suffix)
        self.must_keep_buffers: OrderedSet[str] = OrderedSet()
        self.store_buffer_names: OrderedSet[str] = OrderedSet()
        self._load_mask: Optional[str] = None
        self._load_other: Union[None, int, float] = None
        # OrderedSet in set_current_node
        self.current_node: Optional[SchedulerNode] = None
        self.node_to_bounds: Optional[dict[torch.fx.Node, ValueRanges[Any]]] = None

        self.removed_buffers: OrderedSet[str] = OrderedSet()
        self.inplaced_to_remove: OrderedSet[str] = OrderedSet()

        # key: the buffer to write
        # value: the buffer to read and whose memory can be reused for
        #   the buffer specified by key
        self.inplace_update_buffers: dict[str, str] = {}
        # Set minimum number of elements processed per thread.
        self.min_elem_per_thread = 1
        self.kernel_name: Optional[str] = None

    @contextlib.contextmanager
    def set_current_node(self, node: SchedulerNode) -> Iterator[None]:
        prior = self.current_node
        self.current_node = node
        self.node_to_bounds = node._body.bounds().get_bounds()
        try:
            yield
        finally:
            self.current_node = prior

    @contextlib.contextmanager
    def swap_buffers(
        self,
        lb: IndentedBuffer,
        cb: Optional[IndentedBuffer] = None,
        sb: Optional[IndentedBuffer] = None,
    ) -> Iterator[None]:
        if cb is None:
            cb = lb
        if disallow_stores := sb is None:
            sb = IndentedBuffer()
        loads = self.loads
        compute = self.compute
        stores = self.stores
        cse = self.cse
        self.loads = lb
        self.compute = cb
        self.stores = sb
        self.cse = cse.scoped_copy()
        try:
            yield
        finally:
            self.loads = loads
            self.compute = compute
            self.stores = stores
            self.cse = cse
            # pyrefly: ignore [unbound-name]
            if disallow_stores:
                assert not sb, "unexpected store inside swap_buffers"

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        raise NotImplementedError

    def indirect_load(self, name: str, index: sympy.Expr) -> CSEVariable:
        """A load the depends on an index we have read"""
        prior = self.loads
        try:
            # put the load in the compute section as it might have deps
            self.loads = self.compute
            return self.load(name, index)
        finally:
            self.loads = prior

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        raise NotImplementedError

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        raise NotImplementedError

    def device_assert_async(self, cond: CSEVariable, msg: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: device_assert_async should be handled by CSEProxy"
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:
        raise NotImplementedError

    def partial_accumulate(
        self,
        name: str,
        reduction_type: ReductionType,
        value: CSEVariable,
        extra_meta: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[CSEVariable, ...], tuple[CSEVariable, ...]], tuple[CSEVariable, ...]
        ],
        values: tuple[CSEVariable, ...],
    ) -> tuple[CSEVariable, ...]:
        raise NotImplementedError

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[CSEVariable, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[CSEVariable, ...]:
        raise NotImplementedError

    def var_ranges(self) -> dict[sympy.Symbol, sympy.Expr]:
        raise NotImplementedError

    def bucketize(
        self,
        values: CSEVariable,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: CSEVariable,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: Optional[tuple[str, sympy.Expr]] = None,
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
        assert isinstance(var, str), type(var)
        assert lower is None or isinstance(lower, str)
        assert upper is None or isinstance(upper, str)
        if lower and upper:
            # The conditions need to be in parens because of Python's operator precedence.
            # It'd be less error-prone to use and/or/not, which is supported by triton
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
    ) -> None:
        raise NotImplementedError

    def index_to_str(self, index: sympy.Expr) -> str:
        raise NotImplementedError

    def __enter__(self) -> Self:
        super().__enter__()
        assert self.overrides
        self.exit_stack.enter_context(
            V.set_ops_handler(CSEProxy(self, self.overrides()))
        )
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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
            scheduler.name_to_buf[buf].defining_op_name()
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
                self.num_store -= 1
                names_to_remove.add(name)

        for name in names_to_remove:
            if name in self.args.inplace_buffers:
                buf = self.args.inplace_buffers[name]
                if isinstance(buf, RemovedArg):
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
        self.args.output_buffers[name] = REMOVED
        self.removed_buffers.add(name)

    def remove_inplace_buffer(self, name: str) -> None:
        log.debug("removing_inplace_buffer(%r)", name)
        self.args.inplace_buffers[name] = REMOVED
        self.removed_buffers.add(name)

    def rename_indexing(
        self, index: Union[list[sympy.Expr], tuple[sympy.Expr, ...], sympy.Expr]
    ) -> sympy.Expr:
        # adds the necessary kernel args for index expressions
        # and renames variables in index expressions to kernel arg names
        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]
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
                    SymT.UNBACKED_FLOAT,
                ),
            )
        }
        return sympy_subs(index, replacements)

    def create_cse_var(self, *args: Any, **kwargs: Any) -> CSEVariable:
        return CSEVariable(*args, **kwargs)

    def arg_name(self, node: IRNode) -> Optional[str]:
        """
        Returns arg name of a given input or output node.
        """
        if node is None:
            return None
        return self.args.arg_name(node.get_name())


@dataclasses.dataclass
class OptimizationContext:
    key: ClassVar[str] = "opt_ctx"

    dtype: Optional[torch.dtype] = None
    ops_name: str = ""


@functools.cache
def jinja2_env() -> Any:
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

    Children classes: TritonTemplate, CUTLASSTemplate
    """

    @staticmethod
    def indent_except_first(
        source: str, num_indents: int, indents_spacing: int = 4
    ) -> str:
        lines = source.splitlines(True)
        if len(lines) > 1:
            lines[1:] = [
                (" " * indents_spacing * num_indents) + line for line in lines[1:]
            ]
        return "".join(lines)

    @staticmethod
    def _template_from_string(source: str) -> Any:
        env = jinja2_env()
        if env is None:
            return None
        env.filters["indent_except_first"] = KernelTemplate.indent_except_first
        from jinja2 import TemplateSyntaxError

        try:
            return env.from_string(source)
        except TemplateSyntaxError as e:

            class DetailedTemplateSyntaxError(TemplateSyntaxError):
                def __init__(self, original_error: TemplateSyntaxError) -> None:
                    super().__init__(
                        # pyrefly: ignore [bad-argument-type]
                        original_error.message,
                        original_error.lineno,
                        original_error.name,
                        original_error.filename,
                    )
                    self.original_error = original_error

                def __str__(self) -> str:
                    error_info = f"Error in template at line {self.lineno}\n"
                    error_info += f"Error message: {self.message}\n"
                    if hasattr(self.original_error, "source"):
                        # pyrefly: ignore [missing-attribute]
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

            raise DetailedTemplateSyntaxError(e) from e

    @staticmethod
    def _fake_get_dtype(
        fake_outs: Union[list[Buffer], Buffer],
    ) -> Callable[[str], torch.dtype]:
        _get_dtype_real = V.graph.get_dtype
        if isinstance(fake_outs, (list, tuple)):
            lookup = {buf.get_name(): buf.get_dtype() for buf in fake_outs}
        else:
            lookup = {fake_outs.get_name(): fake_outs.get_dtype()}

        def get_dtype(name: str) -> torch.dtype:
            result = lookup.get(name)
            if result is not None:
                return result
            return _get_dtype_real(name)

        return get_dtype

    def __init__(self, name: str, hash: Optional[str] = None) -> None:
        self.name = name
        self._hash = hash

    @property
    def uid(self) -> str:
        """
        entry point to override for templates to ensure a uid e.g. through a prefix

        the purpose of this is that every KernelTemplate/ExternKernelChoice is unique
        in the system, but reproducible e.g. restarting pytorch should yield the same id
        """
        # TODO(coconutruben): add some central registration to assert on global uniqueness
        return self.name

    @property
    def src_hash(self) -> Union[str, None]:
        """
        source hash for a Template.

        Templates can optionally provide a src hash to make it easier to cache/validate that
        a template has not changed from one version to another. Override this if that detection
        is different for your specific Template
        """
        return self._hash

    def choice_or_none(self, **kwargs: Any) -> Optional[ChoiceCaller]:
        """
        Maybe generates a new ChoiceCaller and returns it, or None if generation fails.

        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """
        temp_choices: list[Any] = []
        result = self.maybe_append_choice(temp_choices, **kwargs)
        if result is None and len(temp_choices) == 1:
            return temp_choices[0]
        return None

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> Optional[NotImplementedError]:
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
            log.info(  # noqa: G200
                "Cannot Append Choice: %s. KernelTemplate type is %s",
                e,
                type(self),
                stack_info=log.getEffectiveLevel() < logging.INFO,
            )
            return e

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        """
        Generates a ChoiceCaller instance from the given arguments.
        """

        raise NotImplementedError


class CSEProxy(DefaultHandler):
    """A ops handler that proxies calls to `kernel` and its
    handler and returns `CSEVariable`s with correct shape and dtype.
    """

    name = "CSEProxy"

    def __init__(self, kernel: Kernel[Any], parent_handler: OpsHandler[Any]):
        super().__init__()
        from ..bounds import ValueRangeAnalysis

        self.vr_analysis = ValueRangeAnalysis()
        self.kernel = kernel
        self.parent_handler = parent_handler

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bounds = self._bound_variable(name, *args, **kwargs)

        value = getattr(self.parent_handler, name)(*args, **kwargs)
        dtype_handler = DtypePropagationOpsHandler()
        shape_handler = ShapePropagationOpsHandler()

        backend = get_current_backend()

        shape_op = getattr(shape_handler, name)
        output_dtype = None
        output_shape = None

        if name == "masked" and backend == "triton":
            output_dtype = value.dtype
            output_shape = value.shape
        elif name == "masked" and backend == "cpp":
            output_dtype = V.interpreter.current_node.meta.get(
                OptimizationContext.key, None
            ).dtype
            # TODO: fix me
            output_shape = None
        elif backend in ("triton", "cpp", "mps"):
            dtype_op = getattr(dtype_handler, name)
            output_dtype = dtype_op(*args, **kwargs)
            output_shape = shape_op(*args, **kwargs)

        if backend in ("triton", "cpp"):
            # maybe there are some exceptions on mps?
            assert output_dtype is not None

        output_idx = 0

        def do_cse(v: Union[str, CSEVariable]) -> CSEVariable:
            # we tree_map over the output, so we need to fetch corresponding dtype
            nonlocal output_idx
            var_dtype: Optional[torch.dtype] = (
                output_dtype[output_idx]
                if isinstance(output_dtype, (list, tuple))
                else output_dtype
            )
            var_shape: BlockShapeType = (
                output_shape[output_idx]  # type: ignore[assignment]
                if isinstance(output_shape, (list, tuple))
                and len(output_shape) > 0
                and isinstance(output_shape[0], (list, tuple))
                else output_shape
            )
            output_idx += 1

            # some cpp op implementations don't set the dtype
            if isinstance(v, CSEVariable):
                if backend == "cpp" and v.dtype is None:
                    v.dtype = var_dtype
                if v.shape is None:
                    v.shape = var_shape

            csevar = V.kernel.cse.generate(
                V.kernel.compute,
                v,
                bounds=bounds,
                dtype=output_dtype,
                shape=output_shape,
            )

            csevar.update_on_args(name, args, kwargs)

            if (
                config.test_configs.runtime_triton_dtype_assert
                or config.test_configs.static_cpp_dtype_assert
            ):
                assert var_dtype is not None
                check_dtype(V.kernel.compute, csevar, var_dtype)

            if config.test_configs.runtime_triton_shape_assert:
                assert output_shape is not None
                check_shape(V.kernel.compute, csevar, output_shape)

            if config.runtime_triton_nan_asserts:
                check_nan(V.kernel.compute, csevar)

            return csevar

        return pytree.tree_map(do_cse, value)

    def _bound_variable(self, name: str, *args: Any, **kwargs: Any) -> ValueRanges[Any]:
        """
        If the variable comes from an FX node, we forward the bound we have already computed
        Else, if the variable when codegen'ing another op, we try to compute its bounds
        """
        from ..bounds import ValueRangeAnalysis
        from ..select_algorithm import TritonTemplateKernel
        from .cutlass.kernel import CUTLASSTemplateKernel

        if isinstance(V.kernel, TritonTemplateKernel):
            return ValueRanges.unknown()

        if isinstance(V.kernel, CUTLASSTemplateKernel):
            return ValueRanges.unknown()

        if isinstance(V.interpreter, NullHandler):
            return ValueRanges.unknown()

        fx_node = V.interpreter.current_node
        if fx_node.target == name and self.kernel.node_to_bounds is not None:
            assert isinstance(self.kernel.node_to_bounds, dict), type(
                self.kernel.node_to_bounds
            )
            return self.kernel.node_to_bounds.get(fx_node, ValueRanges.unknown())
        elif config.compute_all_bounds and hasattr(ValueRangeAnalysis, name):
            # These create lots of inner strings. We would need to compute the bounds at the ops
            # We will also likely not get much from computing VRs on these nodes
            if any(s in fx_node.target for s in ("set_indirect", "reduction", "scan")):
                return ValueRanges.unknown()

            # We assume that the inputs come from `ops.` and are not strings. If you want to generate
            # intermediary strings, wrap them in CSE variables with properly initialised bounds.

            # If there is no FX bound but we know how to compute one we do so
            assert not kwargs

            def arg_to_bound(x: Any) -> Any:
                if isinstance(x, CSEVariable):
                    return x.bounds
                elif isinstance(x, sympy.Expr):
                    return bound_sympy(x)
                else:
                    return x

            arg_bounds = list(map(arg_to_bound, args))
            return getattr(self.vr_analysis, name)(*arg_bounds)
        return ValueRanges.unknown()

    def indirect_indexing(
        self,
        var: CSEVariable,
        size: Union[sympy.Expr, int],
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        if isinstance(size, int):
            size = sympy.Integer(size)
        assert isinstance(size, sympy.Expr), (type(size), size)
        # Skip CSE since this doesn't return an expression

        if var.bounds.lower < 0:
            if wrap_neg:
                stm = ops.add(var, ops.index_expr(size, torch.long))
                # Mixed negative and non-negative
                if var.bounds.upper >= 0:
                    lt = ops.lt(var, 0)
                    stm = ops.where(lt, stm, var)
            else:
                stm = var

            # Propagate bounds as we know how to compute them properly
            new_bounds = ValueRanges.unknown()
            if var.bounds != ValueRanges.unknown() and isinstance(size, sympy.Number):
                # Take the negative part of the bound and add size to it
                # Then take union of that and the positive part
                # This is a tighter bound than that of a generic ops.where, as we have info on the cond
                neg_bounds = var.bounds & ValueRanges(-int_oo, -1)
                new_bounds = ValueRanges(
                    neg_bounds.lower + size, neg_bounds.upper + size
                )
                # We don't have a good way of representing the empty range
                if var.bounds.upper >= 0:
                    pos = var.bounds & ValueRanges(0, int_oo)
                    new_bounds = new_bounds | pos

            var = self.kernel.cse.generate(
                self.kernel.compute,
                stm,
                bounds=new_bounds,
                dtype=var.dtype,
                shape=var.shape,
            )

        sympy_var = self.parent_handler.indirect_indexing(var, size, check)
        if generate_assert(check):
            assert_lower = not (var.bounds.lower >= 0)
            # value ranges cannot x < s when x and s are symbols
            assert_upper = not isinstance(size, sympy.Number) or not (
                var.bounds.upper < size
            )
            self.kernel.check_bounds(sympy_var, size, assert_lower, assert_upper)
        return sympy_var

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        return self.kernel.check_bounds(expr, size, lower, upper)

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        if name in self.kernel.cse.invalidated_stores:
            # A load from an invalidated store requires us to
            # keep the actual buffer around
            V.kernel.must_keep_buffers.add(name)
        if free_symbol_is_type(index, SymT.TMP):
            return self.kernel.indirect_load(name, index)
        store_cache = self.kernel.cse.store_cache
        if name in store_cache:
            return store_cache[name]
        out = self.kernel.load(name, index)
        # count load that is not in the store_cache, and also not in the
        # cse cache.
        if out.use_count == 1:
            self.kernel.num_load += 1
        return out

    def _update_store_cache(self, name: str, value: CSEVariable) -> None:
        self.kernel.cse.store_cache[name] = value
        if self.kernel.current_node and name in V.graph.name_to_buffer:
            buf = self.kernel.current_node.get_output(name)
            for other_name in buf.get_mutations():
                self.kernel.cse.store_cache[other_name] = value

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        self.kernel.store_buffer_names.add(name)
        if mode is None:
            self._update_store_cache(name, value)
        if name not in V.graph.removed_buffers:
            self.kernel.store(name, index, value, mode=mode)
            self.kernel.num_store += 1

    def device_assert_async(self, cond: CSEVariable, msg: str) -> None:
        self.kernel.device_assert_async(cond, msg)

    # pyrefly: ignore [bad-override]
    def partial_accumulate(self, *args: Any) -> None:
        self.kernel.partial_accumulate(*args)

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        self.kernel.store_buffer_names.add(name)
        self._update_store_cache(name, value)

        if name not in V.graph.removed_buffers:
            self.kernel.num_store += 1
            return self.kernel.store_reduction(name, index, value)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:
        self.kernel.num_reduction += 1
        return self.kernel.reduction(dtype, src_dtype, reduction_type, value)

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[CSEVariable, ...], tuple[CSEVariable, ...]],
            tuple[CSEVariable, ...],
        ],
        values: tuple[CSEVariable, ...],
    ) -> tuple[CSEVariable, ...]:
        return self.kernel.scan(dtypes, combine_fn, values)

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[CSEVariable, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[CSEVariable, ...]:
        return self.kernel.sort(dtypes, values, stable, descending)

    def bucketize(
        self,
        values: CSEVariable,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: CSEVariable,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: Optional[tuple[str, sympy.Expr]] = None,
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
        return self.kernel.bucketize(
            values,
            boundaries,
            boundary_indices,
            indexing_dtype,
            right,
            sorter,
            sorter_indices,
        )
