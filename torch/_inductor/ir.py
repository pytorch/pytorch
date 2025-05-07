from __future__ import annotations

import contextlib
import dataclasses
import functools
import itertools
import logging
import textwrap
import traceback
import typing
from collections.abc import Generator, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from enum import Enum
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    overload,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import assert_never, Never, TypeAlias
from unittest.mock import patch

import sympy
from sympy import Expr, Integer, Symbol

import torch._export.serde.schema as export_schema
import torch._library.utils as library_utils
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._higher_order_ops.auto_functionalize import can_auto_functionalize
from torch._inductor import metrics
from torch._prims_common import (
    compute_required_storage_length,
    is_boolean_dtype,
    is_float_dtype,
    make_channels_last_strides_for,
    StrideType,
)
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import (
    _remove_effect_token_unbacked_bindings,
    compute_unbacked_bindings,
    free_symbols,
    free_unbacked_symbols,
    IterateExprs,
    rebind_unbacked,
    resolve_unbacked_bindings,
    ShapeEnv,
    statically_known_true,
    SymTypes,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import SymT

from . import config, dependencies
from .codegen.common import (
    BackendFeature,
    CodegenSymbol,
    get_scheduling_for_device,
    index_prevent_reordering,
)
from .dependencies import (
    Dep,
    extract_free_symbols,
    extract_input_node_reduction_ranges,
    extract_read_writes,
    var_builder,
)
from .loop_body import LoopBody
from .ops_handler import OpCounterCSE, OpCountResult, ReductionType, StoreMode
from .runtime.benchmarking import benchmarker
from .runtime.hints import DeviceProperties, ReductionHint
from .utils import (
    argsort,
    argsort_sym,
    cache_on_self,
    ceildiv,
    convert_shape_to_inductor,
    convert_shape_to_symint,
    developer_warning,
    get_dtype_size,
    get_kernel_metadata,
    GPU_ALIGN_BYTES,
    ir_dataclass,
    is_dynamic,
    is_gpu,
    sympy_dot,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    sympy_product,
    sympy_subs,
    tensor_is_aligned,
)
from .virtualized import ops, OpsValue, V


if TYPE_CHECKING:
    from torch._library.fake_class_registry import FakeScriptObject
    from torch.fx.node import Node

    from .codegen.cuda.cuda_template import CUDATemplate
    from .graph import GraphLowering
    from .utils import IndentedBuffer

else:
    CUDATemplate: TypeAlias = object


try:
    import triton

    triton_version = triton.__version__
    has_triton = True
except ImportError:
    triton_version = None
    has_triton = False


_T = TypeVar("_T")
_U = TypeVar("_U")
_V = TypeVar("_V")

_IntLike: TypeAlias = Union[int, Expr]
_NumLike: TypeAlias = Union[int, float, Expr]

log = logging.getLogger(__name__)
indent = functools.partial(textwrap.indent, prefix="  ")
aten = torch.ops.aten

""" [Note: Inductor IR]

Inductor's IR is produced by executing 'lowering' code (see lowering.py).  Each
lowering is registered to a particular aten operator, and expects inputs that
correspond to the aten schema.  However, in place of torch Tensor inputs, lowerings
expect Inductor TensorBox inputs.

TensorBox IR represents torch tensors.  Tensors are sometimes single objects owning
storage, and sometimes views of another Tensor's storage.  Mutating tensor operations
(such as add_()) affect the underlying storage and any associated views.  Other operations
(such as .t_()) update metadata about the current view but don't modify the underlying storage.

To model this in Inductor, the IR distinguishes between TensorBox, View, StorageBox and Buffer.

TensorBox is the top level IR construct that any lowering should produce and maps to a torch.Tensor
output from an operation.  But just as torch.Tensors take different forms, TensorBox IR can
reference View IR or directly reference StorageBox IRs.

Some Inductor lowerings produce new sets of 'Box'es, while others (such as .t() or other view ops)
may take an existing TensorBox and point it to a new underlying View IR.

Tensors that directly own storage are represented as a chain of:
TensorBox -> StorageBox -> Buffer
where Buffer is a simple (1D) allocation, and StorageBox introduces the concept of a Layout.

If you mutate the data of such a tensor, we swing the StorageBox pointer to point to a new buffer
(leaving the old buffer unmodified and functionalizing the operation).

Tensors backed by views add one more indirection to the IR.
TensorBox -> View -> StorageBox -> Buffer
In these cases, the underlying StorageBox/Buffer will be shared with the pre-view TensorBox.

Computation is represented by Operation nodes, with each operation producing 1
or more output Buffers. In the case of mutations, these will be new Buffers that have the
mutated buffer listed in its get_mutation_names().

It is also possible to have an InputBuffer for which there is no corresponding Operation,
e.g. it may be a graph input or compile time constant.

"""


_NodeOrNodes: TypeAlias = Union[
    int,
    "TensorBox",
    dict[str, "TensorBox"],
    "Symbol",
    "IRNode",
    Sequence[
        Optional[Union[int, dict[str, "TensorBox"], "TensorBox", "Symbol", "IRNode"]]
    ],
]


@dataclasses.dataclass(frozen=True)
class GraphPartitionSignature:
    # symbol inputs that are neccessary for codegen
    symbol_inputs: OrderedSet[sympy.Symbol]

    # mapping from partition input name to IRNode or Expr. Need the name str since
    # we cannot get name from Expr.
    input_nodes: dict[str, Union[IRNode, sympy.Expr, TorchBindObject]]
    output_nodes: list[IRNode]

    # mapping from partition input name to a boolean for whether deallocating it
    # in the partition function
    input_deallocation: dict[str, bool]
    skip_cudagraph: bool

    # name of constants read/written by the graph partition
    constant_names: list[str]


def validate_ir(node_or_nodes: Optional[_NodeOrNodes]) -> None:
    def _check_tensorbox(nodes: Optional[_NodeOrNodes]) -> None:
        # Could expand this to check deeper properties
        # (e.g. TensorBox points to View or StorageBox)
        if nodes is None:
            pass
        elif isinstance(nodes, (list, tuple)):
            for node in nodes:
                _check_tensorbox(node)
        elif isinstance(nodes, dict):
            for node in nodes.values():
                _check_tensorbox(node)
        else:
            assert isinstance(
                nodes,
                (
                    ExpandView,
                    DynamicScalar,
                    AssertScalar,
                    TensorBox,
                    sympy.logic.boolalg.Boolean,
                    Expr,
                    int,
                    EffectfulKernel,
                    ShapeAsConstantBuffer,
                ),
            ), (
                f"Found {type(nodes)}, which is not a supported top level IR node. See [Note: Inductor IR]"
            )

    # Be picky about the accepted data structure (don't use pytree here)
    _check_tensorbox(node_or_nodes)


def ops_wrapper(name: str) -> Callable[..., OpsValue]:
    assert isinstance(name, str)

    def fn(*args: object, **kwargs: object) -> OpsValue:
        return getattr(ops, name)(*args, **kwargs)

    return fn


def inverse_reorder(order: Sequence[int]) -> Callable[[Sequence[_T]], Sequence[_T]]:
    inv_order = dict(zip(order, range(len(order))))

    def reindex(index: Sequence[_T]) -> Sequence[_T]:
        assert len(index) == len(inv_order)
        return [index[inv_order[i]] for i in range(len(index))]

    return reindex


def same_reorder(order: Sequence[int]) -> Callable[[Sequence[_T]], Sequence[_T]]:
    def reindex(index: Sequence[_T]) -> Sequence[_T]:
        assert len(index) == len(order)
        return [index[order[i]] for i in range(len(index))]

    return reindex


def fuse_reindexing(
    reindex1: Callable[[Sequence[_U]], Sequence[_V]],
    reindex2: Callable[[Sequence[_T]], Sequence[_U]],
) -> Callable[[Sequence[_T]], Sequence[_V]]:
    def reindex(index: Sequence[_T]) -> Sequence[_V]:
        return reindex1(reindex2(index))

    return reindex


def get_free_symbols(x: IterateExprs, unbacked_only: bool) -> OrderedSet[sympy.Symbol]:
    if unbacked_only:
        return free_unbacked_symbols(x)
    else:
        return free_symbols(x)


NHWC_STRIDE_ORDER = [3, 0, 2, 1]
NHWDC_STRIDE_ORDER = [4, 0, 3, 2, 1]


def get_fill_order(
    seq: Sequence[Union[int, torch.SymInt, Expr]], shape_env: Optional[ShapeEnv] = None
) -> Sequence[int]:
    """
    Convert strides to fill order (argsort)
    """
    if shape_env is None:
        sorted_idx: Sequence[int] = argsort(seq)
    else:
        # argsort_sym handles unbacked symints (with the help of the shape_env)
        sorted_idx = argsort_sym(shape_env, seq)
    return sorted_idx


def stride_order2fill_order(order: Sequence[Union[int, Integer]]) -> Sequence[int]:
    """
    Convert stride order to fill order
    For channel last format,

    stride order = [3, 0, 2, 1] and fill order = [1, 3, 2, 0]
    """
    lookup = {pos: idx for idx, pos in enumerate(order)}
    fill_order = [lookup[i] for i in range(len(order))]
    return fill_order


def get_stride_order(
    seq: Sequence[Union[int, torch.SymInt, Expr]], shape_env: Optional[ShapeEnv] = None
) -> Sequence[int]:
    """
    Convert strides to stride order
    """
    sorted_idx: Sequence[int] = get_fill_order(seq, shape_env)
    out = [0 for _ in range(len(seq))]
    for i, elem in enumerate(sorted_idx):
        out[elem] = i
    return out


@overload
def ir_node_to_tensor(x: Literal[None], guard_shape: bool = True) -> None: ...


@overload
def ir_node_to_tensor(x: IRNode, guard_shape: bool = True) -> torch.Tensor: ...


def ir_node_to_tensor(
    x: Optional[IRNode], guard_shape: bool = True
) -> Optional[torch.Tensor]:
    if x is None:
        return None

    shape_fn: Callable[[Union[int, Expr]], Union[int, Expr]]
    if not guard_shape:
        shape_fn = V.graph.sizevars.size_hint
    else:
        shape_fn = identity
    size = [shape_fn(s) for s in x.get_size()]
    stride: StrideType
    if is_storage_and_layout(x):
        stride = [shape_fn(s) for s in x.get_layout().stride]
    else:
        stride = FlexibleLayout.contiguous_strides(size)
    dtype = x.get_dtype()
    device = x.get_device()
    size = convert_shape_to_symint(size)
    stride = convert_shape_to_symint(stride)
    with V.graph.sizevars.shape_env.suppress_guards():
        t = torch.empty_strided(
            size=size, stride=stride, dtype=dtype, device=device
        ).zero_()
    return t


def may_convert_to_optional(
    value: Optional[Sequence[_T]],
) -> Optional[Sequence[Optional[_T]]]:
    if isinstance(value, list) and not value:
        # [None] makes sure the cpp wrapper codegen will generate something like
        # {std::nullopt} instead of {}
        return [None]
    return value


def get_device_type(
    x: Union[IRNode, OutputSpec, torch.device, None, str],
) -> Optional[str]:
    if isinstance(x, str) or x is None:
        return x
    elif isinstance(x, torch.device):
        return x.type
    elif isinstance(x, (IRNode, OutputSpec)):
        return get_device_type(x.get_device())
    assert_never(f"get_device_type({x}: {type(x).__name__})")


def is_triton(x: Union[IRNode, torch.device, None, str]) -> bool:
    device = get_device_type(x)
    # Special case cpu and cuda as using the method below
    # to determine if the scheduler is a triton scheduler subclass
    # requires instantiating a scheduler for them
    if device in ["cpu", "cuda"]:
        if getattr(config, f"{device}_backend") == "triton":
            return True
        return False
    if (
        device is None
        or (device_scheduling := get_scheduling_for_device(device)) is None
    ):
        return False
    from .codegen.triton import TritonScheduling

    assert isinstance(device_scheduling, type)
    return issubclass(device_scheduling, TritonScheduling)


def is_cpu(x: Union[IRNode, torch.device, None, str]) -> bool:
    return get_device_type(x) == "cpu"


def is_aligned_realized_tensor(x: Union[Buffer, TensorBox], alignment: int) -> bool:
    if not isinstance(x, IRNode) or x.maybe_get_stride() is None:
        return False

    aligned_strides = all(
        (V.graph.sizevars.size_hint(x.get_stride()[i]) % alignment) == 0
        for i in range(len(x.get_stride()) - 1)
    )
    # if the last dim size is <= 1, stride doesnt matter
    aligned_last_dim = (
        V.graph.sizevars.size_hint(x.get_stride()[-1]) == 1
        or V.graph.sizevars.size_hint(x.get_size()[-1]) <= 1
    )
    return aligned_last_dim and aligned_strides


def significant_strides_equal(
    strides1: Sequence[_IntLike],
    strides2: Sequence[_IntLike],
    shape: Sequence[_IntLike],
) -> bool:
    """
    Returns true if the strides are equal, ignoring dimensions of size 1 .
    """
    assert len(shape) == len(strides1) and len(strides1) == len(strides2)
    for dim, s1, s2 in zip(shape, strides1, strides2):
        if V.graph.sizevars.statically_known_leq(dim, 1):  # type: ignore[arg-type]
            continue

        if not V.graph.sizevars.statically_known_equals(
            s1, s2
        ) and not V.graph.sizevars.symbolic_hint(s1) == V.graph.sizevars.symbolic_hint(
            s2
        ):
            return False

    return True


def try_match_insignificant_strides(
    tensor: Union[TensorBox, BaseView],
    strides: Sequence[Union[int, torch.SymInt]],
) -> Union[TensorBox, BaseView]:
    """
    Tries to match the strides of the tensor to those in the meta_strides. Strides of insignificant
    dimensions - size 0 or 1 - will be updated.

    If there are real stride differences (NHWC vs NCHW), or the tensor is not realized, then the input will be returned
    """
    if not is_storage_and_layout(tensor):
        return tensor

    if all(
        V.graph.sizevars.statically_known_equals(s1, s2)
        for s1, s2 in zip(strides, tensor.get_stride())
    ):
        return tensor  # type: ignore[arg-type]

    if not significant_strides_equal(strides, tensor.get_stride(), tensor.get_size()):
        return tensor

    storage, old_layout = as_storage_and_layout(tensor)
    new_stride = [*old_layout.stride]
    for i, s in enumerate(tensor.get_size()):
        if V.graph.sizevars.statically_known_leq(s, 1):  # type: ignore[arg-type]
            new_stride[i] = strides[i]

    new_layout = FixedLayout(
        old_layout.device,
        old_layout.dtype,
        old_layout.size,
        new_stride,
        old_layout.offset,
    )
    return TensorBox(ReinterpretView(data=storage, layout=new_layout))


class IRNode:
    _current_origins: ClassVar[OrderedSet[Any]] = OrderedSet()

    # NB: These are kinda weird,
    origins: OrderedSet[Any] = dataclasses.field(init=False)
    traceback: Optional[list[str]] = dataclasses.field(init=False)
    origin_node: Optional[torch.fx.Node] = dataclasses.field(init=False)

    @staticmethod
    @contextlib.contextmanager
    def current_origins(origins: OrderedSet[Node]) -> Generator[None, None, None]:
        old = IRNode._current_origins
        IRNode._current_origins = old | origins
        try:
            yield
        finally:
            IRNode._current_origins = old

    def _post_init_setattr(self, attr: str, value: Any) -> None:
        # Intended for use in __post_init__ for enforcing an invariant on a dataclass
        # If you must, can also be used for setting provenance info
        # We would like to try and minimize these usages though
        object.__setattr__(self, attr, value)

    def __post_init__(self) -> None:
        self._post_init_setattr("origins", OrderedSet(self._current_origins))
        self._post_init_setattr(
            "traceback", traceback.format_stack() if config.debug_ir_traceback else None
        )
        self._post_init_setattr("origin_node", None)

    def get_read_names(self) -> OrderedSet[str]:
        return OrderedSet(dep.name for dep in self.get_reads())

    def get_traceback(self) -> Optional[list[str]]:
        return self.traceback

    def get_origin_node(self) -> Optional[torch.fx.Node]:
        return self.origin_node

    def get_defining_op(self) -> Optional[Operation]:
        return None

    def common_repr(self, shorten: bool = True) -> Sequence[str]:
        origins = f"origins={getattr(self, 'origins', '')}"
        if shorten and len(origins) > 64:
            # this can get *very* long
            origins = f"{origins[:61]}..."
        return [origins]

    def str_helper(
        self, lines: Sequence[object], shorten: bool = True, multiline: bool = True
    ) -> str:
        lines = list(lines) + list(self.common_repr(shorten))
        lines = list(map(str, lines))
        if multiline:
            new_lines = indent(",\n".join(lines))
            return f"{type(self).__name__}(\n{new_lines}\n)"
        else:
            return f"{type(self).__name__}({lines})"

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def maybe_get_dtype(self) -> Optional[torch.dtype]:
        try:
            return self.get_dtype()
        except NotImplementedError:
            return None

    def get_layout(self) -> Layout:
        raise NotImplementedError(f"get_layout() is not implemented by {type(self)}!")

    def maybe_get_layout(self) -> Optional[Layout]:
        try:
            return self.get_layout()
        except NotImplementedError:
            return None

    def get_output_spec(self) -> OutputSpec:
        return self.get_layout()

    def maybe_get_output_spec(self) -> Optional[OutputSpec]:
        try:
            return self.get_output_spec()
        except NotImplementedError:
            return None

    def has_tensor_output(self) -> bool:
        """True for single tensor output (excludes MultiOutput)"""
        return isinstance(self.maybe_get_output_spec(), Layout)

    def get_size(self) -> Sequence[Expr]:
        raise NotImplementedError(f"get_size() is not implemented by {type(self)}!")

    def maybe_get_size(self) -> Optional[Sequence[_IntLike]]:
        try:
            return self.get_size()
        except NotImplementedError:
            return None

    @property
    def shape(self) -> Union[_IntLike, sympy.Rel, Sequence[_IntLike]]:
        return self.get_size()

    def get_numel(self) -> Expr:
        return sympy_product(self.get_size())

    def is_zero_elements(self) -> bool:
        return V.graph.sizevars.is_expr_static_and_true(sympy.Eq(self.get_numel(), 0))

    def realize(self) -> Optional[str]:
        """
        If the IRNode refers to data which has not been materialized (e.g.,
        it is a Pointwise/Reduction that could potentially have more
        compute fused into it), realize the IRNode into physical memory,
        ending the possibility of fusing into it, but allowing, e.g., multiple
        users to access the data without having to recompute.

        Check StorageBox.realize for a particularly notable implementation.

        TODO(ezyang): I think, in principle, every IRNode should have an
        implementation of this, and most of the time no-op is OK, but you
        really do have to audit each IRNode for this, so for now, raise
        an error if it's not implemented.  Note that some code in graph.py
        will catch this thrown error and suppress it with a warning.
        """
        raise NotImplementedError(f"realize NYI on {type(self)}")

    def codegen_reference(self, writer: Optional[IndentedBuffer] = None) -> str:
        raise NotImplementedError(f"codegen_reference NYI on {type(self)}")

    def get_device(self) -> Optional[torch.device]:
        return None

    def get_device_or_error(self) -> torch.device:
        device = self.get_device()
        assert device is not None
        return device

    def has_exceeded_max_reads(self) -> bool:
        return False

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        raise NotImplementedError(type(self).__name__)

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        raise NotImplementedError(type(self).__name__)

    def get_stride(self) -> Sequence[_IntLike]:
        raise NotImplementedError(type(self).__name__)

    def maybe_get_stride(self) -> Optional[Sequence[_IntLike]]:
        try:
            return self.get_stride()
        except NotImplementedError:
            return None

    def get_name(self) -> str:
        raise NotImplementedError(type(self).__name__)

    def maybe_get_name(self) -> Optional[str]:
        try:
            return self.get_name()
        except NotImplementedError:
            return None

    def has_large_inner_fn(self, threshold: Optional[int] = None) -> bool:
        return False

    def mark_reuse(self, users: int) -> None:
        pass

    def realize_hint(self) -> None:
        pass

    def unwrap_view(self) -> IRNode:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout(self) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_stride_order(
        self, order: list[int], allow_padding: bool = False
    ) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_fill_order(self, order: list[int]) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_same_order(self, stride: list[_IntLike]) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_exact_strides(
        self, exact_strides: list[_IntLike], allow_padding: bool = False
    ) -> None:
        raise NotImplementedError(type(self).__name__)

    def get_read_writes(self) -> dependencies.ReadWrites:
        raise NotImplementedError(type(self).__name__)

    def get_reads(self) -> OrderedSet[Dep]:
        return self.get_read_writes().reads

    def num_reads(self) -> int:
        return len(self.get_reads())

    def get_storage_numel(self) -> _IntLike:
        raise NotImplementedError(type(self).__name__)

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        raise NotImplementedError(type(self).__name__)

    def get_reduction_type(self) -> Optional[str]:
        raise NotImplementedError(type(self).__name__)

    def get_reduction_size(self) -> Sequence[sympy.Expr]:
        raise NotImplementedError(type(self).__name__)

    def is_extern(self) -> bool:
        return False

    def is_no_op(self) -> bool:
        return False

    def constant_to_device(self, device: torch.device) -> IRNode:
        raise NotImplementedError(type(self).__name__)

    def get_mutation_names(self) -> Sequence[str]:
        raise NotImplementedError(type(self).__name__)

    def get_operation_name(self) -> str:
        raise NotImplementedError(type(self).__name__)

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        raise NotImplementedError(type(self).__name__)

    if TYPE_CHECKING:

        @property
        def dtype(self) -> torch.dtype: ...


@ir_dataclass(frozen=False)
class Operation:
    def __post_init__(self) -> None:
        self.operation_name: Optional[str] = None

    def get_device(self) -> Optional[torch.device]:
        raise NotImplementedError

    def get_origin_node(self) -> Optional[torch.fx.Node]:
        assert hasattr(self, "origin_node")
        return self.origin_node

    def get_origins(self) -> OrderedSet[Any]:
        assert hasattr(self, "origins")
        return self.origins

    def get_operation_name(self) -> str:
        assert self.operation_name is not None
        return self.operation_name

    def is_extern(self) -> bool:
        return False

    def is_no_op(self) -> bool:
        return False

    def get_read_writes(self) -> dependencies.ReadWrites:
        raise NotImplementedError

    def is_user_of(self, name: str) -> bool:
        return name in self.get_read_names()

    def get_read_names(self) -> OrderedSet[str]:
        return OrderedSet(dep.name for dep in self.get_reads())

    def get_reads(self) -> OrderedSet[Dep]:
        return self.get_read_writes().reads

    def get_outputs(self) -> list[Buffer]:
        raise NotImplementedError

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        """
        When unbacked_only=True:
        Returns the unbacked symbols which are required to be in scope in
        order to successfully perform codegen for this buffer.  For example,
        a buffer that corresponds to an extern kernel call that takes i0 as
        an argument would return {i0} here.  This is used to generate necessary
        dependencies that ensure we actually bind i0 in codegen before you
        try to use it.

        Note that this is NOT transitive; in particular, if this buffer takes
        in as input another buffer with dynamic shape (e.g., (i0,)), we will
        not report it here, because you will already have a dependency
        on that buffer, which will eventually have a dependency on i0 if
        necessary.

        When unbacked_only=False:
        Similar to `unbacked_only=True` but including all free symbols
        instead of only free unbacked symbols.
        """
        return OrderedSet()

    def get_workspace_size(self) -> int:
        """
        Gets extra global memory size needed by this buffer.
        Some algorithms (e.g. group gemm) may require extra global memory in the generated code.
        """
        return 0


@ir_dataclass
class Loops(IRNode):
    device: torch.device
    dtype: torch.dtype
    inner_fn: Callable[..., Any]
    ranges: Sequence[_IntLike]

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet().union(
            *(get_free_symbols(e, unbacked_only) for e in self.ranges),
            self.inner_fn_free_symbols(unbacked_only),
        )

    def _to_str(self, names: Sequence[str]) -> str:
        return self.str_helper(
            [
                f"'{self.device.type}'",
                str(self.dtype),
                self.inner_fn_str(),
            ]
            + [f"{name}={getattr(self, name)}" for name in names]
            + [f"origin_node={self.origin_node!r}"]
        )

    def __post_init__(self) -> None:
        super().__post_init__()

    def __str__(self) -> str:
        return self._to_str(("ranges",))

    __repr__ = __str__

    def get_device(self) -> Optional[torch.device]:
        return self.device

    def get_origin_node(self) -> Optional[torch.fx.Node]:
        return self.origin_node

    def get_size(self) -> Sequence[Expr]:
        return self.ranges

    def get_pointwise_size(self) -> Sequence[Expr]:
        return self.ranges

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> TensorBox:
        origin_node = kwargs.pop("origin_node", None)
        tb = kwargs.pop("traceback", None)
        r = cls(*args, **kwargs)
        # Need to explicitly set origin_node here to propagate it down.
        # todo(chilli): I think it would be better for IRNode to directly set
        # origin_node
        r._post_init_setattr("origin_node", origin_node)
        r._post_init_setattr("traceback", tb or r.traceback)
        return TensorBox.create(r)

    @staticmethod
    def _index(ranges: Sequence[_IntLike], prefix: SymT = SymT.INDEX) -> Sequence[Expr]:
        return [
            sympy.S.Zero if s == 1 else sympy_index_symbol_with_prefix(prefix, n)
            for n, s in enumerate(ranges)
        ]

    @cache_on_self
    def inner_fn_opcount(self) -> OpCountResult:
        opcounter = OpCounterCSE(V.MockHandler())
        with (
            V.set_ops_handler(opcounter),
            patch.object(FlexibleLayout, "allow_indexing", True),
        ):
            self.inner_fn(*self.inner_fn_args())
            return opcounter.getvalue()

    def inner_fn_args(self) -> Sequence[Sequence[_IntLike]]:
        return (self._index(self.ranges),)

    @cache_on_self
    def inner_fn_str(self) -> str:
        return V.KernelFormatterHandler.ir_to_string(
            self.inner_fn, *self.inner_fn_args()
        )

    def has_large_inner_fn(self, threshold: Optional[int] = None) -> bool:
        if threshold is None:
            threshold = 0
        threshold = max(threshold, config.realize_opcount_threshold)
        return self.inner_fn_opcount().num_ops > threshold

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        index = self._index(self.ranges)
        return extract_free_symbols(self.inner_fn, index, unbacked_only=unbacked_only)

    def get_reads(self) -> OrderedSet[Dep]:
        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.get_reduction_type():
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                    self.get_reduction_size(),
                ).reads
            else:
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                ).reads

    def get_read_names(self) -> OrderedSet[str]:
        return OrderedSet(self.inner_fn_opcount().read_buffers)

    def num_reads(self) -> int:
        return len(self.inner_fn_opcount().read_buffers)

    def get_reduction_size(self) -> Sequence[sympy.Expr]:
        raise NotImplementedError(
            f"get_reduction_size() is not implemented by {type(self)}!"
        )

    def get_reduction_type(self) -> Optional[str]:
        raise NotImplementedError(
            f"get_reduction_type() is not implemented by {type(self)}!"
        )

    def constant_to_device(self, device: torch.device) -> IRNode:
        raise NotImplementedError(
            f"constant_to_device() is not implemented by {type(self)}!"
        )


def nop_loader_fn(idx: Union[Expr, Sequence[Expr]], *, dtype: torch.dtype) -> OpsValue:
    if dtype.is_floating_point:
        return ops.constant(float("nan"), dtype)
    else:
        return ops.constant(0, dtype)


@ir_dataclass
class Pointwise(Loops):
    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # Make zero-element loops into a no-op
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.dtype)

        return self.inner_fn

    def get_reduction_size(self) -> Sequence[sympy.Expr]:
        return []

    def get_reduction_type(self) -> Optional[str]:
        return None

    def store_output(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
    ) -> None:
        loader = self.make_loader()
        return ops.store(output_name or "unnamed", indexer(vars), loader(vars))

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(
            device=device, dtype=self.dtype, inner_fn=loader, ranges=self.ranges
        )


@ir_dataclass
class Scatter(Pointwise):
    output_indexer: Callable[[Sequence[Expr]], Expr]
    scatter_mode: StoreMode = None

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Scatter(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
            output_indexer=self.output_indexer,
            scatter_mode=self.scatter_mode,
        )

    def store_output(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
    ) -> None:
        loader = self.make_loader()
        if output_name is None:
            output_name = "unnamed"
        return ops.store(
            output_name,
            indexer(self.output_indexer(vars)),
            loader(vars),
            mode=self.scatter_mode,
        )


REDUCTION_COMBINE_FN: dict[str, Callable[..., OpsValue]] = {
    "any": ops_wrapper("logical_or"),
    "max": ops_wrapper("maximum"),
    "min": ops_wrapper("minimum"),
    "prod": ops_wrapper("mul"),
    "sum": ops_wrapper("add"),
    "xor_sum": ops_wrapper("bitwise_xor"),
}


def get_reduction_combine_fn(
    reduction_type: str, dtype: torch.dtype, arg_break_ties_left: bool = True
) -> Callable[..., object]:
    if reduction_type in REDUCTION_COMBINE_FN:
        return REDUCTION_COMBINE_FN[reduction_type]

    elif reduction_type in ("argmax", "argmin"):

        def argmax_combine_fn(
            a: tuple[object, object], b: tuple[object, object]
        ) -> tuple[OpsValue, OpsValue]:
            a_value, a_index = a
            b_value, b_index = b

            if reduction_type == "argmin":
                mask = ops.lt(a_value, b_value)
            else:
                mask = ops.gt(a_value, b_value)

            equal = ops.eq(a_value, b_value)
            if is_float_dtype(dtype):
                a_isnan = ops.ne(a_value, a_value)
                b_isnan = ops.ne(b_value, b_value)
                mask = ops.logical_or(mask, ops.gt(a_isnan, b_isnan))
                equal = ops.logical_or(equal, ops.logical_and(a_isnan, b_isnan))

            tie = (
                ops.lt(a_index, b_index)
                if arg_break_ties_left
                else ops.gt(a_index, b_index)
            )
            mask = ops.logical_or(mask, ops.logical_and(equal, tie))
            return (
                ops.where(mask, a_value, b_value),
                ops.where(mask, a_index, b_index),
            )

        return argmax_combine_fn

    elif reduction_type == "welford_combine":

        def welford_combine_fn(
            a: tuple[OpsValue, OpsValue, OpsValue],
            b: tuple[OpsValue, OpsValue, OpsValue],
        ) -> tuple[OpsValue, OpsValue, OpsValue]:
            a_mean, a_m2, a_weight = a
            b_mean, b_m2, b_weight = b

            delta = b_mean - a_mean
            new_weight = a_weight + b_weight
            w2_over_w = b_weight / new_weight
            return (
                a_mean + delta * w2_over_w,
                a_m2 + b_m2 + delta * delta * a_weight * w2_over_w,
                new_weight,
            )

        return welford_combine_fn

    else:
        raise NotImplementedError(f"unknown reduction_type={reduction_type}")


@ir_dataclass
class Reduction(Loops):
    reduction_ranges: Sequence[_IntLike]
    reduction_type: ReductionType
    # self.dtype represents the dst dtype
    src_dtype: torch.dtype
    reduction_hint: ReductionHint

    def __str__(self) -> str:
        return self._to_str(("ranges", "reduction_ranges", "reduction_type"))

    __repr__ = __str__

    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        return super().get_free_symbol_uses(unbacked_only) | OrderedSet().union(
            *(get_free_symbols(e, unbacked_only) for e in self.reduction_ranges)
        )

    def get_reduction_size(self) -> Sequence[sympy.Expr]:
        return self.reduction_ranges

    def get_reduction_type(self) -> Optional[str]:
        return self.reduction_type

    def store_reduction(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
        reduction_vars: Sequence[Symbol],
    ) -> None:
        value = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        return ops.store_reduction(output_name or "unnamed", indexer(vars), value)

    def index_length(self) -> int:
        return len(self.ranges) + len(self.reduction_ranges)

    def inner_fn_args(self) -> Sequence[Sequence[Expr]]:
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.R0_INDEX)
        return (index, rindex)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.R0_INDEX)
        return extract_free_symbols(
            self.inner_fn, index, rindex, unbacked_only=unbacked_only
        )

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Reduction(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
            reduction_ranges=self.reduction_ranges,
            reduction_type=self.reduction_type,
            src_dtype=self.src_dtype,
            reduction_hint=ReductionHint.DEFAULT,
        )

    @staticmethod
    def num_splits(
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., OpsValue],
        ranges: Sequence[_IntLike],
        reduction_ranges: Sequence[_IntLike],
        reduction_type: Union[ReductionType, Literal["scan"]],
        reduction_numel: Expr,
        input_node: Optional[IRNode] = None,
    ) -> tuple[ReductionHint, _IntLike]:
        def _is_static(x: object) -> bool:
            return isinstance(x, (int, Integer))

        reduction_numel_hint = V.graph.sizevars.symbolic_hint(reduction_numel)
        numel_hint = V.graph.sizevars.symbolic_hint(sympy_product(ranges))

        should_split = reduction_type == "scan" or (
            not V.graph.has_feature(device, BackendFeature.REDUCE_TO_SINGLE_ELEMENT)
            and reduction_type
            not in (
                "argmax",
                "argmin",
            )
            and config.split_reductions
        )
        if not (_is_static(reduction_numel_hint) and _is_static(numel_hint)):
            # We don't support unbacked symints
            return ReductionHint.DEFAULT, 1

        props = DeviceProperties.create(device)
        num_sm = props.multi_processor_count
        min_elements_per_thread = 32
        if should_split:
            inner_reduction_splits: Callable[[int, int], int] = functools.partial(
                V.choices.reduction_split_factor, device, inner_reduction=True
            )
            outer_reduction_splits: Callable[[int, int], int] = functools.partial(
                V.choices.reduction_split_factor, device, inner_reduction=False
            )
        else:

            def inner_reduction_splits(
                reduction_numel_hint: int,
                numel_hint: int,
            ) -> int:
                return 1

            outer_reduction_splits = inner_reduction_splits

        # easy cases
        if numel_hint == 1:
            split = inner_reduction_splits(reduction_numel_hint, numel_hint)
            if split == 1:
                # No need to split.
                return ReductionHint.INNER, split
            if input_node is not None and isinstance(input_node, TensorBox):
                with patch.object(FlexibleLayout, "allow_indexing", True):
                    (
                        new_ranges,
                        new_reduction_ranges,
                    ) = extract_input_node_reduction_ranges(input_node)
                if new_ranges is not None and new_reduction_ranges is not None:
                    extracted_numel_hint = V.graph.sizevars.symbolic_hint(
                        sympy_product(new_ranges + new_reduction_ranges)
                    )
                    if reduction_numel_hint == extracted_numel_hint:
                        log.debug(
                            "Use previous IRNode's range and reduction_ranges instead of split. "
                            "current ranges: %s, current reduction ranges: %s, current split: %d, "
                            "new ranges: %s, new reduction ranges: %s",
                            ranges,
                            reduction_ranges,
                            split,
                            new_ranges,
                            new_reduction_ranges,
                        )
                        # If the input_node or its dependent nodes are also Reduction nodes,
                        # use reduction_sizes of this node or its dependent nodes directly.
                        return ReductionHint.INNER, -1
            return ReductionHint.INNER, split
        if (
            reduction_numel_hint <= min_elements_per_thread
            or numel_hint >= num_sm * 2 * 32
        ):
            return ReductionHint.DEFAULT, 1

        r = Reduction(
            device=device,
            dtype=dst_dtype,
            inner_fn=inner_fn,
            ranges=ranges,
            reduction_ranges=reduction_ranges,
            reduction_type=reduction_type if reduction_type != "scan" else "sum",
            src_dtype=src_dtype,
            reduction_hint=ReductionHint.DEFAULT,
        )

        def get_read_indices(r: Reduction) -> tuple[Sequence[Expr], bool]:
            cb = ComputedBuffer(
                name=None,
                layout=FlexibleLayout(
                    device=r.get_device(),
                    dtype=r.get_dtype(),
                    size=r.get_size(),
                ),
                data=r,
            )
            read_writes = cb.get_read_writes()
            # try finding the full size producer
            # TODO this will fail for something like ((1, N) * (N, 1)).sum()
            # this would also possibly be wrong for producers with the different contiguity but we hope those cases are rare
            assert read_writes.range_vars is not None
            range_vars = [
                r
                for r in read_writes.range_vars
                if isinstance(r, Expr) and not isinstance(r, sympy.Number)
            ]
            indices = []
            changed = False
            for md in sorted(read_writes.reads, key=lambda x: x.name):
                if all(r in md.index.free_symbols for r in range_vars):
                    indices.append(md.index)
                    if md.name in V.graph.name_to_buffer:
                        buf = V.graph.name_to_buffer[md.name]
                        original_stride = getattr(buf.layout, "stride", None)
                        buf.decide_layout()
                        if getattr(buf.layout, "stride", None) != original_stride:
                            changed = True
            return indices, changed

        indices, changed = get_read_indices(r)
        if changed:
            indices, _ = get_read_indices(r)

        if len(indices) == 0:
            # TODO determine splits when all inputs are broadcast
            return ReductionHint.DEFAULT, 1

        (_, reduction_vars), ranges1 = dependencies.index_vars_squeeze(
            r.get_size(), r.get_reduction_size()
        )
        num_outer = 0
        num_inner = 0
        for i in indices:
            j = V.graph.sizevars.simplify_with_ranges(i, ranges1)
            strides = V.graph.sizevars.stride_hints(
                j, reduction_vars, list(ranges1.keys())
            )
            outer = all(s > 1 for s in strides)
            if outer:
                num_outer += 1
            else:
                num_inner += 1
        if num_inner > num_outer:
            return ReductionHint.INNER, inner_reduction_splits(
                reduction_numel_hint, numel_hint
            )
        else:
            return ReductionHint.OUTER, outer_reduction_splits(
                reduction_numel_hint, numel_hint
            )

    @staticmethod
    def _unroll_reduction_fn(
        inner_fn: Callable[[Sequence[_IntLike], Sequence[_IntLike]], OpsValue],
        reduction_ranges: Sequence[_IntLike],
        reduction_type: str,
        src_dtype: torch.dtype,
    ) -> Callable[[Sequence[_IntLike]], OpsValue]:
        """Convert inner_fn from a reduction to an pointwise"""
        reduction_ranges = [
            V.graph.sizevars.evaluate_static_shape(x) for x in reduction_ranges
        ]

        combine_fn = get_reduction_combine_fn(reduction_type, src_dtype)

        def fn(index: Sequence[_IntLike]) -> Any:
            return functools.reduce(
                combine_fn,
                (
                    value_fn(index, rindex)
                    for rindex in itertools.product(
                        *[range(x) for x in reduction_ranges]
                    )
                ),
            )

        value_fn: Callable[[Sequence[_IntLike], Sequence[_IntLike]], Any]
        if reduction_type in ("argmin", "argmax"):
            flatten_index = FixedLayout(
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                reduction_ranges,
                FlexibleLayout.contiguous_strides(reduction_ranges),
            ).make_indexer()

            def value_fn(
                index: Sequence[_IntLike], rindex: Sequence[_IntLike]
            ) -> tuple[OpsValue, OpsValue]:
                rindex = [sympy.expand(i) for i in rindex]
                return (
                    inner_fn(index, rindex),
                    ops.index_expr(flatten_index(rindex), torch.int64),
                )

            return lambda index: fn(index)[1]
        else:
            value_fn = inner_fn
            return fn

    @classmethod
    def create(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        reduction_type: ReductionType,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: Optional[IRNode] = None,
    ) -> TensorBox:
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        if reduction_numel == 0:
            # N.B. This is a hack to generate the literal of the given type
            # Ideally, we should be fixing `def constant` in triton.py
            # but it breaks due to hardcoded dtypes in other places
            def py_cnst(val: object) -> Union[bool, float, int]:
                if dst_dtype == torch.bool:
                    return bool(val)
                elif dst_dtype.is_floating_point:
                    assert isinstance(val, typing.SupportsFloat)
                    return float(val)
                else:
                    assert isinstance(val, typing.SupportsInt)
                    return int(val)

            rtypes_to_inits = {
                "sum": py_cnst(0),
                "xor_sum": py_cnst(0),
                "prod": py_cnst(1),
                "any": py_cnst(0),
                # "all" is desugared to `!any(!val)`
            }

            assert reduction_type in rtypes_to_inits.keys(), (
                f"{reduction_type} not supported for zero-dimension tensors!"
            )

            def const_fn(index: int) -> OpsValue:
                return ops.constant(rtypes_to_inits[reduction_type], dst_dtype)

            return Pointwise.create(
                device=device,
                dtype=src_dtype,
                inner_fn=const_fn,
                ranges=list(ranges),
            )

        if reduction_numel == 1:
            # this reduction is actually a pointwise op
            if reduction_type in ("argmin", "argmax"):

                def fn(index: int) -> OpsValue:
                    return ops.constant(0, dst_dtype)

            else:

                def fn(index: int) -> OpsValue:
                    reduction_index = [sympy.S.Zero for _ in reduction_ranges]
                    return inner_fn(index, reduction_index)

            return Pointwise.create(
                device=device, dtype=dst_dtype, inner_fn=fn, ranges=ranges
            )

        if (
            isinstance(reduction_numel, Integer)
            and V.graph.sizevars.size_hint(reduction_numel)
            < config.unroll_reductions_threshold
            and (sympy_product(ranges) != 1 or is_gpu(device.type))
        ):
            # NB: This works around https://github.com/pytorch/pytorch/issues/140457
            # since turning reductions into pointwise ops can exacerbate this problem
            return Pointwise.create(
                device=device,
                dtype=dst_dtype,
                inner_fn=cls._unroll_reduction_fn(
                    inner_fn, reduction_ranges, reduction_type, src_dtype
                ),
                ranges=ranges,
            )

        # triton doesn't support reduce to single element well, so break it up
        hint, split = cls.num_splits(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            reduction_numel,
            input_node,
        )
        # intermediate reduction in split can contain complex indexing,
        # and num_splits will fail to correctly set the hint
        # reuse the passed hint if available
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split == -1:
            assert input_node is not None
            new_ranges, new_reduction_ranges = extract_input_node_reduction_ranges(
                input_node
            )
            assert new_ranges is not None
            assert new_reduction_ranges is not None
            return cls.create_multilayer_existing_ranges(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                new_ranges,
                new_reduction_ranges,
                reduction_type,
                reduction_hint,
            )
        elif split > 1:
            # triton doesn't support reduce to single element well, so break it up
            return cls.create_multilayer(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                split,
                reduction_hint,
                input_node,
            )

        return TensorBox.create(
            Reduction(
                device=device,
                dtype=dst_dtype,
                inner_fn=inner_fn,
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
            )
        )

    @staticmethod
    def default_accumulator(
        reduction_type: str, dtype: torch.dtype
    ) -> Union[_NumLike, Sequence[_NumLike]]:
        if reduction_type in ("max", "argmax"):
            if is_float_dtype(dtype):
                return float("-inf")
            elif is_boolean_dtype(dtype):
                return False
            else:
                return torch.iinfo(dtype).min
        if reduction_type in ("min", "argmin"):
            if is_float_dtype(dtype):
                return float("inf")
            elif is_boolean_dtype(dtype):
                return True
            else:
                return torch.iinfo(dtype).max

        zero = False if is_boolean_dtype(dtype) else 0
        one = True if is_boolean_dtype(dtype) else 1
        return {
            "sum": zero,
            "prod": one,
            "xor_sum": zero,
            "any": zero,
            "welford_reduce": (zero, zero, zero),
            "welford_combine": (zero, zero, zero),
            "online_softmax_reduce": (float("-inf"), zero),
        }[reduction_type]

    @staticmethod
    def default_value(
        reduction_type: str, dtype: torch.dtype
    ) -> Union[_NumLike, Sequence[_NumLike]]:
        if reduction_type == "welford_reduce":
            return 0
        return Reduction.default_accumulator(reduction_type, dtype)

    @staticmethod
    def _multilayer_second_step_hint(
        split: _IntLike, numel_hint: int, reduction_hint: ReductionHint
    ) -> ReductionHint:
        if split == -1:
            return reduction_hint
        if split <= 512 and numel_hint <= 512 and reduction_hint == ReductionHint.OUTER:
            return ReductionHint.OUTER_TINY
        if (
            split <= 1024
            and numel_hint <= 256
            and reduction_hint == ReductionHint.OUTER
        ):
            return ReductionHint.OUTER_TINY

        return reduction_hint

    @classmethod
    def check_for_split_dense_dim_reindexing(
        cls, reduction_numel: _IntLike, input_node: Optional[IRNode]
    ) -> Optional[int]:
        """
        If we are reducing over the full tensor, and it is non-dense in the last dimension,
        reindex so we reduce over the dense dimension. initially just handle complete
        reduction case
        """
        if input_node is None:
            return None

        if not V.graph.sizevars.statically_known_equals(
            input_node.get_numel(), reduction_numel
        ):
            return None

        input_node.realize()
        try:
            # finalize layout
            as_storage_and_layout(input_node)
        except NotImplementedError:
            return None

        strides = input_node.get_stride()

        for i, s in enumerate(strides[:-1]):
            if V.graph.sizevars.statically_known_equals(s, 1):
                return i

        return None

    @classmethod
    def _multilayer_wrap_loader(
        cls,
        loader: Callable[..., OpsValue],
        reduction_ranges: Sequence[_IntLike],
        reduction_numel: _IntLike,
        split: _IntLike,
        block_size: _IntLike,
        default: Union[_NumLike, Sequence[_NumLike]],
        input_node: Optional[IRNode] = None,
    ) -> Callable[..., object]:
        dense_index = cls.check_for_split_dense_dim_reindexing(
            reduction_numel, input_node
        )
        reindex = View.dynamic_reshape_indexer(
            reduction_ranges, [reduction_numel], dense_index
        )
        need_mask = not V.graph.sizevars.is_expr_static_and_true(
            sympy.Eq(reduction_numel % split, 0)
        )

        def wrapper_fn(
            index: Sequence[Symbol], reduction_index: Sequence[Symbol]
        ) -> OpsValue:
            (reduction_index,) = reduction_index
            *new_index, reduction_block = index
            indices = block_size * reduction_block + reduction_index

            def body() -> OpsValue:
                return loader(new_index, reindex([indices]))

            if need_mask:
                mask = ops.lt(
                    ops.index_expr(indices, torch.int32),
                    ops.index_expr(reduction_numel, torch.int32),
                )
                return ops.masked(mask, body, default)
            else:
                return body()

        return wrapper_fn

    @classmethod
    def _multilayer_wrap_loader_existing_ranges(
        cls,
        loader: Callable[[Sequence[sympy.Expr], Sequence[sympy.Expr]], OpsValue],
        original_ranges: Sequence[Expr],
        original_reduction_ranges: Sequence[Expr],
        new_ranges: Sequence[Integer],
        new_reduction_ranges: Sequence[Integer],
    ) -> Callable[[Sequence[sympy.Expr], Sequence[sympy.Expr]], OpsValue]:
        assert all(r == 1 for r in original_ranges), (
            f"Only enabled for numel_hint == 1, found {original_ranges=}"
        )
        reindex = View.dynamic_reshape_indexer(
            original_reduction_ranges, tuple(new_ranges) + tuple(new_reduction_ranges)
        )

        def wrapper_fn(
            merged_index: Sequence[sympy.Expr],
            new_reduction_index: Sequence[sympy.Expr],
        ) -> OpsValue:
            original_idx = merged_index[: len(original_ranges)]
            new_index = merged_index[len(original_ranges) :]
            return loader(
                original_idx,
                reindex(tuple(new_index) + tuple(new_reduction_index)),
            )

        return wrapper_fn

    @classmethod
    def create_multilayer_helper(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        wrapper_fn: Callable[..., Any],
        original_ranges: Sequence[Expr],
        original_reduction_ranges: Sequence[Expr],
        new_ranges: list[Expr],
        new_reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        split: _IntLike,
        reduction_hint: ReductionHint,
    ) -> TensorBox:
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        # triton will automatically compute reductions in fp32 if reducing over fp16/bf16
        # within the kernel. keep the intermediate in fp32 so as to keep the whole reduction
        # in fp32 and not reduce precision by breaking up the kernel into multiple layers
        intermediate_dtype = (
            dst_dtype
            if dst_dtype not in (torch.float16, torch.bfloat16)
            else torch.float
        )
        intermediate = Reduction.create(
            device,
            intermediate_dtype,
            src_dtype,
            wrapper_fn,
            new_ranges,
            new_reduction_ranges,
            reduction_type,
            reduction_hint,
        )
        intermediate.realize()
        intermediate_loader = intermediate.make_loader()

        def intermediate_fn(
            index: Sequence[_IntLike], reduction_index: Sequence[_IntLike]
        ) -> OpsValue:
            return intermediate_loader([*index, *reduction_index])

        numel_hint = V.graph.sizevars.size_hint(sympy_product(original_ranges))
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )

        assert original_ranges == new_ranges[: len(original_ranges)]
        return TensorBox.create(
            Reduction(
                device=device,
                dtype=dst_dtype,
                inner_fn=intermediate_fn,
                ranges=original_ranges,
                reduction_ranges=new_ranges[len(original_ranges) :],
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
            )
        )

    @classmethod
    def create_multilayer(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        reduction_type: ReductionType,
        split: _IntLike,
        reduction_hint: ReductionHint,
        input_node: Optional[IRNode] = None,
    ) -> TensorBox:
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        # TODO(jansel): realize the reduction so we can do dynamic indexing
        reduction_numel = sympy_product(reduction_ranges)
        block_size = FloorDiv(reduction_numel + (split - 1), split)
        default = cls.default_value(reduction_type, dst_dtype)
        wrapper_fn = cls._multilayer_wrap_loader(
            inner_fn,
            reduction_ranges,
            reduction_numel,
            split,
            block_size,
            default,
            input_node,
        )

        return cls.create_multilayer_helper(
            device,
            dst_dtype,
            src_dtype,
            wrapper_fn,
            ranges,
            reduction_ranges,
            [*ranges, split],
            [block_size],
            reduction_type,
            split,
            reduction_hint,
        )

    @classmethod
    def create_multilayer_existing_ranges(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        original_ranges: Sequence[Expr],
        original_reduction_ranges: Sequence[Expr],
        new_ranges: list[Integer],
        new_reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        reduction_hint: ReductionHint,
    ) -> TensorBox:
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        wrapper_fn = cls._multilayer_wrap_loader_existing_ranges(
            inner_fn,
            original_ranges,
            original_reduction_ranges,
            new_ranges,
            new_reduction_ranges,
        )
        return cls.create_multilayer_helper(
            device,
            dst_dtype,
            src_dtype,
            wrapper_fn,
            original_ranges,
            original_reduction_ranges,
            [*original_ranges, *new_ranges],
            new_reduction_ranges,
            reduction_type,
            -1,
            reduction_hint,
        )


INNER_FN_TY = Callable[[Sequence[Expr], Sequence[Expr]], OpsValue]


class MultiOutputReduction(Reduction):
    output_index: int

    def __init__(
        self,
        device: torch.device,
        dst_dtype: torch.dtype,
        inner_fns: Union[INNER_FN_TY, Sequence[INNER_FN_TY]],
        ranges: Sequence[Integer],
        reduction_ranges: Sequence[Integer],
        reduction_type: ReductionType,
        src_dtype: torch.dtype,
        reduction_hint: ReductionHint,
        output_index: int,
    ):
        if callable(inner_fns):
            inner_fns = (inner_fns,)

        loader: Callable[[Sequence[Expr], Sequence[Expr]], Any]
        if len(inner_fns) == 1:
            loader = inner_fns[0]
        else:

            def loader(
                idx: Sequence[Expr], reduction_idx: Sequence[Expr]
            ) -> tuple[OpsValue, ...]:
                return tuple(fn(idx, reduction_idx) for fn in inner_fns)

        super().__init__(
            device=device,
            dtype=dst_dtype,
            inner_fn=loader,
            ranges=ranges,
            reduction_ranges=reduction_ranges,
            reduction_type=reduction_type,
            src_dtype=src_dtype,
            reduction_hint=reduction_hint,
        )
        self.output_index = output_index

    def store_reduction(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
        reduction_vars: Sequence[Symbol],
    ) -> None:
        values = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        assert isinstance(values, (tuple, list)), f"{type(values)}"
        value = values[self.output_index]
        return ops.store_reduction(output_name or "unnamed", indexer(vars), value)


class OnlineSoftmaxReduction(MultiOutputReduction):
    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        num_output: int,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: Optional[IRNode] = None,
    ) -> Sequence[TensorBox]:
        """
        Create the reduction disregarding splitting.
        """
        results = tuple(
            TensorBox.create(
                MultiOutputReduction(
                    device,
                    dst_dtype,
                    inner_fn,
                    ranges,
                    reduction_ranges,
                    "online_softmax_reduce",  # type: ignore[arg-type]
                    src_dtype,
                    reduction_hint,
                    output_idx,
                )
            )
            for output_idx in range(num_output)
        )
        for t in results:
            t.realize()
        return results


class WelfordReduction(MultiOutputReduction):
    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fns: Sequence[Callable[..., Any]],
        ranges: list[Integer],
        reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
    ) -> Sequence[TensorBox]:
        assert reduction_type in ("welford_reduce", "welford_combine")

        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        def const(val: int) -> TensorBox:
            def inner_fn(idx: Sequence[Expr]) -> OpsValue:
                return ops.constant(
                    val,
                    dtype,
                )

            return Pointwise.create(
                device=device,
                dtype=dtype,
                inner_fn=inner_fn,
                ranges=list(ranges),
            )

        if reduction_numel == 0:
            mean = const(0)
            m2 = const(0)
            weight = const(0)
            return mean, m2, weight

        if reduction_numel == 1:

            def copy(
                loader: Callable[[Sequence[Expr], Sequence[Expr]], OpsValue],
            ) -> TensorBox:
                def inner_fn(idx: Sequence[Expr]) -> OpsValue:
                    reduction_index = [sympy.S.Zero for _ in reduction_ranges]
                    return loader(idx, reduction_index)

                return Pointwise.create(
                    device=device,
                    dtype=dtype,
                    inner_fn=inner_fn,
                    ranges=list(ranges),
                )

            if reduction_type == "welford_reduce":
                return copy(inner_fns[0]), const(0), const(1)
            else:
                return tuple(copy(fn) for fn in inner_fns)

        # TODO: Unrolled reduction
        # if (
        #     isinstance(reduction_numel, Integer)
        #     and V.graph.sizevars.size_hint(reduction_numel)
        #     < config.unroll_reductions_threshold
        #     and sympy_product(ranges) != 1
        # ):
        #     return Pointwise.create(
        #         device,
        #         dst_dtype,
        #         cls._unroll_reduction_fn(
        #             inner_fn, reduction_ranges, reduction_type, src_dtype,
        #         ),
        #         ranges,
        #     )

        # triton doesn't support reduce to single element well, so break it up
        hint, split = Reduction.num_splits(
            device,
            dtype,
            dtype,
            inner_fns[0],
            ranges,
            reduction_ranges,
            reduction_type=reduction_type,
            reduction_numel=reduction_numel,
        )
        # intermediate reduction in split can contain complex indexing,
        # and num_splits will fail to correctly set the hint
        # reuse the passed hint if available
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split > 1:
            # triton doesn't support reduce to single element well, so break it up
            return cls.create_multilayer(
                device,
                dtype,
                inner_fns,
                ranges,
                reduction_ranges,
                reduction_type,
                split,
                reduction_hint,
            )

        results = [
            TensorBox.create(
                WelfordReduction(
                    device,
                    dtype,
                    inner_fns,
                    ranges,
                    reduction_ranges,
                    reduction_type,
                    dtype,
                    reduction_hint,
                    output_idx,
                )
            )
            for output_idx in range(3)
        ]
        for t in results:
            t.realize()
        return results

    @staticmethod
    def default_value(
        reduction_type: str, dtype: torch.dtype
    ) -> Union[_NumLike, Sequence[_NumLike]]:
        return (0, 0, 0)

    @classmethod
    def create_multilayer(  # type: ignore[override]
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fns: Sequence[Callable[..., Any]],
        ranges: list[Integer],
        reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        split: _IntLike,
        reduction_hint: ReductionHint,
    ) -> Sequence[TensorBox]:
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        reduction_numel = sympy_product(reduction_ranges)
        need_mask = not V.graph.sizevars.is_expr_static_and_true(
            sympy.Eq(reduction_numel % split, 0)
        )

        if need_mask and reduction_type != "welford_combine":
            # If we need mask, then "welford_reduce" doesn't work because
            # masked inputs shouldn't count towards the welford weight

            def constant(
                idx: Sequence[Expr], reduction_idx: Sequence[Expr], value: int
            ) -> OpsValue:
                return ops.constant(value, dtype)

            return cls.create_multilayer(
                device=device,
                dtype=dtype,
                inner_fns=(
                    inner_fns[0],
                    partial(constant, value=0),
                    partial(constant, value=1),
                ),
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type="welford_combine",
                split=split,
                reduction_hint=reduction_hint,
            )

        block_size = FloorDiv(reduction_numel + (split - 1), split)
        intermediates = WelfordReduction.create(
            device,
            dtype,
            tuple(
                cls._multilayer_wrap_loader(
                    loader,
                    reduction_ranges,
                    reduction_numel,
                    split,
                    block_size,
                    default=0,
                )
                for loader in inner_fns
            ),
            [*ranges, split],
            [block_size],
            reduction_type,
            reduction_hint,
        )
        for i in intermediates:
            i.realize()

        def intermediate_loader_fn(
            index: Sequence[Expr],
            reduction_index: Sequence[Expr],
            loader: Callable[[Sequence[Expr]], OpsValue],
        ) -> OpsValue:
            return loader([*index, *reduction_index])

        numel_hint = V.graph.sizevars.size_hint(sympy_product(ranges))
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )
        return WelfordReduction.create(
            device,
            dtype,
            tuple(
                partial(intermediate_loader_fn, loader=i.make_loader())
                for i in intermediates
            ),
            ranges,
            [split],
            # welford_reduce turns one input into three outputs, which are combined with welford_combine
            "welford_combine",
            reduction_hint,
        )


@ir_dataclass
class Scan(Loops):
    scan_ranges: list[Integer]
    size: list[Integer]
    combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]]
    reindex: Callable[[Sequence[_IntLike], Sequence[_IntLike]], Sequence[_IntLike]]
    reduction_hint: ReductionHint
    output_index: int
    # output_index indexes the following tuples
    dtypes: tuple[torch.dtype, ...]
    inner_fns: tuple[Callable[..., Any], ...]

    # HACK we mimick reduction

    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        # TODO: Can combine_fn/reindex close over unbacked symbols? If so, we
        # need to explicitly represent the closure so we can pull out unbacked
        # symbols here
        return (
            super().get_free_symbol_uses(unbacked_only)
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.scan_ranges)
            )
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.size)
            )
        )

    def __post_init__(self) -> None:
        assert len(self.ranges) + len(self.scan_ranges) == len(self.size)
        super().__post_init__()

    def store_reduction(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[_IntLike]], Never],
        vars: Sequence[Expr],
        scan_vars: Sequence[Symbol],
    ) -> None:
        idx = self.reindex(vars, scan_vars)
        values = tuple(inner_fn(idx) for inner_fn in self.inner_fns)
        result = ops.scan(self.dtypes, self.combine_fn, values)
        return ops.store(
            output_name or "unnamed", indexer(idx), result[self.output_index]
        )

    def get_reduction_type(self) -> Optional[str]:
        # return self.scan_op
        return "custom"

    def get_reduction_size(self) -> Sequence[sympy.Expr]:
        return self.scan_ranges

    def get_size(self) -> Sequence[Expr]:
        return self.size

    def get_pointwise_size(self) -> Sequence[Expr]:
        return self.ranges

    def index_length(self) -> int:
        return len(self.ranges) + len(self.scan_ranges)

    def inner_fn_args(self) -> Sequence[Sequence[_IntLike]]:
        index = self._index(self.ranges)
        rindex = self._index(self.scan_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return (idx,)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        index = self._index(self.ranges)
        rindex = self._index(self.scan_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return extract_free_symbols(self.inner_fn, idx, unbacked_only=unbacked_only)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dtypes: tuple[torch.dtype, ...],
        inner_fns: tuple[Callable[[Sequence[Expr]], Any], ...],
        size: list[Integer],
        axis: int,
        combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]],
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        *,
        # Whether we have the option to fallback to aten
        can_fallback_to_aten: bool = True,
        **kwargs: Any,
    ) -> Sequence[Optional[TensorBox]]:
        pointwise_ranges = [*size[:axis], *size[axis + 1 :]]
        scan_ranges = [size[axis]]

        if not V.graph.has_feature(device, BackendFeature.SCAN):
            return [None] * len(dtypes)

        if len(dtypes) > 1 and not V.graph.has_feature(
            device, BackendFeature.TUPLE_REDUCTION
        ):
            return [None] * len(dtypes)

        sizevars = V.graph.sizevars
        scan_numel = sizevars.simplify(sympy_product(scan_ranges))

        assert len(dtypes) == len(inner_fns)

        # Scan with a single element is just a copy
        if sizevars.is_expr_static_and_true(sympy.Le(scan_numel, 1)):
            return [
                Pointwise.create(
                    device=device,
                    dtype=dtypes[output_index],
                    inner_fn=inner_fns[output_index],
                    ranges=size,
                )
                for output_index in range(len(dtypes))
            ]

        reduction_hint, num_splits = cls.num_splits(
            device=device,
            dtype=dtypes[0],
            inner_fn=inner_fns[0],
            axis=axis,
            pointwise_ranges=pointwise_ranges,
            scan_ranges=scan_ranges,
            combine_fn=combine_fn,
            scan_numel=scan_numel,
        )
        scan_type = Scan
        if num_splits > 1:
            supports_split = (
                torch.version.hip is None or (has_triton and triton_version >= "3.3.0")
            ) and (len(dtypes) == 1)
            if not supports_split:
                if can_fallback_to_aten:
                    # Fallback to ATen
                    return [None] * len(dtypes)
                else:
                    num_splits = 1
            else:
                scan_type = SplitScan

        def reindex(index: Sequence[Expr], scan_index: Sequence[Expr]) -> list[Expr]:
            assert len(scan_index) == len(scan_ranges)
            assert len(index) == len(pointwise_ranges)
            return [*index[:axis], *scan_index, *index[axis:]]

        results = [
            TensorBox.create(
                scan_type(
                    device=device,
                    dtype=dtypes[output_index],
                    dtypes=dtypes,
                    inner_fn=inner_fns[output_index],
                    inner_fns=inner_fns,
                    size=size,
                    ranges=pointwise_ranges,
                    scan_ranges=scan_ranges,
                    combine_fn=combine_fn,
                    reindex=reindex,
                    reduction_hint=reduction_hint,
                    output_index=output_index,
                    **kwargs,
                )
            )
            for output_index in range(len(dtypes))
        ]

        for result in results:
            result.realize()

        return results

    @classmethod
    def num_splits(
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fn: Callable[[Sequence[Expr]], OpsValue],
        axis: int,
        pointwise_ranges: list[Integer],
        scan_ranges: list[Integer],
        combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]],
        scan_numel: Expr,
    ) -> tuple[ReductionHint, _IntLike]:
        # TODO: custom splitting heuristic for scan
        def wrapper_fn(idx: Sequence[Expr], reduction_idx: Sequence[Expr]) -> OpsValue:
            return inner_fn([*idx[:axis], *reduction_idx, *idx[axis:]])

        return Reduction.num_splits(
            device=device,
            dst_dtype=dtype,
            src_dtype=dtype,
            inner_fn=wrapper_fn,
            ranges=pointwise_ranges,
            reduction_ranges=scan_ranges,
            reduction_type="scan",
            reduction_numel=scan_numel,
        )


# This signifies a scan op that should go through TritonSplitScanKernel codegen on CUDA.
@ir_dataclass
class SplitScan(Scan):
    pass


@ir_dataclass
class Sort(Loops):
    # Sorts a tuple of key, value pairs
    sort_ranges: list[Integer]
    size: list[Integer]
    reindex: Callable[[Sequence[Expr], Sequence[Expr]], Sequence[Expr]]
    reduction_hint: ReductionHint
    output_index: int
    # output_index indexes the following tuples
    dtypes: tuple[torch.dtype, ...]
    inner_fns: tuple[Callable[..., Any], ...]

    stable: bool
    descending: bool

    # HACK we mimick reduction

    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        return (
            super().get_free_symbol_uses(unbacked_only)
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.sort_ranges)
            )
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.size)
            )
        )

    def __post_init__(self) -> None:
        assert len(self.ranges) + len(self.sort_ranges) == len(self.size)
        super().__post_init__()

    def store_reduction(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Expr],
        vars: Sequence[Expr],
        reduction_vars: Sequence[Expr],
    ) -> None:
        idx = self.reindex(vars, reduction_vars)
        values = tuple(inner_fn(idx) for inner_fn in self.inner_fns)
        result = ops.sort(self.dtypes, values, self.stable, self.descending)
        return ops.store(
            output_name or "unnamed", indexer(idx), result[self.output_index]
        )

    def get_reduction_type(self) -> Optional[str]:
        return "sort"

    def get_reduction_size(self) -> Sequence[Expr]:
        return self.sort_ranges

    def get_size(self) -> Sequence[Expr]:
        return self.size

    def get_pointwise_size(self) -> Sequence[Expr]:
        return self.ranges

    def index_length(self) -> int:
        return len(self.ranges) + len(self.sort_ranges)

    def inner_fn_args(self) -> Sequence[Sequence[Expr]]:
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return (idx,)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return extract_free_symbols(self.inner_fn, idx, unbacked_only=unbacked_only)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dtypes: tuple[torch.dtype, ...],
        inner_fns: tuple[Callable[[list[Expr]], Any], ...],
        size: list[Integer],
        axis: int,
        stable: bool,
        descending: bool,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        **kwargs: Any,
    ) -> Sequence[Optional[TensorBox]]:
        pointwise_ranges = [*size[:axis], *size[axis + 1 :]]
        sort_ranges = [size[axis]]

        if not V.graph.has_feature(device, BackendFeature.SORT):
            return [None] * len(dtypes)

        sizevars = V.graph.sizevars
        sort_numel = sizevars.simplify(sympy_product(sort_ranges))

        # Heuristic, smallest rblock where triton usually outperforms aten.sort
        # It also isn't bandwidth bound so fusion is unlikely to help.
        max_rblock = 512
        is_persistent_kernel = (
            config.triton.persistent_reductions
            and sizevars.is_expr_static_and_true(sympy.Le(sort_numel, max_rblock))
        )
        if not is_persistent_kernel:
            # We only support persistent triton kernels
            return [None] * len(dtypes)

        assert len(dtypes) == len(inner_fns)

        # Sort with a single element is just a copy
        if sizevars.is_expr_static_and_true(sympy.Le(sort_numel, 1)):
            return [
                Pointwise.create(
                    device=device,
                    dtype=dtypes[output_index],
                    inner_fn=inner_fns[output_index],
                    ranges=size,
                )
                for output_index in range(len(dtypes))
            ]

        def reindex(index: Sequence[Expr], sort_index: Sequence[Expr]) -> list[Expr]:
            assert len(sort_index) == len(sort_ranges)
            assert len(index) == len(pointwise_ranges)
            return [*index[:axis], *sort_index, *index[axis:]]

        results = [
            TensorBox.create(
                Sort(
                    device=device,
                    dtype=dtypes[output_index],
                    dtypes=dtypes,
                    inner_fn=inner_fns[output_index],
                    inner_fns=inner_fns,
                    size=size,
                    ranges=pointwise_ranges,
                    sort_ranges=sort_ranges,
                    reindex=reindex,
                    reduction_hint=reduction_hint,
                    output_index=output_index,
                    stable=stable,
                    descending=descending,
                    **kwargs,
                )
            )
            for output_index in range(len(dtypes))
        ]

        for result in results:
            result.realize()

        return results


def is_storage_and_layout(x: IRNode) -> bool:
    try:
        as_storage_and_layout(x, freeze=False)
        return True
    except NotImplementedError:
        return False


def is_contiguous_storage_and_layout(x: IRNode) -> bool:
    try:
        _buffer, layout = as_storage_and_layout(x, freeze=False)
        # pad the stride here so we will NOT claim an tensor as contiguous
        # if a padding is gonna happen.
        if layout.should_pad_strides():
            layout.pad_strides()
        return layout.is_contiguous()
    except NotImplementedError:
        return False


def as_storage_and_layout(
    x: IRNode,
    freeze: bool = True,
    want_contiguous: bool = False,
    stride_order: Optional[Sequence[Union[int, Integer]]] = None,
    allow_padding: bool = False,
    exact_strides: Optional[Sequence[Union[int, Integer]]] = None,
) -> tuple[StorageBox, Layout]:
    """
    Try to simplify x into a StorageBox and a Layout.

    allow_padding only affect how we apply stride_order. When allow_padding
    is True, we have the freedom to add padding when applying the stride_order.
    """
    if isinstance(x, TensorBox):
        return as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
            allow_padding=allow_padding,
            exact_strides=exact_strides,
        )
    if isinstance(x, StorageBox):
        _, layout = as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
            allow_padding=allow_padding,
            exact_strides=exact_strides,
        )
        return x, x.data.get_layout()
    if isinstance(x, Buffer):
        if freeze:
            if want_contiguous:
                x.freeze_layout()
                assert x.get_layout().is_contiguous()
            elif stride_order is not None:
                x.freeze_layout_with_stride_order(
                    stride_order, allow_padding=allow_padding
                )
            elif exact_strides is not None:
                x.freeze_layout_with_exact_strides(
                    exact_strides, allow_padding=allow_padding
                )
            else:
                x.decide_layout()
        return StorageBox(x), x.get_layout()
    if isinstance(x, ReinterpretView):
        # making the base of x contiguous or stride_ordered will not necessarily make
        # the ReinterpretView either, so don't pass along those arguments
        buffer, _ = as_storage_and_layout(
            x.data,
            freeze=freeze,
        )
        return buffer, x.layout
    raise NotImplementedError


def is_stride_order_storage_and_layout(
    x: IRNode, stride_order: Sequence[Union[int, Integer]]
) -> bool:
    try:
        _buffer, layout = as_storage_and_layout(x, freeze=False)
        return layout.is_stride_ordered(stride_order)
    except NotImplementedError:
        return False


def is_unaligned(node: IRNode) -> bool:
    if isinstance(node, (TensorBox, StorageBox)):
        return is_unaligned(node.data)

    if isinstance(node, ReinterpretView):
        layout = node.layout
        has_unaligned_layout = not statically_known_true(
            layout.offset * get_dtype_size(layout.dtype) % GPU_ALIGN_BYTES == 0
        )
        return is_unaligned(node.data) or has_unaligned_layout

    if isinstance(node, Buffer):
        return node.get_name() in V.graph.unaligned_buffers

    # assume to be aligned otherwise
    return False


@ir_dataclass
class BaseView(IRNode):
    data: IRNode

    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        return self.data.get_free_symbol_uses(unbacked_only)

    def make_reindexer(self) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        raise NotImplementedError(f"make_reindexer NYI on {self}")

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        inner = self.data.make_indexer()
        reindex = self.make_reindexer()

        def indexer(idx: Sequence[Expr]) -> Expr:
            return inner(reindex(idx))

        return indexer

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        inner = self.data.make_loader()
        reindex = self.make_reindexer()

        def loader(idx: Sequence[Expr]) -> OpsValue:
            return inner(reindex(idx))

        return loader

    @property
    def dtype(self) -> torch.dtype:
        return self.data.get_dtype()

    def get_layout(self) -> Layout:
        return self.data.get_layout()

    def get_device(self) -> Optional[torch.device]:
        return self.data.get_device()

    def get_origin_node(self) -> Optional[torch.fx.Node]:
        return None

    def get_name(self) -> str:
        return self.data.get_name()

    def get_pointwise_size(self) -> Sequence[Expr]:
        return self.get_size()

    def mark_reuse(self, users: int) -> None:
        return self.data.mark_reuse(users)

    def has_exceeded_max_reads(self) -> bool:
        return self.data.has_exceeded_max_reads()

    def realize(self) -> Optional[str]:
        return self.data.realize()

    def realize_hint(self):  # type: ignore[no-untyped-def]
        return self.data.realize_hint()

    def get_storage_numel(self):  # type: ignore[no-untyped-def]
        return self.data.get_storage_numel()

    def is_extern(self) -> bool:
        return self.data.is_extern()  # type: ignore[attr-defined]

    def is_module_buffer(self) -> bool:
        return self.data.is_module_buffer()  # type: ignore[attr-defined]

    def get_read_names(self) -> OrderedSet[str]:
        return self.data.get_read_names()

    def get_reads(self) -> OrderedSet[Dep]:
        with patch.object(FlexibleLayout, "allow_indexing", True):
            return extract_read_writes(
                self.make_loader(),
                self.get_size(),  # type: ignore[arg-type]
            ).reads

    def unwrap_view(self):  # type: ignore[no-untyped-def]
        x: IRNode = self
        while isinstance(x, BaseView):
            x = x.data
        return x

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(
            device=device,
            dtype=self.get_dtype(),
            inner_fn=loader,
            ranges=self.get_size(),
        )


@ir_dataclass
class ExpandView(BaseView):
    size: list[Expr]

    @staticmethod
    def _normalize_size(x, new_size):  # type: ignore[no-untyped-def]
        """Replace `-1` with correct sizes"""
        sizevars = V.graph.sizevars
        new_size = list(map(sympy.expand, new_size))
        old_size = x.get_size()
        old_size = [None] * (len(new_size) - len(old_size)) + list(old_size)
        assert len(new_size) == len(old_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                assert old_size[i] is not None
                new_size[i] = old_size[i]
            elif old_size[i] is None or V.graph.sizevars.shape_env.evaluate_expr(
                sympy.Eq(old_size[i], 1), size_oblivious=True
            ):
                pass
            else:
                # Sanity check: Expect broadcast compatibility
                #
                # NB: new_size[i] == old_size[i] is expected to already be
                # guarded because the meta formula was expected to have taught
                # us this equality.
                assert sizevars.size_hint(new_size[i] - old_size[i], fallback=0) == 0, (
                    "Broadcast failed in ExpandView({x.get_size()}, {new_size}) on dimension {i}"
                )
        return new_size

    @classmethod
    def create(cls, x, new_size):  # type: ignore[no-untyped-def]
        new_size = cls._normalize_size(x, new_size)

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            skip = len(new_size) - len(old_layout.size)
            assert skip >= 0
            new_stride = [sympy.S.Zero] * skip
            for stride, size in zip(old_layout.stride, old_layout.size):
                new_stride.append(
                    stride
                    if not V.graph.sizevars.shape_env.evaluate_expr(
                        sympy.Eq(size, 1), size_oblivious=True
                    )
                    else sympy.S.Zero
                )
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                list(new_size),
                new_stride,
                old_layout.offset,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        return ExpandView(data=x, size=new_size)

    def get_size(self) -> Sequence[Expr]:
        return self.size

    def make_reindexer(self):  # type: ignore[no-untyped-def]
        target = self.get_size()
        actual = self.data.get_size()
        skip = len(target) - len(actual)

        def reindex(index):  # type: ignore[no-untyped-def]
            index = list(index[skip:])
            assert len(index) == len(actual)
            for i in range(len(actual)):
                if actual[i] == 1:
                    # zero out broadcast dimension
                    index[i] = sympy.S.Zero
            return index

        return reindex


@ir_dataclass
class PermuteView(BaseView):
    dims: list[Expr]

    @classmethod
    def create(cls, x, dims):  # type: ignore[no-untyped-def]
        dims = cls._map_neg_dims(dims)
        assert OrderedSet(dims) == OrderedSet(range(len(dims)))

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                [old_layout.size[i] for i in dims],
                [old_layout.stride[i] for i in dims],
                old_layout.offset,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        return PermuteView(data=x, dims=dims)

    @classmethod
    def _map_neg_dims(cls, dims):  # type: ignore[no-untyped-def]
        return [dim if dim >= 0 else len(dims) + dim for dim in dims]

    def get_size(self) -> Sequence[Expr]:
        assert OrderedSet(self._map_neg_dims(self.dims)) == OrderedSet(
            range(len(self.dims))
        )
        size = self.data.get_size()
        return [size[i] for i in self.dims]

    def make_reindexer(self):  # type: ignore[no-untyped-def]
        inv = {j: i for i, j in enumerate(self.dims)}
        inv = [inv[i] for i in range(len(self.dims))]
        assert OrderedSet(inv) == OrderedSet(range(len(self.dims)))

        def reindex(index):  # type: ignore[no-untyped-def]
            return [index[i] for i in inv]

        return reindex


@ir_dataclass
class SqueezeView(BaseView):
    @classmethod
    def create(cls, x, *, dim=None):  # type: ignore[no-untyped-def]
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_size = []
            new_stride = []
            if dim is not None:
                assert isinstance(dim, int), "expected integer dim argument"
                assert 0 <= dim and dim < len(old_layout.size)

            for i, (size, stride) in enumerate(zip(old_layout.size, old_layout.stride)):
                if dim is None:
                    if size != 1:
                        new_size.append(size)
                        new_stride.append(stride)
                else:
                    if i != dim:
                        new_size.append(size)
                        new_stride.append(stride)
                    else:
                        assert size == 1, "expected squeezed size to be 1"

            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        if dim is None:
            # redirect to a generic view
            return View.create(x, [s for s in x.get_size() if s != 1])
        else:
            assert x.get_size()[dim] == 1
            return View.create(x, [s for i, s in enumerate(x.get_size()) if i != dim])

    @staticmethod
    def squeezer(size: Sequence[sympy.Expr]):  # type: ignore[no-untyped-def]
        new_size = [s for s in size if s != 1]
        not_one = [i for i, s in enumerate(size) if s != 1]
        length = len(size)

        def reindex(index: list[sympy.Expr]) -> tuple[sympy.Expr, ...]:
            assert len(index) == len(not_one), f"{index} {not_one}"
            new_index = [sympy.S.Zero] * length
            for idx, s in zip(not_one, index):
                new_index[idx] = s
            return tuple(new_index)

        return new_size, reindex

    def __init__(self, data) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("use SqueezeView.create()")


@ir_dataclass
class GenericView(BaseView):
    size: list[Expr]
    reindex: Callable[..., Any]

    def make_reindexer(self):  # type: ignore[no-untyped-def]
        return self.reindex

    def reindex_str(self) -> str:
        index_old = [
            sympy_index_symbol_with_prefix(SymT.INDEX, n) for n in range(len(self.size))
        ]
        index_new = list(self.reindex(index_old))
        return f"lambda {', '.join(map(str, index_old))}: {index_new}"

    def __str__(self) -> str:
        return self.str_helper(
            [self.data, f"size={self.size}", f"reindex={self.reindex_str()}"]
        )

    __repr__ = __str__

    @classmethod
    def create(cls, x, new_size, reindex):  # type: ignore[no-untyped-def]
        return cls(data=x, size=list(new_size), reindex=reindex)

    def get_size(self) -> Sequence[Expr]:
        return self.size


@ir_dataclass
class View(GenericView):
    @staticmethod
    def handle_negative_index(idx, size):  # type: ignore[no-untyped-def]
        idx = sympy.expand(idx)
        size = sympy.expand(size)
        evaluate_expr = V.graph.sizevars.shape_env.evaluate_expr
        if evaluate_expr(sympy.Lt(idx, 0)):
            idx = idx + size
        return idx

    @classmethod
    def create(cls, x, new_size):  # type: ignore[no-untyped-def]
        assert isinstance(new_size, (tuple, list))
        old_size, new_size = cls.resolve_negative_size(x.get_size(), new_size)

        # Skip pointless views
        if V.graph.sizevars.statically_known_list_equals(old_size, new_size):
            return x

        unbacked_symbols_in_sizes = False
        if (
            len(free_unbacked_symbols(old_size)) > 0
            or len(free_unbacked_symbols(new_size)) > 0
        ):
            unbacked_symbols_in_sizes = True

        if 0 in new_size:

            def fake_reindex(index):  # type: ignore[no-untyped-def]
                return tuple([0] * len(old_size))

            return cls(data=x, size=list(new_size), reindex=fake_reindex)
        # TODO: a new class for FixedTransferLayout that output layout is constrained by input layout
        elif is_contiguous_storage_and_layout(x) or unbacked_symbols_in_sizes:
            if unbacked_symbols_in_sizes and (not is_contiguous_storage_and_layout(x)):
                # realize x; otherwise, the dynamic_reshape_indexer below will fail
                # due to the size_hint's inability to process unbacked SymInts
                # TODO: unbacked should not diverge from backed in determining striding
                # Need to require contiguous here instead of realize, see:
                # https://github.com/pytorch/pytorch/issues/145561
                x = ExternKernel.require_exact_strides(
                    x, FlexibleLayout.contiguous_strides(x.get_size())
                )

            storage, old_layout = as_storage_and_layout(x, want_contiguous=True)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                FlexibleLayout.contiguous_strides(new_size),
                old_layout.offset,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        reindex = cls.dynamic_reshape_indexer(old_size, new_size)
        return cls(data=x, size=list(new_size), reindex=reindex)

    @staticmethod
    def resolve_negative_size(old_size, new_size):  # type: ignore[no-untyped-def]
        new_size = [V.graph.sizevars.simplify(x) for x in new_size]
        old_size = [V.graph.sizevars.simplify(x) for x in old_size]

        new_size = list(new_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                new_size[i] = sympy.S.One
                new_size[i] = CleanDiv(sympy_product(old_size), sympy_product(new_size))
                break

        V.graph.sizevars.guard_equals(sympy_product(old_size), sympy_product(new_size))
        return old_size, new_size

    @classmethod
    def dynamic_reshape_indexer(
        cls,
        old_size: Sequence[_IntLike],
        new_size: Sequence[_IntLike],
        dense_dim: Optional[int] = None,  # type: ignore[no-untyped-def]
    ) -> Callable[[Sequence[_T]], Sequence[_V]]:
        try:
            reindex = cls._dynamic_reshape_indexer(old_size, new_size, dense_dim)
        except (AssertionError, IndexError):
            # optimistic algorithm failed, lets do a fallback
            flat = [sympy_product(old_size)]
            reindex1 = cls._dynamic_reshape_indexer(old_size, flat)
            reindex2 = cls._dynamic_reshape_indexer(flat, new_size)
            reindex = fuse_reindexing(reindex1, reindex2)
        return reindex

    @staticmethod
    def _dynamic_reshape_indexer(old_size, new_size, dense_dim: Optional[int] = None):  # type: ignore[no-untyped-def]
        """
        Perform a reshape entirely by modifying indexing math
        """
        size_hint = V.graph.sizevars.size_hint
        # TODO: These symbols may not escape, if they don't assert so and
        # treat them as temporary
        vars = [
            sympy_index_symbol_with_prefix(SymT.VIEW, i) for i in range(len(new_size))
        ]

        stack_new = list(zip(vars, new_size))
        stack_old = list(old_size)

        # process the dense dim first
        reordering_dense_dim = (
            dense_dim is not None
            and dense_dim != len(stack_old) - 1
            and len(new_size) == 1
        )
        if reordering_dense_dim:
            assert dense_dim is not None  # mypy
            old_dim = stack_old.pop(dense_dim)
            stack_old.append(old_dim)

        view_expr = []
        while stack_new and stack_old:
            size_old = stack_old.pop()
            var, size_new = stack_new.pop()
            if size_old == 1:
                view_expr.append(sympy.S.Zero)
                stack_new.append((var, size_new))  # re-add
            elif size_new == 1:
                stack_old.append(size_old)  # re-add
            elif size_hint(size_new) == size_hint(size_old):
                view_expr.append(var)
                V.graph.sizevars.guard_equals(size_new, size_old)
            elif size_hint(size_new) < size_hint(size_old):
                while size_hint(size_new) < size_hint(size_old):
                    var2, size_new2 = stack_new.pop()
                    var = var2 * size_new + var
                    size_new = size_new * size_new2
                view_expr.append(var)
                V.graph.sizevars.guard_equals(size_new, size_old)
            elif size_hint(size_new) > size_hint(size_old):
                divisor = sympy.S.One
                modulus = size_old
                view_expr.append(ModularIndexing(var, divisor, modulus))
                divisor = divisor * modulus
                while size_hint(size_new) > size_hint(size_old):
                    modulus = stack_old.pop()
                    view_expr.append(ModularIndexing(var, divisor, modulus))
                    divisor = divisor * modulus
                    size_old = size_old * modulus
                V.graph.sizevars.guard_equals(size_new, size_old)
            else:
                raise AssertionError

        while stack_old:
            size_old = stack_old.pop()
            V.graph.sizevars.guard_equals(size_old, 1)
            view_expr.append(sympy.S.Zero)

        while stack_new:
            var, size_new = stack_new.pop()
            V.graph.sizevars.guard_equals(size_new, 1)

        if dense_dim is not None and len(new_size) == 1:
            view_expr.reverse()
            # Move the last expression (dense dim) to its original position
            dense_expr = view_expr.pop()
            view_expr.insert(dense_dim, dense_expr)
        else:
            view_expr.reverse()

        assert len(view_expr) == len(old_size)

        def reindex(index):  # type: ignore[no-untyped-def]
            assert len(index) == len(vars), (len(index), len(vars))
            replacements = dict(zip(vars, index))
            return tuple(sympy_subs(x, replacements) for x in view_expr)

        return reindex


@ir_dataclass
class ReinterpretView(BaseView):
    """Pretend our storage has a different layout"""

    layout: Layout

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.data, BaseView):
            object.__setattr__(self, "data", self.data.unwrap_view())

    def __str__(self) -> str:
        return self.str_helper(
            [
                self.data,
                self.layout,
            ]
        )

    __repr__ = __str__

    def get_name(self):  # type: ignore[no-untyped-def]
        return self.data.get_name()

    def get_device(self) -> Optional[torch.device]:
        return self.layout.device

    def get_origin_node(self) -> Optional[torch.fx.Node]:
        return None

    @property
    def dtype(self):  # type: ignore[no-untyped-def]
        return self.layout.dtype

    def get_size(self) -> Sequence[Expr]:
        return list(self.layout.size)

    def get_stride(self):  # type: ignore[no-untyped-def]
        return list(self.layout.stride)

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        def loader(index: Sequence[Expr]) -> OpsValue:
            indexer = self.layout.make_indexer()
            tmp_loader = ops.load(self.get_name(), indexer(index))
            if self.layout.dtype != self.data.dtype:
                return ops.to_dtype_bitcast(tmp_loader, self.dtype, self.data.dtype)
            else:
                return tmp_loader

        return loader

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        return self.layout.make_indexer()

    def get_layout(self) -> Layout:
        return self.layout

    def freeze_layout(self):  # type: ignore[no-untyped-def]
        pass

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return (
            get_free_symbols(self.layout.size, unbacked_only)
            | get_free_symbols(self.layout.stride, unbacked_only)
            | get_free_symbols(self.layout.offset, unbacked_only)
        )

    def codegen_reference(self, writer: Optional[IndentedBuffer] = None) -> str:
        # reinterpret_tensor is similar to as_strided except:
        # - offset is added to the existing offset (rather than replacing it)
        # - view tracking is disabled similar to unsafe_view
        return V.graph.wrapper_code.codegen_reinterpret_view(
            self.data,
            self.layout.size,
            self.layout.stride,
            self.layout.offset,
            writer.writeline if writer is not None else V.graph.wrapper_code.writeline,
            dtype=self.layout.dtype,
        )

    def num_reads(self) -> int:
        return 1


@ir_dataclass
class DtypeView(BaseView):
    """Pretend our storage has a different type"""

    target_dtype: torch.dtype

    @classmethod
    def create(cls, x, new_dtype):  # type: ignore[no-untyped-def]
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                new_dtype,
                old_layout.size,
                old_layout.stride,
                old_layout.offset,
            )
            return ReinterpretView(data=storage, layout=new_layout)
        return DtypeView(data=x, target_dtype=new_dtype)

    def __str__(self) -> str:
        return self.str_helper([self.data, self.target_dtype])

    __repr__ = __str__

    @property
    def dtype(self):  # type: ignore[no-untyped-def]
        return self.target_dtype

    def get_size(self) -> Sequence[Expr]:
        return self.data.get_size()

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        inner = self.data.make_loader()

        def loader(idx):  # type: ignore[no-untyped-def]
            return ops.to_dtype_bitcast(inner(idx), self.target_dtype, self.data.dtype)

        return loader


class SliceView(View):
    @classmethod
    def normalize_start_end(cls, x, dim, start, end):  # type: ignore[no-untyped-def]
        """
        Normalize start and end such that both are in the range
        [0, x.get_size()[dim]] and start <= end.
        """
        sizevars = V.graph.sizevars
        dim_size = x.get_size()[dim]

        if any(free_unbacked_symbols(x) for x in (start, end, dim_size)):
            min_func = sympy.Min
            max_func = sympy.Max
        else:
            min_func = sizevars.evaluate_min
            max_func = sizevars.evaluate_max

        def clamp(x, lower, upper):  # type: ignore[no-untyped-def]
            clamped_lower = (
                x if sizevars.statically_known_geq(x, lower) else max_func(x, lower)
            )
            clamped_full = (
                clamped_lower
                if sizevars.statically_known_leq(clamped_lower, upper)
                else min_func(clamped_lower, upper)
            )
            return clamped_full

        def clamp_wrap(val, lower, upper, default):  # type: ignore[no-untyped-def]
            if val is None:
                return default
            val = cls.handle_negative_index(val, dim_size)
            return clamp(val, lower, upper)

        start = clamp_wrap(start, 0, dim_size, 0)
        end = clamp_wrap(end, start, dim_size, dim_size)
        return start, end

    @classmethod
    def create(cls, x, dim, start, end, step=1, clamp=True):  # type: ignore[no-untyped-def]
        step = sympy.expand(step)
        assert isinstance(step, sympy.Expr) or step > 0
        try:
            if start == 0 and end >= 2**63 - 1 and step == 1:
                return x
        except TypeError:
            pass

        new_size = list(x.get_size())

        # NB: Ordinarily we default to clamping.
        # We only don't clamp for split_with_sizes. For split_with_sizes, sizes should be already valid
        # failing in this situation is ok, since invalid sizes could trigger silent errors.
        if clamp:
            start, end = cls.normalize_start_end(x, dim, start, end)

        new_size[dim] = FloorDiv(end - start + (step - 1), step)

        if is_storage_and_layout(x):
            # Fast path
            storage, old_layout = as_storage_and_layout(x)
            new_stride = list(old_layout.stride)
            new_stride[dim] = new_stride[dim] * step
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset + old_layout.stride[dim] * start,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        def reindex(index):  # type: ignore[no-untyped-def]
            assert len(index) == len(new_size), f"wrong ndim {index} {new_size}"
            index = list(index)
            index[dim] = index[dim] * step + start
            return index

        # redirect to a generic view
        return SliceView(data=x, size=new_size, reindex=reindex)


@ir_dataclass
class BaseConstant(IRNode):
    dtype: torch.dtype
    device: torch.device

    def get_size(self) -> Sequence[Expr]:
        return ()

    def get_device(self) -> Optional[torch.device]:
        return self.device

    def get_origin_node(self) -> Optional[torch.fx.Node]:
        return None

    def get_reads(self) -> OrderedSet[Dep]:
        return OrderedSet()


@ir_dataclass
class Constant(BaseConstant):
    value: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        def loader(index: Sequence[Expr]) -> OpsValue:
            return ops.constant(self.value, self.dtype)

        return loader

    def realize(self) -> Optional[str]:
        pass

    def constant_to_device(self, device: torch.device) -> IRNode:
        return Constant(value=self.value, dtype=self.dtype, device=device)


@ir_dataclass
class IndexingConstant(BaseConstant):
    index: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        def loader(index: Sequence[Expr]) -> OpsValue:
            return ops.index_expr(self.index, self.dtype)

        return loader

    def constant_to_device(self, device: torch.device) -> IRNode:
        return IndexingConstant(index=self.index, dtype=self.dtype, device=device)


def is_contiguous_strides_for_shape(
    stride: Sequence[_IntLike], shape: Sequence[_IntLike]
) -> bool:
    return all(
        size == 1 or left == right
        for left, right, size in zip(
            stride, FlexibleLayout.contiguous_strides(shape), shape
        )
    )


def get_align_for_dtype(dtype: torch.dtype) -> int:
    return config.padding_alignment_bytes // dtype.itemsize


class OutputSpec:
    """Abstract base for Layout, MultiOutputLayout, NoneLayout.
    Represents the memory layout of the output of an Operation."""

    def get_device(self) -> Optional[torch.device]:
        raise NotImplementedError(type(self).__name__)

    def storage_size(self) -> int:
        raise NotImplementedError(type(self).__name__)


@ir_dataclass
class Layout(OutputSpec):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: list[Expr],
        stride: Optional[list[Expr]] = None,
        offset: Expr = Integer(0),
    ) -> None:
        if stride is None:
            stride = FlexibleLayout.contiguous_strides(size)
        self.device = device
        self.dtype = dtype
        assert len(size) == len(stride), f"size={size}, stride={stride}"
        assert all(isinstance(s, (Expr, int)) for s in size)
        self.size: list[Expr] = size
        self.stride: list[Expr] = stride
        self.offset: Expr = offset

    def __str__(self) -> str:
        offset = ""
        if self.offset != 0:
            offset = f", offset={self.offset}"

        device_index_str = "" if self.device.index is None else f":{self.device.index}"
        return (
            f"{type(self).__name__}('{self.device.type}{device_index_str}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}{offset})"
        )

    __repr__ = __str__

    def get_device(self) -> torch.device:
        return self.device

    def get_example(self) -> torch.Tensor:
        with V.fake_mode:
            return torch.empty_strided(
                convert_shape_to_symint(self.size),
                convert_shape_to_symint(self.stride),
                dtype=self.dtype,
                device=self.device,
            )

    def is_contiguous(self) -> bool:
        return is_contiguous_strides_for_shape(self.stride, self.size)

    @staticmethod
    def is_channels_last_contiguous(
        shape: Sequence[_IntLike], strides: Sequence[_IntLike]
    ) -> bool:
        ndim = len(shape)
        if ndim not in [4, 5] or shape[1] == 1:
            return False
        for left, right, size in zip(
            strides, make_channels_last_strides_for(shape), shape
        ):
            if size != 1 and left != right:
                return False
        return True

    def is_transposed(self) -> bool:
        for left, right, size in zip(
            self.stride,
            reversed(FlexibleLayout.contiguous_strides(list(reversed(self.size)))),
            self.size,
        ):
            if size != 1 and left != right:
                return False
        return True

    def is_stride_ordered(self, order) -> bool:  # type: ignore[no-untyped-def]
        assert len(self.stride) == len(order)

        # ignore dimensions of size 1, they dont affect layout
        non_1_indices = [
            i
            for i, dim in enumerate(self.size)
            if V.graph.sizevars.size_hint(dim, fallback=2) != 1
        ]

        stride = [self.stride[i] for i in non_1_indices]
        order = [order[i] for i in non_1_indices]

        def sorted_indices(arr):  # type: ignore[no-untyped-def]
            sorted_arr = sorted(arr)
            return [sorted_arr.index(element) for element in arr]

        # since we may have removed dimensions, need to re-sort & re-index order
        order = sorted_indices(order)

        # reorder the stride given order
        stride_ordered = [-1] * len(order)
        for i in range(len(order)):
            stride_ordered[order[i]] = stride[i]
        # check if it is in ascending order
        for i in range(len(order) - 1):
            expr = stride_ordered[i] > stride_ordered[i + 1]
            if not isinstance(expr, bool):
                expr = V.graph._shape_env.evaluate_expr(
                    stride_ordered[i] > stride_ordered[i + 1], size_oblivious=True
                )
            if expr:
                return False
        return True

    def is_channels_last_stride_ordered(self):  # type: ignore[no-untyped-def]
        # create channels_last order(NCHW, NCDHW, the C is the first order).
        order = [0] + list(reversed(range(1, len(self.stride) - 1)))
        order = [len(order)] + order
        return self.is_stride_ordered(order)

    @staticmethod
    def _pad_strides(in_strides, size, dtype):  # type: ignore[no-untyped-def]
        """
        The padding does not change stride order but makes sure all strides larger
        than the threshold are multiple of align.
        """
        align = get_align_for_dtype(dtype)
        if len(in_strides) == 0:
            return in_strides

        if not config.pad_channels_last and Layout.is_channels_last_contiguous(
            size, in_strides
        ):
            return in_strides

        current_fx_node = V.get_current_node()
        if hasattr(current_fx_node, "meta") and current_fx_node.meta.get(
            "dislike_padding", False
        ):
            return in_strides

        # get_stride_order does not work with dynamic shape. Also we can not
        # statically decide if a padding is needed or how much padding we should
        # do for dynamic shape.
        #
        # Skip padding the strides for dynamic shape for now.
        if not all(
            isinstance(s, (int, sympy.Integer))
            for s in itertools.chain(in_strides, size)
        ):
            return in_strides

        stride_order = get_stride_order(in_strides)
        fill_order = stride_order2fill_order(stride_order)

        new_strides = [0 for _ in range(len(in_strides))]
        # since we pad when the layout is flexible, we can decide the
        # smallest stride to be 1.
        new_strides[fill_order[0]] = 1

        padded = False
        for rank, idx in enumerate(fill_order[1:], start=1):
            prev_idx = fill_order[rank - 1]
            stride = new_strides[prev_idx] * size[prev_idx]

            if stride > config.padding_stride_threshold and stride % align != 0:
                stride = ceildiv(stride, align) * align
                padded = True
            new_strides[idx] = stride

        if not padded:
            # Consider a tensor with shape [256, 1, 5, 5]
            # Avoid strides like [25, 5, 5, 1] being padded to equivalent strides
            # [25, 25, 5, 1].
            return in_strides

        metrics.num_comprehensive_padding += 1
        return new_strides

    def pad_strides(self):  # type: ignore[no-untyped-def]
        assert isinstance(self, FlexibleLayout)
        assert self.stride is not None
        self.stride = self._pad_strides(self.stride, self.size, self.dtype)

    def should_pad_strides(self):  # type: ignore[no-untyped-def]
        return config.comprehensive_padding and isinstance(self, FlexibleLayout)

    def as_fixed(self):  # type: ignore[no-untyped-def]
        if isinstance(self, FixedLayout):
            return self

        if self.should_pad_strides():
            self.pad_strides()
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            self.stride,
            self.offset,
        )

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        assert FlexibleLayout.allow_indexing, (
            f"convert {type(self).__name__} to FixedLayout first"
        )
        return self.as_fixed().make_indexer()

    def __eq__(self, other) -> bool:  # type: ignore[no-untyped-def]
        return (
            self.device == other.device
            and self.dtype == other.dtype
            and self.size == other.size
            and self.stride == other.stride
            and self.offset == other.offset
        )

    def storage_size(self) -> sympy.Expr:
        return compute_required_storage_length(self.size, self.stride, self.offset)


class FixedLayout(Layout):
    """A Tensor layout we cannot change"""

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        """A closure containing math to read a given element"""

        def indexer(index):  # type: ignore[no-untyped-def]
            assert len(index) == len(self.stride)
            assert len(index) == len(self.size)
            result = self.offset
            for idx, stride, sz in zip(index, self.stride, self.size):
                if sz != 1:
                    result = result + idx * stride
            return result

        return indexer


class FlexibleLayout(Layout):
    """A Tensor layout we are allowed to change"""

    allow_indexing = False

    # WARNING!  This doesn't handle zero size tensors correctly
    @staticmethod
    def contiguous_strides(sizes):  # type: ignore[no-untyped-def]
        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.S.One]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    @staticmethod
    def fill_ordered(sizes, order):  # type: ignore[no-untyped-def]
        """
        Create a stride based on the order the dimensions should be filled in.

        In this format, channels last would be:
            [1, 3, 2, 0]
        """
        assert OrderedSet(range(len(sizes))) == OrderedSet(order), (sizes, order)
        next_stride = sympy.S.One
        strides = [None] * len(order)

        for i in order:
            strides[i] = next_stride
            next_stride = next_stride * sizes[i]
        return strides

    @staticmethod
    def stride_ordered(sizes, order):  # type: ignore[no-untyped-def]
        """
        Create a stride based on the sorted order of a permuted range.

        In this format, channels last would be:
            [3, 0, 2, 1]
        """
        assert OrderedSet(range(len(sizes))) == OrderedSet(order)
        fill_order = stride_order2fill_order(order)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @staticmethod
    def stride_ordered_for_memory_format(sizes, memory_format):  # type: ignore[no-untyped-def]
        """
        Create a stride based on a memory format.

        Memory format is translasted into a stride order,
        so channels_last is the same as:
            FlexibleLayout.stride_ordered(sizes, [3, 0, 2, 1])

        This interface does not support memory_format `torch.preserve_format`
        which should be used to deduce a format from another source
        """
        if memory_format == torch.channels_last:
            return FlexibleLayout.stride_ordered(sizes, NHWC_STRIDE_ORDER)
        elif memory_format == torch.channels_last_3d:
            return FlexibleLayout.stride_ordered(sizes, NHWDC_STRIDE_ORDER)
        elif memory_format == torch.contiguous_format:
            return FlexibleLayout.contiguous_strides(sizes)
        else:
            log.debug(
                "stride_ordered_for_memory_format, unsuppored memory_format: %s",
                memory_format,
            )
            raise NotImplementedError

    @staticmethod
    def same_ordered(sizes, stride):  # type: ignore[no-untyped-def]
        """
        Create a stride that has the same stride order as given stride

        For example, if given stride is [1000, 1, 100, 10],
        the fill order should be [1, 3, 2, 0]
        """
        assert len(sizes) == len(stride)
        stride = [V.graph.sizevars.size_hint(x) for x in stride]
        fill_order = sorted(range(len(stride)), key=stride.__getitem__)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    def as_stride_order(self, order, allow_padding=False):  # type: ignore[no-untyped-def]
        new_stride = self.stride_ordered(self.size, order)
        if self.should_pad_strides() and allow_padding:
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)

        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
        )

    def as_exact_strides(self, exact_strides, allow_padding=False):  # type: ignore[no-untyped-def]
        new_stride = exact_strides
        if self.should_pad_strides() and allow_padding:
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)

        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
        )

    def as_fill_order(self, order):  # type: ignore[no-untyped-def]
        new_stride = self.fill_ordered(self.size, order)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
        )

    def as_same_order(self, stride):  # type: ignore[no-untyped-def]
        new_stride = self.same_ordered(self.size, stride)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
        )

    def __init__(self, device, dtype, size, stride_order=None) -> None:  # type: ignore[no-untyped-def]
        if stride_order:
            strides = FlexibleLayout.fill_ordered(size, stride_order)
        else:
            strides = FlexibleLayout.contiguous_strides(size)
        super().__init__(device, dtype, size, strides)


class NonOwningLayout(Layout):
    """Is a view into the storage of another tensor"""

    def __init__(self, view: Union[BaseView, TensorBox]) -> None:
        layout = view.get_layout()
        super().__init__(
            layout.device,
            layout.dtype,
            layout.size,
            layout.stride,
        )
        self.view = view

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        return self.as_fixed().make_indexer()

    def maybe_guard_aligned(self):  # type: ignore[no-untyped-def]
        offset = self.view.get_layout().offset
        if offset == 0:
            return True
        from .utils import ALIGNMENT

        return V.graph.sizevars.statically_known_multiple_of(offset, ALIGNMENT)


class CommBufferType(Enum):
    SYMM_MEM = "symm_mem"


class CommBufferLayout(FixedLayout):
    """
    A layout that signifies the buffer is a comm buffer.
    In terms of striding, the layout is identical to `FixedLayout`.

    Buffers with this layout do not participate in in-place reuse - it can be
    neither the source nor the target for in-place reuse.

    For detailed motivation and usage of this layout, see
    NOTE [lowering-time collective optimization].
    """

    comm_buffer_type: CommBufferType
    group_name: str

    def __init__(
        self,
        layout: FlexibleLayout,
        comm_buffer_type: CommBufferType,
        group_name: str,
    ):
        if not isinstance(layout, FlexibleLayout):
            raise AssertionError(
                "A `CommBufferLayout` can only be initialized with "
                f"a `FlexibleLayout` (got {layout})."
            )

        fixed = layout.as_fixed()
        super().__init__(
            device=fixed.device,
            dtype=fixed.dtype,
            size=fixed.size,
            stride=fixed.stride,
            offset=fixed.offset,
        )
        self.comm_buffer_type = comm_buffer_type
        self.group_name = group_name


@ir_dataclass
class NoneLayout(OutputSpec):
    # This is janky, I figured out what fields to populate by just running
    # the model I was interested in and adding properties/methods as needed.
    # This doesn't inherit from Layout because Layout assumes you have stuff
    # like sizes, but I don't really have anything here.
    #
    # If you have an ir.Node with NoneLayout, you probably need to setup
    # dependencies manually in scheduler

    device: Optional[torch.device]
    size: list[int] = dataclasses.field(default_factory=lambda: [0])
    stride: list[int] = dataclasses.field(default_factory=lambda: [0])

    def storage_size(self) -> int:
        return 0

    def as_fixed(self):  # type: ignore[no-untyped-def]
        return self

    def get_device(self) -> Optional[torch.device]:
        return self.device


class MutationLayoutSHOULDREMOVE(Layout):
    def __init__(self, target: IRNode) -> None:
        super().__init__(
            target.get_device_or_error(),
            target.get_dtype(),
            target.get_size(),  # type: ignore[arg-type]
            None,
        )
        self.target = target
        name = self.get_buffer().get_name()
        V.graph.mark_buffer_mutated(name)

    @property
    def stride(self) -> list[Expr]:
        return self.real_layout().stride

    @stride.setter
    def stride(self, value: Never) -> None:
        pass  # ignore setting of stride

    def storage_size(self) -> sympy.Expr:
        return self.real_layout().storage_size()

    def get_buffer(self) -> Buffer:
        def unwrap_views(target):  # type: ignore[no-untyped-def]
            if isinstance(target, MutationLayoutSHOULDREMOVE):
                return unwrap_views(target.target)
            if isinstance(target, BaseView):
                return unwrap_views(target.unwrap_view())
            if isinstance(target, MutableBox):
                return unwrap_views(target.data)
            return target

        result = unwrap_views(self.target)
        assert isinstance(result, Buffer), (
            "MutationLayoutSHOULDREMOVE must refer to a buffer"
        )
        return result

    def real_layout(self):  # type: ignore[no-untyped-def]
        return self.get_buffer().layout

    @classmethod
    def realize_into(cls, src, dst, unsafe_alias=False):  # type: ignore[no-untyped-def]
        dst.realize()
        # NOTE: We must realize users of `dst` before we realize `src`, since
        # realization order determines scheduling order. Otherwise, src's
        # mutation would be scheduled before the existing users of dst!
        V.graph.mark_buffer_mutated(dst.get_name())

        if isinstance(src, TensorBox):
            src = src.data

        # We copy the contents of src into dst. In most cases this should
        # be fused into a single kernel by the scheduler.
        # NOTE: We cannot change src's layout to mutate dst directly as this
        # would alias src to dst, which is not correct as further mutations to
        # dst would effect users of src. However if there are no more users of
        # dst, we can alias src to dst.
        src.realize_hint()

        if not unsafe_alias:
            src = Pointwise.create(
                device=src.get_device(),
                dtype=src.get_dtype(),
                inner_fn=src.make_loader(),
                ranges=[
                    V.graph.sizevars.guard_equals(a, b)
                    for a, b in zip(src.get_size(), dst.get_size())
                ],
            ).data

        src.realize()
        assert isinstance(src.data.layout, FlexibleLayout)
        src.data.layout = MutationLayoutSHOULDREMOVE(dst)
        return src.data

    def as_fixed(self):  # type: ignore[no-untyped-def]
        return self

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        return self.target.make_indexer()


@ir_dataclass(frozen=False)
class Buffer(IRNode, CodegenSymbol):
    # Name is sometimes None; e.g., ForceInPlace, where there isn't
    # a meaningful name
    name: Optional[str]
    layout: OutputSpec

    # Multi-output buffers will define 'outputs: List[Buffer]'. Confusingly,
    # MultiOutput does NOT define this!

    def __post_init__(self) -> None:
        super().__post_init__()
        self._post_init_setattr("origin_node", None)

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        return self.get_layout().make_indexer()

    def get_name(self) -> str:
        assert self.name, self
        return self.name

    def get_example(self) -> Union[torch.Tensor, sympy.Symbol]:
        if isinstance(self.layout, Layout):
            return self.layout.get_example()
        raise NotImplementedError(type(self.layout).__name__)

    def get_device(self) -> Optional[torch.device]:
        return self.get_output_spec().get_device()

    def get_defining_op(self) -> Optional[Operation]:
        return None

    @property
    def dtype(self) -> torch.dtype:
        return self.get_layout().dtype

    def get_size(self) -> Sequence[Expr]:
        return [*self.get_layout().size]

    def get_stride(self) -> list[Expr]:
        return [*self.get_layout().stride]

    def get_offset(self) -> Expr:
        return self.get_layout().offset

    def get_layout(self) -> Layout:
        if isinstance(self.layout, Layout):
            return self.layout
        raise NotImplementedError(type(self.layout).__name__)

    def get_output_spec(self) -> OutputSpec:
        return self.layout

    def get_storage_numel(self):  # type: ignore[no-untyped-def]
        return self.get_numel()

    def freeze_layout(self):  # type: ignore[no-untyped-def]
        if isinstance(self.layout, Layout) and not isinstance(
            self.layout, NonOwningLayout
        ):
            self.layout = self.layout.as_fixed()

    def freeze_layout_with_stride_order(self, order, allow_padding=False) -> None:  # type: ignore[no-untyped-def]
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_stride_order(order, allow_padding=allow_padding)

    def freeze_layout_with_fill_order(self, order) -> None:  # type: ignore[no-untyped-def]
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_fill_order(order)

    def freeze_layout_with_same_order(self, stride) -> None:  # type: ignore[no-untyped-def]
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_same_order(stride)

    def freeze_layout_with_exact_strides(  # type: ignore[no-untyped-def]
        self, exact_strides, allow_padding=False
    ) -> None:
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_exact_strides(
            exact_strides, allow_padding=allow_padding
        )

    def is_zero_elements(self):  # type: ignore[no-untyped-def]
        return V.graph.sizevars.is_expr_static_and_true(sympy.Eq(self.get_numel(), 0))

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # Loading from a zero-element buffer is a no-op
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.get_dtype())

        def loader(index):  # type: ignore[no-untyped-def]
            indexer = self.make_indexer()
            return ops.load(self.name or "unnamed", indexer(index))

        return loader

    def codegen_reference(self, writer: Optional[IndentedBuffer] = None) -> str:
        return self.get_name()

    def decide_layout(self):  # type: ignore[no-untyped-def]
        pass

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        if isinstance(self.layout, NonOwningLayout):
            return [self.layout.view.get_name()]
        return ()

    def get_mutation_names(self) -> Sequence[str]:
        if isinstance(self.layout, MutationLayoutSHOULDREMOVE):
            return [self.layout.target.get_name()]
        return ()

    def get_read_names(self) -> OrderedSet[str]:
        return OrderedSet([self.get_name()])

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def realize(self) -> Optional[str]:
        pass

    def should_allocate(self) -> bool:
        # Returns False by default.
        return False


@ir_dataclass(frozen=False)
class OperationBuffer(Buffer, Operation):
    # An operation that produces a single output buffer
    def get_outputs(self) -> list[Buffer]:
        return [self]

    def get_defining_op(self) -> Operation:
        return self

    # Skip implementation in Buffer
    get_operation_name = Operation.get_operation_name

    def __post_init__(self) -> None:
        Buffer.__post_init__(self)
        Operation.__post_init__(self)


class InputBuffer(Buffer):
    def num_reads(self) -> int:
        return 1


class DonatedBuffer(InputBuffer):
    """
    Represents a donated buffer which is a saved tensor that is not alias to any
    fwd inputs, fwd user outputs, and bwd outputs. We generally cannot inplace
    reuse the input tensor memory during backward since it might be used in another
    function. However, donated buffer can be inplace reused during backward
    to save memory.
    """


class ConstantBuffer(InputBuffer):
    override_device: Optional[torch.device] = None

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        def loader(index: Sequence[Expr]) -> OpsValue:
            indexer = self.get_layout().make_indexer()
            return ops.load(
                V.graph.constant_name(self.get_name(), self.override_device),
                indexer(index),
            )

        return loader

    def constant_to_device(self, device: torch.device) -> IRNode:
        return ConstantBuffer(
            name=V.graph.constant_name(self.get_name(), device), layout=self.layout
        )


@ir_dataclass
class NoneAsConstantBuffer(IRNode):
    def get_reads(self) -> OrderedSet[Dep]:
        return OrderedSet()

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def codegen_reference(self, writer: Optional[IndentedBuffer] = None) -> str:
        return V.graph.wrapper_code.none_str

    def get_output_spec(self) -> OutputSpec:
        return NoneLayout(device=None)

    def has_tensor_output(self) -> bool:
        return False


@ir_dataclass
class ShapeAsConstantBuffer(IRNode):
    expr: Expr

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return get_free_symbols(self.expr, unbacked_only)

    def codegen_reference(self, writer: Optional[IndentedBuffer] = None) -> str:
        return V.graph.wrapper_code.codegen_sizevar(self.expr)

    def has_tensor_output(self) -> bool:
        return False


@ir_dataclass(frozen=False)
class ComputedBuffer(OperationBuffer):
    data: Loops

    def get_computed_buffer_name(self) -> Optional[str]:
        """
        Returns self.name if it exists, otherwise returns the name of the data node if that exists.
        If neither exist, returns None.
        """
        if self.name is not None:
            return self.name
        if hasattr(self.data, "name"):
            return self.data.name
        return None

    def num_reads(self) -> int:
        return self.data.num_reads()

    def get_reads(self) -> OrderedSet[Dep]:
        return self.data.get_reads()

    def get_read_names(self) -> OrderedSet[str]:
        return self.data.get_read_names()

    def get_read_writes(self) -> dependencies.ReadWrites:
        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.data.get_reduction_type():
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_pointwise_size(),  # type: ignore[arg-type]
                    self.data.get_reduction_size(),  # type: ignore[arg-type]
                )
            else:
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_size(),  # type: ignore[arg-type]
                )

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # Ordinarily, we'd like to just peek at the arguments list,
        # but ComputedBuffers have no argument list.
        #
        # Morally, this logic needs to be synchronized with the
        # KernelArgs.size calls, which are responsible for making symbols make
        # there way as kernel arguments (and it is precisely passing in one of
        # those symbols that establishes a dependency).  However, we haven't
        # started codegen yet so we can't directly reuse that logic.
        #
        # For now, I'm just yoloing with the size of the buffer.  Not sure if
        # it is enough.
        #
        # One thing you might wonder is if this is enough for a ComputedBuffer
        # denoting a reduction over i0.  Empirically, it is enough, but for an
        # unusual reason: we only need accurate dependencies for item() call,
        # but it's impossible to end up with a reduction over i0 from an
        # item() call without a regular non-reduction buffer first.
        return (
            get_free_symbols(self.get_size(), unbacked_only)
            | get_free_symbols(self.get_stride(), unbacked_only)
            | get_free_symbols(self.get_offset(), unbacked_only)
            | self.data.get_free_symbol_uses(unbacked_only)
        )

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        if (
            not self.get_reduction_type()
            and self.name not in V.graph.mutated_buffers
            and self.num_reads() == 0
        ):
            # inline this op rather than generating ops.load()
            return self.data.make_loader()
        return super().make_loader()

    def get_store_function(self) -> Callable[..., None]:
        indexer = self.get_layout().as_fixed().make_indexer()
        if isinstance(self.data, (Reduction, Scan, Sort)):
            return partial(self.data.store_reduction, self.name, indexer)
        else:
            assert isinstance(self.data, Pointwise)
            return partial(self.data.store_output, self.name, indexer)

    def get_fill_order(self) -> Optional[list[int]]:
        """
        If our layout is still flexible, try to determine the stride order based on stride orders of reads.

        TODO(jansel): A better algorithm here would look at downstream consumers of this
                      value and try to do global graph-level layout optimization.
                      This is also something just begging to be autotuned.
        """
        if isinstance(self.layout, FlexibleLayout):
            (index_vars, reduction_vars), _ = dependencies.index_vars_squeeze(
                self.data.get_pointwise_size(), self.data.get_reduction_size()
            )
            reads = self.get_read_writes().reads
            # only consider reads to buffer of same size
            # ignore StarDeps because they don't contribute stride information
            assert all(
                isinstance(r, (dependencies.StarDep, dependencies.MemoryDep))
                for r in reads
            )
            reads = [
                sympy_subs(r.index, {v: sympy.S.Zero for v in reduction_vars if v != 0})
                for r in reads
                if isinstance(r, dependencies.MemoryDep)
            ]

            if reads:
                if isinstance(self.data, (Scan, Sort)):
                    indices = self.data.reindex(index_vars, reduction_vars)
                else:
                    indices = index_vars
                stride_lengths = [
                    V.graph.sizevars.stride_hints(expr, indices) for expr in reads
                ]
                from .scheduler import pick_loop_order

                return pick_loop_order(stride_lengths, self.get_size())

        return None

    def decide_layout(self) -> None:
        if isinstance(self.layout, FlexibleLayout):
            order = self.get_fill_order()
            if order:
                self.freeze_layout_with_fill_order(order)
            else:
                self.freeze_layout()

    @cache_on_self
    def get_default_sizes_body(
        self,
    ) -> tuple[
        tuple[list[sympy.Expr], list[sympy.Expr]],
        LoopBody,
        tuple[list[sympy.Expr], list[sympy.Expr]],
    ]:
        args, var_ranges = dependencies.index_vars_squeeze(
            self.data.get_pointwise_size(), self.data.get_reduction_size(), prefix="q"
        )
        with patch.object(ConstantBuffer, "override_device", self.get_device()):
            body = LoopBody(
                self.get_store_function(),
                (args if self.get_reduction_type() else args[:1]),
                var_ranges,
                *args,
            )
        index_vars = []
        reduce_vars: list[Any] = []
        index_size = []
        reduce_size = []
        for v, s in var_ranges.items():
            if v in args[0]:
                assert not reduce_vars
                index_vars.append(v)
                index_size.append(s)
            else:
                assert v in args[1]
                reduce_vars.append(v)
                reduce_size.append(s)
        return (index_size, reduce_size), body, (index_vars, reduce_vars)

    def simplify_and_reorder(
        self,
        extra_indexing_constraints: Optional[tuple[dict[Any, Any], list[Any]]] = None,
        recompute_sizes_body_func: Optional[Callable[..., Any]] = None,
    ) -> tuple[tuple[list[sympy.Expr], list[sympy.Expr]], LoopBody]:
        """
        This is a main place where we do loop transformations in a
        backend-agnostic way.

        Here we:
            1) Remove any 1 dimensions
            2) Fuse contiguous dimensions together
            3) Reorder dimensions based on stride orders

        Optional argument extra_indexing_constraints can be used to append additional
        indexing expressions to existing ones derived from buffer's body. This can be useful
        to fuse scheduler nodes with compatible ranges, e.g. (s0*s1*...,) and (s0, s1, s2, ...)
        on CPU by preventing indexing simplifications and obtaining index/reduce ranges for
        the scheduler node compatible with other nodes.
        Optional argument recompute_sizes_body_func can be used to recompute sizes and body
        on the default body. This can be useful to append additional loop transformations.
        """
        (
            (index_size, reduce_size),
            body,
            (index_vars, reduce_vars),
        ) = self.get_default_sizes_body()

        if recompute_sizes_body_func:
            (
                (index_size, reduce_size),
                body,
                (index_vars, reduce_vars),
            ) = recompute_sizes_body_func(
                (index_size, reduce_size), body, (index_vars, reduce_vars)
            )

        index_formulas = [*body.indexing_exprs.values()]
        if extra_indexing_constraints is not None:
            assert (
                isinstance(extra_indexing_constraints, tuple)
                and len(extra_indexing_constraints) == 2
            )
            extra_indexing_ranges, extra_indexing_expr = extra_indexing_constraints
            assert isinstance(extra_indexing_ranges, dict)
            assert isinstance(extra_indexing_expr, list)
            assert all(isinstance(f, Expr) for f in extra_indexing_expr)

            expected_var_ranges = body.var_ranges
            assert expected_var_ranges == extra_indexing_ranges, (
                expected_var_ranges,
                extra_indexing_ranges,
            )
            # remove already existing expressions
            extra_indexing_expr = [
                e for e in extra_indexing_expr if e not in index_formulas
            ]
            index_formulas += extra_indexing_expr

        memory_addrs = [*body.get_write_exprs()]
        if not V.graph.has_feature(self, BackendFeature.PREFER_STORE_LOOP_ORDER):
            memory_addrs.extend(body.get_read_exprs())

        def simplify_and_reorder(x_vars, support_vars, sizes, simplify_loops):  # type: ignore[no-untyped-def]
            sizes, reindex0, reindex1 = self._apply_loop_reordering(
                x_vars, support_vars, sizes, memory_addrs
            )
            # for NHWC: reindex0([0,1,2,3]) = [0,2,3,1], reindex1([0,1,2,3]) = [0,3,2,1]
            x_vars = reindex0(x_vars)

            if simplify_loops:
                sizes, reindex2, _prune = V.graph.sizevars._simplify_loops(
                    x_vars,
                    sizes,
                    index_prevent_reordering(index_formulas, x_vars, sizes),
                )
                reindex = fuse_reindexing(reindex1, reindex2)
            else:
                reindex = reindex1
            return sizes, reindex, reindex1

        support_vars = index_vars + reduce_vars
        should_merge_loops = (
            not is_gpu(get_device_type(self)) or not config.loop_ordering_after_fusion
        )
        iter_ranges, iter_reindex, _ = simplify_and_reorder(
            index_vars,
            support_vars,
            index_size,
            should_merge_loops,
        )

        # Like iteration dimensions, we may also want to delay merging reduction dimensions.
        # E.g., if we reduce a tensor [M, N, K] for its M and N dimensions followed by a pointwise
        # kernel, merging M and N dimension too early makes it hard to decide what loop order
        # we should pick for the piontwise kernel so that it is fusible with the reduction.
        reduce_ranges, reduce_reindex, _ = simplify_and_reorder(
            reduce_vars, support_vars, reduce_size, should_merge_loops
        )

        # retrace the loop body with simplification and reordering applied
        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            iter_ranges,
            reduce_ranges,
            prefix="p",
        )
        body = LoopBody(
            body,
            [iter_reindex(iter_vars), reduce_reindex(reduce_vars)],
            var_ranges,
            iter_vars,
            reduce_vars,
        )
        return (iter_ranges, reduce_ranges), body

    @staticmethod
    def _apply_loop_reordering(  # type: ignore[no-untyped-def]
        index_vars,
        support_vars,
        sizes,
        memory_addrs,
        priority_idx=None,
    ):
        """
        Shuffle the order of loops around to hopefully improve performance.
        """
        from .scheduler import pick_loop_order

        if priority_idx is None:
            priority_idx = []

        try:
            strides = [
                V.graph.sizevars.stride_hints(expr, index_vars, support_vars)
                for expr in memory_addrs
            ]
            assert len(strides) == len(memory_addrs) and len(strides[0]) == len(
                index_vars
            )
            order = list(reversed(pick_loop_order(strides, sizes, priority_idx)))
        except Exception:
            if config.debug:
                log.warning(
                    "Did not simplify complex index:\n%s\n%s",
                    dict(zip(index_vars, sizes)),
                    memory_addrs,
                )
            order = list(range(len(sizes)))
        sizes = [sizes[i] for i in order]
        return sizes, same_reorder(order), inverse_reorder(order)

    def get_reduction_size(self) -> Sequence[sympy.Expr]:
        return self.data.get_reduction_size()

    def get_reduction_type(self) -> Optional[str]:
        return self.data.get_reduction_type()

    def is_no_op(self) -> bool:
        return self.data.is_zero_elements()

    def should_allocate(self) -> bool:
        return True

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        return self.data.constant_to_device(device)


class TemplateBuffer(OperationBuffer):
    """
    Represents a Triton (in the future other type) of template operator
    that we can fuse an epilogue onto.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[..., Any],
    ) -> None:
        super().__init__(name=None, layout=layout)
        self.inputs = InputsKernel.unwrap_storage(inputs)
        self.make_kernel_render = make_kernel_render
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_read_writes(self) -> dependencies.ReadWrites:
        return self.extract_read_writes(normalize=True)

    def extract_read_writes(self, normalize):  # type: ignore[no-untyped-def]
        name = self.get_name()
        indexer = self.get_layout().make_indexer()

        def dummy(index, rindex):  # type: ignore[no-untyped-def]
            assert len(rindex) == 0
            return ops.store(name, indexer(index), "fake")

        deps = dependencies.extract_read_writes(
            dummy, self.get_size(), (), normalize=normalize
        )

        for inp in self.inputs:
            indexer = inp.layout.make_indexer()

            def dummy(index, rindex):  # type: ignore[no-untyped-def]
                assert len(rindex) == 0
                ops.load(inp.get_name(), indexer(index))

            deps.reads |= dependencies.extract_read_writes(
                dummy, inp.get_size(), (), normalize=True
            ).reads

        return deps

    def get_reduction_size(self) -> Sequence[sympy.Expr]:
        return sympy.S.One

    def get_reduction_type(self) -> Optional[str]:
        return None

    def should_allocate(self) -> bool:
        return True

    def simplify_and_reorder(  # type: ignore[no-untyped-def]
        self,
        extra_indexing_constraints: Optional[tuple[dict[Any, Any], list[Any]]] = None,
        recompute_sizes_body_func: Optional[Callable[..., Any]] = None,
    ):
        return (
            (
                self.get_size(),
                (),
            ),
            None,
        )


class TritonTemplateBuffer(TemplateBuffer):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        layout,
        inputs,
        make_kernel_render,
        mutated_inputs: Optional[Iterable[IRNode]] = None,
        allowed_prologue_inps: Optional[OrderedSet[str]] = None,
    ) -> None:
        """
        NOTE:[TritonTemplates with multiple outputs]
        We want the ability for TritonTemplates to output multiple tensors. Triton
        kernels have no notion of outputs and this is done by creating tensors that
        are then mutated by the kernel. Currenlty our STORE_OUTPUT codegen doesn't
        support creating multinode outputs for triton templates.
        We work around this by creating an extra input buffer during the lowering
        and we mark them as mutated inputs.
        """
        super().__init__(layout, inputs, make_kernel_render)
        self.mutated_inputs = mutated_inputs
        self.outputs: list[Buffer] = [self]
        if mutated_inputs is not None:
            # Ensure that the mutated inputs are only allowed for certain nodes
            allowed_set = (
                torch.ops.higher_order.flex_attention,
                torch.ops.higher_order.flex_attention_backward,
            )
            current_node = V.graph.current_node.target
            assert current_node in allowed_set, (
                f"Mutated inputs are only allowed for {allowed_set} but got {current_node}"
            )
            device = self.inputs[0].get_device()
            self.outputs += [
                MutationOutput(NoneLayout(device=device), buf, self)
                for buf in mutated_inputs
            ]

        self.allowed_prologue_inps = (
            allowed_prologue_inps if allowed_prologue_inps else OrderedSet()
        )

    def get_outputs(self) -> list[Buffer]:
        return self.outputs

    def get_allowed_prologue_inps(self) -> OrderedSet[str]:
        return self.allowed_prologue_inps

    def __str__(self) -> str:
        out = f"TritonTemplateBuffer(layout={self.layout})"
        return out


PrimitiveInfoType = Union[int, float, bool, str, list[Union[int, str, float, bool]]]


class ChoiceCaller:
    """
    Represents a possible choice used in autotune_process.py.
    During autotuning, self.benchmark() is first called to get benchmark result,
    and if this choice is selected, self.output_node() is called to get the output_node.

    Children classes: TritonTemplateCaller, CUDATemplateCaller.
    """

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        description: str,
    ) -> None:
        super().__init__()
        self.name = name
        self.layout = layout
        self.input_nodes = input_nodes
        # An additional description used to describe the choice (useful for
        # knowing what autotuning is choosing)
        self.description = description

    def benchmark(self, *args, out) -> float:  # type: ignore[no-untyped-def]
        algo = self.to_callable()
        return benchmarker.benchmark(algo, args, {"out": out})

    def call_name(self) -> str:
        raise NotImplementedError

    def to_callable(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def hash_key(self) -> str:
        raise NotImplementedError

    def output_node(self) -> TensorBox:
        raise NotImplementedError

    def info_dict(self) -> dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {}

    def autoheuristic_id(self) -> str:
        return "unsupported_choice"


class TritonTemplateCallerBase(ChoiceCaller):
    def get_make_kernel_render(self) -> Any:
        raise NotImplementedError


class MultiTemplateBuffer(TritonTemplateBuffer):
    """
    Represents a Buffer with multiple backing implementation choices.

    Choices can be TritonTemplates or ExternKernels. During scheduling if there is a potential
    epilogue we will benchmark each of the choices with the epilogue to determine an implementation.
    Otherwise, the fastest base choice will be chosen.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: list[IRNode],
        choice_timings: Callable[[], dict[ChoiceCaller, float]],
        unfiltered_choices: list[ChoiceCaller],
        allowed_prologue_inps: OrderedSet[str],
    ) -> None:
        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=None,
            allowed_prologue_inps=allowed_prologue_inps,
        )
        self._choice_timings_fn = choice_timings
        self._choice_timings: Optional[dict[ChoiceCaller, float]] = None
        self.original_inputs = inputs
        self._output_plannable = all(
            isinstance(choice, TritonTemplateCallerBase)
            or (
                isinstance(choice, torch._inductor.select_algorithm.ExternKernelCaller)
                and choice.has_out_variant
            )
            for choice in unfiltered_choices
        )

    @property
    def output_plannable(self) -> bool:
        """
        Are all possible choices TritonTemplates or Extern Kernels with out variants
        """
        return self._output_plannable

    @property
    def choice_timings(self) -> dict[ChoiceCaller, float]:
        if self._choice_timings is None:
            self._choice_timings = self._choice_timings_fn()
        return self._choice_timings

    @contextlib.contextmanager
    def swap_as_triton_caller(self, caller: TritonTemplateCallerBase):  # type: ignore[no-untyped-def]
        assert isinstance(caller, torch._inductor.select_algorithm.TritonTemplateCaller)
        assert self.layout == caller.layout

        render = self.make_kernel_render
        self.make_kernel_render = caller.get_make_kernel_render()
        try:
            yield
        finally:
            self.make_kernel_render = render

    def finalize_as_triton_caller(self, caller: TritonTemplateCallerBase) -> None:
        assert isinstance(caller, torch._inductor.select_algorithm.TritonTemplateCaller)
        assert self.get_size() == caller.layout.size
        assert self.get_stride() == caller.layout.stride
        self.make_kernel_render = caller.get_make_kernel_render()

    def get_min_choice(self) -> tuple[ChoiceCaller, float]:
        min_choice = min(self.choice_timings, key=self.choice_timings.get)  # type: ignore[arg-type]
        return (min_choice, self.choice_timings[min_choice])


class CUDATemplateBuffer(TemplateBuffer):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        layout,
        inputs,
        make_kernel_render,
        workspace_size: int,
        template: CUDATemplate,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        # Global memory (in bytes) needed for this template.
        self.workspace_size = workspace_size
        self.template = template

    def get_workspace_size(self):  # type: ignore[no-untyped-def]
        return self.workspace_size if self.workspace_size is not None else 0


class CppTemplateBuffer(TemplateBuffer):
    def __init__(self, layout, inputs, make_kernel_render, template, choice) -> None:  # type: ignore[no-untyped-def]
        super().__init__(layout, inputs, make_kernel_render)
        self.template = template
        self.choice = choice
        self.outputs: Optional[list[Buffer]] = None

    def get_layout(self) -> Layout:
        if isinstance(self.layout, MultiOutputLayout):
            assert isinstance(self.outputs, Iterable)
            first_output = self.outputs[0]
            assert isinstance(first_output, Buffer)
            layout = first_output.layout
            assert isinstance(layout, Layout)
            return layout
        else:
            return super().get_layout()


@ir_dataclass(frozen=False)
class InputsKernel(OperationBuffer):
    inputs: list[Buffer]

    def get_read_writes(self) -> dependencies.ReadWrites:
        reads = OrderedSet[dependencies.Dep]()
        StarDep = dependencies.StarDep
        for input in self.inputs:
            if isinstance(input, list):
                reads.update(StarDep(x.get_name()) for x in input)
            elif isinstance(input, ShapeAsConstantBuffer):
                # Skip creating dependncy for symbolics as they're visible globally
                continue
            else:
                reads.add(StarDep(input.get_name()))

        writes = OrderedSet[dependencies.Dep](
            StarDep(buf.get_name()) for buf in self.get_outputs()
        )

        return dependencies.ReadWrites(
            reads=reads,
            writes=writes,
            index_exprs=OrderedSet(),
        )

    def get_reads(self) -> OrderedSet[Dep]:
        return self.get_read_writes().reads

    @classmethod
    def unwrap_storage_for_input(cls, x: IRNode) -> IRNode:
        if isinstance(x, TensorBox):
            x = x.data
        if isinstance(x, StorageBox):
            x = x.data
        if isinstance(x, BaseView) and not isinstance(x, ReinterpretView):
            x = ExternKernel.realize_input(x)
        if isinstance(x, TensorBox):
            # when converting to ReinterpretView fails in the
            # realize_input call above, the result will be wrapped
            # into TensorBox / StorageBox pair as a result of the
            # cls.copy_input call; so we should unwrap recursively
            return cls.unwrap_storage_for_input(x)
        if isinstance(x, TorchBindObject):
            return x
        assert isinstance(x, (Buffer, ReinterpretView)), x
        return x

    @staticmethod
    def unwrap_storage(inputs):  # type: ignore[no-untyped-def]
        inputs_new = []
        for x in inputs:
            if isinstance(x, list):
                x = [InputsKernel.unwrap_storage_for_input(i) for i in x]
            else:
                x = InputsKernel.unwrap_storage_for_input(x)
            inputs_new.append(x)
        return inputs_new

    def is_extern(self) -> bool:
        return True

    def num_reads(self) -> int:
        return 1


class NopKernel(InputsKernel):
    def is_no_op(self) -> bool:
        return True

    def get_reads(self) -> OrderedSet[Dep]:
        return OrderedSet()


class ConcatKernel(NopKernel):
    """
    There isn't actually a real kernel for concat, we just change the
    storage for the upstream data.
    """

    @classmethod
    def create(cls, inputs, dim):  # type: ignore[no-untyped-def]
        device = inputs[0].get_device()
        dtype = inputs[0].get_dtype()
        new_size = list(inputs[0].get_size())
        offsets_start = [0]
        offsets_end = [new_size[dim]]
        assert 0 <= dim < len(new_size)
        for i in range(1, len(inputs)):
            input_size = inputs[i].get_size()
            offsets_start.append(new_size[dim])
            assert len(input_size) == len(new_size)
            assert inputs[i].get_dtype() == dtype
            assert inputs[i].get_device() == device
            for j in range(len(new_size)):
                if j == dim:
                    new_size[j] = new_size[j] + input_size[j]
                else:
                    new_size[j] = V.graph.sizevars.guard_equals(
                        new_size[j], input_size[j]
                    )
            offsets_end.append(new_size[dim])

        output_stride = FlexibleLayout.contiguous_strides(new_size)
        if config.comprehensive_padding:
            # Ensure the output stride matches the alignment requirements
            output_stride = Layout._pad_strides(
                output_stride, new_size, inputs[0].dtype
            )

        # If any of the inputs is in CL format, use CL format for the output
        for i in range(len(inputs)):
            x = inputs[i]
            if is_storage_and_layout(x):
                layout = x.get_layout()
                if isinstance(
                    layout, FixedLayout
                ) and Layout.is_channels_last_contiguous(layout.size, layout.stride):
                    # use CL stride for the output
                    output_stride = make_channels_last_strides_for(new_size)
                    break
        any_input_is_storage_and_layout = any(is_storage_and_layout(x) for x in inputs)
        fx_node_args = V.graph.current_node.args[0]
        assert isinstance(fx_node_args, list)
        # If any of the inputs has meta tensor and the meta tensor is in CL format, use CL format for the output
        if any_input_is_storage_and_layout is False and any(
            "val" in arg.meta
            and (
                arg.meta["val"].is_contiguous(memory_format=torch.channels_last)
                or arg.meta["val"].is_contiguous(memory_format=torch.channels_last_3d)
            )
            for arg in fx_node_args
        ):
            output_stride = make_channels_last_strides_for(new_size)

        concat_kernel = ConcatKernel(
            name=None,
            layout=FixedLayout(
                device=device,
                dtype=dtype,
                size=new_size,
                stride=output_stride,
            ),
            inputs=[],
        )
        kernel = StorageBox(concat_kernel)
        op_names = []
        for i in range(len(inputs)):
            input_buffer = cls.realize_into(
                inputs[i],
                SliceView.create(
                    kernel, dim, offsets_start[i], offsets_end[i], clamp=False
                ),
            )
            concat_kernel.inputs.append(input_buffer)

            if isinstance(inputs[i].data, BaseView):
                input_unwrapped = inputs[i].data.unwrap_view()
            else:
                input_unwrapped = inputs[i].data

            if (
                input_unwrapped.is_input_buffer()
                and is_gpu(inputs[i].get_device().type)
                and not is_dynamic(input_buffer)
            ):
                op_names.append(input_buffer.get_operation_name())

        if len(op_names) > 1 and V.graph.has_feature(device, BackendFeature.FOREACH):
            V.graph.register_operation_list(op_names)

        concat_kernel.name = V.graph.register_buffer(concat_kernel)
        concat_kernel.inputs = cls.unwrap_storage(concat_kernel.inputs)
        V.graph.register_operation(concat_kernel)

        return kernel

    @classmethod
    def can_realize_into_without_copy(cls, src, dst=None):  # type: ignore[no-untyped-def]
        if isinstance(src, TensorBox):
            # unwrap a TensorBox
            return cls.can_realize_into_without_copy(src.data, dst)

        if isinstance(src.data, MultiTemplateBuffer):
            if (
                not isinstance(src.data.layout, FixedLayout)
                or not src.data.output_plannable
            ):
                return False

            # we call can_realize_into_without_copy in cat lowering before we've decided
            # on output format, optimistically assume layout matches
            if dst is None:
                return True

            # otherwise, check equality of layouts
            if not len(src.get_stride()) == len(dst.get_stride()):
                return False

            return all(
                V.graph.sizevars.statically_known_equals(s1, s2)
                for s1, s2 in zip(src.get_stride(), dst.get_stride())
            )

        return isinstance(src.data.layout, FlexibleLayout) and not isinstance(
            src.data, ExternKernelAlloc
        )

    @classmethod
    def realize_into(cls, src, dst):  # type: ignore[no-untyped-def]
        # Attempt to turn this into a ReinterpretView rather than assert.
        # This has concessions around layout, as as_storage_and_layout
        # can cause us to go from flexible to fixed layout.
        if not isinstance(dst, ReinterpretView):
            if is_storage_and_layout(dst):
                storage, layout = as_storage_and_layout(dst)
                dst = ReinterpretView(data=storage, layout=layout)
        assert isinstance(dst, ReinterpretView), dst
        if isinstance(src, TensorBox):
            # unwrap a TensorBox
            return cls.realize_into(src.data, dst)

        if isinstance(src, StorageBox):
            src.realize()
            # ExternKernelAlloc has specific requirements for output layout, should create a copy
            assert hasattr(src.data, "layout")
            if cls.can_realize_into_without_copy(src, dst):
                src.data.layout = NonOwningLayout(dst)
                return src.data
        # introduce a copy
        pw = Pointwise.create(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=[
                V.graph.sizevars.guard_equals(a, b)
                for a, b in zip(src.get_size(), dst.get_size())
            ],
        )
        return cls.realize_into(pw, dst)

    def should_allocate(self) -> bool:
        return True


@ir_dataclass(frozen=False)
class ExternKernel(InputsKernel):
    constant_args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    output_view: Optional[ReinterpretView] = None
    python_kernel_name: Optional[str] = None
    cpp_kernel_name: Optional[str] = None
    # FIXME: in some cases we sill need to explicitly pass in ordered_kwargs_for_cpp_kernel
    # We shouldn't need to do this since the information can be retrieved from op_overload._schema.
    ordered_kwargs_for_cpp_kernel: Iterable[str] = dataclasses.field(
        default_factory=list
    )
    op_overload: Optional[
        Union[torch._ops.OpOverload, torch._ops.HigherOrderOperator]
    ] = None
    arg_properties: Optional[list[dict[str, Any]]] = None
    kwarg_properties: Optional[dict[str, dict[str, Any]]] = None
    unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] = dataclasses.field(
        default_factory=dict
    )
    mutation_outputs: list[MutationOutput] = dataclasses.field(default_factory=list)

    def __init__(  # type: ignore[no-untyped-def]
        self,
        name,
        layout,
        inputs,
        constant_args=(),
        kwargs=None,
        output_view=None,
        python_kernel_name=None,
        cpp_kernel_name=None,
        ordered_kwargs_for_cpp_kernel=(),
        op_overload=None,
    ) -> None:
        super().__init__(
            name=name,
            layout=layout,
            inputs=inputs,
        )
        self.constant_args = constant_args
        self.kwargs = kwargs if kwargs else {}
        self.output_view = output_view
        self.op_overload = op_overload
        self.set_cpp_kernel_name(cpp_kernel_name)
        self.set_python_kernel_name(python_kernel_name)
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        self.collect_arg_kwarg_properties()
        self.unbacked_bindings = {}
        self.mutation_outputs = []
        self.fx_node = V.graph.current_node

    def get_outputs(self) -> list[Buffer]:
        return [self, *self.mutation_outputs]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def collect_arg_kwarg_properties(self):  # type: ignore[no-untyped-def]
        # if self.op_overload is torch._ops.OpOverload, we can use its schema to collect additional
        # information for args and kwargs, e.g. type and default value, to help with the cpp wrapper codegen
        self.arg_properties = (
            [
                {
                    "name": x.name,
                    "type": x.real_type,
                    "default_value": x.default_value,
                }
                for x in self.op_overload._schema.arguments
                if not x.kwarg_only
            ]
            if isinstance(self.op_overload, torch._ops.OpOverload)
            else [{} for i in range(len(self.inputs))]
        )
        self.allarg_properties = (
            {
                x.name: {"type": x.real_type, "default_value": x.default_value}
                for x in self.op_overload._schema.arguments
            }
            if isinstance(self.op_overload, torch._ops.OpOverload)
            else {}
        )
        # FIXME: self.kwargs does not always match kwargs defined in schema, so sometimes
        # ordered_kwargs_for_cpp_kernel is explicilty passed in.
        if isinstance(self.op_overload, torch._ops.OpOverload):
            if not self.ordered_kwargs_for_cpp_kernel:
                self.ordered_kwargs_for_cpp_kernel = [
                    x.name for x in self.op_overload._schema.arguments if x.kwarg_only
                ]
            self.schema_kwargs = [
                x for x in self.op_overload._schema.arguments if x.kwarg_only
            ]

    def decide_layout(self):  # type: ignore[no-untyped-def]
        if isinstance(self.layout, FlexibleLayout):
            self.apply_constraint()
            self.freeze_layout()

    def codegen_comment(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        origin_str, _detailed_origin_str = get_kernel_metadata(self, wrapper)
        if origin_str:
            wrapper.make_comment(origin_str)

    def codegen(self, wrapper):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def set_cpp_kernel_name(self, cpp_kernel_name: Optional[str] = None) -> None:
        self.cpp_kernel_name = cpp_kernel_name
        if not V.graph.cpp_wrapper or not isinstance(
            self.op_overload, torch._ops.OpOverload
        ):
            return

        kernel = self.op_overload
        if self.cpp_kernel_name is None:
            # Try to construct cpp_kernel_name from op_overload
            if kernel.namespace == "aten":
                # Calling with the default kernel name can lead to ambiguous behavior like the following example.
                # repeat_interleave(const at::Tensor & repeats, std::optional<int64_t> output_size=std::nullopt)
                # repeat_interleave(const at::Tensor & self, int64_t repeats,
                #       std::optional<int64_t> dim=std::nullopt, std::optional<int64_t> output_size=std::nullopt)
                opname = (
                    kernel.__name__.split(".")[0]
                    if kernel._overloadname == "default"
                    else kernel.__name__.replace(".", "_")
                )
                self.cpp_kernel_name = f"at::_ops::{opname}::call"
            else:
                self.cpp_kernel_name = kernel._schema.name

    def set_python_kernel_name(self, python_kernel_name: Optional[str]) -> None:
        self.python_kernel_name = python_kernel_name
        if python_kernel_name is not None:
            return

        kernel = self.op_overload
        if kernel is None:
            pass
        elif isinstance(kernel, torch._ops.HigherOrderOperator):
            self.python_kernel_name = f"torch.ops.higher_order.{kernel.__name__}"
        else:
            self.python_kernel_name = (
                f"{kernel.__module__.replace('._ops.', '.ops.')}.{kernel.__name__}"
            )

    def get_kernel_name(self):  # type: ignore[no-untyped-def]
        device = d.type if (d := self.get_device()) else V.graph.device_type
        return (
            V.graph.wrapper_code.get_c_shim_func_name(self.cpp_kernel_name, device)  # type: ignore[attr-defined]
            if V.graph.cpp_wrapper
            else self.python_kernel_name
        )

    @staticmethod
    def copy_input(x):  # type: ignore[no-untyped-def]
        pw = Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=x.make_loader(),
            ranges=x.get_size(),
            origin_node=x.get_origin_node(),
            traceback=x.get_traceback(),
        )
        pw.realize()
        return pw

    @classmethod
    def process_kernel(  # type: ignore[no-untyped-def]
        cls, kernel, *args, **kwargs
    ) -> tuple[
        Any,
        list[Any],
        list[Any],
        Callable[[Any, Any], Any],
        Optional[dict[sympy.Symbol, pytree.KeyPath]],
    ]:
        binded_args = {"args": args, "kwargs": kwargs}

        args_flat, args_spec = pytree.tree_flatten(binded_args)

        is_arg_tensor = []
        # tensor_args can be either tensor or torchbind objects
        tensor_args = []
        non_tensor_args: list[Any] = []
        for arg in args_flat:
            is_arg_tensor.append(
                isinstance(arg, IRNode) and not isinstance(arg, GeneratorState)
            )
            if is_arg_tensor[-1]:
                tensor_args.append(arg)
            else:
                if isinstance(arg, sympy.Expr):
                    arg = V.graph.sizevars.shape_env.create_symintnode(arg, hint=None)
                non_tensor_args.append(arg)

        def unflatten_args(new_tensor_args, new_non_tensor_args):  # type: ignore[no-untyped-def]
            result = []
            it_tensors = iter(new_tensor_args)
            it_non_tensors = iter(new_non_tensor_args)
            for is_tensor in is_arg_tensor:
                if is_tensor:
                    result.append(next(it_tensors))
                else:
                    result.append(next(it_non_tensors))
            r = pytree.tree_unflatten(result, args_spec)
            return r.get("args", []), r.get("kwargs", {})

        tensor_args = [cls.realize_input(x) for x in tensor_args]

        # freeze layout otherwise our output stride calculation might
        # become incorrect
        for x in tensor_args:
            if is_storage_and_layout(x):
                as_storage_and_layout(x, freeze=True)

        # Rerun fake tensor propagation, because Inductor may have changed the
        # strides of inputs and we need to determine accurately what the
        # output stride will be.
        example_args: list[
            Union[
                torch.Tensor, torch._C.ScriptObject, FakeScriptObject, torch.Generator
            ]
        ] = []

        # We need to retain the constant values of fake tensors that we originally
        # propagated the graph with, because for some operators running without a
        # constant would trigger an error / DataDependentException
        for x in tensor_args:
            # if x is a view of a constant, we need to realize the view
            # (we can't pass the constant into the kernel directly)
            if not isinstance(x, BaseView) and x.get_name() in V.graph.constants:
                example_args.append(V.graph.constants[x.get_name()])
            elif (
                not isinstance(x, BaseView)
                and x.get_name() in V.graph.torchbind_constants
            ):
                example_args.append(V.graph.torchbind_constants[x.get_name()])
            elif isinstance(x, TorchBindObject):
                example_args.append(x.get_value())
            elif isinstance(x, torch._inductor.ir.GeneratorState):
                device_index = x.device.index
                assert x.device.type == "cuda" and device_index is not None
                example_args.append(
                    torch.cuda.default_generators[device_index].clone_state()
                )
            else:
                example_args.append(ir_node_to_tensor(x, guard_shape=True))

        new_args, new_kwargs = unflatten_args(example_args, non_tensor_args)
        example_output = kernel(*new_args, **new_kwargs)

        unbacked_bindings: Optional[dict[sympy.Symbol, pytree.KeyPath]] = None
        if shape_env := V.fake_mode.shape_env:
            node_meta_val = V.current_node.meta.get("val")
            ctx = nullcontext()
            if V.current_node.target == torch._higher_order_ops.effects.with_effects:
                # remove the first effect token in meta["val"] and meta["unbacked_bindings"]
                node_meta_val = node_meta_val[1]
                ctx = _remove_effect_token_unbacked_bindings(V.current_node)  # type: ignore[assignment]

            with ctx:
                rebind_unbacked(shape_env, V.current_node, example_output)
            unbacked_bindings = compute_unbacked_bindings(
                shape_env, example_output, node_meta_val
            )

        example_out_li = (
            [example_output]
            if not isinstance(example_output, (list, tuple))
            else example_output
        )
        for t in example_out_li:
            if isinstance(t, torch.Tensor) and t.is_sparse:
                msg = "sparsity not handled. Please file issue for sparse inference weights."
                if stack_trace := V.graph.current_node.meta.get("stack_trace", None):
                    msg = f"{msg} Found from : \n {stack_trace}"
                V.graph.disable_cudagraphs_reason = msg

        return (
            example_output,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            unbacked_bindings,
        )

    @classmethod
    def convert_to_reinterpret_view(cls, x):  # type: ignore[no-untyped-def]
        """
        In order to pass this to an extern kernel we need a
        ReinterpretView not a View.  This allows us to avoid some
        unneeded copies.
        """
        assert isinstance(x, BaseView)
        if isinstance(x, ReinterpretView):
            return x

        # NOTE: Don't use extract_read_writes here as it fails when
        # make_loader() inlines the computation
        x_unwrap_view = x.unwrap_view()
        buf = V.graph.get_buffer(x_unwrap_view.get_name())
        assert buf is not None
        x_unwrap_view_fx_node = buf.get_origin_node()
        # Prefer channels last format according to how the format is set from eager.
        if (
            x_unwrap_view_fx_node is not None
            and "val" in x_unwrap_view_fx_node.meta
            and isinstance(x_unwrap_view.layout, FlexibleLayout)
            and (
                x_unwrap_view_fx_node.meta["val"].is_contiguous(
                    memory_format=torch.channels_last
                )
                or x_unwrap_view_fx_node.meta["val"].is_contiguous(
                    memory_format=torch.channels_last_3d
                )
            )
        ):
            x_unwrap_view.freeze_layout_with_same_order(
                make_channels_last_strides_for(x_unwrap_view.get_size())
            )
        else:
            x_unwrap_view.freeze_layout()

        index_args, var_ranges = dependencies.index_vars_squeeze(
            x.get_size(),
            prefix="r",  # type: ignore[arg-type]
        )
        range_vars = index_args[0]
        index = x.make_indexer()(range_vars)

        index = V.graph.sizevars.simplify_with_ranges(index, var_ranges)
        strides = V.graph.sizevars.stride_vars(index, range_vars)
        offset = V.graph.sizevars.offset_var(index, range_vars)
        expected = sympy_dot(range_vars, strides) + offset

        if index != expected:
            log.debug(
                "convert_to_reinterpret_view failed: stride=%s offset=%s index=%s",
                strides,
                offset,
                index,
            )
            raise NotImplementedError

        return ReinterpretView(
            data=x.data,
            layout=FixedLayout(
                device=x.get_device_or_error(),
                dtype=x.get_dtype(),
                size=x.get_size(),  # type: ignore[arg-type]
                stride=strides,
                offset=offset,
            ),
        )

    @classmethod
    def realize_input(cls, x):  # type: ignore[no-untyped-def]
        if x is None:
            return NoneAsConstantBuffer()
        if isinstance(x, (sympy.Expr, sympy.logic.boolalg.Boolean, int)):
            return ShapeAsConstantBuffer(expr=x)
        if isinstance(x, Constant):
            return V.graph.add_tensor_constant(
                torch.tensor(x.value, dtype=x.get_dtype(), device=x.get_device())
            )
        if isinstance(x, ConstantBuffer):
            return x
        if isinstance(x, TensorBox):
            return cls.realize_input(x.data)
        if isinstance(x, ReinterpretView):
            return ReinterpretView(
                data=cls.realize_input(x.data), layout=x.get_layout()
            )
        if isinstance(x, BaseView):
            x.realize()
            if is_storage_and_layout(x.unwrap_view()):
                try:
                    return cls.convert_to_reinterpret_view(x)
                except NotImplementedError:
                    pass
        if isinstance(x, StorageBox):
            # TODO(jansel): impose layout preference on realized buffer
            x.realize()
            return x
        if isinstance(x, (NonTensorObj, ShapeAsConstantBuffer)):
            return x
        return cls.copy_input(x)

    @classmethod
    def require_stride1(cls, x):  # type: ignore[no-untyped-def]
        if is_storage_and_layout(x):
            if len(x.get_stride()) == 0:
                return x
            for stride in x.get_stride():
                if stride == 1:
                    return x
        return cls.copy_input(x)

    @classmethod
    def require_strides(  # type: ignore[no-untyped-def]
        cls,
        x,
        order: Optional[Sequence[int]] = None,
        exact_strides: Optional[Sequence[_IntLike]] = None,
        allow_padding=False,
    ):
        assert order is not None or exact_strides is not None
        # Layout generally doesn't matter, but some consuming external ops might have requirements
        if x.get_numel() in (0, 1) and not exact_strides:
            return x

        # require x to have the layout
        if is_storage_and_layout(x):
            if isinstance(x.get_layout(), FlexibleLayout):
                if order:
                    # If the the FlexibleLayout already has the size and stride in the required order,
                    # freeze it to a FixedLayout by using its current size and stride.
                    # The behavior of using its current size and stride or the given order can be different
                    # if the size and stride has ambiguilty, for example for a 4D input where the iC = 1:
                    # size=[s0, 1, 28, 28], stride=[784, 784, 28, 1]. If the required order is [3, 0, 2, 1] (channels last),
                    # the current size and stride already satisfies this order.
                    # However by freezing it to the required order, the layout will be changed to:
                    # size=[s0, 1, 28, 28], stride=[784, 1, 28, 1]), which is not actually necessary.

                    # fix flexiblelayout to be FixedLayout with stride_order
                    as_storage_and_layout(
                        x,
                        freeze=True,
                        want_contiguous=False,
                        stride_order=(
                            get_stride_order(
                                V.graph.sizevars.size_hints(x.get_layout().stride)
                            )
                            if is_stride_order_storage_and_layout(x, order)
                            else order
                        ),
                        allow_padding=allow_padding,
                    )
                    return x
                else:
                    # If the exact_strides is given, freeze the FlexibleLayout to a FixedLayout with the exact_strides.
                    as_storage_and_layout(
                        x,
                        freeze=True,
                        want_contiguous=False,
                        stride_order=None,
                        allow_padding=allow_padding,
                        exact_strides=exact_strides,
                    )
                    return x
            elif isinstance(x.get_layout(), (FixedLayout, NonOwningLayout)) and (
                (order and x.get_layout().is_stride_ordered(order))
                or (
                    exact_strides
                    and significant_strides_equal(
                        exact_strides, x.get_layout().stride, x.get_size()
                    )
                )
            ):
                return (
                    try_match_insignificant_strides(x, exact_strides)
                    if exact_strides is not None
                    else x
                )
            elif isinstance(x.get_layout(), MutationLayoutSHOULDREMOVE):
                if isinstance(x.get_layout().real_layout(), FlexibleLayout):
                    raise AssertionError(
                        "the MutationLayoutSHOULDREMOVE's real layout shouldn't be FlexibleLayout"
                    )
                elif isinstance(x.get_layout().real_layout(), FixedLayout) and (
                    (order and x.get_layout().real_layout().is_stride_ordered(order))
                    or (
                        exact_strides
                        and significant_strides_equal(
                            exact_strides,
                            x.get_layout().real_layout().stride,
                            x.get_size(),
                        )
                    )
                ):
                    return x

        # TODO - Storage to InputBuffer
        if isinstance(x, InputBuffer) and (
            (order and x.get_layout().is_stride_ordered(order))
            or (
                exact_strides
                and significant_strides_equal(
                    exact_strides, x.get_layout().stride, x.get_size()
                )
            )
        ):
            return x
        if (
            isinstance(x, TensorBox)
            and isinstance(x.data, BaseView)
            and not isinstance(x.data, ReinterpretView)
            and is_storage_and_layout(x.unwrap_view())
            and not isinstance(x.unwrap_view().data, ExternKernelAlloc)  # type: ignore[attr-defined]
        ):
            try:
                x.data = cls.convert_to_reinterpret_view(x.data)
                if order:
                    return cls.require_stride_order(
                        x, order, allow_padding=allow_padding
                    )
                elif exact_strides:
                    return cls.require_exact_strides(
                        x, exact_strides, allow_padding=allow_padding
                    )
            except NotImplementedError:
                pass

        # Preserve ExpandView representation that would be lost during copy_input
        # Without representation of the expand in inductor IR, in codegen we end up
        # launching a grid for the full size tensor and doing redundant computation
        # across expanded dims.
        # TODO: could also be good to have a codegen fix to recognize overlapping elements

        expanded_dims: Optional[list[int]] = None
        orig_size = x.get_size()
        if exact_strides is not None:
            sizevars = V.graph.sizevars
            expanded_dims = [
                i
                for i in range(len(x.get_size()))
                if sizevars.statically_known_equals(exact_strides[i], 0)
                and sizevars.statically_known_geq(x.get_size()[i], 2)
            ]

            for dim in expanded_dims:
                x = torch._inductor.lowering.slice_(x, dim, 0, 1)

        # Although this is a clone, inductor is good about fusing clones into previous
        # operations if they weren't realized and their layouts were flexible.
        x = cls.copy_input(x)

        as_storage_and_layout(
            x,
            freeze=True,
            want_contiguous=False,
            stride_order=order,
            allow_padding=allow_padding,
            exact_strides=exact_strides,
        )
        if order:
            assert is_stride_order_storage_and_layout(x, order)
        elif expanded_dims:
            assert orig_size is not None and exact_strides is not None
            x = torch._inductor.lowering.expand(x, orig_size)
            # the expand will sometimes may change insignificant strides, so match them back
            return try_match_insignificant_strides(x, exact_strides)

        return x

    @classmethod
    def require_exact_strides(cls, x, exact_strides, allow_padding=False):  # type: ignore[no-untyped-def]
        return cls.require_strides(
            x, exact_strides=exact_strides, allow_padding=allow_padding
        )

    @classmethod
    def require_stride_order(cls, x, order, allow_padding=False):  # type: ignore[no-untyped-def]
        return cls.require_strides(x, order=order, allow_padding=allow_padding)

    @classmethod
    def require_channels_last(cls, x):  # type: ignore[no-untyped-def]
        return cls.require_stride_order(x, NHWC_STRIDE_ORDER)

    @classmethod
    def require_channels_last_3d(cls, x):  # type: ignore[no-untyped-def]
        return cls.require_stride_order(x, NHWDC_STRIDE_ORDER)

    @classmethod
    def require_contiguous(cls, x):  # type: ignore[no-untyped-def]
        return cls.require_stride_order(x, list(reversed(range(len(x.get_size())))))

    def apply_constraint(self) -> None:
        pass

    def fill_non_provided_args(self, args, kwargs):  # type: ignore[no-untyped-def]
        # Previously, we want to maintain forward-compatibility by skipping
        # default args in the serialized artifacts in fbcode. However,
        # some of our shim interfaces require default values being OrderedSet.
        # Discussed with Sherlock offline and we decided to allow serializing
        # default args into the C++ wrapper code for now. We will refine this
        # part if we see real FC requirement. More details related to FC
        # can be found at:
        # https://docs.google.com/document/d/1FzWm-sHYwmRi3x_g036kOxd99KaYquUsA-L5JwOn8ys/edit?usp=sharing
        assert isinstance(args, (list, tuple))
        if isinstance(args, tuple):
            args = list(args)
        assert self.arg_properties, "ExternKernel.arg_properties should not be empty"

        n_args = len(args)
        n_pos_args = len(self.arg_properties)
        # For cpp wrapper, if some positional args are not provided, we need to check
        # if they're in the kwargs or use their default value
        if n_args < n_pos_args:
            log.debug(
                "%s has %d unprovided positional arguments. "
                "Will check if they are in the keyword arguments or will use default values.",
                self.op_overload,
                n_pos_args - n_args,
            )
            for i in range(n_args, n_pos_args):
                arg_name = self.arg_properties[i]["name"]
                args.append(
                    kwargs[arg_name]
                    if arg_name in kwargs
                    else self.arg_properties[i]["default_value"]
                )
        return args

    def codegen_const_args(self, names: Optional[list[str]] = None):  # type: ignore[no-untyped-def]
        if V.graph.cpp_wrapper:
            result = []
            # Aten ops follow the convention that tensor args are before non-tensor args,
            # in which case the following 'len(self.inputs) + i' logic works. But this
            # may not be true for other ops, and if that is the case, caller needs to
            # pass in a list of const arg names for arg_properties lookup.
            name_to_arg_properties = None
            if names and self.arg_properties:
                assert len(self.constant_args) == len(names), (
                    "names passed to codegen_const_args does not match self.constant_args"
                )
                name_to_arg_properties = {
                    arg.get("name"): arg for arg in self.arg_properties
                }

            for i, x in enumerate(self.constant_args):
                if name_to_arg_properties is not None:
                    prop = name_to_arg_properties.get(names[i])  # type: ignore[index]
                    type_ = prop.get("type") if prop else None
                else:
                    idx = len(self.inputs) + i
                    type_ = (
                        self.arg_properties[idx].get("type")
                        if self.arg_properties and idx < len(self.arg_properties)
                        else None
                    )
                result.append(V.graph.wrapper_code.val_to_arg_str(x, type_))
            return result
        else:
            return map(V.graph.wrapper_code.val_to_arg_str, self.constant_args)

    def codegen_args(self):  # type: ignore[no-untyped-def]
        if V.graph.cpp_wrapper and self.op_overload is not None:
            # cpp wrapper needs special logic to fill in missing args with default values
            inputs = self.fill_non_provided_args(
                [*self.inputs, *self.constant_args], self.kwargs
            )
            # fill_non_provided_args has handled constant args, so no need to codegen for that later
            need_codegen_constant_args = False
        else:
            inputs = self.inputs
            need_codegen_constant_args = True

        args = []
        for i, x in enumerate(inputs):
            if V.graph.cpp_wrapper:
                assert self.arg_properties and i < len(self.arg_properties), (
                    "Invalid access to ExternKernel.arg_properties"
                )
                type_ = self.arg_properties[i].get("type")
                args.append(V.graph.wrapper_code.val_to_arg_str(x, type_))
            else:
                args.append(V.graph.wrapper_code.val_to_arg_str(x))
        if need_codegen_constant_args:
            args.extend(self.codegen_const_args())
        return args

    def get_kwargs_value(self, arg_name, **kwargs):  # type: ignore[no-untyped-def]
        """Given an argument name, queries for values in (in order):
        1. any provided kwargs for this function.
        2. the class self.kwargs member.
        3. any available default arguments in self.allarg_properties."""
        if arg_name in kwargs:
            return kwargs.get(arg_name)
        if arg_name in self.kwargs:
            return self.kwargs.get(arg_name)
        if self.allarg_properties and arg_name in self.allarg_properties:
            return self.allarg_properties.get(arg_name).get("default_value")  # type: ignore[union-attr]
        raise AssertionError(f"{arg_name} not in self.allarg_properties")

    def codegen_kwargs(self, skip_out=False):  # type: ignore[no-untyped-def]
        if V.graph.cpp_wrapper:
            if self.op_overload is not None and len(self.schema_kwargs) == 0:
                # All the args should have been generated by fill_non_provided_args in codegen_args
                return []

            kwargs = []
            for arg_name in self.ordered_kwargs_for_cpp_kernel:
                if skip_out and arg_name == "out":
                    # ExternKernelOut has its own logic for inserting the out parameter
                    continue

                v = self.get_kwargs_value(arg_name)
                if isinstance(v, sympy.Expr):
                    kwargs.append(v)
                else:
                    type_ = (
                        self.allarg_properties.get(arg_name).get("type")  # type: ignore[union-attr]
                        if self.allarg_properties and arg_name in self.allarg_properties
                        else None
                    )
                    kwargs.append(V.graph.wrapper_code.val_to_arg_str(v, type_))
        else:
            kwargs = [
                f"{k}={V.graph.wrapper_code.val_to_arg_str(v)}"
                for k, v in self.kwargs.items()
            ]
        return kwargs

    def codegen_size_asserts(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        if config.size_asserts and not V.graph.cpp_wrapper:
            # comparing strides for 0 size tensor is tricky. Ignore them for now.
            if sympy_product(self.get_size()) == 0:
                return
            size = V.graph.wrapper_code.codegen_shape_tuple(self.get_size())
            stride = V.graph.wrapper_code.codegen_shape_tuple(self.get_stride())

            wrapper.writeline(
                f"assert_size_stride({self.get_name()}, {size}, {stride})"
            )

    def codegen_alignment_asserts(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        if config.alignment_asserts and not V.graph.cpp_wrapper:
            name = self.get_name()
            aligned = name not in V.graph.unaligned_buffers
            if aligned:
                wrapper.writeline(f"assert_alignment({name}, {GPU_ALIGN_BYTES})")
            else:
                wrapper.writeline(f"# buffer {name} is assumed to be not aligned")

    def get_group_stride(self):  # type: ignore[no-untyped-def]
        """
        get output sizes and strides, for template_codegen
        """
        _size = self.get_size()
        _stride = self.get_stride()
        # iter_ranges = _size of output tensor, reduce_range = [] because no reduction
        return [_size, []], _stride

    def canonicalize(self):  # type: ignore[no-untyped-def]
        """
        Manually get canonicalization of the output index
        """
        # manually generate index formula for conv
        sizevars = V.graph.sizevars
        sizes = self.get_size()
        strides = self.get_stride()
        strides = [sizevars.size_hint(x) for x in strides]
        # TODO: I can't tell if the symbols here are temporary
        index_vars = [sympy_index_symbol(f"d{i}") for i in range(len(sizes))]
        # reorder index vars according to stride
        index_order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        lookup = {pos: idx for idx, pos in enumerate(index_order)}
        order = [lookup[i] for i in range(len(lookup))]
        index_vars = [index_vars[i] for i in order]
        indexer = self.make_indexer()
        index = indexer(index_vars)

        new_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, [index]
        )

        # assign new variables each dimension to deal with numbering mismatches
        # d0, d1, d2 could become d0, d2 -- which won't match d0, d1
        _, add_var = var_builder("c")
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))

        index = sympy_subs(sympy.expand(index), replacement)
        return index, tuple(new_sizes)

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # NB: It's not necessary to check regular inputs as we automatically
        # have dependencies on them
        maybe_get_symbols = (
            maybe_free_unbacked_symbols if unbacked_only else maybe_free_symbols
        )
        r = OrderedSet[sympy.Symbol]()
        for arg in self.constant_args:
            r |= maybe_get_symbols(arg)
        for arg in self.kwargs.values():
            r |= maybe_get_symbols(arg)
        return r

    def __str__(self) -> str:
        kernel_name = getattr(self, "python_kernel_name", None)
        lines = [
            f"python_kernel_name={kernel_name!r}",
        ]
        lines += [
            f"{field.name}={getattr(self, field.name)}"
            for field in dataclasses.fields(self)
        ]
        lines.append(f"origin_node={self.origin_node!r}")
        return self.str_helper(lines)

    __repr__ = __str__


@ir_dataclass(frozen=False)
class ExternKernelOut(ExternKernel):
    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        wrapper.generate_extern_kernel_out(self)

    def __init__(  # type: ignore[no-untyped-def]
        self,
        layout,
        inputs,
        constant_args=(),
        kwargs=None,
        output_view=None,
        python_kernel_name=None,
        cpp_kernel_name=None,
        ordered_kwargs_for_cpp_kernel=(),
        op_overload=None,
    ) -> None:
        super().__init__(
            None,
            layout,
            self.unwrap_storage(inputs),
            constant_args,
            kwargs or {},
            None,
            python_kernel_name,
            cpp_kernel_name,
            ordered_kwargs_for_cpp_kernel,
            op_overload,
        )
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def should_allocate(self) -> bool:
        return True


class RandomSeeds(ExternKernelOut):
    def __init__(self, count: int, device: torch.device) -> None:
        limits = torch.iinfo(torch.int64)
        super().__init__(
            layout=FixedLayout(
                device=device,
                dtype=torch.int64,
                size=[count],
            ),
            inputs=[],
            constant_args=[limits.min, limits.max, [count]],
            python_kernel_name="aten.randint.low_out",
            # FIXME: Ideally we should only use at::_ops::randint_low_out::call here,
            # but the signature is different from is at::randint_out. Again,
            # we can simplify the code when only keeping an ABI-compatible version.
            cpp_kernel_name="at::_ops::randint_low_out::call",
            op_overload=aten.randint.low_out,
        )


class ExternKernelAlloc(ExternKernel):
    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        wrapper.generate_extern_kernel_alloc(self)

    def __init__(  # type: ignore[no-untyped-def]
        self,
        layout,
        inputs,
        constant_args=(),
        kwargs=None,
        python_kernel_name=None,
        cpp_kernel_name=None,
        ordered_kwargs_for_cpp_kernel=(),
        op_overload=None,
    ) -> None:
        super().__init__(
            None,
            layout,
            self.unwrap_storage(inputs),
            constant_args,
            kwargs or {},
            None,
            python_kernel_name,
            cpp_kernel_name,
            ordered_kwargs_for_cpp_kernel,
            op_overload,
        )
        # We need output buffers for generating kernel arguments in the
        # abi-compatible mode, where we retrieve outputs by pass each individual
        # output through the abi-compatible interface.
        self.outputs: Sequence[Any] = []
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def should_allocate(self) -> bool:
        return False

    def apply_constraint(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class MutationOutput(Buffer):
    """
    An output buffer that represents the mutation of a pre-existing buffer
    """

    def __init__(self, layout, mutated_node, mutating_node: Operation) -> None:  # type: ignore[no-untyped-def]
        super().__init__(name=None, layout=layout)
        mutated_node_name = mutated_node.get_name()
        V.graph.mark_buffer_mutated(mutated_node_name)
        self.mutation_names = [mutated_node_name]
        self.mutating_node: Operation = mutating_node
        self.name = V.graph.register_buffer(self)

    def get_defining_op(self) -> Operation:
        return self.mutating_node

    def get_mutation_names(self):  # type: ignore[no-untyped-def]
        return self.mutation_names

    def should_allocate(self) -> bool:
        return False


class TMADescriptor(ExternKernel):
    """
    An IR node representing a host-side TMA descriptor in the Triton API
    (the ones obtained via create_{1d,2d}_tma_descriptor calls). Mostly
    useful for user-defined Triton kernels relying on host-side TMA; but
    can, in principle, be used for Inductor's Triton templates, too.
    """

    # as TMA descriptors are immutable,
    # we can dedup them by the input args
    _CACHE: dict[Any, TMADescriptor] = {}

    @classmethod
    def create(  # type: ignore[no-untyped-def]
        cls,
        tensor: IRNode,
        dims: list[Union[int, torch.SymInt]],
        block_dims: list[Union[int, torch.SymInt]],
        element_size: Optional[int] = None,
    ):
        key = (id(tensor), dims, block_dims, element_size)
        if key not in cls._CACHE:
            cls._CACHE[key] = TMADescriptor(tensor, dims, block_dims, element_size)
        return cls._CACHE[key]

    def __init__(
        self,
        tensor: IRNode,
        dims: list[Union[int, torch.SymInt]],
        block_dims: list[Union[int, torch.SymInt]],
        element_size: Optional[int] = None,
    ) -> None:
        assert len(dims) in (1, 2)
        assert len(dims) == len(block_dims)

        if element_size is None:
            element_size = tensor.get_dtype().itemsize

        self.tensor = tensor
        self.dims = dims
        self.block_dims = block_dims
        self.element_size = element_size
        self.rank = len(self.dims)

        inputs = [tensor]
        constant_args = [
            *self.dims,
            *self.block_dims,
            self.element_size,
        ]

        super().__init__(
            None,
            # link back to the underlying tensor in terms of ownership
            # to avoid getting the underlying tensor deleted *before*
            # the TMADescriptor node can be deleted.
            NonOwningLayout(
                ReinterpretView(
                    data=tensor,
                    layout=tensor.get_layout(),
                )
            ),
            inputs,
            tuple(constant_args),
            None,
        )

        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        wrapper.generate_tma_descriptor(self)


class SubgraphBuffer(ExternKernel):
    def __init__(
        self,
        layout: Layout,
        input_nodes: list[Buffer],
        gm: torch.fx.GraphModule,
        example_inputs: list[Any],
        subgraph_name: str,
    ):
        super().__init__(None, layout, input_nodes)
        self.gm = gm
        self.example_inputs = example_inputs
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

        self.subgraph = V.graph.make_subgraph(
            self.gm, self.example_inputs, subgraph_name
        )
        import torch._inductor.config as inductor_config

        with V.set_graph_handler(self.subgraph):
            # Don't bother autotuning on Triton here
            with inductor_config.patch(  # type: ignore[no-untyped-def]
                max_autotune=False,
                max_autotune_gemm=False,
                max_autotune_gemm_backends="ATEN",
            ):
                self.subgraph.run(*self.example_inputs)

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        class CodegenGraph:
            def __init__(self, graph: GraphLowering):
                self.graph = graph
                self.name = graph.name

        outer_inputs = [t.codegen_reference() for t in self.inputs]

        wrapper.codegen_subgraph_with_flattened_outputs(
            CodegenGraph(self.subgraph),
            outer_inputs,
            [self.name],
        )


class UserDefinedTritonKernel(ExternKernel):
    def get_kernel_and_metadata(self):  # type: ignore[no-untyped-def]
        from triton.runtime.autotuner import Autotuner

        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        kernel = kernel_side_table.get_kernel(self.kernel_idx)
        configs = []
        restore_value_args: list[str] = []
        reset_to_zero_args: list[str] = []
        if isinstance(kernel, Autotuner):
            # https://github.com/triton-lang/triton/pull/5083
            # changes kernel.restore_idx to kernel.restore_value
            if hasattr(kernel, "restore_idx"):
                restore_value_args.extend(
                    kernel.fn.arg_names[i] for i in kernel.restore_idx
                )
            else:
                assert hasattr(kernel, "restore_value")
                restore_value_args.extend(kernel.restore_value)

            if hasattr(kernel, "reset_idx"):
                for i in kernel.reset_idx:
                    reset_to_zero_args.append(kernel.fn.arg_names[i])
            else:
                assert hasattr(kernel, "reset_to_zero")
                reset_to_zero_args.extend(kernel.reset_to_zero)

            configs = kernel.configs
            kernel = kernel.fn
        return kernel, configs, restore_value_args, reset_to_zero_args

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        from torch._inductor.utils import triton_version_uses_attrs_dict

        (
            kernel,
            configs,
            restore_value_args,
            reset_to_zero_args,
        ) = self.get_kernel_and_metadata()

        # Definition of kernel
        (
            new_name,
            triton_meta,
            extra_launch_args,
        ) = wrapper.define_user_defined_triton_kernel(
            kernel,
            configs,
            self.kwargs,
            restore_value_args,
            reset_to_zero_args,
            self.grid,
        )
        named_args = {
            k: self.get_kwargs_value(k) for k in self.ordered_kwargs_for_cpp_kernel
        }
        constexpr_names = OrderedSet([kernel.arg_names[i] for i in kernel.constexprs])

        args: list[Any] = []
        arg_types: list[Any] = []
        raw_keys_filtered: list[Any] = []
        raw_args_filtered: list[Any] = []
        for name, arg in itertools.chain(
            named_args.items(), zip(itertools.repeat(""), extra_launch_args)
        ):
            raw_keys_filtered.append(name)
            raw_args_filtered.append(arg)
            if isinstance(arg, IRNode):
                args.append(arg.codegen_reference())
                arg_types.append(arg.get_dtype())
            elif isinstance(arg, (int, float, bool, sympy.Expr)):
                args.append(arg)
                arg_types.append(type(arg))
            elif name in constexpr_names:
                # insert a dummy value for constexpr args of unsupported type
                # constexprs will end up getting baked into the kernel at compile time
                args.append(-1)
                arg_types.append(int)
            elif arg is None:
                """
                Filter out None args.

                see https://github.com/pytorch/pytorch/issues/115344

                Two cases for a None arg:
                1. The arg is already tl.constexpr, so leave it in
                2. The arg is not tl.constexpr so we have to remove it
                """
                if triton_version_uses_attrs_dict():
                    args.append(-1)
                    arg_types.append(int)
                else:
                    raw_keys_filtered.pop()
                    raw_args_filtered.pop()
            else:
                raise NotImplementedError(f"Unsupported arg type: {type(arg)}: {arg}")

        self.codegen_comment(wrapper)
        wrapper.generate_kernel_call(
            new_name,
            args,
            arg_types=arg_types,
            raw_args=raw_args_filtered,
            raw_keys=raw_keys_filtered,
            triton_meta=triton_meta,
            triton=True,
            device=self.get_device(),
            original_fxnode_name=self.fx_node.name,
        )

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # add unbacked symbols used in the grid to the ones used
        # in the kwargs (the latter is generated by ExternKernel)
        return super().get_free_symbol_uses(unbacked_only) | get_free_symbols(
            self.grid, unbacked_only
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(  # type: ignore[no-untyped-def]
        self, *, kernel_idx, grid, tma_descriptor_metadata, kernel_args
    ) -> None:
        inputs = []
        kwargs = {}
        constant_args = []
        for k, v in kernel_args.items():
            if isinstance(v, TensorBox):
                t = InputsKernel.unwrap_storage_for_input(self.realize_input(v))
                if k in tma_descriptor_metadata:
                    t = TMADescriptor.create(t, *tma_descriptor_metadata[k])
                inputs.append(t)
                kwargs[k] = t
            else:
                constant_args.append(v)
                kwargs[k] = v

        assert len(inputs) != 0
        self.device = inputs[0].get_device()

        super().__init__(
            None,
            NoneLayout(device=self.device),
            inputs,
            tuple(constant_args),
            kwargs,
        )
        self.kernel_idx = kernel_idx
        self.grid = grid

        kernel, configs, _, _ = self.get_kernel_and_metadata()

        # If we are autotuning, not all arguments will be passed
        self.ordered_kwargs_for_cpp_kernel = [
            arg for arg in kernel.arg_names if arg in kernel_args
        ]

        from torch._higher_order_ops.triton_kernel_wrap import identify_mutated_tensors

        autotuned_kwargs = configs[0].kwargs if len(configs) > 0 else {}
        self.mutable_args = [
            kernel_args[key]
            for key in identify_mutated_tensors(
                kernel, {**kernel_args, **autotuned_kwargs}
            )
        ]

        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=self.device), buf, self)
            for buf in self.mutable_args
        ]
        V.graph.register_operation(self)

    def get_outputs(self) -> list[Buffer]:
        return list(self.mutation_outputs)

    def get_device(self) -> Optional[torch.device]:
        return self.device


class InplaceBernoulliFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        (x,) = (t.codegen_reference() for t in self.inputs)

        if V.graph.cpp_wrapper:
            # Inductor doesn't really support aten Generator, so the Generator kwarg is always NULL here,
            # which needs to be explicitly generated for cpp wrapper
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}, NULL){wrapper.ending}"
            )
        else:
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}){wrapper.ending}"
            )

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self):  # type: ignore[no-untyped-def]
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(self, op_overload, x, *constant_args) -> None:  # type: ignore[no-untyped-def]
        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage([x]),
            constant_args,
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(x.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


# Used to deal with torch.complex types
class InplaceCopyFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        (dst, src, non_blocking) = self.codegen_args()
        wrapper.codegen_device_copy(src, dst, non_blocking)

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self):  # type: ignore[no-untyped-def]
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(  # type: ignore[no-untyped-def]
        self,
        layout,
        inputs,
        constant_args,
    ) -> None:
        super().__init__(
            None,
            layout,
            inputs,
            constant_args,
            python_kernel_name="aten.copy_",
            cpp_kernel_name="aoti_torch_copy_",
        )
        V.graph.mark_buffer_mutated(inputs[0].get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(cls, dst, src, non_blocking: bool = False):  # type: ignore[no-untyped-def]
        inputs = [cls.realize_input(t) for t in [dst, src]]
        constant_args = (non_blocking,)
        result = InplaceCopyFallback(
            NoneLayout(device=dst.get_device()),
            inputs,
            constant_args,
        )
        return result


class MutatingFirstArgExternKernel(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        argrefs = [
            *(t.codegen_reference() for t in self.inputs),
            *map(repr, self.constant_args),
        ]
        wrapper.writeline(
            f"{self.get_kernel_name()}({', '.join(argrefs)}){wrapper.ending}"
        )

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self):  # type: ignore[no-untyped-def]
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def has_side_effects(self) -> bool:
        return True


class ResizeStorageBytes(MutatingFirstArgExternKernel):
    def __init__(self, variable, new_size) -> None:  # type: ignore[no-untyped-def]
        assert isinstance(new_size, int), "TODO: dynamic shapes"
        super().__init__(
            None,
            NoneLayout(device=variable.get_device()),
            self.unwrap_storage([variable]),
            constant_args=(new_size,),
        )
        V.graph.mark_buffer_mutated(variable.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        self.python_kernel_name = "inductor_ops.resize_storage_bytes_"
        self.cpp_kernel_name = "torch::inductor::resize_storage_bytes_"
        V.graph.never_reuse_buffers.add(variable.data.get_name())


class SetSourceTensorKernel(ExternKernelAlloc):
    def __init__(self, self_tensor, storage_tensor) -> None:  # type: ignore[no-untyped-def]
        storage_tensor.freeze_layout()
        super().__init__(
            storage_tensor.get_layout(),
            [self_tensor, storage_tensor],
            python_kernel_name="torch.ops.aten.set_.source_Tensor",
            op_overload=torch.ops.aten.set_.source_Tensor,
        )
        V.graph.never_reuse_buffers.add(self_tensor.data.get_name())
        V.graph.never_reuse_buffers.add(storage_tensor.get_name())
        V.graph.never_reuse_buffers.add(self.get_name())
        device = storage_tensor.get_device()
        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=device), self_tensor, self),
            MutationOutput(NoneLayout(device=device), storage_tensor, self),
        ]

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        return [self.inputs[0].get_name(), self.inputs[1].get_name()]


class ScatterFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly.
    This class handles both aten.scatter_ and aten.scatter_reduce_.
    It also handle the case `src` being a scalar properly.
    """

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        reduce = self.kwargs["reduce"]
        if V.graph.cpp_wrapper:
            # Follow aten/src/ATen/native/ReductionType.h:get_operator_enum
            get_operator_enum = {"add": "sum", "multiply": "prod"}
            if reduce in get_operator_enum:
                reduce = get_operator_enum[reduce]

        if self.src_is_tensor:
            (x, index, src) = (t.codegen_reference() for t in self.inputs)
        else:
            (x, index) = (t.codegen_reference() for t in self.inputs)
            src = self.constant_args[1]
        wrapper.generate_scatter_fallback(
            x,
            [x, self.constant_args[0], index, src],
            self.cpp_kernel_name,
            self.python_kernel_name,
            self.src_is_tensor,
            reduce,
            self.codegen_kwargs(),
        )

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self):  # type: ignore[no-untyped-def]
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(  # type: ignore[no-untyped-def]
        self,
        op_overload,
        x,
        dim: int,
        index,
        src,
        *,
        reduce: Optional[str] = None,
        include_self: bool = True,
    ) -> None:
        self.src_is_tensor = isinstance(src, TensorBox)

        constant_args: tuple[Any, ...]
        if self.src_is_tensor:
            tensors = [self.realize_input(t) for t in [x, index, src]]
            constant_args = (dim,)
        else:
            tensors = [self.realize_input(t) for t in [x, index]]
            constant_args = (dim, src)

        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage(tensors),
            constant_args,
            {"reduce": reduce, "include_self": include_self},
            python_kernel_name=str(op_overload),
            ordered_kwargs_for_cpp_kernel=["reduce", "include_self"],
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(x.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class IndexPutFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation and indices properly
    """

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        (x, values, *valid_indices) = (t.codegen_reference() for t in self.inputs)
        indices = []
        iter_valid_indices = iter(valid_indices)
        for i, _ in enumerate(self.indices):
            if self.indices[i] is not None:
                indices.append(next(iter_valid_indices))
            else:
                indices.append(V.graph.wrapper_code.none_str)

        wrapper.generate_index_put_fallback(
            self.get_kernel_name(), x, indices, values, *self.codegen_const_args()
        )

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self):  # type: ignore[no-untyped-def]
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(self, op_overload, x, indices, values, accumulate) -> None:  # type: ignore[no-untyped-def]
        self.indices = indices
        valid_indices = [i for i in indices if i is not None]
        tensors = [self.realize_input(x) for x in [x, values, *valid_indices]]
        cpp_kernel_name = "aoti_torch_index_put_out"
        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage(tensors),
            (accumulate,),
            python_kernel_name="aten.index_put_",
            cpp_kernel_name=cpp_kernel_name,
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(self.inputs[0].get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class DeviceCopy(ExternKernelOut):
    @classmethod
    def create(cls, x, device, non_blocking):  # type: ignore[no-untyped-def]
        if (
            not x.is_extern()
            and all(r in V.graph.constants for r in x.get_read_names())
            and not config.aot_inductor.use_runtime_constant_folding
        ):
            return x.constant_to_device(device)

        V.graph.add_device_info(device)
        V.graph.add_device_info(x.get_device())

        developer_warning("DeviceCopy in input program")
        constant_args = (non_blocking,)
        return DeviceCopy(
            FlexibleLayout(
                device=device,
                dtype=x.get_dtype(),
                size=x.get_size(),
            ),
            [cls.realize_input(x)],
            constant_args,
        )

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        args = self.codegen_args()
        assert len(args) == 2
        if self.output_view:
            wrapper.codegen_device_copy(
                args[0], self.output_view.codegen_reference(), args[1]
            )
        else:
            wrapper.codegen_device_copy(args[0], self.codegen_reference(), args[1])


class DynamicScalar(ExternKernel):
    """
    The result of a call to aten._local_scalar_dense.
    """

    def get_reads(self) -> OrderedSet[Dep]:
        return OrderedSet()

    def should_allocate(self) -> bool:
        return False

    def __init__(self, sym, keypath, data) -> None:  # type: ignore[no-untyped-def]
        data.realize()
        super().__init__(
            None, NoneLayout(device=torch.device("cpu")), self.unwrap_storage([data])
        )
        self.sym = sym
        self.keypath = keypath

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet([self.sym])

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        wrapper.codegen_dynamic_scalar(self)


class AssertScalar(ExternKernel):
    """
    The result of a call to aten._assert_scalar
    """

    def get_reads(self) -> OrderedSet[Dep]:
        return OrderedSet()

    def should_allocate(self) -> bool:
        return False

    def __init__(self, scalar, msg) -> None:  # type: ignore[no-untyped-def]
        super().__init__(
            # Buffer(name, layotu)
            None,
            NoneLayout(device=torch.device("cpu")),
            # InputsKernel(inputs)
            [],
        )
        self.scalar = scalar
        self.msg = msg

    def has_side_effects(self) -> bool:
        return True

    def get_free_symbol_uses(self, unbacked_only: bool = False):  # type: ignore[no-untyped-def]
        return get_free_symbols(self.scalar, unbacked_only)

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        if not config.scalar_asserts:
            return
        # NB: It is EXTREMELY important not to simplify the scalar under assertion here,
        # because simplify is done with respect to runtime asserts.  So if you have
        # "u0 == 0" in the runtime asserts, if you subsequently try to
        # simplify(u0 == 0), you will get True (because we've already runtime assert'ed
        # that it's true).  But we're code generating the actual runtime assert here!!
        symbol = next(iter(self.get_free_symbol_uses(unbacked_only=True)))
        if V.graph.cpp_wrapper:
            symbol_str = f"std::to_string({symbol})"
            sizevar = V.graph.wrapper_code.codegen_cpp_sizevar(
                self.scalar, simplify=False
            )
            # TODO: when we start compiling in C++20, annotate with [[unlikely]].
            wrapper.writeline(
                f'if (!({sizevar})) {{ throw std::runtime_error("Expected {self.msg} but received " + {symbol_str}); }}'
            )
        else:
            sizevar = V.graph.wrapper_code.codegen_python_sizevar(
                self.scalar, simplify=False
            )
            wrapper.writeline(f"if not ({sizevar}):")
            wrapper.writeline(f"    raise RuntimeError({repr(self.msg)})")
            # No one should ever use this buffer, but for uniformity
            # define the variable and assign it None
            wrapper.writeline(f"{self.get_name()} = None")


@ir_dataclass(frozen=False)
class ExternKernelNode:
    name: str
    node: export_schema.Node


class FallbackKernel(ExternKernelAlloc):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs=None,
        *,
        unbacked_bindings=None,
    ) -> None:
        super().__init__(
            layout,
            tuple(tensor_args),
            tuple(nontensor_args),
            op_overload=kernel,
        )

        self.use_runtime_dispatch = False
        self.unbacked_bindings = unbacked_bindings

        assert isinstance(
            kernel,
            (
                torch._ops.OpOverload,
                torch._ops.HigherOrderOperator,
            ),
        ), f"Fails to create FallbackKernel for {kernel}: {type(kernel)} not supported"
        self.op_overload = kernel
        self.unflatten_args = unflatten_args
        self.kwargs = {} if kwargs is None else kwargs
        V.graph.warn_fallback(self.python_kernel_name)  # type: ignore[arg-type]

        # args that are aliased
        self.alias_names: list[str] = []
        # args that are mutated AND returned from the op
        self.mutation_names: list[str] = []

        if isinstance(self.op_overload, torch._ops.HigherOrderOperator):
            # We assume here that HOPs with FallbackKernel are functional.
            # This may not always be true! HOPs must individually opt-in to
            # FallbackKernel, so please check this if you opt-in.
            return

        if "_c10d_functional" in self.op_overload.name():
            # _c10d_functional kernels are lowered into _CollectiveKernel which
            # derives from FallbackKernel for the cpp codegen. The kernels
            # don't pass the can_auto_functionalize check, but their mutation
            # is handled properly by _CollectiveKernel.
            return

        schema = self.op_overload._schema

        # NOTE: [FallbackKernel supported operators]
        # We only support three types of operators:
        # - functional ops
        # - view ops
        # - inplace aten ops
        # - mutating ops that are auto-functionalizable. That is,
        # the operator may mutate any number of inputs, but its outputs
        # may not alias any of the inputs.
        #
        # The unsupported cases usually do not show up here (because
        # AOTAutograd functionalized them away); the only way for an in-place
        # op to show up here is if a lowering or pass introduced it.
        if torch._library.utils.mutates_and_returns_first_arg(self.op_overload):
            self.mutation_names.append(tensor_args[0].get_name())
            return

        if schema.is_mutable and not can_auto_functionalize(kernel):
            raise NotImplementedError(
                f"NYI: Can't generate FallbackKernel for {kernel}"
            )

        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)

        def handle_aliasing_and_mutation(info, arg) -> None:  # type: ignore[no-untyped-def]
            # Assertions to make sure we didn't mismatch args
            if isinstance(info.type, torch.ListType):
                assert isinstance(arg, (list, tuple))
            if library_utils.is_tensor_like_type(info.type):
                # PyTorch also accepts None and scalar types for args marked as "Tensor".
                # We're not going to check all of them here.
                assert not isinstance(arg, (tuple, list))

            if arg is None:
                return
            if info.alias_info is None:
                return

            def add_alias(t) -> None:  # type: ignore[no-untyped-def]
                self.alias_names.append(t.get_name())
                if info.alias_info.is_write:
                    self.mutation_outputs.append(
                        MutationOutput(NoneLayout(device=t.get_device()), t, self)
                    )

            if library_utils.is_tensorlist_like_type(info.type):
                if arg is not None:
                    for optional_tensor_arg in arg:
                        add_alias(optional_tensor_arg)
            else:
                assert library_utils.is_tensor_like_type(info.type)
                add_alias(arg)

        for info, arg in torch._library.utils.zip_schema(schema, args, kwargs):
            handle_aliasing_and_mutation(info, arg)

    def get_read_writes(self) -> dependencies.ReadWrites:
        read_writes = super().get_read_writes()

        if self.op_overload is torch._prims.rng_prims.graphsafe_run_with_rng_state:
            for arg in self.constant_args:
                if isinstance(arg, GeneratorState):
                    read_writes = read_writes.with_read(
                        dependencies.StarDep(arg.get_name())
                    )

        return read_writes

    def codegen_unbacked_symbol_defs(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        return wrapper.codegen_unbacked_symbol_defs_for_outputs(
            self.get_name(), self.outputs, getattr(self, "unbacked_bindings", None)
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            resolved = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            )
            assert resolved is not None
            return resolved.keys()  # type: ignore[return-value]
        else:
            return OrderedSet()

    def codegen_args(self):  # type: ignore[no-untyped-def]
        @dataclasses.dataclass
        class Shim:
            ref: Any

            def __repr__(self) -> str:
                return self.ref

        tensor_args = [Shim(x.codegen_reference()) for x in self.inputs]
        args, kwargs = self.unflatten_args(tensor_args, self.constant_args)
        if V.graph.cpp_wrapper and isinstance(self.op_overload, torch._ops.OpOverload):
            args = self.fill_non_provided_args(args, kwargs)
            args = [
                V.graph.wrapper_code.val_to_arg_str(x, param.real_type)
                for param, x in zip(self.op_overload._schema.arguments, args)
            ]
        else:
            args = [V.graph.wrapper_code.val_to_arg_str(x) for x in args]

        # let self.codegen_kwargs handle kwargs
        self.kwargs.update(kwargs)
        return args

    @staticmethod
    def find_device(tensor_args, example_output):  # type: ignore[no-untyped-def]
        non_torch_bind_tensor_args = (
            [t for t in tensor_args if not isinstance(t, TorchBindObject)]
            if tensor_args
            else None
        )
        if non_torch_bind_tensor_args:
            devices = [arg.get_device() for arg in tensor_args if arg.get_device()]
            return devices[0]
        if isinstance(example_output, torch.Tensor):
            return example_output.device
        if isinstance(example_output, (list, tuple)):
            device_set = OrderedSet(
                FallbackKernel.find_device(None, x) for x in example_output
            )
            # Remove None
            devices = [device for device in device_set if device]
            if len(devices) == 1:
                return devices[0]
            for device in devices:
                if is_gpu(device.type):
                    return device
            return devices[0]
        return None

    def has_side_effects(self):  # type: ignore[no-untyped-def]
        if isinstance(self.op_overload, torch._ops.HigherOrderOperator):
            return False
        return get_schema_info(self.op_overload).is_mutable()

    def get_inputs_that_alias_output(self):  # type: ignore[no-untyped-def]
        return self.alias_names

    def get_mutation_names(self):  # type: ignore[no-untyped-def]
        assert len(self.mutation_names) <= 1
        return self.mutation_names

    # ProxyExecutor Design Note
    # We export the ExternFallbackNodes (for custom ops) into a serialized file
    # and run it with a host side proxy executor to address the ABI problem
    # This is currently only implemented for fbcode. Eventually, we will also make this work for OSS.
    # Detailed design doc can be found at
    # https://docs.google.com/document/d/1wC4DOZFaYym2t1Esz0X5yxlLI3RDnSiyRbUus3bkJ64/edit?usp=sharing
    def export_extern_kernel_node(self):  # type: ignore[no-untyped-def]
        log.debug(
            "Extern kernel node added for node %s with target %s.",
            self.get_name(),
            self.op_overload,
        )

        assert isinstance(self, FallbackKernel)
        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
        args = self.fill_non_provided_args(args, kwargs)
        ordered_kwargs = [
            self.get_kwargs_value(key, **kwargs)
            for key in self.ordered_kwargs_for_cpp_kernel
        ]
        target = self.op_overload

        if not V.graph.aot_mode:
            # No need to serialize in the cpp wrapper JIT mode
            return [*args, *ordered_kwargs]

        serializer = GraphModuleSerializer(None, None)  # type: ignore[arg-type]
        named_arguments = serializer.serialize_inputs(target, args, kwargs)

        # serialize_outputs
        def handle_single_output(return_type, output):  # type: ignore[no-untyped-def]
            if isinstance(return_type, (torch.TensorType, torch.NoneType)):
                # For single Tensor or None
                out = output
                if isinstance(output, (list, tuple)):
                    assert len(output) == 1
                    out = output[0]
                if isinstance(return_type, torch.TensorType):
                    return export_schema.Argument.create(
                        as_tensor=export_schema.TensorArgument(name=out.get_name())
                    )
                else:  # NoneType
                    assert out is None
                    return export_schema.Argument.create(as_none=True)
            elif isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.TensorType
            ):
                # For single TensorList
                return export_schema.Argument.create(
                    as_tensors=[
                        export_schema.TensorArgument(name=out.get_name())
                        for out in output
                    ]
                )
            else:
                raise RuntimeError(f"Unsupported return type {type(return_type)}")

        if isinstance(target, torch._higher_order_ops.torchbind.CallTorchBind):
            returns = target.schema(args[0], args[1]).returns  # type: ignore[union-attr]
        else:
            returns = target._schema.returns  # type: ignore[union-attr]
        if len(returns) == 1:
            # NOTE: [special handling of all_reduce_coalesced_'s return value]
            # all_reduce_coalesced_ return a list of tensors via self.mutation_outputs
            outputs = self.outputs if self.outputs else self.mutation_outputs
            return_type = returns[0].real_type
            output_arguments = [handle_single_output(return_type, outputs)]
        else:
            # For tuple returns, e.g "-> (Tensor, Tensor)" or "-> (Tesnor, Tensor[])"
            # Not generating output args for self.mutation_outputs
            output_arguments = [
                handle_single_output(return_schema.real_type, output)
                for return_schema, output in zip(returns, self.outputs)
            ]

        node = ExternKernelNode(
            name=self.get_name(),
            node=export_schema.Node(
                target=self.op_overload.name(),  # type: ignore[union-attr]
                inputs=named_arguments,
                outputs=output_arguments,
                metadata={},
            ),
        )

        V.graph.extern_kernel_nodes.append(node)

        return [*args, *ordered_kwargs]

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        kernel = self.op_overload
        if kernel.namespace == "aten":  # type: ignore[union-attr]
            # Aten Fallback Ops
            assert isinstance(kernel, torch._ops.OpOverload)
            if V.graph.cpp_wrapper:
                from torchgen.aoti.fallback_ops import inductor_fallback_ops

                if str(kernel) not in inductor_fallback_ops:
                    # C shim v2 is torchgen-ed, which should cover all aten ops.
                    # If you do hit a missed op, please update fallback_ops.py.
                    log.warning(
                        "%s is missing a c-shim implementation, using proxy executor as fallback",
                        kernel,
                    )
                    self.use_runtime_dispatch = True
        elif kernel.namespace == "_quantized":  # type: ignore[union-attr]
            # Internal Quantized Fallback Ops
            assert isinstance(kernel, torch._ops.OpOverload)
        elif V.graph.cpp_wrapper:
            # For non-aten OpOverload, i.e. custom ops
            self.use_runtime_dispatch = True

        def do_runtime_dispatch() -> None:
            args = None
            exported_args = self.export_extern_kernel_node()

            wrapper.generate_fallback_kernel_with_runtime_lookup(
                self.get_name(),
                self.python_kernel_name,
                self.cpp_kernel_name,
                args,
                self.op_overload,
                exported_args,
                # NOTE: [special handling of all_reduce_coalesced_'s return value]
                self.outputs if self.outputs else self.mutation_outputs,
            )

        def is_number(t: torch.JitType) -> bool:
            return isinstance(t, torch.NumberType) or (
                isinstance(t, torch.OptionalType)
                and isinstance(t.getElementType(), torch.NumberType)
            )

        self.codegen_comment(wrapper)
        if self.use_runtime_dispatch:
            do_runtime_dispatch()
        else:
            args = [*self.codegen_args(), *self.codegen_kwargs()]
            if (
                V.graph.cpp_wrapper
                and isinstance(kernel, torch._ops.OpOverload)
                and any(
                    "c10::complex" in arg_str and is_number(op_arg.real_type)
                    for arg_str, op_arg in zip(args, kernel._schema.arguments)
                )
            ):
                # Handle the special case where a complex number is input to a
                # cpp_wrapper C-shim kernel.  If the corresponding argument is a number,
                # the torchgen-created shim API will use type "double", which cannot be
                # converted to from a c10::complex.  In these cases, fallback to runtime
                # dispatch.
                do_runtime_dispatch()
            else:
                wrapper.generate_fallback_kernel(self)
                if isinstance(self.layout, Layout):
                    self.codegen_size_asserts(wrapper)
                    self.codegen_alignment_asserts(wrapper)

        self.codegen_unbacked_symbol_defs(wrapper)

    @staticmethod
    def tensor_to_layout(output: torch.Tensor):  # type: ignore[no-untyped-def]
        return FixedLayout(
            output.device,
            output.dtype,
            convert_shape_to_inductor(output.size()),
            convert_shape_to_inductor(output.stride()),
        )

    @classmethod
    def create(cls, kernel, *args, **kwargs):  # type: ignore[no-untyped-def]
        fake_incorrect_kernels = (aten._fused_moving_avg_obs_fq_helper_functional,)
        context: AbstractContextManager[None] = (
            V.graph.fake_mode if kernel not in fake_incorrect_kernels else nullcontext()  # type: ignore[assignment]
        )
        with context:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, *args, **kwargs)

        # We need this extra check for input alignment since the example
        # inputs we created are always aligned.
        has_unaligned_input = any(is_unaligned(arg) for arg in tensor_args)

        device = cls.find_device(tensor_args, example_output)

        if not device and isinstance(
            kernel, torch._higher_order_ops.torchbind.CallTorchBind
        ):
            # use CPU device for torchbind methods that don't take in or output any tensor, e.g. size()
            device = torch.device("cpu")

        if example_output is None:
            packed = cls(
                NoneLayout(device=device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings=unbacked_bindings,
            )

        else:
            assert device, "Not sure where to find device info"
            packed = cls(
                MultiOutputLayout(device=device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings=unbacked_bindings,
            )

        def generate_output(output, indices):  # type: ignore[no-untyped-def]
            if isinstance(output, (list, tuple)):
                return type(output)(
                    generate_output(output[i], indices + [(type(output), i)])
                    for i in range(len(output))
                )
            elif isinstance(output, dict):
                return {
                    key: generate_output(val, indices + [(type(output), key)])
                    for key, val in output.items()
                }
            elif isinstance(output, torch.Tensor):
                buf = MultiOutput(
                    cls.tensor_to_layout(output),
                    packed,
                    indices,
                )
                if (
                    config.assume_unaligned_fallback_output
                    or has_unaligned_input
                    or not tensor_is_aligned(output)
                ):
                    V.graph.unaligned_buffers.add(buf.name)  # type: ignore[arg-type]
                return buf
            elif isinstance(output, int):
                return output
            elif isinstance(output, torch.SymInt):
                return output.node.expr
            else:
                assert output is None, (
                    f"FallbackKernel output type {type(output)} is not supported"
                )
                return None

        outputs = generate_output(example_output, [])
        if isinstance(outputs, (list, tuple, dict)):
            packed.outputs = outputs  # type: ignore[assignment]
        else:
            packed.outputs = [outputs]
        return outputs

    def apply_constraint(self):  # type: ignore[no-untyped-def]
        return super().apply_constraint()


@ir_dataclass(frozen=False)
class ComplexView(FallbackKernel):
    """View a complex number as two dtyped numbers or vice versa"""

    def should_allocate(self) -> bool:
        return False

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        # Signal to codegen that our output buffer isn't safe to reuse
        return [self.inputs[0].get_name()]

    def __init__(  # type: ignore[no-untyped-def]
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        *,
        unbacked_bindings=None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            unbacked_bindings=unbacked_bindings,
        )


@ir_dataclass
class MultiOutputLayout(OutputSpec):
    device: torch.device

    def get_device(self) -> Optional[torch.device]:
        return self.device


class MultiOutput(ExternKernel):
    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        wrapper.codegen_multi_output(self)
        if not self.skip_size_stride_alignment_checks:
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)

    def __init__(  # type: ignore[no-untyped-def]
        self,
        layout: OutputSpec,
        input,
        indices: list[tuple[Any, ...]],
        skip_size_stride_alignment_checks=False,
    ) -> None:
        super().__init__(None, layout, [input], ())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        self.indices = indices
        self.skip_size_stride_alignment_checks = skip_size_stride_alignment_checks

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return self.inputs[0].get_free_symbol_uses(unbacked_only)

    def should_allocate(self) -> bool:
        if len(self.inputs) == 1 and (
            isinstance(self.inputs[0], CppTemplateBuffer)  # Grouped GEMM
        ):
            return True
        return False

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        return [
            inp.get_name()
            for inp in self.inputs
            if isinstance(inp, FallbackKernel)
            and len(inp.get_inputs_that_alias_output()) > 0
        ]


# We just use a normal dataclass for MutableBox/TensorBox/StorageBox since
# they're mainly lowering-time constructs that we expect to mutate and such.
@dataclasses.dataclass
class MutableBox(IRNode):
    """
    TensorBox / StorageBox allow in-place mutation of Tensors
    """

    data: IRNode

    def has_exceeded_max_reads(self) -> bool:
        return self.data.has_exceeded_max_reads()

    def get_device(self) -> Optional[torch.device]:
        return self.data.get_device()

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        return self.data.make_loader()

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        return self.data.make_indexer()

    def get_stride(self) -> Sequence[_IntLike]:
        return self.data.get_stride()

    def get_name(self) -> str:
        return self.data.get_name()

    def has_large_inner_fn(self, threshold: Optional[int] = None) -> bool:
        return self.data.has_large_inner_fn(threshold)

    def mark_reuse(self, users: int) -> None:
        return self.data.mark_reuse(users)

    def realize_hint(self) -> None:
        return self.data.realize_hint()

    def unwrap_view(self) -> IRNode:
        return self.data.unwrap_view()

    def freeze_layout(self) -> None:
        return self.data.freeze_layout()

    def freeze_layout_with_stride_order(
        self, order: list[int], allow_padding: bool = False
    ) -> None:
        return self.data.freeze_layout_with_stride_order(order, allow_padding)

    def freeze_layout_with_fill_order(self, order: list[int]) -> None:
        return self.data.freeze_layout_with_fill_order(order)

    def freeze_layout_with_same_order(self, stride: list[_IntLike]) -> None:
        return self.data.freeze_layout_with_same_order(stride)

    def freeze_layout_with_exact_strides(
        self, exact_strides: list[_IntLike], allow_padding: bool = False
    ) -> None:
        return self.data.freeze_layout_with_exact_strides(exact_strides, allow_padding)

    def get_read_writes(self) -> dependencies.ReadWrites:
        return self.data.get_read_writes()

    def get_reads(self) -> OrderedSet[Dep]:
        return self.data.get_reads()

    def num_reads(self) -> int:
        return self.data.num_reads()

    def get_storage_numel(self) -> _IntLike:
        return self.data.get_storage_numel()

    def get_reduction_type(self) -> Optional[str]:
        return self.data.get_reduction_type()

    def get_reduction_size(self) -> Sequence[sympy.Expr]:
        return self.data.get_reduction_size()

    def is_extern(self) -> bool:
        return self.data.is_extern()

    def is_no_op(self) -> bool:
        return self.data.is_no_op()

    def constant_to_device(self, device: torch.device) -> IRNode:
        return self.data.constant_to_device(device)

    def get_mutation_names(self) -> Sequence[str]:
        return self.data.get_mutation_names()

    def get_operation_name(self) -> str:
        return self.data.get_operation_name()

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        return self.data.get_inputs_that_alias_output()

    def realize(self) -> Optional[str]:
        return self.data.realize()

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return self.data.get_free_symbol_uses(unbacked_only)

    def get_read_names(self) -> OrderedSet[str]:
        return self.data.get_read_names()

    def get_defining_op(self) -> Optional[Operation]:
        return self.data.get_defining_op()

    def codegen_reference(self, writer: Optional[IndentedBuffer] = None) -> str:
        return self.data.codegen_reference(writer)

    @property
    def layout(self) -> OutputSpec:
        # we intentionally call get_output_spec (rather than get_layout) since Buffer.layout is an OutputSpec
        return self.data.get_output_spec()

    def get_layout(self) -> Layout:
        return self.data.get_layout()

    def get_output_spec(self) -> OutputSpec:
        return self.data.get_output_spec()

    def get_size(self) -> Sequence[Expr]:
        return self.data.get_size()

    @property
    def dtype(self):  # type: ignore[no-untyped-def]
        return self.data.dtype

    def __str__(self) -> str:
        if isinstance(self.data, MutableBox):
            line0 = f"{type(self).__name__}({type(self.data).__name__}("
            endl = "))"
            inner = self.data.data
        else:
            line0 = f"{type(self).__name__}("
            inner = self.data
            endl = ")"

        lines = [
            line0,
            indent(str(inner)),
            endl,
        ]
        return "\n".join(lines)

    __repr__ = __str__


class TensorBox(MutableBox):
    @staticmethod
    def create(data):  # type: ignore[no-untyped-def]
        if isinstance(data, ShapeAsConstantBuffer):
            return data
        return TensorBox(StorageBox(data))


class StorageBox(MutableBox):
    def is_input_buffer(self):  # type: ignore[no-untyped-def]
        if isinstance(self.data, (InputBuffer, ReinterpretView)):
            return self.data.get_name() in V.graph.graph_inputs
        return False

    def is_module_buffer(self):  # type: ignore[no-untyped-def]
        return (
            isinstance(self.data, (ConstantBuffer))
            and self.data.get_name() in V.graph.constants
        )

    def realize(self) -> Optional[str]:
        if isinstance(
            self.data,
            (
                ComputedBuffer,
                InputsKernel,
                InputBuffer,
                ReinterpretView,
                TemplateBuffer,
            ),
        ):
            return self.data.get_name()
        assert isinstance(self.data, (Pointwise, Reduction, Scan, Sort)), type(
            self.data
        )
        origin_node = self.data.get_origin_node()
        traceback = self.data.get_traceback()
        self.data = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=self.data.get_device(),
                dtype=self.data.get_dtype(),
                size=self.data.get_size(),
            ),
            data=self.data,
        )
        self.data.name = V.graph.register_buffer(self.data)
        V.graph.register_operation(self.data)
        self.data.origins = self.origins
        self.data.origin_node = origin_node
        self.data.traceback = traceback
        return self.data.name

    def realize_hint(self) -> None:
        """
        Called on buffers we expect to be forced to realize later.
        """
        if (
            isinstance(self.data, (Pointwise, Reduction))
            and self.data.inner_fn_opcount().nontrivial_read_count > 1
        ):
            self.realize()

    def has_exceeded_max_reads(self) -> bool:
        return isinstance(self.data, Pointwise) and (
            self.num_reads() > config.realize_acc_reads_threshold
            or self.has_large_inner_fn()
        )

    def should_realize_on_reuse(self, users):  # type: ignore[no-untyped-def]
        """
        A heuristic to decide if we should realize a tensor
        that is used multiple times.
        """
        if users > 1 and isinstance(self.data, (Pointwise, Reduction)):
            if is_cpu(self.data):
                # Heuristic for realizing reused result of heavy ops on cpu
                opcount = self.data.inner_fn_opcount()
                heavy_ops = ["exp", "sigmoid"]  # a list of heavy ops
                if any(x in opcount.used_ops for x in heavy_ops):
                    return True
            return (
                self.num_reads() > config.realize_reads_threshold
                or self.has_large_inner_fn()
            )
        return False

    def mark_reuse(self, users: int) -> None:
        if self.should_realize_on_reuse(users):
            self.realize()

    def num_reads(self):  # type: ignore[no-untyped-def]
        return self.data.num_reads()


@ir_dataclass(frozen=False)
class Subgraph(IRNode):
    name: str
    graph_module: torch.fx.GraphModule
    graph: Optional[GraphLowering] = None


def _has_aliased_buffers(buffers: Sequence[IRNode]) -> bool:
    buffers = [
        buffer.unwrap_view() if isinstance(buffer, ReinterpretView) else buffer
        for buffer in buffers
    ]
    # assuming the same buffer is represented by the same IRNode object
    return len(OrderedSet(id(buffer) for buffer in buffers)) < len(buffers)


@ir_dataclass(frozen=False)
class InvokeSubgraph(ExternKernel):
    subgraph: Optional[Subgraph] = None
    operands: Optional[list[TensorBox]] = None
    outputs: Optional[list[MultiOutput]] = None

    def __init__(
        self, subgraph: Subgraph, operands: list[TensorBox], layout: MultiOutputLayout
    ) -> None:
        super().__init__(
            name=None,
            layout=layout,
            inputs=operands,
        )
        self.subgraph = subgraph
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(cls, subgraph: Subgraph, *operands):  # type: ignore[no-untyped-def]
        # TODO(anijain2305) - Support sym expr as operands in future.
        fx_operands = V.graph.current_node.args[2:]
        fake_operands = [x.meta["val"] for x in fx_operands]  # type: ignore[union-attr]

        # Realize the inputs. Also intermediates can have different strides than
        # the inputs of the subgraph. So, force the intermediates to have same
        # strides as that of subgraph inputs.
        operands = [cls.realize_input(x) for x in operands]

        def handle_sym_expr(stride):  # type: ignore[no-untyped-def]
            return [s.node.expr if isinstance(s, torch.SymInt) else s for s in stride]

        new_operands = []
        for idx, operand in enumerate(operands):
            if isinstance(operand, ShapeAsConstantBuffer):
                new_operands.append(operand)
            else:
                example_stride = handle_sym_expr(fake_operands[idx].stride())
                new_operands.append(cls.require_exact_strides(operand, example_stride))

        operands = new_operands

        if subgraph.graph is None:
            # create and lower subgraphs
            subgraph.graph = V.graph.make_subgraph(
                gm=subgraph.graph_module,
                example_inputs=fake_operands,
                subgraph_name=subgraph.name,
            )
            with V.set_graph_handler(subgraph.graph):
                subgraph.graph.run(*fake_operands)

        outputs = subgraph.graph.graph_outputs

        # Find the device - operands could be integers from shapes, so we can't
        # use operands[0]
        device = None
        for operand in operands:
            if not isinstance(operand, ShapeAsConstantBuffer):
                device = operand.get_device()
                break
        assert device is not None

        invoke_subgraph = InvokeSubgraph(
            subgraph=subgraph,
            operands=operands,
            layout=MultiOutputLayout(device=device),
        )

        def create_output(output: IRNode, ind: int):
            if isinstance(output, (ShapeAsConstantBuffer, NoneAsConstantBuffer)):
                return output
            else:
                return MultiOutput(
                    FixedLayout(
                        device=output.get_device(),
                        dtype=output.get_dtype(),
                        size=output.get_size(),  # type: ignore[arg-type]
                        stride=output.get_stride(),
                        offset=output.get_layout().offset,
                    ),
                    invoke_subgraph,
                    [(list, ind)],
                    skip_size_stride_alignment_checks=True,
                )

        outputs = [create_output(output, i) for i, output in enumerate(outputs)]
        invoke_subgraph.outputs = outputs
        return outputs

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        wrapper.codegen_invoke_subgraph(self)


@ir_dataclass(frozen=False)
class Conditional(ExternKernel):
    predicate: Optional[IRNode] = None
    operands: Optional[list[Union[TensorBox, ShapeAsConstantBuffer]]] = None
    true_subgraph: Optional[Subgraph] = None
    false_subgraph: Optional[Subgraph] = None
    outputs: Optional[list[MultiOutput]] = None

    def __init__(
        self,
        predicate: IRNode,
        operands: list[Union[TensorBox, ShapeAsConstantBuffer]],
        true_subgraph: Subgraph,
        false_subgraph: Subgraph,
        layout: MultiOutputLayout,
        unbacked_bindings: Optional[dict[sympy.Symbol, pytree.KeyPath]],
    ) -> None:
        self.predicate = predicate
        self.operands = operands
        self.true_subgraph = true_subgraph
        self.false_subgraph = false_subgraph

        sym_args, tensor_args = _split_by_sym_type([predicate] + operands)

        super().__init__(
            name=None,
            layout=layout,
            inputs=tensor_args,
            constant_args=sym_args,
        )
        if unbacked_bindings is not None:
            self.unbacked_bindings = unbacked_bindings

        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(  # type: ignore[no-untyped-def]
        cls,
        predicate: TensorBox,
        true_fn: Subgraph,
        false_fn: Subgraph,
        operands: list[Union[TensorBox, ShapeAsConstantBuffer]],
    ):
        predicate = cls.realize_input(predicate)
        operands = [cls.realize_input(x) for x in operands]
        fx_operands = V.graph.current_node.args[-1]
        fake_operands = [x.meta["val"] for x in fx_operands]  # type: ignore[union-attr]

        for subgraph in (true_fn, false_fn):
            if subgraph.graph is None:
                # create and lower subgraphs
                subgraph.graph = V.graph.make_subgraph(
                    gm=subgraph.graph_module,
                    example_inputs=fake_operands,
                    subgraph_name=subgraph.name,
                )
                with V.set_graph_handler(subgraph.graph):
                    subgraph.graph.run(*fake_operands)

        true_outputs = true_fn.graph.graph_outputs  # type: ignore[union-attr]
        false_outputs = false_fn.graph.graph_outputs  # type: ignore[union-attr]

        for name, outputs in (("true_fn", true_outputs), ("false_fn", false_outputs)):
            if _has_aliased_buffers(true_outputs):
                raise AssertionError(
                    "Output aliasing is currently not supported in compiled torch.cond. "
                    f"The outputs of the {name} subgraph of torch.cond are aliased: {outputs}"
                )

        # make sure true and false outputs are structurally equivalent
        assert len(true_outputs) == len(false_outputs), (true_outputs, false_outputs)
        for i, (to, fo) in enumerate(zip(true_outputs, false_outputs)):
            assert to.get_device() == fo.get_device(), (i, to, fo)
            assert to.get_dtype() == fo.get_dtype(), (i, to, fo)
            assert to.get_layout().offset == fo.get_layout().offset, (i, to, fo)

        device = next(
            o.get_device()
            for o in [predicate] + operands
            if not isinstance(o, ShapeAsConstantBuffer)
        )
        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env,
            V.graph.current_node.meta.get("unbacked_bindings", None),
        )
        assert device is not None, "cannot determine device"
        conditional = Conditional(
            predicate=predicate,
            operands=operands,
            true_subgraph=true_fn,
            false_subgraph=false_fn,
            layout=MultiOutputLayout(device=device),
            unbacked_bindings=unbacked_bindings,
        )

        def _maybe_expr(s: Union[int, torch.SymInt]) -> Union[int, sympy.expr]:
            if isinstance(s, int):
                return s
            return s.node.expr

        outputs = [
            MultiOutput(
                FixedLayout(
                    device=output.get_device(),
                    dtype=output.get_dtype(),
                    size=[_maybe_expr(sz) for sz in merged_output.size()],
                    stride=[_maybe_expr(sz) for sz in merged_output.stride()],
                    offset=output.get_layout().offset,
                ),
                conditional,
                [(list, i)],
            )
            # as the true and false outputs are equivalent,
            # we can use either of them here as a "template"
            for i, (output, merged_output) in enumerate(
                zip(true_outputs, V.graph.current_node.meta["val"])
            )
        ]

        conditional.outputs = outputs  # type: ignore[assignment]
        return outputs

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        wrapper.codegen_conditional(self)
        wrapper.codegen_unbacked_symbol_defs_for_outputs(
            self.get_name(), self.outputs, getattr(self, "unbacked_bindings", {})
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            resolved = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            )
            assert resolved is not None
            return resolved.keys()  # type: ignore[return-value]
        else:
            return OrderedSet()


def _split_by_sym_type(
    args: list[Any],
) -> tuple[list[ShapeAsConstantBuffer], list[Any]]:
    non_sym_args = []
    sym_args = []
    for arg in args:
        if isinstance(arg, ShapeAsConstantBuffer):
            sym_args.append(arg.expr)
        else:
            non_sym_args.append(arg)

    return sym_args, non_sym_args


@ir_dataclass(frozen=False)
class WhileLoop(ExternKernel):
    carried_inputs: Optional[list[Union[TensorBox, ShapeAsConstantBuffer]]] = None
    additional_inputs: Optional[list[Union[TensorBox, ShapeAsConstantBuffer]]] = None
    cond_subgraph: Optional[Subgraph] = None
    body_subgraph: Optional[Subgraph] = None
    outputs: Optional[list[MultiOutput]] = None

    def __init__(
        self,
        carried_inputs: list[Union[TensorBox, ShapeAsConstantBuffer]],
        additional_inputs: list[Union[TensorBox, ShapeAsConstantBuffer]],
        cond_subgraph: Subgraph,
        body_subgraph: Subgraph,
        layout: MultiOutputLayout,
    ) -> None:
        self.carried_inputs = carried_inputs
        self.additional_inputs = additional_inputs
        self.cond_subgraph = cond_subgraph
        self.body_subgraph = body_subgraph

        sym_args, tensor_args = _split_by_sym_type(carried_inputs + additional_inputs)
        super().__init__(
            name=None,
            layout=layout,
            inputs=tensor_args,
            constant_args=sym_args,
        )

        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(  # type: ignore[no-untyped-def]
        cls,
        cond_fn: Subgraph,
        body_fn: Subgraph,
        carried_inputs: list[Union[TensorBox, ShapeAsConstantBuffer]],
        additional_inputs: list[Union[TensorBox, ShapeAsConstantBuffer]],
    ):
        from torch._higher_order_ops.utils import check_input_alias_and_mutation

        def _require_exact_strides(
            tensor_boxes: list[TensorBox | ShapeAsConstantBuffer],
            fake_tensors: list[Union[int, torch.SymInt, torch.Tensor]],
        ) -> list[TensorBox | ShapeAsConstantBuffer]:
            assert len(tensor_boxes) == len(fake_tensors)
            ret = []
            for tb, fk in zip(tensor_boxes, fake_tensors):
                if isinstance(fk, torch.Tensor):
                    ret.append(
                        ExternKernel.require_exact_strides(
                            tb, fk.stride(), allow_padding=False
                        )
                    )
                else:
                    ret.append(tb)
            return ret

        fx_carried_inputs = V.graph.current_node.args[-2]
        fx_additional_inputs = V.graph.current_node.args[-1]
        fx_all_inputs = fx_carried_inputs + fx_additional_inputs  # type: ignore[operator]
        fake_all_inputs = [x.meta["val"] for x in fx_all_inputs]  # type: ignore[union-attr]
        fake_carried_inputs = [x.meta["val"] for x in fx_carried_inputs]  # type: ignore[union-attr]
        fake_additional_inputs = [x.meta["val"] for x in fx_additional_inputs]  # type: ignore[union-attr]

        carried_inputs = [cls.realize_input(x) for x in carried_inputs]
        carried_inputs = _require_exact_strides(carried_inputs, fake_carried_inputs)
        additional_inputs = [cls.realize_input(x) for x in additional_inputs]
        additional_inputs = _require_exact_strides(
            additional_inputs, fake_additional_inputs
        )
        all_inputs = carried_inputs + additional_inputs

        for subgraph in (cond_fn, body_fn):
            if subgraph.graph is None:
                # create and lower subgraphs
                subgraph.graph = V.graph.make_subgraph(
                    gm=subgraph.graph_module,
                    example_inputs=fx_all_inputs,  # type: ignore[arg-type]
                    subgraph_name=subgraph.name,
                )
                with V.set_graph_handler(subgraph.graph):
                    subgraph.graph.run(*fake_all_inputs)
                    # For body_fn, we require its output to have the exact same stride
                    # as inputs because the previous output is the input of next iteration.
                    #
                    # This cannot be automatically done in graph lowering because body_fn's graph outputs
                    # are not user-facing so the special handling for strides of user-facing output in graph
                    # lowering is not applicable.
                    if subgraph is body_fn:
                        assert len(subgraph.graph.graph_outputs) == len(
                            fake_carried_inputs
                        )
                        subgraph.graph.graph_outputs = _require_exact_strides(  # type: ignore[assignment]
                            subgraph.graph.graph_outputs,  # type: ignore[arg-type]
                            fake_carried_inputs,
                        )

        cond_outputs = cond_fn.graph.graph_outputs  # type: ignore[union-attr]
        body_outputs = body_fn.graph.graph_outputs  # type: ignore[union-attr]

        if _has_aliased_buffers(body_outputs):
            raise AssertionError(
                "Output aliasing is currently not supported in compiled torch.while_loop. "
                f"The outputs of the body_fn subgraph of torch.while_loop are aliased: {body_outputs}"
            )

        # make sure cond_fn returns a boolean scalar Tensor
        assert len(cond_outputs) == 1, cond_outputs
        p = cond_outputs[0]
        if not isinstance(p, ShapeAsConstantBuffer):
            assert p.get_dtype() == torch.bool, p
            assert len(p.get_size()) == 0, p

        assert len(all_inputs) > 0, (
            "torch.while_loop is assumed to have at least one operand."
        )

        device = all_inputs[0].get_device()

        assert device is not None  # to make linter happy
        # make sure carried_inputs and body outputs are structurally equivalent
        assert len(carried_inputs) == len(body_outputs), (carried_inputs, body_outputs)
        for i, (op, bo) in enumerate(zip(carried_inputs, body_outputs)):

            def _guard_list_equals(
                lhs_exprs: Sequence[Union[int, Any]],
                rhs_exprs: Sequence[Union[int, Any]],
            ) -> None:
                for lhs, rhs in zip(lhs_exprs, rhs_exprs):
                    V.graph.sizevars.guard_equals(lhs, rhs)

            _guard_list_equals(op.get_size(), bo.get_size())
            _guard_list_equals(op.get_stride(), bo.get_stride())
            # assume all carried_inputs and outputs are on the same device
            # as the MultiOutputLayout below requires single device
            assert op.get_device() == bo.get_device(), (i, op, bo, device)
            assert op.get_dtype() == bo.get_dtype(), (i, op, bo)
            assert op.get_layout().offset == bo.get_layout().offset, (i, op, bo)

        while_loop = WhileLoop(
            carried_inputs=carried_inputs,
            additional_inputs=additional_inputs,
            cond_subgraph=cond_fn,
            body_subgraph=body_fn,
            # asserted above that there is at least one operand
            layout=MultiOutputLayout(device=device),
        )

        assert body_fn.graph is not None and isinstance(
            body_fn.graph.module, torch.fx.GraphModule
        )  # to make linter happy

        # Handling input mutations
        mutated_idxs = check_input_alias_and_mutation(
            body_fn.graph.module, fake_all_inputs
        )[0]
        mutated_idx_set = OrderedSet(mutated_idxs)
        mutated_inputs = [all_inputs[idx] for idx in mutated_idx_set]
        real_outputs = {
            idx: out
            for idx, out in enumerate(body_outputs)
            if idx not in mutated_idx_set
        }
        real_outputs = [
            MultiOutput(
                FixedLayout(
                    device=output.get_device(),
                    dtype=output.get_dtype(),
                    size=output.get_size(),
                    stride=output.get_stride(),
                    offset=output.get_layout().offset,
                ),
                while_loop,
                [(list, idx)],
            )
            for idx, output in real_outputs.items()
        ]
        while_loop.outputs = real_outputs
        while_loop.mutation_outputs = [
            MutationOutput(inp.layout, inp, while_loop)  # type: ignore[union-attr]
            for inp in mutated_inputs
        ]

        outputs_iter = iter(real_outputs)
        mutated_inputs_iter = iter(mutated_inputs)
        all_outputs = [
            next(mutated_inputs_iter) if idx in mutated_idx_set else next(outputs_iter)
            for idx in range(len(body_outputs))
        ]
        for inp, out in zip(carried_inputs, all_outputs):
            if inp.get_name() in V.graph.graph_inputs:
                # if a carried input of the while_loop is a graph input,
                # it can be returned as is when the number of iterations
                # is zero. due to this, we can't (generally) reuse the
                # output buffers corresponding to the graph inputs, as
                # the inputs may end up being mutated.
                V.graph.never_reuse_buffers.add(out.get_name())
        return all_outputs

    def codegen(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        wrapper.codegen_while_loop(self)


class EffectfulKernel(FallbackKernel):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs=None,
        *,
        unbacked_bindings=None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )

        from torch._higher_order_ops.effects import get_effect_key

        uncovered_args = [
            a.value if isinstance(a, TorchBindObject) else a for a in tensor_args
        ]
        effect_type = get_effect_key(kernel, (*nontensor_args, *uncovered_args), kwargs)
        assert effect_type is not None
        self.effect_type = effect_type
        self.prev_effect_buffer = V.graph.effectful_ops.get(effect_type, None)
        V.graph.effectful_ops[effect_type] = self

    def get_read_writes(self) -> dependencies.ReadWrites:
        read_writes = super().get_read_writes()

        if self.prev_effect_buffer is not None:
            read_writes.reads.add(
                dependencies.StarDep(self.prev_effect_buffer.get_name())
            )

        return read_writes

    def has_side_effects(self) -> bool:
        return True


class NonTensorObj(IRNode):
    pass


@ir_dataclass
class TorchBindObject(NonTensorObj):
    name: str
    value: Union[FakeScriptObject, torch.ScriptObject]

    def get_name(self):  # type: ignore[no-untyped-def]
        return self.name

    def codegen_reference(self, writer: Optional[IndentedBuffer] = None) -> str:
        return self.name

    def get_value(self) -> Union[FakeScriptObject, torch.ScriptObject]:
        return self.value

    def get_real_obj(self) -> torch.ScriptObject:
        if isinstance(self.value, torch.ScriptObject):
            return self.value
        else:
            return self.value.real_obj

    def get_buf_bytes(self) -> int:
        # Returns the sum of all tensors in the flattened object
        real_script_obj = self.get_real_obj()
        flat_dict = dict(real_script_obj.__obj_flatten__())  # type: ignore[attr-defined]
        flat_elems = pytree.tree_flatten(flat_dict)[0]
        flat_sizes = [
            x.element_size() * x.numel()
            for x in flat_elems
            if isinstance(x, torch.Tensor)
        ]
        return functools.reduce(lambda x, y: x + y, flat_sizes, 0)


@ir_dataclass
class GeneratorState(NonTensorObj):
    name: str
    device: torch.device

    def get_name(self):  # type: ignore[no-untyped-def]
        return self.name

    def codegen_reference(self, writer: Optional[IndentedBuffer] = None) -> str:
        return self.name


class _CollectiveKernel(FallbackKernel):
    def should_allocate(self) -> bool:
        return False

    def has_side_effects(self) -> bool:
        return True

    # This is identical to FallbackKernel.set_cpp_kernel(), minus the
    # part that checks against input aliasing and mutation.
    def set_cpp_kernel_name(self, cpp_kernel_name: Optional[str] = None) -> None:
        assert type(self.op_overload) is torch._ops.OpOverload, (
            "Setting cpp kernel needs a valid op_overload"
        )
        kernel = self.op_overload
        self.cpp_kernel_name = kernel._schema.name

        self.ordered_kwargs_for_cpp_kernel = [
            x.name for x in kernel._schema.arguments if x.kwarg_only
        ]

    # NOTE: [In-Place Collective Safety]
    # Between the initiation and completion of an in-place collective, the
    # input buffers are subject to both volatile reads and volatile writes.
    # They must not be read, written to or reused by another kernel. To ensure
    # the constraints, we model collective -> wait_tensor as as two-step
    # mutation of the input buffers.
    @classmethod
    def create_inplace(  # type: ignore[no-untyped-def]
        cls, kernel, inputs: Union[TensorBox, list[TensorBox]], *args, **kwargs
    ) -> None:
        with V.graph.fake_mode:
            (
                _example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, inputs, *args, **kwargs)
        assert not unbacked_bindings, f"{kernel} {unbacked_bindings}"
        for tensor_arg in tensor_args:
            tensor_arg.realize()

        device = tensor_args[0].get_device()
        packed = cls(
            NoneLayout(device=device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )

        inps = pytree.tree_leaves(inputs)
        packed.mutation_outputs.extend(
            [MutationOutput(NoneLayout(device=device), buf, packed) for buf in inps]
        )

        # For inplace collective ops, the input is guaranteed to be alias of the returned value of op.
        packed.alias_names.extend([inp.get_name() for inp in inps])
        if "out" in kwargs:
            packed.mutation_outputs.append(
                MutationOutput(NoneLayout(device=device), kwargs["out"], packed)
            )
            # For out-variant collective ops, the `out=` arg is guaranteed to be alias of the returned value of op.
            packed.alias_names.append(kwargs["out"].get_name())

    # NOTE: [Out-of-Place Collective Safety]
    # Between the initiation and completion of an out-of-place collective:
    #
    # Input buffers:
    # - Are subject to volatile reads
    # - Can be read by another kernel
    # - Must not be written to or reused by another kernel
    #
    # Output buffers:
    # - Are subject to volatile writes
    # - Must not be read, written to or reused by another kernel
    #
    # To ensure the safety of input buffers without sacrificing read
    # availability, we add input buffers as read deps of wait_tensor kernels.
    #
    # To ensure the safety of output buffers, we model wait_tensor as a
    # mutation to the output buffer. Note we also assumes the user program being
    # correct and the output buffer is not consumed by kernels other than
    # wait_tensor.
    #
    # TODO(yifu): add a pre-grad pass to validate the correctness of collective
    # usage in the user program.
    @classmethod
    def create_out_of_place(  # type: ignore[no-untyped-def]
        cls, kernel, inputs: Union[TensorBox, list[TensorBox]], *args, **kwargs
    ):
        with V.graph.fake_mode:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, inputs, *args, **kwargs)
        assert not unbacked_bindings, f"{kernel}, {unbacked_bindings}"
        for tensor_arg in tensor_args:
            tensor_arg.realize()

        if isinstance(example_output, list):
            device = cls.find_device(tensor_args, example_output)
            packed = cls(
                MultiOutputLayout(device=device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            packed.outputs = [
                MultiOutput(
                    cls.tensor_to_layout(tensor),
                    packed,
                    [(list, i)],
                )
                for i, tensor in enumerate(example_output)
            ]
            for buf, tensor in zip(packed.outputs, example_output):
                if config.assume_unaligned_fallback_output or not tensor_is_aligned(
                    tensor
                ):
                    V.graph.unaligned_buffers.add(buf.name)  # type: ignore[arg-type]
            return packed.outputs
        else:
            packed = cls(
                cls.tensor_to_layout(example_output),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            if config.assume_unaligned_fallback_output or not tensor_is_aligned(
                example_output
            ):
                V.graph.unaligned_buffers.add(packed.name)  # type: ignore[arg-type]
            packed.outputs = [packed]
            return packed


class _WaitKernel(_CollectiveKernel):
    def get_volatile_reads(self):  # type: ignore[no-untyped-def]
        inp = self.inputs[0]
        if isinstance(inp, _CollectiveKernel):
            # Out-of-place single-output
            return [inp.inputs[0]]
        elif isinstance(inp, MultiOutput):
            # This can be two things:
            # 1. Out-of-place multi-output coll
            # 2. In-place coll with inputs coming from another MultiOutput
            coll = inp.inputs[0]
            # Case 1
            if isinstance(coll, _CollectiveKernel):
                _, idx = inp.indices[0]
                return [coll.inputs[idx]]
            # Case 2
            return []
        else:
            # In-place requires no additional deps handling for volatile
            # reads since the inputs are mutated.
            return []

    @classmethod
    def create_wait(cls, kernel, inp: TensorBox) -> None:  # type: ignore[no-untyped-def]
        with V.graph.fake_mode:
            (
                _example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, inp)
        assert not unbacked_bindings, f"{kernel} {unbacked_bindings}"
        packed = cls(
            NoneLayout(device=inp.get_device()),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )
        packed.mutation_outputs.append(
            MutationOutput(NoneLayout(device=inp.get_device()), inp, packed)
        )

    def get_read_writes(self) -> dependencies.ReadWrites:
        read_writes = super().get_read_writes()
        # See [Out-of-Place Collective Safety].
        volatile_reads = self.get_volatile_reads()
        for vr in volatile_reads:
            read_writes.reads.add(dependencies.StarDep(vr.get_name()))
        return read_writes


# NB: recursive structure here reflects val_to_arg_str, avoid
# calling free_unbacked_symbols on "exotic" types that don't get pexpr
# treatment
def maybe_free_unbacked_symbols(s: object) -> OrderedSet[Symbol]:
    if isinstance(s, (SymTypes, Expr)):
        # This branch should be impossible in return position
        return free_unbacked_symbols(s)
    elif isinstance(s, (tuple, list)):
        r = OrderedSet[sympy.Symbol]()
        for t in s:
            r |= maybe_free_unbacked_symbols(t)
        return r
    elif isinstance(s, torch.Tensor):
        # This branch is impossible in constant-args position
        return free_unbacked_symbols(s)
    else:
        return OrderedSet()


def maybe_free_symbols(s: object) -> OrderedSet[Symbol]:
    if isinstance(s, (SymTypes, Expr)):
        # This branch should be impossible in return position
        return free_symbols(s)
    elif isinstance(s, (tuple, list)):
        r = OrderedSet[sympy.Symbol]()
        for t in s:
            r |= maybe_free_symbols(t)
        return r
    elif isinstance(s, torch.Tensor):
        # This branch is impossible in constant-args position
        return free_symbols(s)
    else:
        return OrderedSet()
