from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import itertools
import logging
import operator
import textwrap
import traceback
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from enum import Enum
from functools import partial
from typing import (
    Any,
    cast,
    ClassVar,
    Literal,
    overload,
    SupportsFloat,
    SupportsInt,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    Union,
)
from typing_extensions import assert_never, Never, override, ParamSpec, Self, TypeIs
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
from torch._inductor.utils import get_free_symbols
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import get_opaque_obj_repr, is_opaque_value
from torch._prims_common import (
    compute_required_storage_length,
    is_boolean_dtype,
    is_contiguous_for_memory_format_or_false,
    is_float_dtype,
    make_channels_last_strides_for,
    StrideType,
)
from torch.fx.experimental.symbolic_shapes import (
    _remove_effect_token_unbacked_bindings,
    compute_unbacked_bindings,
    free_symbols,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    has_free_unbacked_symbols,
    IterateExprs,
    rebind_unbacked,
    resolve_unbacked_bindings,
    ShapeEnv,
    SymTypes,
)
from torch.fx.node import Node
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, Mod, ModularIndexing
from torch.utils._sympy.symbol import SymT

from . import config, dependencies
from .codegen.common import (
    BackendFeature,
    CodegenSymbol,
    get_scheduling_for_device,
    index_prevent_reordering,
    Kernel,
)
from .dependencies import (
    Dep,
    extract_free_symbols,
    extract_input_node_reduction_ranges,
    extract_read_writes,
    SymbolUsageCollectorOpsHandler,
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
    cache_on_self_and_args,
    ceildiv,
    convert_shape_to_inductor,
    convert_shape_to_symint,
    developer_warning,
    do_bench_using_profiling,
    dtype_from_size,
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
    from torch.fx.experimental.symbolic_shapes import SympyBoolean
    from torch.fx.node import Argument

    from .codegen.cutlass.template import CUTLASSTemplate
    from .codegen.wrapper import PythonWrapperCodegen
    from .graph import GraphLowering
    from .utils import IndentedBuffer

else:
    CUTLASSTemplate: TypeAlias = object


try:
    import triton

    triton_version = triton.__version__
    has_triton = True
except ImportError:
    triton_version = None
    has_triton = False


_P = ParamSpec("_P")
_T = TypeVar("_T")
_U = TypeVar("_U")
_V = TypeVar("_V")

_IntLike: TypeAlias = int | Expr
_NumLike: TypeAlias = int | float | Expr

_OpOverloads: TypeAlias = torch._ops.OpOverload | torch._ops.HigherOrderOperator

log = logging.getLogger("torch._inductor.ir")
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
        Union[int, dict[str, "TensorBox"], "TensorBox", "Symbol", "IRNode"] | None
    ],
]


def _is_static(x: object) -> TypeIs[int | Integer]:
    return isinstance(x, (int, Integer))


@dataclasses.dataclass(frozen=True)
class GraphPartitionSignature:
    # symbol inputs that are necessary for codegen
    symbol_inputs: OrderedSet[sympy.Symbol]

    # mapping from partition input name to IRNode or Expr. Need the name str since
    # we cannot get name from Expr.
    input_nodes: dict[str, IRNode | sympy.Expr | TorchBindObject]
    output_nodes: list[IRNode]

    # mapping from partition input name to a boolean for whether deallocating it
    # in the partition function
    input_deallocation: dict[str, bool]
    skip_cudagraph: bool

    # name of constants read/written by the graph partition
    constant_names: list[str]


def validate_ir(node_or_nodes: _NodeOrNodes | None) -> None:
    def _check_tensorbox(nodes: _NodeOrNodes | None) -> None:
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
                    OpaqueMultiOutput,
                ),
            ), (
                f"Found {type(nodes)}, which is not a supported top level IR node. See [Note: Inductor IR]"
            )

    # Be picky about the accepted data structure (don't use pytree here)
    _check_tensorbox(node_or_nodes)


def ops_wrapper(name: str) -> Callable[..., OpsValue]:
    assert isinstance(name, str), type(name)

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


NHWC_STRIDE_ORDER = [3, 0, 2, 1]
NHWDC_STRIDE_ORDER = [4, 0, 3, 2, 1]


def get_fill_order(
    seq: Sequence[int | torch.SymInt | Expr], shape_env: ShapeEnv | None = None
) -> Sequence[int]:
    """
    Convert strides to fill order (argsort)
    """
    if shape_env is None or all(isinstance(s, (int, sympy.Integer)) for s in seq):
        sorted_idx: Sequence[int] = argsort(seq)
    else:
        # argsort_sym handles unbacked symints (with the help of the shape_env)
        sorted_idx = argsort_sym(shape_env, seq)
    return sorted_idx


def stride_order2fill_order(order: Sequence[int | Integer]) -> Sequence[int]:
    """
    Convert stride order to fill order
    For channel last format,

    stride order = [3, 0, 2, 1] and fill order = [1, 3, 2, 0]
    """
    lookup = {pos: idx for idx, pos in enumerate(order)}
    fill_order = [lookup[i] for i in range(len(order))]
    return fill_order


def get_stride_order(
    seq: Sequence[int | torch.SymInt | Expr], shape_env: ShapeEnv | None = None
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
def ir_node_to_tensor(x: None, replace_symbols_with_hints: bool = False) -> None: ...


@overload
def ir_node_to_tensor(
    x: IRNode, replace_symbols_with_hints: bool = False
) -> torch.Tensor: ...


def ir_node_to_tensor(
    x: IRNode | None, replace_symbols_with_hints: bool = False
) -> torch.Tensor | None:
    # When replace_symbols_with_hints=False (default), sizes/strides remain as
    # symbolic expressions, so downstream operations on the resulting tensor (e.g.,
    # shape comparisons inside a kernel's meta function) may install guards. When
    # True, symbolic expressions are replaced with concrete integer hints via
    # size_hint, preventing any downstream guards.
    if x is None:
        return None

    shape_fn: Callable[[int | Expr], int | Expr]
    if replace_symbols_with_hints:
        shape_fn = V.graph.sizevars.optimization_hint
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
    # pyrefly: ignore [bad-assignment]
    stride = convert_shape_to_symint(stride)
    with V.graph.sizevars.shape_env.suppress_guards():
        t = torch.empty_strided(
            size=size, stride=stride, dtype=dtype, device=device
        ).zero_()
    return t


def may_convert_to_optional(
    value: Sequence[_T] | None,
) -> Sequence[_T | None] | None:
    if isinstance(value, list) and not value:
        # [None] makes sure the cpp wrapper codegen will generate something like
        # {std::nullopt} instead of {}
        return [None]
    return value


def get_device_type(
    x: IRNode | OutputSpec | torch.device | None | str,
) -> str | None:
    if isinstance(x, str) or x is None:
        return x
    elif isinstance(x, torch.device):
        return x.type
    elif isinstance(x, (IRNode, OutputSpec)):
        return get_device_type(x.get_device())
    # pyrefly: ignore [bad-argument-type]
    assert_never(f"get_device_type({x}: {type(x).__name__})")


def is_triton(x: IRNode | torch.device | None | str) -> bool:
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

    assert isinstance(device_scheduling, type), type(device_scheduling)
    return issubclass(device_scheduling, TritonScheduling)


def is_cpu(x: IRNode | torch.device | None | str) -> bool:
    return get_device_type(x) == "cpu"


def is_aligned_realized_tensor(x: Buffer | TensorBox, alignment: int) -> bool:
    if (
        not isinstance(x, IRNode)
        or x.maybe_get_stride() is None
        or free_unbacked_symbols(x.get_stride())
        or free_unbacked_symbols(x.get_size())
    ):
        return False

    aligned_strides = sympy.And(
        *(sympy.Eq(Mod(s, alignment), 0) for s in x.get_stride()[:-1])
    )
    aligned_last_dim = sympy.Or(
        sympy.Eq(x.get_stride()[-1], 1), sympy.Le(x.get_size()[-1], 1)
    )
    is_aligned = sympy.And(aligned_strides, aligned_last_dim)

    # Make sure to guard to recompile when necessary.
    return V.graph.sizevars.guard_or_false(is_aligned)


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
        if V.graph.sizevars.statically_known_leq(dim, 1):
            continue

        if not V.graph.sizevars.guard_or_false(sympy.Eq(s1, s2)):
            return False
    return True


def try_match_insignificant_strides(
    tensor: IRNode,
    strides: Sequence[int | torch.SymInt],
) -> IRNode:
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
        return tensor

    if not significant_strides_equal(strides, tensor.get_stride(), tensor.get_size()):
        return tensor

    storage, old_layout = as_storage_and_layout(tensor)
    new_stride = [*old_layout.stride]
    for i, s in enumerate(tensor.get_size()):
        if V.graph.sizevars.statically_known_leq(s, 1):
            new_stride[i] = strides[i]

    new_layout = FixedLayout(
        old_layout.device,
        old_layout.dtype,
        old_layout.size,
        new_stride,
        old_layout.offset,
        old_layout.is_pinned,
    )
    return TensorBox(ReinterpretView(data=storage, layout=new_layout))


def gm_original_output_strides(gm: torch.fx.GraphModule) -> None:
    output_node = gm.graph.find_nodes(op="output")[0]
    output_node.meta["user_visible_output_idxs"] = [
        idx for idx, _ in enumerate(output_node.args)
    ]
    from torch._inductor.compile_fx import record_original_output_strides

    record_original_output_strides(gm)


def get_symbolic_inputs(inputs: Sequence[IRNode]) -> list[Expr]:
    sym_vars: OrderedSet[Expr] = OrderedSet()
    for inp in inputs:
        sym_vars |= get_free_symbols(inp.get_size(), unbacked_only=False)
        sym_vars |= get_free_symbols(inp.get_stride(), unbacked_only=False)

    return list(sym_vars)


def try_get_name(x):
    if isinstance(x, TensorBox):
        x = x.data
    if isinstance(x, BaseView):
        x = x.unwrap_view()
    if isinstance(x, StorageBox):
        x = x.data
    return x.get_name() if isinstance(x, Buffer) else None


class IRNode:
    """Base class for all intermediate representation (IR) nodes in TorchInductor.

    Note:
        This is an abstract base class. Most methods raise NotImplementedError
        and must be overridden by concrete subclasses.
    """

    _current_origins: ClassVar[OrderedSet[Any]] = OrderedSet()

    # NB: These are kinda weird,
    origins: OrderedSet[Any] = dataclasses.field(init=False)
    # traces back to where the IRNode is created in Inductor
    traceback: list[str] | None = dataclasses.field(init=False)
    origin_node: torch.fx.Node | None = dataclasses.field(init=False)
    # Annotations dict for storing metadata (e.g., KernelTemplateChoice)
    annotations: dict[str, Any] = dataclasses.field(init=False)

    @staticmethod
    @contextlib.contextmanager
    def current_origins(origins: OrderedSet[Node]) -> Generator[None, None, None]:
        old = IRNode._current_origins
        IRNode._current_origins = old | origins
        try:
            yield
        finally:
            IRNode._current_origins = old

    @staticmethod
    def is_realized_node(node: IRNode) -> bool:
        return isinstance(
            node,
            (
                ComputedBuffer,
                InputsKernel,
                InputBuffer,
                ReinterpretView,
                TemplateBuffer,
            ),
        )

    def wrap_for_lowering(self) -> IRNode:
        return TensorBox.create(self)

    def _post_init_setattr(self, attr: str, value: Any) -> None:
        # Intended for use in __post_init__ for enforcing an invariant on a dataclass
        # If you must, can also be used for setting provenance info
        # We would like to try and minimize these usages though
        object.__setattr__(self, attr, value)

    def __post_init__(self) -> None:
        origins = OrderedSet(self._current_origins)
        self._post_init_setattr("origins", origins)
        self._post_init_setattr(
            "traceback", traceback.format_stack() if config.debug_ir_traceback else None
        )
        self._post_init_setattr("origin_node", None)
        # Annotations dict for storing metadata (e.g., KernelTemplateChoice)
        self._post_init_setattr("annotations", {})

    def get_read_names(self) -> OrderedSet[str]:
        return OrderedSet(dep.name for dep in self.get_reads())

    def get_traceback(self) -> list[str] | None:
        return self.traceback

    def get_origin_node(self) -> torch.fx.Node | None:
        return self.origin_node

    def get_defining_op(self) -> Operation | None:
        return None

    def get_subgraphs(self) -> list[Subgraph]:
        """Return subgraphs contained in this node"""
        return []

    def get_stack_traces(self) -> OrderedSet[str]:
        # Return stack traces to user model code
        # A single IRNode could correspond to multiple lines of code
        stack_traces: OrderedSet[str] = OrderedSet()
        origins = self.origins
        if isinstance(self, ExternKernel):
            origin_node = self.get_origin_node()
            if self.origin_node:
                origins = OrderedSet([origin_node])
        for node in origins:
            if hasattr(node, "stack_trace") and node.stack_trace:
                # nodes in the backward graph don't have mapping to pre_grad_graph
                stack_traces.add(node.stack_trace)
            else:
                pre_grad_nodes = (
                    torch._inductor.debug._inductor_post_to_pre_grad_nodes.get(
                        "postToPre",
                        {},
                        # pyrefly: ignore [missing-attribute]
                    ).get(node.name, [])
                )
                if not isinstance(pre_grad_nodes, list):
                    continue
                for node_name in pre_grad_nodes:
                    stack_trace = (
                        torch._inductor.debug._inductor_pre_grad_node_stack_trace.get(
                            node_name, None
                        )
                    )
                    if stack_trace:
                        stack_traces.add(stack_trace)
        return stack_traces

    def common_repr(self, shorten: bool = True) -> Sequence[str]:
        origins = f"origins={getattr(self, 'origins', '')}"
        if shorten and len(origins) > 64:
            # this can get *very* long
            origins = f"{origins[:61]}..."
        if not self.get_stack_traces():
            return [origins]

        stack_trace_str = []
        for stack_trace in self.get_stack_traces():
            stack_trace_str.append("stack_traces = {")
            stack_trace_str += stack_trace.split("\n")
            stack_trace_str.append("}")
        return [origins] + stack_trace_str

    def str_helper(
        self, lines: Sequence[object], shorten: bool = True, multiline: bool = True
    ) -> str:
        lines = list(lines) + list(self.common_repr(shorten))
        lines = list(map(str, lines))
        if multiline:
            # pyrefly: ignore [no-matching-overload]
            new_lines = indent(",\n".join(lines))
            return f"{type(self).__name__}(\n{new_lines}\n)"
        else:
            return f"{type(self).__name__}({lines})"

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def maybe_get_dtype(self) -> torch.dtype | None:
        try:
            return self.get_dtype()
        except NotImplementedError:
            return None

    def get_layout(self) -> Layout:
        raise NotImplementedError(f"get_layout() is not implemented by {type(self)}!")

    def maybe_get_layout(self) -> Layout | None:
        try:
            return self.get_layout()
        except NotImplementedError:
            return None

    def get_output_spec(self) -> OutputSpec:
        return self.get_layout()

    def maybe_get_output_spec(self) -> OutputSpec | None:
        try:
            return self.get_output_spec()
        except NotImplementedError:
            return None

    def has_tensor_output(self) -> bool:
        """True for single tensor output (excludes MultiOutput)"""
        return isinstance(self.maybe_get_output_spec(), Layout)

    def get_size(self) -> Sequence[Expr]:
        raise NotImplementedError(f"get_size() is not implemented by {type(self)}!")

    def maybe_get_size(self) -> Sequence[_IntLike] | None:
        try:
            return self.get_size()
        except NotImplementedError:
            return None

    @property
    def shape(self) -> _IntLike | sympy.Rel | Sequence[_IntLike]:
        return self.get_size()

    def get_numel(self) -> Expr:
        return sympy_product(self.get_size())

    def is_zero_elements(self) -> bool:
        return V.graph.sizevars.statically_known_true(sympy.Eq(self.get_numel(), 0))

    def realize(self) -> str | None:
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

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        raise NotImplementedError(f"codegen_reference NYI on {type(self)}")

    def get_device(self) -> torch.device | None:
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

    def maybe_get_stride(self) -> Sequence[_IntLike] | None:
        try:
            return self.get_stride()
        except NotImplementedError:
            return None

    def get_name(self) -> str:
        raise NotImplementedError(type(self).__name__)

    def maybe_get_name(self) -> str | None:
        try:
            return self.get_name()
        except NotImplementedError:
            return None

    def is_input_buffer(self) -> bool:
        try:
            return self.get_name() in V.graph.graph_inputs
        except NotImplementedError:
            return False

    def has_large_inner_fn(self, threshold: int | None = None) -> bool:
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
        self, order: Sequence[int], allow_padding: bool = False
    ) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_fill_order(self, order: Sequence[int]) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_same_order(self, stride: Sequence[_IntLike]) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_exact_strides(
        self, exact_strides: Sequence[_IntLike], allow_padding: bool = False
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

    def get_reduction_type(self) -> str | None:
        raise NotImplementedError(type(self).__name__)

    def get_reduction_size(self) -> Sequence[Expr]:
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
        self.operation_name: str | None = None
        self._config_patches: dict[str, Any] = {}

    def get_device(self) -> torch.device | None:
        raise NotImplementedError

    def get_origin_node(self) -> torch.fx.Node | None:
        assert hasattr(self, "origin_node")
        return self.origin_node

    def get_origins(self) -> OrderedSet[Any]:
        assert hasattr(self, "origins")
        return self.origins

    def get_operation_name(self) -> str:
        assert self.operation_name is not None
        return self.operation_name

    def get_config_patches(self) -> dict[str, Any]:
        """Get config patches for this operation (e.g., coordinate_descent_tuning)."""
        return self._config_patches

    def set_config_patches(self, patches: dict[str, Any]) -> None:
        """Set config patches for this operation."""
        self._config_patches = patches

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
class BaseConstant(IRNode):
    dtype: torch.dtype
    device: torch.device

    def get_size(self) -> Sequence[Expr]:
        return ()

    def get_device(self) -> torch.device | None:
        return self.device

    def get_origin_node(self) -> torch.fx.Node | None:
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

    def realize(self) -> str | None:
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
    expected_stride = 1
    expected_stride_max = 1
    for x, y in reversed(tuple(zip(shape, stride))):
        if x == 1:
            continue

        if not V.graph.sizevars.statically_known_equals(
            y, expected_stride
        ) and not V.graph.sizevars.statically_known_equals(y, expected_stride_max):
            return False

        expected_stride_max *= sympy.Max(1, x)
        expected_stride *= x

    return True


def get_align_for_dtype(dtype: torch.dtype) -> int:
    return config.padding_alignment_bytes // dtype.itemsize


class OutputSpec:
    """Abstract base for Layout, MultiOutputLayout, NoneLayout.
    Represents the memory layout of the output of an Operation."""

    def get_device(self) -> torch.device | None:
        raise NotImplementedError(type(self).__name__)

    def storage_size(self) -> int:
        raise NotImplementedError(type(self).__name__)

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        raise NotImplementedError(type(self).__name__)


@ir_dataclass
class Layout(OutputSpec):
    """
    Layout base class

    Carries tensor meta-information including offset and
    whether it is pinned.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: Sequence[Expr],
        stride: Sequence[Expr] | None = None,
        offset: Expr = Integer(0),
        is_pinned: bool = False,
    ) -> None:
        if stride is None:
            stride = FlexibleLayout.contiguous_strides(size)
        # pyrefly: ignore [read-only]
        self.device = device
        self.dtype = dtype
        assert len(size) == len(stride), f"size={size}, stride={stride}"
        assert all(isinstance(s, (Expr, int)) for s in size)
        self._size = size
        self._stride = stride
        self._offset = offset
        self.is_pinned = is_pinned
        # is_pinned implies cpu
        assert (not self.is_pinned) or (self.device.type == "cpu"), (
            "Only CPU tensors can be pinned"
        )

    @property
    def size(self) -> Sequence[Expr]:
        return self._size

    @size.setter
    def size(self, value: Sequence[Expr]) -> None:
        self._size = value

    @property
    def stride(self) -> Sequence[Expr]:
        return self._stride

    @stride.setter
    def stride(self, value: Sequence[Expr]) -> None:
        self._stride = value

    @property
    def offset(self) -> Expr:
        return self._offset

    @offset.setter
    def offset(self, value: Expr) -> None:
        self._offset = value

    def __str__(self) -> str:
        offset = ""
        if self.offset != 0:
            offset = f", offset={self.offset}"

        device_index_str = "" if self.device.index is None else f":{self.device.index}"
        is_pinned_str = ""
        if self.is_pinned:
            is_pinned_str = f", is_pinned={self.is_pinned}"
        return (
            f"{type(self).__name__}('{self.device.type}{device_index_str}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}{offset}{is_pinned_str})"
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
                pin_memory=self.is_pinned,
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
            # pyrefly: ignore [bad-specialization]
            strides,
            # pyrefly: ignore [bad-specialization]
            make_channels_last_strides_for(shape),
            shape,
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

    def is_stride_ordered(self, order: Sequence[int]) -> bool:
        assert len(self.stride) == len(order)

        # ignore dimensions of size 1, they dont affect layout
        non_1_indices = [
            i
            for i, dim in enumerate(self.size)
            if V.graph.sizevars.optimization_hint(dim, fallback=2) != 1
        ]

        stride = [self.stride[i] for i in non_1_indices]
        order: Sequence[int] = [order[i] for i in non_1_indices]

        def sorted_indices(arr: Sequence[int]) -> Sequence[int]:
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

    def is_channels_last_stride_ordered(self) -> bool:
        # create channels_last order(NCHW, NCDHW, the C is the first order).
        order = [0] + list(reversed(range(1, len(self.stride) - 1)))
        order = [len(order)] + order
        return self.is_stride_ordered(order)

    @staticmethod
    def _pad_strides(
        in_strides: Sequence[int], size: Sequence[Expr], dtype: torch.dtype
    ) -> Sequence[int]:
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

        # Skip padding the strides for dynamic shapes based on config.pad_dynamic_shape
        # Checking both shape and strides, as there are cases where only one is dynamic
        is_dynamic = not all(
            isinstance(s, (int, sympy.Integer))
            for s in itertools.chain(in_strides, size)
        )
        if not config.pad_dynamic_shapes and is_dynamic:
            return in_strides

        shape_env = V.graph._shape_env if hasattr(V.graph, "_shape_env") else None

        def contains_unbacked_symints(expr: sympy.Expr | int) -> bool:
            if shape_env is None:
                return False
            if not isinstance(expr, sympy.Expr):
                return False
            return any(shape_env.is_unbacked_symint(s) for s in expr.free_symbols)

        # Skip padding the strides when it contains unbacked symints for now.
        if shape_env and any(contains_unbacked_symints(s) for s in in_strides):
            return in_strides

        stride_order = get_stride_order(in_strides, shape_env)
        fill_order = stride_order2fill_order(stride_order)

        new_strides = [0 for _ in range(len(in_strides))]
        # since we pad when the layout is flexible, we can decide the
        # smallest stride to be 1.
        new_strides[fill_order[0]] = 1

        padded = False
        for rank, idx in enumerate(fill_order[1:], start=1):
            prev_idx = fill_order[rank - 1]
            stride = new_strides[prev_idx] * size[prev_idx]
            # Static stride and meets padding conditions OR
            # Dynamic stride and config.pad_dynamic_shape=True
            require_padding = (
                isinstance(stride, (int, sympy.Integer))
                and stride > config.padding_stride_threshold
                and stride % align != 0
            ) or (isinstance(stride, sympy.Expr) and config.pad_dynamic_shapes)
            new_strides[idx] = stride
            if require_padding:
                new_strides[idx] = ceildiv(stride, align) * align
                padded = True

        if not padded:
            # Consider a tensor with shape [256, 1, 5, 5]
            # Avoid strides like [25, 5, 5, 1] being padded to equivalent strides
            # [25, 25, 5, 1].
            return in_strides

        # pyrefly: ignore [bad-assignment]
        metrics.num_comprehensive_padding += 1
        return new_strides

    def pad_strides(self) -> None:
        assert isinstance(self, FlexibleLayout), type(self)
        assert self.stride is not None
        self.stride = self._pad_strides(self.stride, self.size, self.dtype)

    def should_pad_strides(self) -> bool:
        return config.comprehensive_padding and isinstance(self, FlexibleLayout)

    def as_fixed(self) -> FixedLayout:
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
            self.is_pinned,
        )

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        assert FlexibleLayout.allow_indexing, (
            f"convert {type(self).__name__} to FixedLayout first"
        )
        return self.as_fixed().make_indexer()

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Layout)
            and self.device == other.device
            and self.dtype == other.dtype
            and self.size == other.size
            and self.stride == other.stride
            and self.offset == other.offset
            and self.is_pinned == other.is_pinned
        )

    def storage_size(self) -> Expr:
        return compute_required_storage_length(self.size, self.stride, self.offset)  # type: ignore[arg-type]

    @cache_on_self_and_args("Layout")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return (
            get_free_symbols(self.size, unbacked_only)
            | get_free_symbols(self.stride, unbacked_only)
            | get_free_symbols(self.offset, unbacked_only)
        )


class FixedLayout(Layout):
    """A Tensor layout we cannot change"""

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        """A closure containing math to read a given element"""
        return _fixed_indexer(self.size, self.stride, self.offset)


class FlexibleLayout(Layout):
    """
    A Tensor layout that we are allowed to change

    Assumption: layout change should NOT add or remove free symbols
    """

    allow_indexing = False

    def get_fixed_layout_without_freezing(self) -> FixedLayout:
        """
        Compute what the strides would be if this layout were frozen,
        without actually modifying the layout. This is used for speculative
        stride computation during Triton template code generation.
        """
        # Create a temporary copy and use as_fixed to keep freezing path in sync
        return copy.deepcopy(self).as_fixed()

    # WARNING!  This doesn't handle zero size tensors correctly
    @staticmethod
    def contiguous_strides(sizes: Sequence[int]) -> list[Expr]:
        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.S.One]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    @staticmethod
    def fill_ordered(sizes: Sequence[int], order: Sequence[int]) -> list[Expr]:
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
    def stride_ordered(sizes: Sequence[int], order: Sequence[int]) -> Sequence[Expr]:
        """
        Create a stride based on the sorted order of a permuted range.

        In this format, channels last would be:
            [3, 0, 2, 1]
        """
        assert OrderedSet(range(len(sizes))) == OrderedSet(order)
        fill_order = stride_order2fill_order(order)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @staticmethod
    def stride_ordered_for_memory_format(
        sizes: Sequence[int], memory_format: torch.memory_format
    ) -> Sequence[Expr]:
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
    def same_ordered(
        sizes: Sequence[int], stride: Sequence[_IntLike]
    ) -> Sequence[Expr]:
        """
        Create a stride that has the same stride order as given stride

        For example, if given stride is [1000, 1, 100, 10],
        the fill order should be [1, 3, 2, 0]
        """
        assert len(sizes) == len(stride)
        stride = V.graph.sizevars.guarding_hints_or_throw(stride)
        fill_order = sorted(range(len(stride)), key=stride.__getitem__)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @property
    def size(self) -> Sequence[Expr]:
        return self._size

    @size.setter
    def size(self, value: Sequence[Expr]) -> None:
        self.assert_free_symbol_uses_unchanged("size", value)
        self._size = value

    @property
    def stride(self) -> Sequence[Expr]:
        return self._stride

    @stride.setter
    def stride(self, value: Sequence[Expr]) -> None:
        self.assert_free_symbol_uses_unchanged("stride", value)
        self._stride = value

    @property
    def offset(self) -> Expr:
        return self._offset

    @offset.setter
    def offset(self, value: Expr) -> None:
        self.assert_free_symbol_uses_unchanged("offset", value)
        self._offset = value

    def as_stride_order(
        self, order: Sequence[int], allow_padding: bool = False
    ) -> FixedLayout:
        new_stride = self.stride_ordered(self.size, order)
        if self.should_pad_strides() and allow_padding:
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)

        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def as_exact_strides(
        self, exact_strides: Sequence[_IntLike], allow_padding: bool = False
    ) -> FixedLayout:
        new_stride = exact_strides
        if self.should_pad_strides() and allow_padding:
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)

        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def as_fill_order(self, order: Sequence[int]) -> FixedLayout:
        new_stride: Sequence[int] = self.fill_ordered(self.size, order)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def as_same_order(self, stride: Sequence[_IntLike]) -> FixedLayout:
        new_stride = self.same_ordered(self.size, stride)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def get_initial_free_symbol_uses(self) -> dict[tuple[str, bool], sympy.Symbol]:
        initial_free_symbols = {}
        for name in ["size", "stride", "offset"]:
            for unbacked_only in [True, False]:
                key = (name, unbacked_only)
                initial_free_symbols[key] = OrderedSet(
                    get_free_symbols(getattr(self, name), unbacked_only)
                )

        return initial_free_symbols

    def assert_free_symbol_uses_unchanged(self, name: str, value: IterateExprs) -> None:
        for unbacked_only in [True, False]:
            old_free_symbols = self.initial_free_symbols[(name, unbacked_only)]
            new_free_symbols = OrderedSet(get_free_symbols(value, unbacked_only))
            assert new_free_symbols == old_free_symbols, (
                f"Expected free symbols unchanged, but got {new_free_symbols} vs {old_free_symbols}"
            )

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: Sequence[Expr],
        stride_order: Sequence[int | Integer] | None = None,
        is_pinned: bool = False,
    ) -> None:
        if stride_order:
            strides = FlexibleLayout.fill_ordered(size, stride_order)
        else:
            strides = FlexibleLayout.contiguous_strides(size)
        super().__init__(device, dtype, size, strides, is_pinned=is_pinned)

        # record the initial free symbols to check that we do not add new free symbols
        # later when modifying sizes, strides, and offsets.
        self.initial_free_symbols = self.get_initial_free_symbol_uses()


class NonOwningLayout(Layout):
    """Is a view into the storage of another tensor"""

    def __init__(self, view: BaseView | TensorBox) -> None:
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

    def maybe_guard_aligned(self) -> bool:
        offset = self.view.get_layout().offset
        if offset == 0:
            return True
        from .utils import ALIGNMENT

        return V.graph.sizevars.statically_known_multiple_of(offset, ALIGNMENT)

    @cache_on_self_and_args("NonOwningLayout")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        assert isinstance(self.view, ReinterpretView)
        box = self.view.data
        assert isinstance(box, StorageBox), type(box)
        input_buffer = box.data
        assert isinstance(input_buffer, Buffer), type(box)
        return input_buffer.layout.get_free_symbol_uses(unbacked_only)


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
        layout: FlexibleLayout | FixedLayout,
        comm_buffer_type: CommBufferType,
        group_name: str,
    ):
        fixed = layout.as_fixed() if isinstance(layout, FlexibleLayout) else layout
        super().__init__(
            device=fixed.device,
            dtype=fixed.dtype,
            size=fixed.size,
            stride=fixed.stride,
            offset=fixed.offset,
            is_pinned=fixed.is_pinned,
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

    device: torch.device | None
    size: list[int] = dataclasses.field(default_factory=lambda: [0])
    stride: list[int] = dataclasses.field(default_factory=lambda: [0])

    def storage_size(self) -> int:
        return 0

    def as_fixed(self) -> OutputSpec:
        return self

    def get_device(self) -> torch.device | None:
        return self.device


class MutationLayoutSHOULDREMOVE(Layout):
    def __init__(self, target: IRNode) -> None:
        super().__init__(
            target.get_device_or_error(),
            target.get_dtype(),
            target.get_size(),
            None,
        )
        self.target = target
        name = self.get_buffer().get_name()
        V.graph.mark_buffer_mutated(name)

    @property
    def stride(self) -> Sequence[Expr]:  # type: ignore[override]
        return self.real_layout().stride

    @stride.setter  # type: ignore[override]
    def stride(self, value: Never) -> None:
        pass  # ignore setting of stride

    def storage_size(self) -> Expr:
        return self.real_layout().storage_size()

    def get_buffer(self) -> Buffer:
        def unwrap_views(target: Any) -> Any:
            if isinstance(target, MutationLayoutSHOULDREMOVE):
                return unwrap_views(target.target)
            if isinstance(target, BaseView):
                return unwrap_views(target.unwrap_view())
            if isinstance(target, MutableBox):
                return unwrap_views(target.data)
            return target

        result = unwrap_views(self.target)
        assert isinstance(result, Buffer), type(result)
        return result

    def real_layout(self) -> Layout:
        layout = self.get_buffer().layout
        assert isinstance(layout, Layout)
        return layout

    @classmethod
    def realize_into(
        cls, src: IRNode, dst: IRNode, unsafe_alias: bool = False
    ) -> IRNode:
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
            node = Pointwise.create(
                device=src.get_device(),
                dtype=src.get_dtype(),
                inner_fn=src.make_loader(),
                ranges=[
                    V.graph.sizevars.check_equals_and_simplify(a, b)
                    for a, b in zip(src.get_size(), dst.get_size())
                ],
            )
            assert isinstance(node, (BaseView, MutableBox))
            src = node.data

        src.realize()
        assert hasattr(src, "data"), src
        assert isinstance(src.data.layout, FlexibleLayout), type(src.data.layout)
        src.data.layout = MutationLayoutSHOULDREMOVE(dst)
        return src.data

    def as_fixed(self) -> Self:  # type: ignore[override]
        return self

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        return self.target.make_indexer()


@ir_dataclass(frozen=False)
class Buffer(IRNode, CodegenSymbol):
    # Name is sometimes None; e.g., ForceInPlace, where there isn't
    # a meaningful name
    name: str | None
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

    def get_example(self) -> torch.Tensor | torch.SymInt:
        if isinstance(self.layout, Layout):
            return self.layout.get_example()
        raise NotImplementedError(type(self.layout).__name__)

    def get_device(self) -> torch.device | None:
        return self.get_output_spec().get_device()

    def get_defining_op(self) -> Operation | None:
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

    def get_storage_numel(self) -> int:
        return self.get_numel()

    def get_is_pinned(self) -> bool:
        return self.get_layout().is_pinned

    def freeze_layout(self) -> None:
        if isinstance(self.layout, Layout) and not isinstance(
            self.layout, NonOwningLayout
        ):
            self.layout = self.layout.as_fixed()

    def freeze_layout_with_stride_order(
        self, order: Sequence[int], allow_padding: bool = False
    ) -> None:
        assert isinstance(self.layout, FlexibleLayout), type(self.layout)
        self.layout = self.layout.as_stride_order(order, allow_padding=allow_padding)

    def freeze_layout_with_fill_order(self, order: Sequence[int]) -> None:
        assert isinstance(self.layout, FlexibleLayout), type(self.layout)
        self.layout = self.layout.as_fill_order(order)

    def freeze_layout_with_same_order(self, stride: Sequence[int]) -> None:
        assert isinstance(self.layout, FlexibleLayout), type(self.layout)
        self.layout = self.layout.as_same_order(stride)

    def freeze_layout_with_exact_strides(
        self, exact_strides: Sequence[int], allow_padding: bool = False
    ) -> None:
        assert isinstance(self.layout, FlexibleLayout), type(self.layout)
        self.layout = self.layout.as_exact_strides(
            exact_strides, allow_padding=allow_padding
        )

    def is_zero_elements(self) -> bool:
        return V.graph.sizevars.statically_known_true(sympy.Eq(self.get_numel(), 0))

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # Loading from a zero-element buffer is a no-op
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.get_dtype())

        def loader(index: Sequence[Expr]) -> OpsValue:
            indexer = self.make_indexer()
            return ops.load(self.name or "unnamed", indexer(index))

        return loader

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        return self.get_name()

    def decide_layout(self) -> None:
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

    @cache_on_self_and_args("Buffer")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def realize(self) -> str | None:
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
    override_device: torch.device | None = None

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

    @cache_on_self_and_args("NoneAsConstantBuffer")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        return V.graph.wrapper_code.none_str

    def get_output_spec(self) -> OutputSpec:
        return NoneLayout(device=None)

    def has_tensor_output(self) -> bool:
        return False


@ir_dataclass
class ShapeAsConstantBuffer(IRNode):
    expr: Expr

    @cache_on_self_and_args("ShapeAsConstantBuffer")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return get_free_symbols(self.expr, unbacked_only)

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        return V.graph.wrapper_code.codegen_sizevar(self.expr)

    def has_tensor_output(self) -> bool:
        return False


@ir_dataclass(frozen=False)
class ComputedBuffer(OperationBuffer):
    """
    Represents a buffer that is computed during kernel execution rather than being an input.
    """

    data: Loops
    _force_realize: ClassVar[bool] = False

    # fields for split reduction
    _split_size: int | None = None
    _original_inner_fn: Callable[..., Any] | None = None
    _original_ranges: Sequence[_IntLike] | None = None
    _original_reduction_ranges: Sequence[_IntLike] | None = None

    @contextlib.contextmanager
    def with_original_inner_fn(self) -> Iterator[None]:
        assert self._split_size is not None
        assert self._original_inner_fn is not None
        assert self._original_ranges is not None
        assert self._original_reduction_ranges is not None

        assert isinstance(self.data, Reduction), f"{type(self.data)}"
        old_data = self.data
        old_layout = self.layout
        try:
            new_data = Reduction(
                device=old_data.device,
                dtype=old_data.dtype,
                inner_fn=self._original_inner_fn,
                ranges=self._original_ranges,
                reduction_ranges=self._original_reduction_ranges,
                reduction_type=old_data.reduction_type,
                src_dtype=old_data.src_dtype,
                reduction_hint=old_data.reduction_hint,
            )
            self.data = new_data
            # this layout does not matter since we skip tl.store
            # later
            self.layout = FixedLayout(
                old_data.device,
                old_data.dtype,
                self._original_ranges,
            )
            self.get_default_sizes_body.clear_cache(self)
            yield
        finally:
            self.data = old_data
            self.layout = old_layout

    @staticmethod
    @contextlib.contextmanager
    def force_realize() -> Iterator[None]:
        old_value = ComputedBuffer._force_realize
        try:
            ComputedBuffer._force_realize = True
            yield
        finally:
            ComputedBuffer._force_realize = old_value

    def get_computed_buffer_name(self) -> str | None:
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
        if not isinstance(self.data, (Reduction, Scan, Sort, Pointwise)):
            return dependencies.ReadWrites(
                reads=OrderedSet(),
                writes=OrderedSet(),
                index_exprs=OrderedSet(),
            )

        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.data.get_reduction_type():
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_pointwise_size(),
                    self.data.get_reduction_size(),
                )
            else:
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_size(),
                )

    @cache_on_self_and_args("ComputedBuffer")
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
        # One thing you might wonder is if this is enough for a ComputedBuffer
        # denoting a reduction over i0.  Empirically, it is enough, but for an
        # unusual reason: we only need accurate dependencies for item() call,
        # but it's impossible to end up with a reduction over i0 from an
        # item() call without a regular non-reduction buffer first.
        result = self.layout.get_free_symbol_uses(
            unbacked_only
        ) | self.data.get_free_symbol_uses(unbacked_only)

        if self.has_store_function():
            result |= self.get_read_writes().get_free_symbol_uses(unbacked_only)
        return result

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        if (
            not self.get_reduction_type()
            and self.name not in V.graph.mutated_buffers
            and self.num_reads() == 0
            and not self._force_realize
        ):
            # inline this op rather than generating ops.load()
            return self.data.make_loader()
        return super().make_loader()

    def has_store_function(self) -> bool:
        return isinstance(self.data, (Reduction, Scan, Sort, Pointwise))

    def get_store_function(self) -> Callable[..., None]:
        indexer = self.get_layout().as_fixed().make_indexer()
        if isinstance(self.data, (Reduction, Scan, Sort)):
            return partial(self.data.store_reduction, self.name, indexer)
        else:
            assert isinstance(self.data, Pointwise), type(self.data)
            return partial(self.data.store_output, self.name, indexer)

    def get_fill_order(self) -> list[int] | None:
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
        tuple[list[Expr], list[Expr]],
        LoopBody,
        tuple[list[Expr], list[Expr]],
    ]:
        args, var_ranges = dependencies.index_vars_squeeze(
            self.get_pointwise_size(), self.get_reduction_size(), prefix="q"
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
        extra_indexing_constraints: tuple[dict[Any, Any], list[Any]] | None = None,
        recompute_sizes_body_func: Callable[..., Any] | None = None,
    ) -> tuple[tuple[list[Expr], list[Expr]], LoopBody | None]:
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
            assert isinstance(extra_indexing_ranges, dict), type(extra_indexing_ranges)
            assert isinstance(extra_indexing_expr, list), type(extra_indexing_expr)
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

        def simplify_and_reorder(
            x_vars: Sequence[sympy.Symbol],
            support_vars: Sequence[sympy.Symbol],
            sizes: Sequence[int],
            simplify_loops: bool,
        ) -> tuple[
            list[int],
            Callable[[Sequence[int]], Sequence[int]],
            Callable[[Sequence[int]], Sequence[int]],
        ]:
            newsizes, reindex0, reindex1 = self._apply_loop_reordering(
                x_vars, support_vars, sizes, memory_addrs
            )

            # When using native matmul, the codegen assumes the following loop order,
            # regardless of the stride of A and B:
            #
            #   for z -> y -> x -> r:  C[z, y, x] += A[z, y, r] * B[z, r, x]
            # or
            #   for z -> x -> y -> r:  C[z, y, x] += A[z, y, r] * B[z, r, x]
            #
            # The critical point is the position of the "z" (batch) axis in bmm.
            # It is fine to swap the y and x axes (e.g., (z, y, x, r) or (z, x, y, r)),
            # but reordering the z axis (e.g., (y, x, z, r)) breaks codegen.
            #
            # Therefore, if loop reordering changes the "z" location in bmm,
            # it should be reverted to the default.
            # This may not always produce the optimal loop order when strides
            # do not align with the default assumption.
            #
            # TODO: Consider extending tl.dot codegen to support arbitrary loop orders.
            if self.get_reduction_type() == "dot" and len(sizes) == 3:
                order = list(range(len(sizes)))  # default order

                # if z axis is not the outermost, use the default reorder.
                if reindex0(order)[0] != 0:
                    newsizes = [sizes[i] for i in order]
                    reindex0 = same_reorder(order)
                    reindex1 = inverse_reorder(order)

            # for NHWC: reindex0([0,1,2,3]) = [0,2,3,1], reindex1([0,1,2,3]) = [0,3,2,1]
            x_vars = reindex0(x_vars)

            if simplify_loops:
                newsizes, reindex2, _prune = V.graph.sizevars._simplify_loops(
                    x_vars,
                    newsizes,
                    index_prevent_reordering(index_formulas, x_vars, newsizes),
                )
                reindex = fuse_reindexing(reindex1, reindex2)
            else:
                reindex = reindex1
            return newsizes, reindex, reindex1

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
    def _apply_loop_reordering(
        index_vars: Sequence[sympy.Symbol],
        support_vars: Sequence[sympy.Symbol],
        sizes: Sequence[int],
        memory_addrs: list[sympy.Expr],
        priority_idx: list[int] | None = None,
    ) -> tuple[
        list[int],
        Callable[[Sequence[int]], Sequence[int]],
        Callable[[Sequence[int]], Sequence[int]],
    ]:
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

    def get_pointwise_size(self) -> Sequence[Expr]:
        return self.data.get_pointwise_size()

    def get_reduction_size(self) -> Sequence[Expr]:
        return self.data.get_reduction_size()

    def get_reduction_type(self) -> str | None:
        return self.data.get_reduction_type()

    def is_no_op(self) -> bool:
        return self.data.is_zero_elements()

    def should_allocate(self) -> bool:
        return True

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        return self.data.constant_to_device(device)


class NonTensorObj(IRNode):
    @cache_on_self_and_args("NonTensorObj")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()


@ir_dataclass
class TorchBindObject(NonTensorObj):
    name: str
    value: FakeScriptObject | torch.ScriptObject

    def get_name(self) -> str:
        return self.name

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        return self.name

    def get_value(self) -> FakeScriptObject | torch.ScriptObject:
        return self.value

    def get_real_obj(self) -> torch.ScriptObject:
        if isinstance(self.value, torch.ScriptObject):
            return self.value
        else:
            return self.value.real_obj

    def get_buf_bytes(self) -> int:
        # Returns the sum of all tensors in the flattened object
        real_script_obj = self.get_real_obj()

        if is_opaque_value(real_script_obj):
            return 0

        assert hasattr(real_script_obj, "__obj_flatten__")
        flat_dict = dict(real_script_obj.__obj_flatten__())
        flat_elems = pytree.tree_flatten(flat_dict)[0]
        flat_sizes = [
            x.element_size() * x.numel()
            for x in flat_elems
            if isinstance(x, torch.Tensor)
        ]
        return functools.reduce(operator.add, flat_sizes, 0)


@ir_dataclass
class OpaqueValueTypeConstant(NonTensorObj):
    """IR node for opaque value type constants that appear directly in graph outputs.

    Unlike TorchBindObject (which references named constants loaded at runtime),
    this inlines the value's repr into the generated code since value types are
    reconstructed from their repr.
    """

    value: Any

    def get_name(self) -> str:
        return repr(self.value)

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        obj_repr, opaque_types = get_opaque_obj_repr(self.value)
        for n, t in opaque_types.items():
            V.graph.opaque_value_type_classes[n] = t
        return obj_repr


@ir_dataclass
class GeneratorState(NonTensorObj):
    name: str
    device: torch.device

    def get_name(self) -> str:
        return self.name

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        return self.name


@ir_dataclass
class OpaqueObjectState(NonTensorObj):
    """
    Represents an opaque object (e.g., ProcessGroup) that is passed through
    as a graph input. Similar to GeneratorState, this wraps the object with
    its placeholder name so codegen can reference it properly.
    """

    name: str
    value: Any  # The actual opaque object (for reference, not used in codegen)

    def get_name(self) -> str:
        return self.name

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        return self.name
