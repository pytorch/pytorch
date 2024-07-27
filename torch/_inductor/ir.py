# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    ContextManager,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from unittest.mock import patch

import sympy
from sympy import Expr, Integer

import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
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
    CallMethodKey,
    compute_unbacked_bindings,
    DivideByKey,
    free_unbacked_symbols,
    rebind_unbacked,
    resolve_unbacked_bindings,
    SymTypes,
)
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import SymT

from . import config, dependencies
from .codegen.common import BackendFeature, index_prevent_reordering
from .dependencies import (
    extract_free_unbacked_symbols,
    extract_input_node_reduction_ranges,
    extract_read_writes,
    var_builder,
)
from .ops_handler import OpCounterCSE
from .runtime.hints import ReductionHint
from .runtime.runtime_utils import do_bench
from .utils import (
    argsort,
    cache_on_self,
    ceildiv,
    convert_shape_to_inductor,
    convert_shape_to_symint,
    developer_warning,
    get_kernel_metadata,
    is_dynamic,
    is_gpu,
    sympy_dot,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    sympy_product,
    sympy_subs,
)
from .virtualized import ops, V


if TYPE_CHECKING:
    from .graph import GraphLowering

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


def validate_ir(node_or_nodes):
    def _check_tensorbox(nodes):
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
                    torch._inductor.ir.ExpandView,
                    DynamicScalar,
                    AssertScalar,
                    TensorBox,
                    sympy.logic.boolalg.Boolean,
                    Expr,
                    int,
                    EffectfulKernel,
                ),
            ), f"Found {type(nodes)}, which is not a supported top level IR node. See [Note: Inductor IR]"

    # Be picky about the accepted data structure (don't use pytree here)
    _check_tensorbox(node_or_nodes)


def ops_wrapper(name):
    assert isinstance(name, str)

    def fn(*args, **kwargs):
        return getattr(ops, name)(*args, **kwargs)

    return fn


def inverse_reorder(order):
    inv_order = dict(zip(order, range(len(order))))

    def reindex(index):
        assert len(index) == len(inv_order)
        return [index[inv_order[i]] for i in range(len(index))]

    return reindex


def same_reorder(order):
    def reindex(index):
        assert len(index) == len(order)
        return [index[order[i]] for i in range(len(index))]

    return reindex


def fuse_reindexing(reindex1, reindex2):
    def reindex(index):
        return reindex1(reindex2(index))

    return reindex


NHWC_STRIDE_ORDER = [3, 0, 2, 1]
NHWDC_STRIDE_ORDER = [4, 0, 3, 2, 1]


def stride_order2fill_order(order):
    """
    Convert stride order to fill order
    For channel last format,

    stride order = [3, 0, 2, 1] and fill order = [1, 3, 2, 0]
    """
    lookup = {pos: idx for idx, pos in enumerate(order)}
    fill_order = [lookup[i] for i in range(len(order))]
    return fill_order


def get_stride_order(seq: Sequence[int]) -> List[int]:
    """
    Convert strides to stride order
    """
    sorted_idx: List[int] = argsort(seq)
    out = [0 for _ in range(len(seq))]
    for i, elem in enumerate(sorted_idx):
        out[elem] = i
    return out


def ir_node_to_tensor(x, guard_shape=True):
    if x is None:
        return None

    shape_fn: Callable[[Expr], Union[int, Expr]]
    if not guard_shape:
        shape_fn = V.graph.sizevars.size_hint
    else:
        shape_fn = identity
    size = [shape_fn(s) for s in x.get_size()]
    stride: StrideType
    if is_storage_and_layout(x):
        stride = [shape_fn(s) for s in x.get_layout().stride]  # type: ignore[misc]
    else:
        stride = FlexibleLayout.contiguous_strides(size)  # type: ignore[arg-type]
    dtype = x.get_dtype()
    device = x.get_device()
    size = convert_shape_to_symint(size)
    stride = convert_shape_to_symint(stride)
    with V.graph.sizevars.shape_env.suppress_guards():
        t = torch.empty_strided(
            size=size, stride=stride, dtype=dtype, device=device
        ).zero_()
    return t


def may_convert_to_optional(value):
    if isinstance(value, list) and not value:
        # [None] makes sure the cpp wrapper codegen will generate something like
        # {std::nullopt} instead of {}
        return [None]
    return value


def get_device_type(x):
    if getattr(x, "get_device", None):
        return get_device_type(x.get_device())
    if isinstance(x, torch.device):
        return x.type
    return None


def is_triton(x):
    return is_gpu(get_device_type(x))


def is_cpu(x):
    return get_device_type(x) == "cpu"


class IRNode:
    _current_origins: ClassVar[Set[Any]] = set()

    @staticmethod
    @contextlib.contextmanager
    def current_origins(origins: Set[torch.fx.Node]):
        old = IRNode._current_origins
        IRNode._current_origins = old | origins
        try:
            yield
        finally:
            IRNode._current_origins = old

    def __post_init__(self):
        self.origins = set(self._current_origins)
        self.traceback = traceback.format_stack() if config.debug_ir_traceback else None

    def get_read_names(self) -> Set[str]:
        raise NotImplementedError(f"NYI on {type(self)}")

    def get_traceback(self):
        return self.traceback

    def get_defining_op(self):
        raise NotImplementedError

    def common_repr(self, shorten=True):
        origins = f"origins={getattr(self, 'origins', '')}"
        if shorten and len(origins) > 64:
            # this can get *very* long
            origins = f"{origins[:61]}..."
        return [origins]

    def str_helper(self, lines, shorten=True, multiline=True):
        lines = lines + self.common_repr(shorten)
        lines = list(map(str, lines))
        if multiline:
            new_lines = indent(",\n".join(lines))
            return f"{type(self).__name__}(\n{new_lines}\n)"
        else:
            return f"{type(self).__name__}({lines})"

    def get_dtype(self):
        return self.dtype

    def get_layout(self):
        raise NotImplementedError(f"get_layout() is not implemented by {type(self)}!")

    def get_size(self):
        raise NotImplementedError(f"get_size() is not implemented by {type(self)}!")

    @property
    def shape(self):
        return self.get_size()

    def get_numel(self):
        return sympy_product(self.get_size())

    def is_zero_elements(self):
        return V.graph.sizevars.is_expr_static_and_true(sympy.Eq(self.get_numel(), 0))  # type: ignore[arg-type]

    def realize(self):
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

    def codegen_reference(self, writer=None):
        raise NotImplementedError(f"codegen_reference NYI on {type(self)}")

    # The abstract method declarations below serve to convince mypy that all IRNode instances have these functions
    # defined, while having no effect at runtime. We cannot create stub implementations here because other parts of
    # the code dynamically check for defined attributes.
    get_device: Callable[[], torch.device]
    dtype: torch.dtype
    get_name: Callable[[], str]
    get_reads: Callable[[], Any]
    get_stride: Callable[[], Any]
    get_storage_numel: Callable[[], Any]
    has_exceeded_max_reads: Callable[[], bool]
    make_loader: Callable[[], Callable[[Any], Any]]
    make_indexer: Callable[[], Callable[[Any], Any]]
    mark_reuse: Callable[[int], None]
    realize_hint: Callable[[], None]
    get_unbacked_symbol_uses: Callable[[], Set[sympy.Symbol]]


@dataclasses.dataclass
class Operation:
    def __post_init__(self):
        self.operation_name: Optional[str] = None

    def get_device(self):
        raise NotImplementedError

    def get_origin_node(self):
        assert hasattr(self, "origin_node")
        return self.origin_node

    def get_origins(self):
        assert hasattr(self, "origins")
        return self.origins

    def get_operation_name(self) -> str:
        assert self.operation_name is not None
        return self.operation_name

    def is_extern(self):
        return False

    def is_no_op(self):
        return False

    def get_read_writes(self):
        raise NotImplementedError

    def is_user_of(self, name):
        return name in self.get_read_names()

    def get_read_names(self) -> Set[str]:
        return {dep.name for dep in self.get_reads()}

    def get_reads(self):
        return self.get_read_writes().reads

    def get_outputs(self) -> List[Buffer]:
        raise NotImplementedError

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        """
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
        """
        return set()

    def get_workspace_size(self):
        """
        Gets extra global memory size needed by this buffer.
        Some algorithms (e.g. group gemm) may require extra global memory in the generated code.
        """
        return 0


@dataclasses.dataclass
class Loops(IRNode):
    device: torch.device
    dtype: torch.dtype
    inner_fn: Callable[..., Any]
    ranges: List[Expr]

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return set().union(
            *(free_unbacked_symbols(e) for e in self.ranges),
            self.inner_fn_free_unbacked_symbols(),
        )

    def __str__(self, names=("ranges",)):
        return self.str_helper(
            [
                f"'{self.device.type}'",
                str(self.dtype),
                self.inner_fn_str(),
            ]
            + [f"{name}={getattr(self, name)}" for name in names]
            + [f"origin_node={self.origin_node!r}"]
        )

    def __post_init__(self):
        super().__post_init__()
        self.origin_node = None

    __repr__ = __str__

    def get_device(self):
        return self.device

    def get_origin_node(self):
        return self.origin_node

    def get_size(self):
        return self.ranges

    def get_pointwise_size(self):
        return self.ranges

    def is_extern(self):
        return False

    @classmethod
    def create(cls, *args, **kwargs):
        origin_node = kwargs.pop("origin_node", None)
        tb = kwargs.pop("traceback", None)
        r = cls(*args, **kwargs)
        r.origin_node = origin_node
        r.traceback = (
            tb or traceback.format_stack() if config.debug_ir_traceback else None
        )
        return TensorBox.create(r)

    @staticmethod
    def _index(ranges, prefix=SymT.INDEX):
        return [
            sympy.Integer(0) if s == 1 else sympy_index_symbol_with_prefix(prefix, n)
            for n, s in enumerate(ranges)
        ]

    @cache_on_self
    def inner_fn_opcount(self):
        opcounter = OpCounterCSE(V.MockHandler())

        with V.set_ops_handler(opcounter), patch.object(
            FlexibleLayout, "allow_indexing", True
        ):
            self.inner_fn(*self.inner_fn_args())
            return opcounter.op_count

    def inner_fn_args(self):
        return (self._index(self.ranges),)

    def inner_fn_str(self):
        return V.KernelFormatterHandler.ir_to_string(
            self.inner_fn, *self.inner_fn_args()
        )

    def has_large_inner_fn(self):
        return self.inner_fn_opcount() > config.realize_opcount_threshold

    def inner_fn_free_unbacked_symbols(self):
        index = self._index(self.ranges)
        return extract_free_unbacked_symbols(self.inner_fn, index)

    def get_reads(self):
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

    def get_read_names(self) -> Set[str]:
        return {dep.name for dep in self.get_reads()}

    def get_reduction_size(self):
        raise NotImplementedError(
            f"get_reduction_size() is not implemented by {type(self)}!"
        )

    def get_reduction_type(self):
        raise NotImplementedError(
            f"get_reduction_type() is not implemented by {type(self)}!"
        )

    def constant_to_device(self, device):
        raise NotImplementedError(
            f"constant_to_device() is not implemented by {type(self)}!"
        )


def nop_loader_fn(idx, *, dtype):
    if dtype.is_floating_point:
        return ops.constant(float("nan"), dtype)
    else:
        return ops.constant(0, dtype)


class Pointwise(Loops):
    def make_loader(self):
        # Make zero-element loops into a no-op
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.dtype)

        return self.inner_fn

    def get_reduction_size(self):
        return []

    def get_reduction_type(self):
        return None

    def store_output(self, output_name, indexer, vars):
        loader = self.make_loader()
        return ops.store(output_name, indexer(vars), loader(vars))

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(device, self.dtype, loader, self.ranges)


@dataclasses.dataclass
class Scatter(Pointwise):
    output_indexer: Callable[[List[Expr]], Expr]
    scatter_mode: Optional[str] = None

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Scatter(
            device,
            self.dtype,
            loader,
            self.ranges,
            self.output_indexer,
            self.scatter_mode,
        )

    def store_output(self, output_name, indexer, vars):
        loader = self.make_loader()
        return ops.store(
            output_name,
            indexer(self.output_indexer(vars)),
            loader(vars),
            mode=self.scatter_mode,
        )


REDUCTION_COMBINE_FN = {
    "any": ops_wrapper("logical_or"),
    "max": ops_wrapper("maximum"),
    "min": ops_wrapper("minimum"),
    "prod": ops_wrapper("mul"),
    "sum": ops_wrapper("add"),
    "xor_sum": ops_wrapper("bitwise_xor"),
}


def get_reduction_combine_fn(reduction_type, dtype, arg_break_ties_left=True):
    if reduction_type in REDUCTION_COMBINE_FN:
        combine_fn = REDUCTION_COMBINE_FN[reduction_type]
    elif reduction_type in {"argmax", "argmin"}:

        def combine_fn(a, b):
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

    elif reduction_type == "welford_combine":

        def combine_fn(a, b):
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

    else:
        raise NotImplementedError(f"unknown reduction_type={reduction_type}")

    return combine_fn


@dataclasses.dataclass
class Reduction(Loops):
    reduction_ranges: List[Expr]
    reduction_type: str
    # self.dtype represents the dst dtype
    src_dtype: torch.dtype
    reduction_hint: ReductionHint

    def __str__(self):
        return Loops.__str__(  # type: ignore[call-arg]
            self, names=("ranges", "reduction_ranges", "reduction_type")
        )

    def __repr__(self):
        return self.__str__()

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return super().get_unbacked_symbol_uses() | set().union(
            *(free_unbacked_symbols(e) for e in self.reduction_ranges)
        )

    def get_reduction_size(self):
        return self.reduction_ranges

    def get_reduction_type(self):
        return self.reduction_type

    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        value = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        return ops.store_reduction(output_name, indexer(vars), value)

    def index_length(self):
        return len(self.ranges) + len(self.reduction_ranges)

    def inner_fn_args(self):
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.RINDEX)
        return (index, rindex)

    def inner_fn_free_unbacked_symbols(self):
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.RINDEX)
        return extract_free_unbacked_symbols(self.inner_fn, index, rindex)

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Reduction(
            device,
            self.dtype,
            loader,
            self.ranges,
            self.reduction_ranges,
            self.reduction_type,
            self.src_dtype,
            ReductionHint.DEFAULT,
        )

    @staticmethod
    def num_splits(
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_numel,
        input_node: Optional[IRNode] = None,
    ):
        def _is_static(x):
            return isinstance(x, (int, sympy.Integer))

        reduction_numel_hint = V.graph.sizevars.symbolic_hint(reduction_numel)
        numel_hint = V.graph.sizevars.symbolic_hint(sympy_product(ranges))

        should_split = (
            not V.graph.has_feature(device, BackendFeature.REDUCE_TO_SINGLE_ELEMENT)
            and reduction_type
            not in {
                "argmax",
                "argmin",
            }
            and config.split_reductions
            # We don't support unbacked symints
            and _is_static(reduction_numel_hint)
            and _is_static(numel_hint)
        )
        if not should_split:
            return ReductionHint.DEFAULT, 1

        device_interface = get_interface_for_device(get_device_type(device))
        device_properties = device_interface.Worker.get_device_properties(device)
        if get_device_type(device) == "xpu":
            num_sm = device_properties.gpu_subslice_count
        else:
            # default is cuda behavior
            num_sm = device_properties.multi_processor_count

        min_elements_per_thread = 32
        max_elements_per_thread = 512
        threads_per_sm = 2048
        min_elements_per_device = min_elements_per_thread * num_sm * threads_per_sm
        max_elements_per_device = max_elements_per_thread * num_sm * threads_per_sm

        def inner_reduction_splits(reduction_numel_hint, numel_hint):
            # do heuristics that's close to eager mode for split inner reduction
            # we leak reduction autotune configs here, and will need to refactor to avoid this later
            num_warps = 8
            num_threads = 32 * num_warps
            if numel_hint >= 2 * num_sm:  # don't split if there are enough outputs
                return 1
            if reduction_numel_hint <= 8192:
                return 1
            if reduction_numel_hint * numel_hint <= min_elements_per_device:
                split_size = min_elements_per_thread
            elif reduction_numel_hint * numel_hint < max_elements_per_device:
                target_blocks = num_sm * threads_per_sm // (2 * num_threads)
                blocks_per_output = (target_blocks + numel_hint - 1) // numel_hint
                tmp_split_size = (
                    reduction_numel_hint + num_threads * blocks_per_output - 1
                ) // (num_threads * blocks_per_output)
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
                if abs(closest - tmp_split_size) < 30:
                    # prefer even splits, but never smalle than min_elements_per_thread
                    split_size = max(closest, min_elements_per_thread)
                else:
                    split_size = tmp_split_size
            else:
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
                if abs(closest - max_elements_per_thread) < 50:
                    # prefer even splits
                    split_size = closest
                else:
                    split_size = max_elements_per_thread
            return (reduction_numel_hint + split_size * num_threads - 1) // (
                split_size * num_threads
            )

        def outer_reduction_splits(reduction_numel_hint, numel_hint):
            # TODO the best heuristic currently has XBLOCK (corresponding to numel_hint) 128
            # extend to even smaller number of outputs
            num_warps = 8
            num_threads = num_warps * 32
            rvals_per_thread = 4  # comes from heuristics, refactor to not leak here
            xvals_per_block = 128
            xblocks = (numel_hint + xvals_per_block - 1) // xvals_per_block
            if reduction_numel_hint * numel_hint < min_elements_per_device:
                split_size = min_elements_per_thread
            elif reduction_numel_hint * numel_hint < max_elements_per_device:
                target_blocks = num_sm * threads_per_sm // (num_threads)
                target_blocks = (target_blocks + xblocks - 1) // xblocks
                tmp_split_size = (
                    reduction_numel_hint + rvals_per_thread * target_blocks - 1
                ) // (rvals_per_thread * target_blocks)
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
                if abs(tmp_split_size - closest) < 20:
                    split_size = max(closest, min_elements_per_thread)
                else:
                    split_size = tmp_split_size
            else:
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
                if abs(closest - max_elements_per_thread) < 50:
                    # prefer even splits
                    split_size = closest
                else:
                    split_size = max_elements_per_thread

            return (reduction_numel_hint + rvals_per_thread * split_size - 1) // (
                rvals_per_thread * split_size
            )

        # easy cases
        if numel_hint == 1:
            split = inner_reduction_splits(reduction_numel_hint, numel_hint)
            if split == 1:
                # No need to split.
                return ReductionHint.INNER, split
            if input_node is not None and isinstance(input_node, TensorBox):
                new_ranges, new_reduction_ranges = extract_input_node_reduction_ranges(
                    input_node
                )
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
            device,
            dst_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            src_dtype,
            ReductionHint.DEFAULT,
        )

        def get_read_indices(r):
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
            range_vars = [
                r
                for r in read_writes.range_vars
                if isinstance(r, sympy.Expr) and not isinstance(r, sympy.Number)
            ]
            indices = []
            changed = False
            for md in sorted(read_writes.reads, key=lambda x: x.name):
                if all(r in md.index.free_symbols for r in range_vars):
                    indices.append(md.index)
                    if md.name in V.graph.name_to_buffer:
                        buf = V.graph.name_to_buffer[md.name]
                        original_stride = buf.layout.stride
                        buf.decide_layout()
                        if buf.layout.stride != original_stride:
                            changed = True
            return indices, changed

        indices, changed = get_read_indices(r)
        if changed:
            indices, _ = get_read_indices(r)

        if len(indices) == 0:
            # TODO determine splits when all inputs are broadcast
            return ReductionHint.DEFAULT, 1

        (_, reduction_vars), ranges = dependencies.index_vars_squeeze(
            r.get_size(), r.get_reduction_size()
        )
        num_outer = 0
        num_inner = 0
        for i in indices:
            i = V.graph.sizevars.simplify_with_ranges(i, ranges)
            strides = V.graph.sizevars.stride_hints(i, reduction_vars, ranges.keys())
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
    def _unroll_reduction_fn(inner_fn, reduction_ranges, reduction_type, src_dtype):
        """Convert inner_fn from a reduction to an pointwise"""
        reduction_ranges = [
            V.graph.sizevars.evaluate_static_shape(x) for x in reduction_ranges
        ]

        combine_fn = get_reduction_combine_fn(reduction_type, src_dtype)

        def fn(index):
            return functools.reduce(
                combine_fn,
                (
                    value_fn(index, rindex)
                    for rindex in itertools.product(
                        *[range(x) for x in reduction_ranges]
                    )
                ),
            )

        if reduction_type in ("argmin", "argmax"):
            flatten_index = FixedLayout(
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                reduction_ranges,
                FlexibleLayout.contiguous_strides(reduction_ranges),
            ).make_indexer()

            def value_fn(index, rindex):
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
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: List[Expr],
        reduction_ranges: List[Expr],
        reduction_type: str,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: Optional[IRNode] = None,
    ):
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        if reduction_numel == 0:
            # N.B. This is a hack to generate the literal of the given type
            # Ideally, we should be fixing `def constant` in triton.py
            # but it breaks due to hardcoded dtypes in other places
            def py_cnst(val):
                return (
                    bool(val)
                    if dst_dtype == torch.bool
                    else float(val)
                    if dst_dtype.is_floating_point
                    else int(val)
                )

            rtypes_to_inits = {
                "sum": py_cnst(0),
                "xor_sum": py_cnst(0),
                "prod": py_cnst(1),
                "any": py_cnst(0),
                # "all" is desugared to `!any(!val)`
            }

            assert (
                reduction_type in rtypes_to_inits.keys()
            ), f"{reduction_type} not supported for zero-dimension tensors!"

            def const_fn(index):
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

                def fn(index):
                    return ops.constant(0, dst_dtype)

            else:

                def fn(index):
                    reduction_index = [sympy.Integer(0) for _ in reduction_ranges]
                    return inner_fn(index, reduction_index)

            return Pointwise.create(device, dst_dtype, fn, ranges)

        if (
            isinstance(reduction_numel, sympy.Integer)
            and V.graph.sizevars.size_hint(reduction_numel)
            < config.unroll_reductions_threshold
            and sympy_product(ranges) != 1
        ):
            return Pointwise.create(
                device,
                dst_dtype,
                cls._unroll_reduction_fn(
                    inner_fn, reduction_ranges, reduction_type, src_dtype
                ),
                ranges,
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
                input_node  # type: ignore[arg-type]
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
            )

        return TensorBox.create(
            Reduction(
                device,
                dst_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                src_dtype,
                reduction_hint,
            )
        )

    @staticmethod
    def default_accumulator(reduction_type, dtype):
        if reduction_type in {"max", "argmax"}:
            if is_float_dtype(dtype):
                return float("-inf")
            elif is_boolean_dtype(dtype):
                return 0
            else:
                return torch.iinfo(dtype).min
        if reduction_type in {"min", "argmin"}:
            if is_float_dtype(dtype):
                return float("inf")
            elif is_boolean_dtype(dtype):
                return 1
            else:
                return torch.iinfo(dtype).max

        return {
            "sum": 0,
            "prod": 1,
            "xor_sum": 0,
            "any": 0,
            "welford_reduce": (0, 0, 0),
            "welford_combine": (0, 0, 0),
        }[reduction_type]

    @staticmethod
    def default_value(reduction_type, dtype):
        if reduction_type == "welford_reduce":
            return 0
        return Reduction.default_accumulator(reduction_type, dtype)

    @staticmethod
    def _multilayer_second_step_hint(
        split: int, numel_hint: int, reduction_hint: ReductionHint
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
    def _multilayer_wrap_loader(
        cls,
        loader,
        reduction_ranges,
        reduction_numel,
        split,
        block_size,
        default,
    ):
        reindex = View.dynamic_reshape_indexer(reduction_ranges, [reduction_numel])
        need_mask = not V.graph.sizevars.is_expr_static_and_true(
            sympy.Eq(reduction_numel % split, 0)  # type: ignore[arg-type]
        )

        def wrapper_fn(index, reduction_index):
            (reduction_index,) = reduction_index
            *new_index, reduction_block = index
            indices = block_size * reduction_block + reduction_index

            def body():
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
        loader,
        original_ranges,
        original_reduction_ranges,
        new_ranges,
        new_reduction_ranges,
        default,
    ):
        assert all(
            r == 1 for r in original_ranges
        ), f"Only enabled for numel_hint == 1, found {original_ranges=}"
        reindex = View.dynamic_reshape_indexer(
            original_reduction_ranges, tuple(new_ranges) + tuple(new_reduction_ranges)
        )

        def wrapper_fn(merged_index, new_reduction_index):
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
        original_ranges: List[Expr],
        original_reduction_ranges: List[Expr],
        new_ranges: List[Expr],
        new_reduction_ranges: List[Expr],
        reduction_type: str,
        split: int,
        reduction_hint: ReductionHint,
    ):
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

        def intermediate_fn(index, reduction_index):
            return intermediate_loader([*index, *reduction_index])

        numel_hint = V.graph.sizevars.size_hint(sympy_product(original_ranges))
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )

        assert original_ranges == new_ranges[: len(original_ranges)]
        return TensorBox.create(
            Reduction(
                device,
                dst_dtype,
                intermediate_fn,
                original_ranges,
                new_ranges[len(original_ranges) :],
                reduction_type,
                src_dtype,
                reduction_hint,
            )
        )

    @classmethod
    def create_multilayer(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: List[Expr],
        reduction_ranges: List[Expr],
        reduction_type: str,
        split: int,
        reduction_hint: ReductionHint,
    ):
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        # TODO(jansel): realize the reduction so we can do dynamic indexing
        reduction_numel = sympy_product(reduction_ranges)
        block_size = FloorDiv(reduction_numel + (split - 1), split)
        default = cls.default_value(reduction_type, dst_dtype)
        wrapper_fn = cls._multilayer_wrap_loader(
            inner_fn, reduction_ranges, reduction_numel, split, block_size, default
        )

        return cls.create_multilayer_helper(
            device,
            dst_dtype,
            src_dtype,
            wrapper_fn,
            ranges,
            reduction_ranges,
            [*ranges, split],  # type: ignore[list-item]
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
        original_ranges: List[Expr],
        original_reduction_ranges: List[Expr],
        new_ranges: List[Expr],
        new_reduction_ranges: List[Expr],
        reduction_type: str,
        reduction_hint: ReductionHint,
    ):
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        default = cls.default_value(reduction_type, dst_dtype)
        wrapper_fn = cls._multilayer_wrap_loader_existing_ranges(
            inner_fn,
            original_ranges,
            original_reduction_ranges,
            new_ranges,
            new_reduction_ranges,
            default,
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


def num_reduction_outputs(reduction_type):
    return 3 if "welford" in reduction_type else 1


class WelfordReduction(Reduction):
    output_index: int

    def __init__(
        self,
        device,
        dtype,
        inner_fns,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_hint,
        output_index,
    ):
        if len(inner_fns) == 1:
            loader = inner_fns[0]
        else:

            def loader(idx, reduction_idx):
                return tuple(fn(idx, reduction_idx) for fn in inner_fns)

        super().__init__(
            device,
            dtype,
            loader,
            ranges,
            reduction_ranges,
            reduction_type,
            dtype,
            reduction_hint,
        )
        self.output_index = output_index

    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        values = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        value = values[self.output_index]
        return ops.store_reduction(output_name, indexer(vars), value)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fns: Sequence[Callable[..., Any]],
        ranges: List[Expr],
        reduction_ranges: List[Expr],
        reduction_type: str,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
    ):
        assert reduction_type in {"welford_reduce", "welford_combine"}

        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        def const(val):
            def inner_fn(idx):
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

            def copy(loader):
                def inner_fn(idx):
                    reduction_index = [sympy.Integer(0) for _ in reduction_ranges]
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
        #     isinstance(reduction_numel, sympy.Integer)
        #     and V.graph.sizevars.size_hint(reduction_numel)
        #     < config.unroll_reductions_threshold
        #     and sympy_product(ranges) != 1
        # ):
        #     return Pointwise.create(
        #         device,
        #         dst_dtype,
        #         cls._unroll_reduction_fn(
        #             inner_fn, reduction_ranges, reduction_type, src_dtype
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
    def default_value(reduction_type, dtype):
        return (0, 0, 0)

    @classmethod
    def create_multilayer(  # type: ignore[override]
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fns: Sequence[Callable[..., Any]],
        ranges: List[Expr],
        reduction_ranges: List[Expr],
        reduction_type: str,
        split: int,
        reduction_hint: ReductionHint,
    ):
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        reduction_numel = sympy_product(reduction_ranges)
        need_mask = not V.graph.sizevars.is_expr_static_and_true(
            sympy.Eq(reduction_numel % split, 0)  # type: ignore[arg-type]
        )

        if need_mask and reduction_type != "welford_combine":
            # If we need mask, then "welford_reduce" doesn't work because
            # masked inputs shouldn't count towards the welford weight

            def constant(idx, reduction_idx, value):
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
            [*ranges, split],  # type: ignore[list-item]
            [block_size],
            reduction_type,
            reduction_hint,
        )
        for i in intermediates:
            i.realize()

        i_loaders = [i.make_loader() for i in intermediates]

        def intermediate_loader_fn(index, reduction_index, loader):
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
            [split],  # type: ignore[list-item]
            # welford_reduce turns one input into three outputs, which are combined with welford_combine
            "welford_combine",
            reduction_hint,
        )


@dataclasses.dataclass
class Scan(Loops):
    scan_ranges: List[Expr]
    size: List[Expr]
    combine_fn: Callable[[Tuple[Any, ...], Tuple[Any, ...]], Tuple[Any, ...]]
    reindex: Callable[[List[Expr], List[Expr]], List[Expr]]
    reduction_hint: ReductionHint
    output_index: int
    # output_index indexes the following tuples
    dtypes: Tuple[torch.dtype, ...]
    inner_fns: Tuple[Callable[..., Any], ...]

    # HACK we mimick reduction

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        # TODO: Can combine_fn/reindex close over unbacked symbols? If so, we
        # need to explicitly represent the closure so we can pull out unbacked
        # symbols here
        return (
            super().get_unbacked_symbol_uses()
            | set().union(*(free_unbacked_symbols(e) for e in self.scan_ranges))
            | set().union(*(free_unbacked_symbols(e) for e in self.size))
        )

    def __post_init__(self):
        assert len(self.ranges) + len(self.scan_ranges) == len(self.size)
        super().__post_init__()

    def store_reduction(self, output_name, indexer, vars, scan_vars):
        idx = self.reindex(vars, scan_vars)
        values = [inner_fn(idx) for inner_fn in self.inner_fns]
        result = ops.scan(self.dtypes, self.combine_fn, values)
        return ops.store(output_name, indexer(idx), result[self.output_index])

    def get_reduction_type(self):
        # return self.scan_op
        return "custom"

    def get_reduction_size(self):
        return self.scan_ranges

    def get_size(self):
        return self.size

    def get_pointwise_size(self):
        return self.ranges

    def index_length(self):
        return len(self.ranges) + len(self.scan_ranges)

    def inner_fn_args(self):
        index = self._index(self.ranges)
        rindex = self._index(self.scan_ranges, SymT.RINDEX)
        idx = self.reindex(index, rindex)
        return (idx,)

    def inner_fn_free_unbacked_symbols(self):
        index = self._index(self.ranges)
        rindex = self._index(self.scan_ranges, SymT.RINDEX)
        idx = self.reindex(index, rindex)
        return extract_free_unbacked_symbols(self.inner_fn, idx)

    @classmethod
    def create(
        cls,
        device: torch.device,
        dtypes: Tuple[torch.dtype, ...],
        inner_fns: Tuple[Callable[[List[Expr]], Any], ...],
        size: List[Expr],
        axis: int,
        combine_fn: Callable[[Tuple[Any, ...], Tuple[Any, ...]], Tuple[Any, ...]],
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        *,
        # Whether we have the option to fallback to aten
        can_fallback_to_aten: bool = True,
        **kwargs,
    ) -> List[Optional[TensorBox]]:
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
        if sizevars.is_expr_static_and_true(sympy.Le(scan_numel, 1)):  # type: ignore[arg-type]
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
            supports_split = torch.version.hip is None and len(dtypes) == 1
            if not supports_split:
                if can_fallback_to_aten:
                    # Fallback to ATen
                    return [None] * len(dtypes)
                else:
                    num_splits = 1
            else:
                scan_type = SplitScan

        def reindex(index, scan_index):
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
        inner_fn: Callable[[List[Expr]], Any],
        axis: int,
        pointwise_ranges: List[Expr],
        scan_ranges: List[Expr],
        combine_fn: Callable[[Tuple[Any, ...], Tuple[Any, ...]], Tuple[Any, ...]],
        scan_numel: Expr,
    ):
        # TODO: custom splitting heuristic for scan
        def wrapper_fn(idx, reduction_idx):
            return inner_fn([*idx[:axis], *reduction_idx, *idx[axis:]])

        return Reduction.num_splits(
            device=device,
            dst_dtype=dtype,
            src_dtype=dtype,
            inner_fn=wrapper_fn,
            ranges=pointwise_ranges,
            reduction_ranges=scan_ranges,
            reduction_type="sum",
            reduction_numel=scan_numel,
        )


# This signifies a scan op that should go through TritonSplitScanKernel codegen on CUDA.
@dataclasses.dataclass
class SplitScan(Scan):
    pass


@dataclasses.dataclass
class Sort(Loops):
    # Sorts a tuple of key, value pairs
    sort_ranges: List[Expr]
    size: List[Expr]
    reindex: Callable[[List[Expr], List[Expr]], List[Expr]]
    reduction_hint: ReductionHint
    output_index: int
    # output_index indexes the following tuples
    dtypes: Tuple[torch.dtype, ...]
    inner_fns: Tuple[Callable[..., Any], ...]

    stable: bool
    descending: bool

    # HACK we mimick reduction

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return (
            super().get_unbacked_symbol_uses()
            | set().union(*(free_unbacked_symbols(e) for e in self.sort_ranges))
            | set().union(*(free_unbacked_symbols(e) for e in self.size))
        )

    def __post_init__(self):
        assert len(self.ranges) + len(self.sort_ranges) == len(self.size)
        super().__post_init__()

    def store_reduction(self, output_name, indexer, vars, sort_vars):
        idx = self.reindex(vars, sort_vars)
        values = [inner_fn(idx) for inner_fn in self.inner_fns]
        result = ops.sort(self.dtypes, values, self.stable, self.descending)
        return ops.store(output_name, indexer(idx), result[self.output_index])

    def get_reduction_type(self):
        return "sort"

    def get_reduction_size(self):
        return self.sort_ranges

    def get_size(self):
        return self.size

    def get_pointwise_size(self):
        return self.ranges

    def index_length(self):
        return len(self.ranges) + len(self.sort_ranges)

    def inner_fn_args(self):
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.RINDEX)
        idx = self.reindex(index, rindex)
        return (idx,)

    def inner_fn_free_unbacked_symbols(self):
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.RINDEX)
        idx = self.reindex(index, rindex)
        return extract_free_unbacked_symbols(self.inner_fn, idx)

    @classmethod
    def create(
        cls,
        device: torch.device,
        dtypes: Tuple[torch.dtype, ...],
        inner_fns: Tuple[Callable[[List[Expr]], Any], ...],
        size: List[Expr],
        axis: int,
        stable: bool,
        descending: bool,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        **kwargs,
    ) -> List[Optional[TensorBox]]:
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
        if sizevars.is_expr_static_and_true(sympy.Le(sort_numel, 1)):  # type: ignore[arg-type]
            return [
                Pointwise.create(
                    device=device,
                    dtype=dtypes[output_index],
                    inner_fn=inner_fns[output_index],
                    ranges=size,
                )
                for output_index in range(len(dtypes))
            ]

        def reindex(index, sort_index):
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


def is_storage_and_layout(x):
    try:
        as_storage_and_layout(x, freeze=False)
        return True
    except NotImplementedError:
        return False


def is_contiguous_storage_and_layout(x):
    try:
        buffer, layout = as_storage_and_layout(x, freeze=False)
        # pad the stride here so we will NOT claim an tensor as contiguous
        # if a padding is gonna happen.
        if layout.should_pad_strides():
            layout.pad_strides()
        return layout.is_contiguous()
    except NotImplementedError:
        return False


def as_storage_and_layout(
    x, freeze=True, want_contiguous=False, stride_order=None, allow_padding=False
):
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
        )
    if isinstance(x, StorageBox) and isinstance(x.data, Buffer):
        if freeze:
            if want_contiguous:
                x.data.freeze_layout()
                assert x.data.layout.is_contiguous()
            elif stride_order is not None:
                x.data.freeze_layout_with_stride_order(
                    stride_order, allow_padding=allow_padding
                )
            else:
                x.data.decide_layout()
        return x, x.data.layout
    if isinstance(x, ReinterpretView):
        # making the base of x contiguous or stride_ordered will not necessarily make
        # the ReinterpretView either, so don't pass along those arguments
        buffer, _ = as_storage_and_layout(
            x.data,
            freeze=freeze,
        )
        return buffer, x.layout
    raise NotImplementedError


as_contiguous_storage_and_layout = functools.partial(
    as_storage_and_layout, want_contiguous=True
)


def is_stride_order_storage_and_layout(x, stride_order):
    try:
        buffer, layout = as_storage_and_layout(x, freeze=False)
        return layout.is_stride_ordered(stride_order)
    except NotImplementedError:
        return False


@dataclasses.dataclass
class BaseView(IRNode):
    data: IRNode

    def get_unbacked_symbol_uses(self):
        return self.data.get_unbacked_symbol_uses()

    def make_reindexer(self):
        raise NotImplementedError(f"make_reindexer NYI on {self}")

    def make_indexer(self):
        inner = self.data.make_indexer()
        reindex = self.make_reindexer()

        def indexer(idx):
            return inner(reindex(idx))

        return indexer

    def make_loader(self):
        inner = self.data.make_loader()
        reindex = self.make_reindexer()

        def loader(idx):
            return inner(reindex(idx))

        return loader

    @property
    def dtype(self):
        return self.data.dtype

    def get_layout(self):
        return self.data.get_layout()

    def get_device(self):
        return self.data.get_device()

    def get_origin_node(self):
        return None

    def get_name(self):
        return self.data.get_name()

    def get_pointwise_size(self):
        return self.get_size()

    def mark_reuse(self, users):
        return self.data.mark_reuse(users)

    def has_exceeded_max_reads(self):
        return self.data.has_exceeded_max_reads()

    def realize(self):
        return self.data.realize()

    def realize_hint(self):
        return self.data.realize_hint()

    def get_storage_numel(self):
        return self.data.get_storage_numel()

    def is_extern(self):
        return self.data.is_extern()  # type: ignore[attr-defined]

    def is_module_buffer(self):
        return self.data.is_module_buffer()  # type: ignore[attr-defined]

    def get_read_names(self) -> Set[str]:
        return self.data.get_read_names()

    def get_reads(self):
        with patch.object(FlexibleLayout, "allow_indexing", True):
            return extract_read_writes(
                self.make_loader(),
                self.get_size(),
            ).reads

    def unwrap_view(self):
        x: IRNode = self
        while isinstance(x, BaseView):
            x = x.data
        return x

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(device, self.get_dtype(), loader, self.get_size())


@dataclasses.dataclass
class ExpandView(BaseView):
    size: List[Expr]

    @staticmethod
    def _normalize_size(x, new_size):
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
            elif old_size[i] is None or old_size[i] == 1:
                pass
            else:
                # Sanity check: Expect broadcast compatibility
                #
                # NB: new_size[i] == old_size[i] is expected to already be
                # guarded because the meta formula was expected to have taught
                # us this equality.
                assert (
                    sizevars.size_hint(new_size[i] - old_size[i], fallback=0) == 0
                ), "Broadcast failed in ExpandView({x.get_size()}, {new_size}) on dimension {i}"
        return new_size

    @classmethod
    def create(cls, x, new_size):
        new_size = cls._normalize_size(x, new_size)

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            skip = len(new_size) - len(old_layout.size)
            assert skip >= 0
            new_stride = [sympy.Integer(0)] * skip
            for stride, size in zip(old_layout.stride, old_layout.size):
                new_stride.append(stride if size != 1 else sympy.Integer(0))
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                list(new_size),
                new_stride,
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        return ExpandView(x, new_size)

    def get_size(self):
        return self.size

    def make_reindexer(self):
        target = self.get_size()
        actual = self.data.get_size()
        skip = len(target) - len(actual)

        def reindex(index):
            index = list(index[skip:])
            assert len(index) == len(actual)
            for i in range(len(actual)):
                if actual[i] == 1:
                    # zero out broadcast dimension
                    index[i] = sympy.Integer(0)
            return index

        return reindex


@dataclasses.dataclass
class PermuteView(BaseView):
    dims: List[Expr]

    @classmethod
    def create(cls, x, dims):
        dims = cls._map_neg_dims(dims)
        assert set(dims) == set(range(len(dims)))

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                [old_layout.size[i] for i in dims],
                [old_layout.stride[i] for i in dims],
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        return PermuteView(x, dims)

    @classmethod
    def _map_neg_dims(cls, dims):
        return [dim if dim >= 0 else len(dims) + dim for dim in dims]

    def get_size(self):
        assert set(self._map_neg_dims(self.dims)) == set(range(len(self.dims)))
        size = self.data.get_size()
        return [size[i] for i in self.dims]

    def make_reindexer(self):
        inv = {j: i for i, j in enumerate(self.dims)}
        inv = [inv[i] for i in range(len(self.dims))]  # type: ignore[index]
        assert set(inv) == set(range(len(self.dims)))

        def reindex(index):
            return [index[i] for i in inv]

        return reindex


class SqueezeView(BaseView):
    @classmethod
    def create(cls, x, *, dim=None):
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
            return ReinterpretView(storage, new_layout)

        if dim is None:
            # redirect to a generic view
            return View.create(x, [s for s in x.get_size() if s != 1])
        else:
            assert x.get_size()[dim] == 1
            return View.create(x, [s for i, s in enumerate(x.get_size()) if i != dim])

    @staticmethod
    def squeezer(size: Tuple[sympy.Expr, ...]):
        new_size = [s for s in size if s != 1]
        not_one = [i for i, s in enumerate(size) if s != 1]
        length = len(size)

        def reindex(index: List[sympy.Expr]) -> Tuple[sympy.Expr, ...]:
            assert len(index) == len(not_one), f"{index} {not_one}"
            new_index = [sympy.Integer(0)] * length
            for idx, s in zip(not_one, index):
                new_index[idx] = s
            return tuple(new_index)

        return new_size, reindex

    def __init__(self, data):
        raise AssertionError("use SqueezeView.create()")


@dataclasses.dataclass
class GenericView(BaseView):
    size: List[Expr]
    reindex: Callable[..., Any]

    def make_reindexer(self):
        return self.reindex

    def reindex_str(self):
        index_old = [
            sympy_index_symbol_with_prefix(SymT.INDEX, n) for n in range(len(self.size))
        ]
        index_new = list(self.reindex(index_old))
        return f"lambda {', '.join(map(str, index_old))}: {index_new}"

    def __str__(self):
        return self.str_helper(
            [self.data, f"size={self.size}", f"reindex={self.reindex_str()}"]
        )

    __repr__ = __str__

    @classmethod
    def create(cls, x, new_size, reindex):
        return cls(x, list(new_size), reindex)

    def get_size(self):
        return self.size


@dataclasses.dataclass
class View(GenericView):
    @staticmethod
    def handle_negative_index(idx, size):
        idx = sympy.expand(idx)
        size = sympy.expand(size)
        evaluate_expr = V.graph.sizevars.shape_env.evaluate_expr
        if evaluate_expr(sympy.Lt(idx, 0)):
            idx = idx + size
        return idx

    @classmethod
    def create(cls, x, new_size):
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

            def fake_reindex(index):
                return tuple([0] * len(old_size))

            return cls(x, list(new_size), fake_reindex)
        # TODO: a new class for FixedTransferLayout that output layout is constrained by input layout
        elif is_contiguous_storage_and_layout(x) or unbacked_symbols_in_sizes:
            if unbacked_symbols_in_sizes and (not is_contiguous_storage_and_layout(x)):
                # realize x; otherwise, the dynamic_reshape_indexer below will fail
                # due to the size_hint's inability to process unbacked SymInts
                x = ExternKernel.realize_input(x)

            storage, old_layout = as_contiguous_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                FlexibleLayout.contiguous_strides(new_size),
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        reindex = cls.dynamic_reshape_indexer(old_size, new_size)
        return cls(x, list(new_size), reindex)

    @staticmethod
    def resolve_negative_size(old_size, new_size):
        new_size = [V.graph.sizevars.simplify(x) for x in new_size]
        old_size = [V.graph.sizevars.simplify(x) for x in old_size]

        new_size = list(new_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                new_size[i] = sympy.Integer(1)
                new_size[i] = CleanDiv(sympy_product(old_size), sympy_product(new_size))
                break

        V.graph.sizevars.guard_equals(sympy_product(old_size), sympy_product(new_size))
        return old_size, new_size

    @classmethod
    def dynamic_reshape_indexer(cls, old_size, new_size):
        try:
            reindex = cls._dynamic_reshape_indexer(old_size, new_size)
        except (AssertionError, IndexError):
            # optimistic algorithm failed, lets do a fallback
            flat = [sympy_product(old_size)]
            reindex1 = cls._dynamic_reshape_indexer(old_size, flat)
            reindex2 = cls._dynamic_reshape_indexer(flat, new_size)
            reindex = fuse_reindexing(reindex1, reindex2)
        return reindex

    @staticmethod
    def _dynamic_reshape_indexer(old_size, new_size):
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

        view_expr = []
        while stack_new and stack_old:
            size_old = stack_old.pop()
            var, size_new = stack_new.pop()
            if size_old == 1:
                view_expr.append(sympy.Integer(0))
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
                divisor = sympy.Integer(1)
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
            V.graph.sizevars.guard_equals(size_old, 1)  # type: ignore[arg-type]
            view_expr.append(sympy.Integer(0))

        while stack_new:
            var, size_new = stack_new.pop()
            V.graph.sizevars.guard_equals(size_new, 1)  # type: ignore[arg-type]

        view_expr.reverse()
        assert len(view_expr) == len(old_size)

        def reindex(index):
            assert len(index) == len(vars), (len(index), len(vars))
            replacements = dict(zip(vars, index))
            return tuple(sympy_subs(x, replacements) for x in view_expr)  # type: ignore[arg-type]

        return reindex


@dataclasses.dataclass
class ReinterpretView(BaseView):
    """Pretend our storage has a different layout"""

    layout: Layout

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.data, BaseView):
            self.data = self.data.unwrap_view()

    def __str__(self):
        return self.str_helper(
            [
                self.data,
                self.layout,
            ]
        )

    __repr__ = __str__

    def get_name(self):
        return self.data.get_name()

    def get_device(self):
        return self.layout.device

    def get_origin_node(self):
        return None

    @property
    def dtype(self):
        return self.layout.dtype

    def get_size(self):
        return list(self.layout.size)

    def get_stride(self):
        return list(self.layout.stride)

    def make_loader(self):
        def loader(index):
            indexer = self.layout.make_indexer()
            tmp_loader = ops.load(self.get_name(), indexer(index))
            if self.layout.dtype != self.data.dtype:
                return ops.to_dtype_bitcast(tmp_loader, self.dtype, self.data.dtype)
            else:
                return tmp_loader

        return loader

    def make_indexer(self):
        return self.layout.make_indexer()

    def get_layout(self):
        return self.layout

    def freeze_layout(self):
        pass

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return (
            free_unbacked_symbols(self.layout.size)
            | free_unbacked_symbols(self.layout.stride)
            | free_unbacked_symbols(self.layout.offset)
        )

    def codegen_reference(self, writer=None):
        # reinterpret_tensor is similar to as_strided except:
        # - offset is added to the existing offset (rather than replacing it)
        # - view tracking is disabled similar to unsafe_view
        return V.graph.wrapper_code.codegen_reinterpret_view(
            self.data,
            self.layout.size,
            self.layout.stride,
            self.layout.offset,
            writer,
            dtype=self.layout.dtype,
        )


@dataclasses.dataclass
class DtypeView(BaseView):
    """Pretend our storage has a different type"""

    target_dtype: torch.dtype

    @classmethod
    def create(cls, x, new_dtype):
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                new_dtype,
                old_layout.size,
                old_layout.stride,
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)
        return DtypeView(x, new_dtype)

    def __str__(self):
        return self.str_helper([self.data, self.target_dtype])

    __repr__ = __str__

    @property
    def dtype(self):
        return self.target_dtype

    def get_size(self):
        return self.data.get_size()

    def make_loader(self):
        inner = self.data.make_loader()

        def loader(idx):
            return ops.to_dtype_bitcast(inner(idx), self.target_dtype, self.data.dtype)

        return loader


class SliceView(View):
    @classmethod
    def normalize_start_end(cls, x, dim, start, end):
        """
        Normalize start and end such that both are in the range
        [0, x.get_size()[dim]] and start <= end.
        """
        sizevars = V.graph.sizevars
        dim_size = x.get_size()[dim]

        if any(free_unbacked_symbols(x) for x in (start, end, dim_size)):

            def clamp(x, lower, upper):
                return sympy.Min(sympy.Max(x, lower), upper)

        else:

            def clamp(x, lower, upper):
                return sizevars.evaluate_min(sizevars.evaluate_max(x, lower), upper)

        def clamp_wrap(val, lower, upper, default):
            if val is None:
                return default
            val = cls.handle_negative_index(val, dim_size)
            return clamp(val, lower, upper)

        start = clamp_wrap(start, 0, dim_size, 0)
        end = clamp_wrap(end, start, dim_size, dim_size)
        return start, end

    @classmethod
    def create(cls, x, dim, start, end, step=1, clamp=True):
        step = sympy.expand(step)
        assert isinstance(step, sympy.Expr) or step > 0
        try:
            if start == 0 and end >= 2**63 - 1 and step == 1:
                return x
        except TypeError:
            pass

        sizevars = V.graph.sizevars
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
            return ReinterpretView(storage, new_layout)

        def reindex(index):
            assert len(index) == len(new_size), f"wrong ndim {index} {new_size}"
            index = list(index)
            index[dim] = index[dim] * step + start
            return index

        # redirect to a generic view
        return SliceView(x, size=new_size, reindex=reindex)


class BaseConstant(IRNode):
    dtype: torch.dtype
    device: torch.device

    def get_size(self):
        return ()

    def get_device(self):
        return self.device

    def get_origin_node(self):
        return None

    def mark_reuse(self, users):
        pass

    def has_exceeded_max_reads(self):
        return False

    def get_reads(self):
        return ()

    def is_extern(self):
        return False


@dataclasses.dataclass
class Constant(BaseConstant):
    value: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self):
        def loader(index):
            return ops.constant(self.value, self.dtype)

        return loader

    def realize(self):
        pass

    def constant_to_device(self, device):
        return Constant(self.value, self.dtype, device)


@dataclasses.dataclass
class IndexingConstant(BaseConstant):
    index: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self):
        def loader(index):
            return ops.index_expr(self.index, self.dtype)

        return loader

    def constant_to_device(self, device):
        return IndexingConstant(self.index, self.dtype, device)


def is_contiguous_strides_for_shape(stride, shape):
    return all(
        size == 1 or left == right
        for left, right, size in zip(
            stride, FlexibleLayout.contiguous_strides(shape), shape
        )
    )


def get_align_for_dtype(dtype):
    """
    CUDA max memory transaction size is 128 bytes for a warp.
    We pick `128 // dtype.itemsize` as alighment so GPU can do coalesced
    memory access.
    """
    return 128 // dtype.itemsize


@dataclasses.dataclass
class Layout(IRNode):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: List[Expr],
        stride: Optional[Sequence[Union[Expr, int]]],
        offset: Expr = Integer(0),
    ):
        assert stride is None or len(size) == len(
            stride
        ), f"size={size}, stride={stride}"
        self.device = device
        self.dtype = dtype
        assert all(isinstance(s, (Expr, int)) for s in size)
        self.size = size
        self._stride = stride
        self.offset = offset

    @property
    def stride(self):
        return self._stride

    def __str__(self):
        offset = ""
        if self.offset != 0:
            offset = f", offset={self.offset}"
        return (
            f"{type(self).__name__}('{self.device.type}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}{offset})"
        )

    __repr__ = __str__

    def is_contiguous(self):
        return is_contiguous_strides_for_shape(self.stride, self.size)

    @staticmethod
    def is_channels_last_contiguous(shape, strides):
        ndim = len(shape)
        if ndim not in [4, 5] or shape[1] == 1:
            return False
        for left, right, size in zip(
            strides, make_channels_last_strides_for(shape), shape  # type: ignore[arg-type]
        ):
            if size != 1 and left != right:
                return False
        return True

    def is_transposed(self):
        for left, right, size in zip(
            self.stride,
            reversed(FlexibleLayout.contiguous_strides(list(reversed(self.size)))),
            self.size,
        ):
            if size != 1 and left != right:
                return False
        return True

    def is_stride_ordered(self, order):
        assert len(self.stride) == len(order)

        # ignore dimensions of size 1, they dont affect layout
        non_1_indices = [
            i
            for i, dim in enumerate(self.size)
            if V.graph.sizevars.size_hint(dim, fallback=2) != 1
        ]

        stride = [self.stride[i] for i in non_1_indices]
        order = [order[i] for i in non_1_indices]

        def sorted_indices(arr):
            sorted_arr = sorted(arr)
            return [sorted_arr.index(element) for element in arr]

        # since we may have removed dimensions, need to re-sort & re-index order
        order = sorted_indices(order)

        # reorder the stride given order
        stride_ordered = [-1] * len(order)
        for i in range(len(order)):
            stride_ordered[order[i]] = V.graph.sizevars.size_hint(stride[i])
        # check if it is in ascending order
        for i in range(len(order) - 1):
            if stride_ordered[i] > stride_ordered[i + 1]:
                return False
        return True

    def is_channels_last_stride_ordered(self):
        # create channels_last order(NCHW, NCDHW, the C is the first order).
        order = [0] + list(reversed(range(1, len(self.stride) - 1)))
        order = [len(order)] + order
        return self.is_stride_ordered(order)

    @staticmethod
    def _pad_strides(in_strides, size, dtype):
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

        # Don't align a too small stride since that causes too much memory increase.
        # Pad too small stride may also cause perf loss. We may result in many tiny data blocks
        # with gaps in between. That causes less coalesced GPU memory access!
        #
        # Initially we pick 320 as the threshold since for alignement=16,
        # that results in at most 5% memory cost.
        #
        # But later on we raise the threshold to 1024 to avoid interfere with persistent reduction.
        # Let's say an inner reduction has a row size 513. Inductor will generate
        # persistent reduction code.
        # If we do padding, the strides are not contiguous any more. Inductor
        # uses a much smaller threshold for persistent reduction in this case and
        # generates potentially worse non-persistent reduction code.
        #
        # This change turns HF AllenaiLongformerBase amp training from a loss of 1.09x to a win of 1.05x.
        # (baseline: 71.09ms, padding w/o this change: 77.38ms, padding with this change: 67.77ms)
        align_stride_threshold = 1024
        padded = False
        for rank, idx in enumerate(fill_order[1:], start=1):
            prev_idx = fill_order[rank - 1]
            stride = new_strides[prev_idx] * size[prev_idx]

            if stride > align_stride_threshold and stride % align != 0:
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

    def pad_strides(self):
        assert isinstance(self, FlexibleLayout)
        assert self._stride is not None
        self._stride = self._pad_strides(self._stride, self.size, self.dtype)

    def should_pad_strides(self):
        return config.comprehensive_padding and isinstance(self, FlexibleLayout)

    def as_fixed(self):
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

    def make_indexer(self):
        assert (
            FlexibleLayout.allow_indexing
        ), f"convert {type(self).__name__} to FixedLayout first"
        return self.as_fixed().make_indexer()

    def __eq__(self, other) -> bool:
        return (
            self.device == other.device
            and self.dtype == other.dtype
            and self.size == other.size
            and self.stride == other.stride
            and self.offset == other.offset
        )

    def storage_size(self) -> sympy.Expr:
        return compute_required_storage_length(self.size, self.stride, self.offset)  # type: ignore[arg-type, return-value]


class FixedLayout(Layout):
    """A Tensor layout we cannot change"""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: Union[List[Expr], List[int]],
        stride: Optional[Sequence[Union[Expr, int]]] = None,
        offset: Union[Expr, int] = Integer(0),
    ):
        if stride is None:
            stride = FlexibleLayout.contiguous_strides(size)
        super().__init__(
            device,
            dtype,
            size,  # type: ignore[arg-type]
            stride,
            offset,  # type: ignore[arg-type]
        )

    def make_indexer(self):
        """A closure containing math to read a given element"""

        def indexer(index):
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
    def contiguous_strides(sizes):
        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.Integer(1)]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    @staticmethod
    def fill_ordered(sizes, order):
        """
        Create a stride based on the order the dimensions should be filled in.

        In this format, channels last would be:
            [1, 3, 2, 0]
        """
        assert set(range(len(sizes))) == set(order), (sizes, order)
        next_stride = sympy.Integer(1)
        strides = [None] * len(order)

        for i in order:
            strides[i] = next_stride
            next_stride = next_stride * sizes[i]
        return strides

    @staticmethod
    def stride_ordered(sizes, order):
        """
        Create a stride based on the sorted order of a permuted range.

        In this format, channels last would be:
            [3, 0, 2, 1]
        """
        assert set(range(len(sizes))) == set(order)
        fill_order = stride_order2fill_order(order)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @staticmethod
    def stride_ordered_for_memory_format(sizes, memory_format):
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
    def same_ordered(sizes, stride):
        """
        Create a stride that has the same stride order as given stride

        For example, if given stride is [1000, 1, 100, 10],
        the fill order should be [1, 3, 2, 0]
        """
        assert len(sizes) == len(stride)
        stride = [V.graph.sizevars.size_hint(x) for x in stride]
        fill_order = sorted(range(len(stride)), key=stride.__getitem__)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    def as_stride_order(self, order, allow_padding=False):
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

    def as_fill_order(self, order):
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

    def as_same_order(self, stride):
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

    def __init__(self, device, dtype, size, stride_order=None):
        if stride_order:
            strides = FlexibleLayout.fill_ordered(size, stride_order)
        else:
            strides = FlexibleLayout.contiguous_strides(size)
        super().__init__(device, dtype, size, strides)


class NonOwningLayout(Layout):
    """Is a view into the storage of another tensor"""

    def __init__(self, view: Union[BaseView, TensorBox]):
        layout = view.get_layout()
        super().__init__(
            layout.device,
            layout.dtype,
            layout.size,
            layout.stride,
        )
        self.view = view

    def make_indexer(self):
        return self.as_fixed().make_indexer()

    def maybe_guard_aligned(self):
        offset = self.view.get_layout().offset
        if offset == 0:
            return True
        from .compile_fx import ALIGNMENT

        return V.graph.sizevars.statically_known_multiple_of(offset, ALIGNMENT)  # type: ignore[arg-type]


class NoneLayout(IRNode):
    # This is janky, I figured out what fields to populate by just running
    # the model I was interested in and adding properties/methods as needed.
    # This doesn't inherit from Layout because Layout assumes you have stuff
    # like sizes, but I don't really have anything here.
    #
    # If you have an ir.Node with NoneLayout, you probably need to setup
    # dependencies manually in scheduler

    def __init__(self, device):
        self.device = device
        self.size = [0]
        self.stride = [0]

    def storage_size(self):
        return 0

    def as_fixed(self):
        return self


class MutationLayoutSHOULDREMOVE(Layout):
    def __init__(self, target: IRNode):
        super().__init__(
            target.get_device(),
            target.get_dtype(),
            target.get_size(),
            None,
        )
        self.target = target
        name = self.get_buffer().get_name()
        V.graph.mark_buffer_mutated(name)

    @Layout.stride.getter  # type: ignore[attr-defined]
    def stride(self):
        return self.real_layout().stride

    def storage_size(self) -> sympy.Expr:
        return self.real_layout().storage_size()

    def get_buffer(self) -> Buffer:
        def unwrap_views(target):
            if isinstance(target, MutationLayoutSHOULDREMOVE):
                return unwrap_views(target.target)
            if isinstance(target, BaseView):
                return unwrap_views(target.unwrap_view())
            if isinstance(target, MutableBox):
                return unwrap_views(target.data)
            return target

        result = unwrap_views(self.target)
        assert isinstance(
            result, Buffer
        ), "MutationLayoutSHOULDREMOVE must refer to a buffer"
        return result

    def real_layout(self):
        return self.get_buffer().layout

    @classmethod
    def realize_into(cls, src, dst, unsafe_alias=False):
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

    def as_fixed(self):
        return self

    def make_indexer(self):
        return self.target.make_indexer()


@dataclasses.dataclass
class Buffer(IRNode):
    # Name is sometimes None; e.g., ForceInPlace, where there isn't
    # a meaningful name
    name: Optional[str]
    layout: Layout

    # Multi-output buffers will define 'outputs: List[Buffer]'. Confusingly,
    # MultiOutput does NOT define this!

    def __post_init__(self):
        super().__post_init__()
        self.origin_node = None

    def make_indexer(self):
        return self.layout.make_indexer()

    def get_name(self) -> str:
        assert self.name, self
        return self.name

    def get_device(self):
        return self.layout.device

    def get_origin_node(self):
        return self.origin_node

    def get_defining_op(self) -> Optional[Operation]:
        return None

    @property
    def dtype(self):
        return getattr(self.layout, "dtype", None)

    def get_size(self):
        return list(self.layout.size)

    def get_stride(self):
        return list(self.layout.stride)

    def get_offset(self):
        return self.layout.offset

    def get_layout(self):
        return self.layout

    def get_storage_numel(self):
        return self.get_numel()

    def is_extern(self):
        return False

    def freeze_layout(self):
        if not isinstance(self.layout, (MultiOutputLayout, NonOwningLayout)):
            self.layout = self.layout.as_fixed()

    def freeze_layout_with_stride_order(self, order, allow_padding=False):
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_stride_order(order, allow_padding=allow_padding)

    def freeze_layout_with_fill_order(self, order):
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_fill_order(order)

    def freeze_layout_with_same_order(self, stride):
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_same_order(stride)

    def is_zero_elements(self):
        return V.graph.sizevars.is_expr_static_and_true(sympy.Eq(self.get_numel(), 0))  # type: ignore[arg-type]

    def make_loader(self):
        # Loading from a zero-element buffer is a no-op
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.get_dtype())

        def loader(index):
            indexer = self.layout.make_indexer()
            return ops.load(self.name, indexer(index))

        return loader

    def codegen_reference(self, writer=None):
        return self.get_name()

    def decide_layout(self):
        pass

    def get_inputs_that_alias_output(self):
        if isinstance(self.layout, NonOwningLayout):
            return [self.layout.view.get_name()]
        return ()

    def get_mutation_names(self):
        if isinstance(self.layout, MutationLayoutSHOULDREMOVE):
            return [self.layout.target.get_name()]
        return ()

    def get_read_names(self) -> Set[str]:
        return {self.get_name()}

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return set()

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    def realize(self):
        pass

    def should_allocate(self):
        # Returns False by default.
        return False


@dataclasses.dataclass
class OperationBuffer(Buffer, Operation):
    # An operation that produces a single output buffer
    def get_outputs(self) -> List[Buffer]:
        return [self]

    def get_defining_op(self) -> Operation:
        return self

    def __post_init__(self):
        Buffer.__post_init__(self)
        Operation.__post_init__(self)


class InputBuffer(Buffer):
    pass


class ConstantBuffer(InputBuffer):
    override_device: Optional[torch.device] = None

    def make_loader(self):
        def loader(index):
            indexer = self.layout.make_indexer()
            return ops.load(
                V.graph.constant_name(self.get_name(), self.override_device),
                indexer(index),
            )

        return loader

    def constant_to_device(self, device):
        return ConstantBuffer(
            V.graph.constant_name(self.get_name(), device), self.layout
        )


class NoneAsConstantBuffer(IRNode):
    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return set()

    def codegen_reference(self, writer=None):
        return V.graph.wrapper_code.none_str


class ShapeAsConstantBuffer(IRNode):
    def __init__(self, shape):
        super().__init__()
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return free_unbacked_symbols(self.shape)

    def codegen_reference(self, writer=None):
        return V.graph.wrapper_code.expr_printer(V.graph.sizevars.simplify(self.shape))


@dataclasses.dataclass
class ComputedBuffer(OperationBuffer):
    data: Loops

    def get_computed_buffer_name(self):
        """
        Returns self.name if it exists, otherwise returns the name of the data node if that exists.
        If neither exist, returns None.
        """
        if self.name is not None:
            return self.name
        if hasattr(self.data, "name"):
            return self.data.name
        return None

    @cache_on_self
    def num_reads(self):
        return len(self.get_read_writes().reads)

    def get_read_writes(self):
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

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
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
            free_unbacked_symbols(self.get_size())
            | free_unbacked_symbols(self.get_stride())
            | free_unbacked_symbols(self.get_offset())
            | self.data.get_unbacked_symbol_uses()
        )

    def make_loader(self):
        # Inline constants and index_expressions
        if (
            hasattr(self.data, "make_loader")
            and self.name not in V.graph.mutated_buffers
            and self.num_reads() == 0
        ):
            # can be inlined
            return self.data.make_loader()
        return super().make_loader()

    def get_store_function(self):
        indexer = self.layout.as_fixed().make_indexer()
        if isinstance(self.data, (Reduction, Scan, Sort)):
            return partial(self.data.store_reduction, self.name, indexer)
        else:
            assert isinstance(self.data, Pointwise)
            return partial(self.data.store_output, self.name, indexer)

    def get_fill_order(self):
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
            reads_bufs = [
                V.graph.name_to_buffer[r.name]
                if r.name in V.graph.name_to_buffer.keys()
                else None
                for r in reads
            ]
            # only consider reads to buffer of same size
            # ignore StarDeps because they don't contribute stride information
            assert all(
                isinstance(r, (dependencies.StarDep, dependencies.MemoryDep))
                for r in reads
            )
            reads = [
                sympy_subs(
                    r.index, {v: sympy.Integer(0) for v in reduction_vars if v != 0}
                )
                for r in reads
                if isinstance(r, dependencies.MemoryDep)
            ]

            if reads:
                if isinstance(self.data, (Scan, Sort)):
                    indices = self.data.reindex(index_vars, reduction_vars)
                else:
                    indices = index_vars
                stride_lengths = [
                    V.graph.sizevars.stride_hints(expr, indices) for expr in reads  # type: ignore[arg-type]
                ]
                from .scheduler import pick_loop_order

                return pick_loop_order(stride_lengths, self.get_size())

        return None

    def decide_layout(self):
        if isinstance(self.layout, FlexibleLayout):
            order = self.get_fill_order()
            if order:
                self.freeze_layout_with_fill_order(order)
            else:
                self.freeze_layout()

    @cache_on_self
    def get_default_sizes_body(self):
        args, var_ranges = dependencies.index_vars_squeeze(
            self.data.get_pointwise_size(), self.data.get_reduction_size(), prefix="q"
        )
        with patch.object(ConstantBuffer, "override_device", self.get_device()):
            body = LoopBody(
                self.get_store_function(),
                (args if self.get_reduction_type() else args[:1]),
                var_ranges,
            )
        index_vars = []
        reduce_vars: List[Any] = []
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
        extra_indexing_constraints: Optional[Tuple[Dict[Any, Any], List[Any]]] = None,
    ):
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
        """
        (
            (index_size, reduce_size),
            body,
            (index_vars, reduce_vars),
        ) = self.get_default_sizes_body()

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

        memory_addrs = [*body.writes_name2expr.values()]
        if not V.graph.has_feature(self, BackendFeature.PREFER_STORE_LOOP_ORDER):
            memory_addrs.extend(body.reads_name2expr.values())

        def simplify_and_reorder(x_vars, support_vars, sizes):
            sizes, reindex0, reindex1 = self._apply_loop_reordering(
                x_vars, support_vars, sizes, memory_addrs
            )
            # for NHWC: reindex0([0,1,2,3]) = [0,2,3,1], reindex1([0,1,2,3]) = [0,3,2,1]
            x_vars = reindex0(x_vars)
            sizes, reindex2, prune = V.graph.sizevars._simplify_loops(
                x_vars,
                sizes,
                index_prevent_reordering(index_formulas, x_vars, sizes),
            )
            reindex = fuse_reindexing(reindex1, reindex2)
            return sizes, reindex, reindex1

        support_vars = index_vars + reduce_vars
        iter_ranges, iter_reindex, _ = simplify_and_reorder(
            index_vars,
            support_vars,
            index_size,
        )
        reduce_ranges, reduce_reindex, _ = simplify_and_reorder(
            reduce_vars, support_vars, reduce_size
        )

        # retrace the loop body with simplification and reordering applied
        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            iter_ranges, reduce_ranges, prefix="z"
        )
        body = LoopBody(
            body, [iter_reindex(iter_vars), reduce_reindex(reduce_vars)], var_ranges
        )
        return (iter_ranges, reduce_ranges), body

    @staticmethod
    def _apply_loop_reordering(
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

    def get_reduction_size(self):
        return self.data.get_reduction_size()

    def get_reduction_type(self):
        return self.data.get_reduction_type()

    def is_no_op(self):
        return self.data.is_zero_elements()

    def should_allocate(self):
        return True

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        return self.data.constant_to_device(device)


class TemplateBuffer(OperationBuffer):
    """
    Represents a Triton (in the future other type) of template operator
    that we can fuse an epilogue onto.
    """

    def __init__(self, layout, inputs, make_kernel_render):
        super().__init__(name=None, layout=layout)
        self.inputs = InputsKernel.unwrap_storage(inputs)
        self.make_kernel_render = make_kernel_render
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_read_writes(self):
        return self.normalized_read_writes()

    def normalized_read_writes(self):
        name = self.get_name()
        indexer = self.layout.make_indexer()

        def dummy(index, rindex):
            assert len(rindex) == 0
            return ops.store(name, indexer(index), "fake")

        deps = dependencies.extract_read_writes(
            dummy, self.get_size(), (), normalize=True
        )
        deps.reads = {dependencies.StarDep(x.get_name()) for x in self.inputs}
        return deps

    def get_reduction_size(self):
        return 1

    def get_reduction_type(self):
        return None

    def is_no_op(self):
        return False

    def should_allocate(self):
        return True

    def simplify_and_reorder(
        self,
        extra_indexing_constraints: Optional[Tuple[Dict[Any, Any], List[Any]]] = None,
    ):
        return (
            (
                self.get_size(),
                (),
            ),
            None,
        )


class TritonTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        debug_extra=None,
        mutated_inputs: Optional[Iterable[IRNode]] = None,
    ):
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
        self.debug_extra = debug_extra
        self.mutated_inputs = mutated_inputs
        self.outputs: List[Buffer] = [self]
        if mutated_inputs is not None:
            # Ensure that the mutated inputs are only allowed for certain nodes
            allowed_set = {
                torch.ops.higher_order.flex_attention,
                torch.ops.higher_order.flex_attention_backward,
            }
            current_node = V.graph.current_node.target
            assert (
                current_node in allowed_set
            ), f"Mutated inputs are only allowed for {allowed_set} but got {current_node}"
            device = self.inputs[0].get_device()
            self.outputs += [
                MutationOutput(NoneLayout(device), buf, self) for buf in mutated_inputs
            ]

    def get_outputs(self) -> List[Buffer]:
        return self.outputs

    def __str__(self):
        out = f"TritonTemplateBuffer(layout={self.layout}, {self.debug_extra})"
        return out


PrimitiveInfoType = Union[int, float, bool, str, List[Union[int, str, float, bool]]]


class ChoiceCaller:
    """
    Represents a possible choice used in autotune_process.py.
    During autotuning, self.benchmark() is first called to get benchmark result,
    and if this choice is selected, self.output_node() is called to get the output_node.

    Children classes: TritonTemplateCaller, CUDATemplateCaller.
    """

    def __init__(self, name, input_nodes, layout):
        super().__init__()
        self.name = name
        self.layout = layout
        self.input_nodes = input_nodes

    def benchmark(self, *args, out) -> float:
        algo = self.to_callable()
        return do_bench(algo, args, {"out": out})

    def call_name(self) -> str:
        raise NotImplementedError

    def to_callable(self):
        raise NotImplementedError

    def hash_key(self) -> str:
        raise NotImplementedError

    def output_node(self) -> TensorBox:
        raise NotImplementedError

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
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
        inputs: List[IRNode],
        choice_timings: Callable[[], Dict[ChoiceCaller, float]],
    ):
        super().__init__(layout=layout, inputs=inputs, make_kernel_render=None)
        self._choice_timings_fn = choice_timings
        self._choice_timings: Optional[Dict[ChoiceCaller, float]] = None
        self.original_inputs = inputs

    @property
    def choice_timings(self) -> Dict[ChoiceCaller, float]:
        if self._choice_timings is None:
            self._choice_timings = self._choice_timings_fn()
        return self._choice_timings

    @contextlib.contextmanager
    def swap_as_triton_caller(self, caller: TritonTemplateCallerBase):
        assert isinstance(caller, torch._inductor.select_algorithm.TritonTemplateCaller)
        assert self.layout == caller.layout

        render = self.make_kernel_render
        self.make_kernel_render = caller.get_make_kernel_render()
        try:
            yield
        finally:
            self.make_kernel_render = render

    def finalize_as_triton_caller(self, caller: TritonTemplateCallerBase):
        assert isinstance(caller, torch._inductor.select_algorithm.TritonTemplateCaller)
        assert self.layout.size == caller.layout.size
        assert self.layout.stride == caller.layout.stride
        self.make_kernel_render = caller.get_make_kernel_render()

    def get_min_choice(self) -> Tuple[ChoiceCaller, float]:
        min_choice = min(self.choice_timings, key=self.choice_timings.get)  # type: ignore[arg-type]
        return (min_choice, self.choice_timings[min_choice])


class CUDATemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        workspace_size: int,
        template: CUDATemplate,  # type: ignore[name-defined]  # noqa: F821
    ):
        super().__init__(layout, inputs, make_kernel_render)
        # Global memory (in bytes) needed for this template.
        self.workspace_size = workspace_size
        self.template = template

    def get_workspace_size(self):
        return self.workspace_size if self.workspace_size is not None else 0


class CppTemplateBuffer(TemplateBuffer):
    def __init__(self, layout, inputs, make_kernel_render, template, choice):
        super().__init__(layout, inputs, make_kernel_render)
        self.template = template
        self.choice = choice


@dataclasses.dataclass
class InputsKernel(OperationBuffer):
    inputs: List[Buffer]

    def get_read_writes(self):
        reads: Set[dependencies.Dep] = set()
        StarDep = dependencies.StarDep
        for input in self.inputs:
            if isinstance(input, list):
                reads.update(StarDep(x.get_name()) for x in input)
            else:
                reads.add(StarDep(input.get_name()))

        writes: Set[dependencies.Dep] = {
            StarDep(buf.get_name()) for buf in self.get_outputs()
        }

        return dependencies.ReadWrites(
            reads=reads,
            writes=writes,
            index_exprs=set(),
        )

    @classmethod
    def unwrap_storage_for_input(cls, x):
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
    def unwrap_storage(inputs):
        inputs_new = []
        for x in inputs:
            if isinstance(x, list):
                x = [InputsKernel.unwrap_storage_for_input(i) for i in x]
            else:
                x = InputsKernel.unwrap_storage_for_input(x)
            inputs_new.append(x)
        return inputs_new

    def is_extern(self):
        return True


class NopKernel(InputsKernel):
    def is_no_op(self):
        return True


class ConcatKernel(NopKernel):
    """
    There isn't actually a real kernel for concat, we just change the
    storage for the upstream data.
    """

    @classmethod
    def create(cls, inputs, dim):
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
    def can_realize_into_without_copy(cls, src):
        if isinstance(src, TensorBox):
            # unwrap a TensorBox
            return cls.can_realize_into_without_copy(src.data)

        return isinstance(src.data.layout, FlexibleLayout) and not isinstance(
            src.data, ExternKernelAlloc
        )

    @classmethod
    def realize_into(cls, src, dst):
        # Attempt to turn this into a ReinterpretView rather than assert.
        # This has concessions around layout, as as_storage_and_layout
        # can cause us to go from flexible to fixed layout.
        if not isinstance(dst, ReinterpretView):
            if is_storage_and_layout(dst):
                storage, layout = as_storage_and_layout(dst)
                dst = ReinterpretView(storage, layout)
        assert isinstance(dst, ReinterpretView), dst
        if isinstance(src, TensorBox):
            # unwrap a TensorBox
            return cls.realize_into(src.data, dst)
        if isinstance(src, StorageBox):
            src.realize()
            # ExternKernelAlloc has specific requirements for output layout, should create a copy
            assert hasattr(src.data, "layout")
            if cls.can_realize_into_without_copy(src):
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

    def should_allocate(self):
        return True


def get_aten_cpp_kernel_name(kernel):
    # Calling with the default kernel name can lead to ambiguous behavior like the following example.
    # repeat_interleave(const at::Tensor & repeats, c10::optional<int64_t> output_size=std::nullopt)
    # repeat_interleave(const at::Tensor & self, int64_t repeats,
    #       c10::optional<int64_t> dim=std::nullopt, c10::optional<int64_t> output_size=std::nullopt)
    if not isinstance(kernel, torch._ops.OpOverload) or kernel.namespace != "aten":
        return None
    opname = (
        kernel.__name__.split(".")[0]
        if kernel._overloadname == "default"
        else kernel.__name__.replace(".", "_")
    )
    return f"at::_ops::{opname}::call"


@dataclasses.dataclass
class ExternKernel(InputsKernel):
    constant_args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
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
    arg_properties: Optional[List[Dict[str, Any]]] = None
    kwarg_properties: Optional[Dict[str, Dict[str, Any]]] = None
    unbacked_bindings: Dict[sympy.Symbol, pytree.KeyPath] = dataclasses.field(
        default_factory=dict
    )
    mutation_outputs: List[MutationOutput] = dataclasses.field(default_factory=list)

    def __init__(
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
    ):
        super().__init__(
            name,
            layout,
            inputs,
        )
        self.constant_args = constant_args
        self.kwargs = kwargs if kwargs else {}
        self.output_view = output_view
        self.python_kernel_name = python_kernel_name
        # If cpp_kernel_name is None, we will try to construct it from op_overload
        self.cpp_kernel_name = cpp_kernel_name or get_aten_cpp_kernel_name(op_overload)
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        self.op_overload = op_overload
        self.collect_arg_kwarg_properties()
        self.unbacked_bindings = {}
        self.mutation_outputs = []
        self.fx_node = V.graph.current_node

    def get_outputs(self) -> List[Buffer]:
        return [self, *self.mutation_outputs]

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    def collect_arg_kwarg_properties(self):
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
        if (
            isinstance(self.op_overload, torch._ops.OpOverload)
            and not self.ordered_kwargs_for_cpp_kernel
        ):
            self.ordered_kwargs_for_cpp_kernel = [
                x.name for x in self.op_overload._schema.arguments if x.kwarg_only
            ]

    def fill_non_provided_args(self, args, kwargs, convert_val_to_str=False):
        # Previously, we want to maintain forward-compatibility by skipping
        # default args in the serialized artifacts in fbcode. However,
        # some of our shim interfaces require default values being set.
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

    def decide_layout(self):
        if isinstance(self.layout, FlexibleLayout):
            self.apply_constraint()
            self.freeze_layout()

    def codegen_comment(self, wrapper):
        origin_str, detailed_origin_str = get_kernel_metadata(self, wrapper)
        if origin_str:
            wrapper.writeline(origin_str)

    def codegen(self, wrapper):
        raise NotImplementedError

    def get_kernel_name(self):
        return (
            (
                V.graph.wrapper_code.get_c_shim_func_name(self.cpp_kernel_name)  # type: ignore[attr-defined]
                if config.abi_compatible
                else self.cpp_kernel_name
            )
            if V.graph.cpp_wrapper
            else self.python_kernel_name
        )

    @staticmethod
    def copy_input(x):
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
    def process_kernel(
        cls, kernel, *args, **kwargs
    ) -> Tuple[
        Any,
        List[Any],
        List[Any],
        Callable[[Any, Any], Any],
        Optional[Dict[sympy.Symbol, pytree.KeyPath]],
    ]:
        binded_args = {"args": args, "kwargs": kwargs}

        args_flat, args_spec = pytree.tree_flatten(binded_args)

        is_arg_tensor = []
        tensor_args = []
        non_tensor_args: List[Any] = []
        for arg in args_flat:
            is_arg_tensor.append(isinstance(arg, IRNode))
            if is_arg_tensor[-1]:
                tensor_args.append(arg)
            else:
                if isinstance(arg, sympy.Expr):
                    arg = V.graph.sizevars.shape_env.create_symintnode(arg, hint=None)
                non_tensor_args.append(arg)

        def unflatten_args(new_tensor_args, new_non_tensor_args):
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
        example_args: List[Union[torch.Tensor, torch._C.ScriptObject]] = []

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
            else:
                example_args.append(ir_node_to_tensor(x, guard_shape=True))

        new_args, new_kwargs = unflatten_args(example_args, non_tensor_args)
        example_output = kernel(*new_args, **new_kwargs)

        unbacked_bindings: Optional[Dict[sympy.Symbol, pytree.KeyPath]] = None
        if shape_env := V.fake_mode.shape_env:
            rebind_unbacked(shape_env, V.current_node, example_output)
            unbacked_bindings = compute_unbacked_bindings(
                shape_env, example_output, V.current_node.meta.get("val")
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
    def convert_to_reinterpret_view(cls, x):
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
            x.get_size(), prefix="r"
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
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=x.get_size(),
                stride=strides,
                offset=offset,
            ),
        )

    @classmethod
    def realize_input(cls, x):
        if x is None:
            return NoneAsConstantBuffer()
        if isinstance(x, (sympy.Expr, sympy.logic.boolalg.Boolean, int)):
            return ShapeAsConstantBuffer(x)
        if isinstance(x, Constant):
            return V.graph.add_tensor_constant(
                torch.tensor(x.value, dtype=x.get_dtype(), device=x.get_device())
            )
        if isinstance(x, ConstantBuffer):
            return x
        if isinstance(x, TensorBox):
            return cls.realize_input(x.data)
        if isinstance(x, ReinterpretView):
            return ReinterpretView(cls.realize_input(x.data), x.get_layout())
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
        if isinstance(x, TorchBindObject):
            return x
        return cls.copy_input(x)

    @classmethod
    def require_stride1(cls, x):
        if is_storage_and_layout(x):
            if len(x.get_stride()) == 0:
                return x
            for stride in x.get_stride():
                if stride == 1:
                    return x
        return cls.copy_input(x)

    @classmethod
    def require_stride_order(cls, x, order, allow_padding=False):
        if x.get_numel() == 0:  # Layout doesn't matter
            return x

        # require x to have the layout as strided_ordered as order
        if is_storage_and_layout(x):
            while isinstance(x.get_layout(), NonOwningLayout):
                x = x.get_layout().view
            if isinstance(x.get_layout(), FlexibleLayout):
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
                    stride_order=get_stride_order(
                        V.graph.sizevars.size_hints(x.get_layout().stride)
                    )
                    if is_stride_order_storage_and_layout(x, order)
                    else order,
                    allow_padding=allow_padding,
                )
                return x
            elif isinstance(
                x.get_layout(), FixedLayout
            ) and x.get_layout().is_stride_ordered(order):
                return x
            elif isinstance(x.get_layout(), MutationLayoutSHOULDREMOVE):
                if isinstance(x.get_layout().real_layout(), FlexibleLayout):
                    raise AssertionError(
                        "the MutationLayoutSHOULDREMOVE's real layout shouldn't be FlexibleLayout"
                    )
                elif isinstance(
                    x.get_layout().real_layout(), FixedLayout
                ) and x.get_layout().real_layout().is_stride_ordered(order):
                    return x

        # TODO - Storage to InputBuffer
        if isinstance(x, InputBuffer) and x.get_layout().is_stride_ordered(order):
            return x
        if (
            isinstance(x, TensorBox)
            and isinstance(x.data, BaseView)
            and not isinstance(x.data, ReinterpretView)
            and is_storage_and_layout(x.unwrap_view())
            and not isinstance(x.unwrap_view().data, ExternKernelAlloc)
        ):
            try:
                x.data = cls.convert_to_reinterpret_view(x.data)
                return cls.require_stride_order(x, order, allow_padding=allow_padding)
            except NotImplementedError:
                pass
        x = cls.copy_input(x)
        as_storage_and_layout(
            x,
            freeze=True,
            want_contiguous=False,
            stride_order=order,
            allow_padding=allow_padding,
        )
        assert is_stride_order_storage_and_layout(x, order)
        return x

    @classmethod
    def require_channels_last(cls, x):
        return cls.require_stride_order(x, NHWC_STRIDE_ORDER)

    @classmethod
    def require_channels_last_3d(cls, x):
        return cls.require_stride_order(x, NHWDC_STRIDE_ORDER)

    @classmethod
    def require_contiguous(cls, x):
        return cls.require_stride_order(x, list(reversed(range(len(x.get_size())))))

    def apply_constraint(self):
        pass

    def codegen_const_args(self):
        if V.graph.cpp_wrapper:
            result = []
            for i, x in enumerate(self.constant_args):
                idx = len(self.inputs) + i
                type_ = (
                    self.arg_properties[i].get("type")
                    if self.arg_properties and idx < len(self.arg_properties)
                    else None
                )
                result.append(
                    V.graph.wrapper_code.val_to_arg_str(x, type_)  # type: ignore[arg-type]
                )
            return result
        else:
            return map(V.graph.wrapper_code.val_to_arg_str, self.constant_args)

    def codegen_args(self):
        args = []
        for i, x in enumerate(self.inputs):
            if isinstance(x, list):
                names = [i.codegen_reference() for i in x]
                codegen_reference = f'[{", ".join(names)}]'
                args.append(codegen_reference)
            else:
                if V.graph.cpp_wrapper:
                    assert self.arg_properties and i < len(
                        self.arg_properties
                    ), "Invalid access to ExternKernel.arg_properties"
                    type_ = self.arg_properties[i].get("type")
                    args.append(
                        V.graph.wrapper_code.val_to_arg_str(  # type: ignore[arg-type]
                            x, type_
                        )
                    )
                else:
                    args.append(x.codegen_reference())
        args.extend(self.codegen_const_args())
        return args

    def get_kwargs_value(self, arg_name):
        if arg_name in self.kwargs:
            return self.kwargs.get(arg_name)
        if self.allarg_properties and self.allarg_properties.get(arg_name):
            return self.allarg_properties.get(arg_name).get("default_value")  # type: ignore[union-attr]
        else:
            raise AssertionError(f"{arg_name} not in self.allarg_properties")

    def codegen_kwargs(self, skip_out=False):
        if V.graph.cpp_wrapper:
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
                    kwargs.append(
                        V.graph.wrapper_code.val_to_arg_str(  # type: ignore[arg-type]
                            v, type_
                        )
                    )
        else:
            kwargs = [
                f"{k}={V.graph.wrapper_code.val_to_arg_str(v)}"  # type: ignore[misc]
                for k, v in self.kwargs.items()
            ]
        return kwargs

    def codegen_size_asserts(self, wrapper):
        if config.size_asserts and not V.graph.cpp_wrapper:
            # comparing strides for 0 size tensor is tricky. Ignore them for now.
            if sympy_product(self.get_size()) == 0:
                return
            size = V.graph.wrapper_code.codegen_shape_tuple(self.get_size())
            stride = V.graph.wrapper_code.codegen_shape_tuple(self.get_stride())
            wrapper.writeline(
                f"assert_size_stride({self.get_name()}, {size}, {stride})"
            )

    def get_group_stride(self):
        """
        get output sizes and strides, for template_codegen
        """
        _size = self.get_size()
        _stride = self.get_stride()
        # iter_ranges = _size of output tensor, reduce_range = [] because no reduction
        return [_size, []], _stride

    def canonicalize(self):
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

        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, [index]
        )

        # assign new variables each dimension to deal with numbering mismatches
        # d0, d1, d2 could become d0, d2 -- which won't match d0, d1
        _, add_var = var_builder("c")
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))

        index = sympy_subs(sympy.expand(index), replacement)  # type: ignore[arg-type]
        return index, tuple(new_sizes)

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        # NB: It's not necessary to check regular inputs as we automatically
        # have dependencies on them
        r = set()
        for arg in self.constant_args:
            r |= maybe_free_unbacked_symbols(arg)
        for arg in self.kwargs.values():
            r |= maybe_free_unbacked_symbols(arg)
        return r

    def __str__(self):
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


@dataclasses.dataclass
class ExternKernelOut(ExternKernel):
    def codegen(self, wrapper):
        self.codegen_comment(wrapper)
        args = [*self.codegen_args(), *self.codegen_kwargs(skip_out=True)]
        kernel_name = self.get_kernel_name()
        if (
            V.graph.cpp_wrapper
            and self.cpp_kernel_name == "torch::inductor::_mm_plus_mm"
        ):
            # For https://github.com/pytorch/pytorch/issues/128474
            kernel_name = (
                "aoti_torch__mm_plus_mm_out"
                if config.abi_compatible
                else "torch::inductor::_mm_plus_mm_out"
            )
        else:
            kernel_name = self.get_kernel_name()
        wrapper.generate_extern_kernel_out(
            kernel_name,
            self.codegen_reference(),
            self.output_view.codegen_reference() if self.output_view else None,
            args,
        )

    def __init__(
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
    ):
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

    def should_allocate(self):
        return True


class RandomSeeds(ExternKernelOut):
    def __init__(self, count: int, device: torch.device):
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
            cpp_kernel_name="at::_ops::randint_low_out::call"
            if config.abi_compatible
            else "at::randint_out",
            op_overload=aten.randint.low_out,
        )


class ExternKernelAlloc(ExternKernel):
    def codegen(self, wrapper):
        self.codegen_comment(wrapper)
        args = [*self.codegen_args(), *self.codegen_kwargs()]
        V.graph.wrapper_code.generate_extern_kernel_alloc(self, args)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kwargs=None,
        python_kernel_name=None,
        cpp_kernel_name=None,
        ordered_kwargs_for_cpp_kernel=(),
        op_overload=None,
    ):
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

    def should_allocate(self):
        return False

    def apply_constraint(self):
        raise NotImplementedError


class MutationOutput(Buffer):
    """
    An output buffer that represents the mutation of a pre-existing buffer
    """

    def __init__(self, layout, mutated_node, mutating_node: Operation):
        super().__init__(name=None, layout=layout)
        mutated_node_name = mutated_node.get_name()
        V.graph.mark_buffer_mutated(mutated_node_name)
        self.mutation_names = [mutated_node_name]
        self.mutating_node: Operation = mutating_node
        self.name = V.graph.register_buffer(self)

    def get_defining_op(self) -> Operation:
        return self.mutating_node

    def get_mutation_names(self):
        return self.mutation_names

    def should_allocate(self):
        return False


class UserDefinedTritonKernel(ExternKernel):
    def get_kernel_and_configs(self):
        from triton.runtime.autotuner import Autotuner

        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        kernel = kernel_side_table.get_kernel(self.kernel_idx)
        configs = []
        if isinstance(kernel, Autotuner):
            configs = kernel.configs
            kernel = kernel.fn
        return kernel, configs

    def codegen(self, wrapper):
        kernel, configs = self.get_kernel_and_configs()

        # Definition of kernel
        new_name, triton_meta = wrapper.define_user_defined_triton_kernel(
            kernel, configs, self.kwargs
        )
        raw_args = [
            self.get_kwargs_value(k) for k in self.ordered_kwargs_for_cpp_kernel
        ]
        # Call to kernel
        self.codegen_comment(wrapper)
        wrapper.generate_user_defined_triton_kernel(
            new_name, raw_args, self.grid, configs, triton_meta, kernel.constexprs
        )

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        # add unbacked symbols used in the grid to the ones used
        # in the kwargs (the latter is generated by ExternKernel)
        return super().get_unbacked_symbol_uses() | free_unbacked_symbols(self.grid)

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    def __init__(self, *, kernel_idx, grid, kernel_args):
        inputs = []
        kwargs = {}
        constant_args = []
        for k, v in kernel_args.items():
            if isinstance(v, TensorBox):
                t = InputsKernel.unwrap_storage_for_input(self.realize_input(v))
                inputs.append(t)
                kwargs[k] = t
            else:
                constant_args.append(v)
                kwargs[k] = v

        assert len(inputs) != 0
        self.device = inputs[0].get_device()

        super().__init__(
            None,
            NoneLayout(self.device),  # type: ignore[arg-type]
            inputs,
            tuple(constant_args),
            kwargs,
        )
        self.kernel_idx = kernel_idx
        self.grid = grid

        kernel, configs = self.get_kernel_and_configs()
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
            MutationOutput(NoneLayout(self.device), buf, self)
            for buf in self.mutable_args
        ]
        V.graph.register_operation(self)

    def get_outputs(self) -> List[Buffer]:
        return list(self.mutation_outputs)

    def get_device(self) -> torch.device:
        return self.device


class InplaceBernoulliFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper):
        (x,) = (t.codegen_reference() for t in self.inputs)

        if V.graph.cpp_wrapper and config.abi_compatible:
            # Inductor doesn't really support aten Generator, so the Generator kwarg is always NULL here,
            # which needs to be explicitly generated for cpp wrapper
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}, NULL){wrapper.ending}"
            )
        else:
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}){wrapper.ending}"
            )

    def should_allocate(self):
        return False

    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    def __init__(self, op_overload, x, *constant_args):
        super().__init__(
            None,
            NoneLayout(x.get_device()),  # type: ignore[arg-type]
            self.unwrap_storage([x]),
            constant_args,
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(x.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        self.python_kernel_name = "aten.bernoulli_"
        if not config.abi_compatible:
            # TODO: this should be simplified once we switch to ABI-compatible only
            self.cpp_kernel_name = "at::native::bernoulli_"


# Used to deal with torch.complex types
class InplaceCopyFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper):
        (dst, src, non_blocking) = self.codegen_args()
        wrapper.writeline(
            f"{self.get_kernel_name()}({dst}, {src}, {non_blocking}){wrapper.ending}"
        )

    def should_allocate(self):
        return False

    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    def __init__(
        self,
        layout,
        inputs,
        constant_args,
    ):
        super().__init__(
            None,
            layout,
            inputs,
            constant_args,
            python_kernel_name="aten.copy_",
            cpp_kernel_name=(
                "aoti_torch_copy_" if config.abi_compatible else "at::_ops::copy_::call"
            ),
        )
        V.graph.mark_buffer_mutated(inputs[0].get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(cls, dst, src, non_blocking: bool = False):
        inputs = [cls.realize_input(t) for t in [dst, src]]
        constant_args = (non_blocking,)
        result = InplaceCopyFallback(
            NoneLayout(dst.get_device()),  # type: ignore[arg-type]
            inputs,
            constant_args,
        )
        return result


class MutatingFirstArgExternKernel(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper):
        argrefs = [
            *(t.codegen_reference() for t in self.inputs),
            *map(repr, self.constant_args),
        ]
        wrapper.writeline(
            f"{self.get_kernel_name()}({', '.join(argrefs)}){wrapper.ending}"
        )

    def should_allocate(self):
        return False

    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    def has_side_effects(self):
        return True


class ResizeStorageBytes(MutatingFirstArgExternKernel):
    def __init__(self, variable, new_size):
        assert isinstance(new_size, int), "TODO: dynamic shapes"
        super().__init__(
            None,
            NoneLayout(variable.get_device()),  # type: ignore[arg-type]
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
    def __init__(self, self_tensor, storage_tensor):
        self_tensor.freeze_layout()
        super().__init__(
            self_tensor.get_layout(),
            [self_tensor, storage_tensor],
            python_kernel_name="torch.ops.aten.set_.source_Tensor",
            op_overload=torch.ops.aten.set_.source_Tensor,
        )
        V.graph.never_reuse_buffers.add(self_tensor.data.get_name())
        V.graph.never_reuse_buffers.add(storage_tensor.get_name())
        V.graph.never_reuse_buffers.add(self.get_name())
        device = storage_tensor.get_device()
        self.mutation_outputs = [
            MutationOutput(NoneLayout(device), self_tensor, self),
            MutationOutput(NoneLayout(device), storage_tensor, self),
        ]

    def get_inputs_that_alias_output(self):
        return [self.inputs[0].get_name(), self.inputs[1].get_name()]


class ScatterFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly.
    This class handles both aten.scatter_ and aten.scatter_reduce_.
    It also handle the case `src` being a scalar properly.
    """

    def codegen(self, wrapper):
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

    def should_allocate(self):
        return False

    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    def __init__(
        self,
        op_overload,
        x,
        dim: int,
        index,
        src,
        *,
        reduce: Optional[str] = None,
        include_self: bool = True,
    ):
        self.src_is_tensor = isinstance(src, TensorBox)

        constant_args: Tuple[Any, ...]
        if self.src_is_tensor:
            tensors = [self.realize_input(t) for t in [x, index, src]]
            constant_args = (dim,)
        else:
            tensors = [self.realize_input(t) for t in [x, index]]
            constant_args = (dim, src)

        super().__init__(
            None,
            NoneLayout(x.get_device()),  # type: ignore[arg-type]
            self.unwrap_storage(tensors),
            constant_args,
            {"reduce": reduce, "include_self": include_self},
            python_kernel_name=str(op_overload),
            ordered_kwargs_for_cpp_kernel=["reduce", "include_self"],
            op_overload=op_overload,
        )
        self.cpp_kernel_name = get_aten_cpp_kernel_name(op_overload)
        V.graph.mark_buffer_mutated(x.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class IndexPutFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation and indices properly
    """

    def codegen(self, wrapper):
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

    def should_allocate(self):
        return False

    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    def __init__(self, op_overload, x, indices, values, accumulate):
        self.indices = indices
        valid_indices = [i for i in indices if i is not None]
        tensors = [self.realize_input(x) for x in [x, values, *valid_indices]]
        cpp_kernel_name = (
            "aoti_torch_index_put_out" if config.abi_compatible else "at::index_put_out"
        )
        super().__init__(
            None,
            NoneLayout(x.get_device()),  # type: ignore[arg-type]
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
    def create(cls, x, device):
        if (
            not x.is_extern()
            and all(r in V.graph.constants for r in x.get_read_names())
            and not config.aot_inductor.use_runtime_constant_folding
        ):
            return x.constant_to_device(device)

        V.graph.add_device_info(device)
        V.graph.add_device_info(x.get_device())

        developer_warning("DeviceCopy in input program")
        return DeviceCopy(
            FlexibleLayout(
                device=device,
                dtype=x.get_dtype(),
                size=x.get_size(),
            ),
            [cls.realize_input(x)],
        )

    def codegen(self, wrapper):
        args = self.codegen_args()
        assert len(args) == 1
        if self.output_view:
            wrapper.codegen_device_copy(args[0], self.output_view.codegen_reference())
        else:
            wrapper.codegen_device_copy(args[0], self.codegen_reference())


class DynamicScalar(ExternKernel):
    """
    The result of a call to aten._local_scalar_dense.
    """

    def get_reads(self):
        return ()

    def should_allocate(self):
        return False

    def __init__(self, sym, keypath, data):
        data.realize()
        super().__init__(None, NoneLayout(torch.device("cpu")), self.unwrap_storage([data]))  # type: ignore[arg-type]
        self.sym = sym
        self.keypath = keypath

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return {self.sym}

    def codegen(self, wrapper):
        wrapper.codegen_dynamic_scalar(self)


class AssertScalar(ExternKernel):
    """
    The result of a call to aten._assert_scalar
    """

    def get_reads(self):
        return ()

    def should_allocate(self):
        return False

    def __init__(self, scalar, msg):
        super().__init__(
            # Buffer(name, layotu)
            None,
            NoneLayout(torch.device("cpu")),  # type: ignore[arg-type]
            # InputsKernel(inputs)
            [],
        )  # type: ignore[arg-type]
        self.scalar = scalar
        self.msg = msg

    def has_side_effects(self):
        return True

    def get_unbacked_symbol_uses(self):
        return free_unbacked_symbols(self.scalar)

    def codegen(self, wrapper):
        if V.graph.cpp_wrapper:
            pass
        else:
            # NB: It is EXTREMELY important not to simplify the scalar under
            # assertion here, because simplify is done with respect to
            # runtime asserts.  So if you have "u0 == 0" in the runtime
            # asserts, if you subsequently try to simplify(u0 == 0), you will
            # get True (because we've already runtime assert'ed that it's
            # true).  But we're code generating the actual runtime assert
            # here!!
            wrapper.writeline(
                f"if not {V.graph.wrapper_code.codegen_python_sizevar(self.scalar, simplify=False)}:"
            )
            wrapper.writeline(f"    raise RuntimeError({repr(self.msg)})")
            # No one should ever use this buffer, but for uniformity
            # define the variable and assign it None
            wrapper.writeline(f"{self.get_name()} = None")


@dataclasses.dataclass
class ExternKernelNode:
    name: str
    node: export_schema.Node


has_c_shim = {
    aten._embedding_bag.default,
    aten._fft_c2c.default,
    aten._scaled_dot_product_efficient_attention.default,
    aten._scaled_dot_product_flash_attention.default,
    aten._scaled_dot_product_cudnn_attention.default,
    aten._scaled_mm.default,
    aten.addmm.out,
    aten.bmm.out,
    aten.copy_.default,
    aten.mm.out,
    aten.repeat_interleave.Tensor,
    aten.nonzero.default,
    aten.view.dtype,
    aten.view_as_real.default,
}


class FallbackKernel(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs=None,
        *,
        unbacked_bindings=None,
    ):
        if (
            kernel == aten.mul.Tensor
            and len(tensor_args) == 1
            and len(nontensor_args) == 1
        ):
            # When aten.mul.Tensor's second arg is constant, cpp wrapper expects
            # to call mul_Scalar. A more proper fix is to do it in decomposition.
            # See https://github.com/pytorch/pytorch/issues/123478
            kernel = aten.mul.Scalar

        super().__init__(
            layout,
            tuple(tensor_args),
            tuple(nontensor_args),
            op_overload=kernel,
        )

        # We need output buffers for generating kernel arguments in the
        # abi-compatible mode, where we retrieve outputs by pass each individual
        # output through the abi-compatible interface.
        self.outputs: Sequence[Any] = []
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
        self.alias_names: List[str] = []
        # args that are mutated AND returned from the op
        self.mutation_names: List[str] = []

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

        schema_args = schema.arguments
        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)

        def handle_aliasing_and_mutation(info, arg):
            # Assertions to make sure we didn't mismatch args
            if isinstance(info.type, torch.ListType):
                assert isinstance(arg, (list, tuple))
            is_optional_tensor = isinstance(
                info.type, torch.OptionalType
            ) and isinstance(info.type.getElementType(), torch.TensorType)
            is_list_tensor = isinstance(info.type, torch.ListType) and isinstance(
                info.type.getElementType(), torch.TensorType
            )
            if is_optional_tensor or isinstance(info.type, torch.TensorType):
                # PyTorch also accepts None and scalar types for args marked as "Tensor".
                # We're not going to check all of them here.
                assert not isinstance(arg, (tuple, list))

            if arg is None:
                return
            if info.alias_info is None:
                return

            def add_alias(t):
                self.alias_names.append(t.get_name())
                if info.alias_info.is_write:
                    self.mutation_outputs.append(
                        MutationOutput(NoneLayout(t.get_device()), t, self)
                    )

            if is_list_tensor:
                for tensor_arg in arg:
                    add_alias(tensor_arg)
            else:
                assert isinstance(info.type, torch.TensorType) or is_optional_tensor
                add_alias(arg)

        for info, arg in torch._library.utils.zip_schema(schema, args, kwargs):
            handle_aliasing_and_mutation(info, arg)

    def codegen_unbacked_symbol_defs(self, wrapper):
        if not hasattr(self, "unbacked_bindings"):
            return

        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env, self.unbacked_bindings
        )

        if not unbacked_bindings:
            return

        for s, keypath in unbacked_bindings.items():

            def go(expr, keypath):
                if keypath == ():
                    return expr

                if (
                    len(keypath) >= 2
                    and isinstance(keypath[0], CallMethodKey)
                    and isinstance(keypath[1], pytree.SequenceKey)
                ):
                    return go(
                        f"{expr}.{keypath[0].name}({keypath[1].idx})", keypath[2:]
                    )
                elif isinstance(keypath[0], CallMethodKey):
                    return go(f"{expr}.{keypath[0].name}()", keypath[1:])
                elif isinstance(keypath[0], pytree.SequenceKey):
                    return go(f"{expr}[{keypath[0].idx}]", keypath[1:])
                elif isinstance(keypath[0], DivideByKey):
                    # TODO: need to assert divisibility
                    # TODO: this is invalid C++ codegen
                    return go(f"{expr}.__floordiv__({keypath[0].divisor})", keypath[1:])
                else:
                    raise AssertionError(f"unrecognized keypath {keypath}")

            def go_outer():
                if V.graph.cpp_wrapper and config.abi_compatible:
                    # Special handling for the top level buffer access,
                    # because self.get_name() is actually never bound; the
                    # individual output arguments are bound by
                    # generate_c_shim_fallback_kernel
                    if len(self.outputs) == 1:
                        return go(self.outputs[0].get_name(), keypath)
                    else:
                        assert isinstance(keypath[0], pytree.SequenceKey)
                        return go(self.outputs[keypath[0].idx].get_name(), keypath[1:])
                else:
                    return go(self.get_name(), keypath)

            wrapper.writeline(
                f"{wrapper.codegen_unbacked_symbol_decl(s)} = {go_outer()}{wrapper.ending}"
            )

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            return resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            ).keys()
        else:
            return set()

    def set_cpp_kernel(self, kernel):
        from .codegen.wrapper import get_cpp_op_schema

        assert (
            not kernel._schema.is_mutable
        ), f"mutable {kernel.__name__} is not supported with cpp_wrapper"

        # These checks are here because ops that return aliasing tensors will
        # return type Tensor& instead of Tensor, but codegen will always write
        # type Tensor on the LHS.
        def is_not_write(arg):
            return arg.alias_info is None or not arg.alias_info.is_write

        assert all(
            is_not_write(x) for x in kernel._schema.arguments
        ), f"{kernel.__name__} with alias_info arguments is not supported with cpp_wrapper"
        assert all(
            is_not_write(x) for x in kernel._schema.returns
        ), f"{kernel.__name__} with alias_info returns is not supported with cpp_wrapper"

        self.cpp_kernel_name = kernel._schema.name
        self.cpp_kernel_overload_name = kernel._schema.overload_name
        self.cpp_kernel_key = f"{self.cpp_kernel_name.replace('::', '_')}_{self.cpp_kernel_overload_name}"  # type: ignore[union-attr]

        self.cpp_op_schema = get_cpp_op_schema(kernel)

    def codegen_args(self):
        @dataclasses.dataclass
        class Shim:
            ref: Any

            def __repr__(self):
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
    def find_device(tensor_args, example_output):
        if tensor_args:
            devices = [arg.get_device() for arg in tensor_args if arg.get_device()]
            return devices[0]
        if isinstance(example_output, torch.Tensor):
            return example_output.device
        if isinstance(example_output, (list, tuple)):
            device_set = {FallbackKernel.find_device(None, x) for x in example_output}
            # Remove None
            devices = [device for device in device_set if device]
            if len(devices) == 1:
                return devices[0]
            for device in devices:
                if is_gpu(device.type):
                    return device
            return devices[0]
        return None

    def has_side_effects(self):
        if isinstance(self.op_overload, torch._ops.HigherOrderOperator):
            return False
        return get_schema_info(self.op_overload).is_mutable()

    def get_inputs_that_alias_output(self):
        return self.alias_names

    def get_mutation_names(self):
        assert len(self.mutation_names) <= 1
        return self.mutation_names

    # ProxyExecutor Design Note
    # We export the ExternFallbackNodes (for custom ops) into a serialized file
    # and run it with a host side proxy executor to address the ABI problem
    # This is currently only implemented for fbcode. Eventually, we will also make this work for OSS.
    # Detailed design doc can be found at
    # https://docs.google.com/document/d/1wC4DOZFaYym2t1Esz0X5yxlLI3RDnSiyRbUus3bkJ64/edit?usp=sharing
    def export_extern_kernel_node(self):
        assert isinstance(self, FallbackKernel)
        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
        args = self.fill_non_provided_args(args, kwargs)
        ordered_kwargs = [
            kwargs.get(key, None) for key in self.ordered_kwargs_for_cpp_kernel
        ]
        if not V.graph.aot_mode:
            # No need to serialize in the cpp wrapper JIT mode
            return [*args, *ordered_kwargs]

        serializer = GraphModuleSerializer(None, None)  # type: ignore[arg-type]
        named_arguments = serializer.serialize_inputs(self.op_overload, args, kwargs)  # type: ignore[arg-type]

        # serialize_outputs
        def handle_single_output(return_type, output):
            if isinstance(return_type, torch.TensorType):
                # For single Tensor
                out = output
                if isinstance(output, (list, tuple)):
                    assert len(output) == 1
                    out = output[0]
                return export_schema.Argument.create(
                    as_tensor=export_schema.TensorArgument(name=out.get_name())
                )
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

        target = self.op_overload
        returns = target._schema.returns  # type: ignore[union-attr]
        if len(returns) == 1:
            return_type = returns[0].real_type
            output_arguments = [handle_single_output(return_type, self.outputs)]
        else:
            # For tuple returns, e.g "-> (Tensor, Tensor)" or "-> (Tesnor, Tensor[])"
            assert isinstance(self.outputs, tuple)
            assert len(returns) == len(self.outputs)
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

    def codegen(self, wrapper):
        kernel = self.op_overload
        if kernel.namespace == "aten":  # type: ignore[union-attr]
            # Aten Fallback Ops
            assert isinstance(kernel, torch._ops.OpOverload)
            if V.graph.cpp_wrapper:
                from torchgen.aoti.fallback_ops import inductor_fallback_ops

                if (
                    config.is_fbcode()
                    and kernel not in has_c_shim
                    # C shim v2 is torchgen-ed, which should cover all aten ops.
                    # If you do hit a missed op, please update gen_aoti_c_shim.py.
                    and config.c_shim_version == "1"
                ) or (
                    config.abi_compatible
                    and config.c_shim_version == "2"
                    and str(kernel) not in inductor_fallback_ops
                ):
                    log.warning(
                        "%s is missing a c-shim implementation, using proxy executor as fallback",
                        kernel,
                    )
                    self.use_runtime_dispatch = True
                    self.set_cpp_kernel(kernel)
            self.python_kernel_name = f"torch.ops.{kernel}"
        elif kernel.namespace == "_quantized":  # type: ignore[union-attr]
            # Internal Quantized Fallback Ops
            assert isinstance(kernel, torch._ops.OpOverload)
            if V.graph.cpp_wrapper:
                self.set_cpp_kernel(kernel)
                if not config.abi_compatible:
                    self.use_runtime_dispatch = True
            else:
                self.python_kernel_name = str(kernel)
        elif isinstance(kernel, torch._ops.HigherOrderOperator):
            self.python_kernel_name = f"torch.ops.higher_order.{kernel.__name__}"
        else:
            # For non-aten OpOverload, i.e. custom ops
            self.python_kernel_name = f"{kernel.__module__.replace('._ops.', '.ops.')}.{kernel.__name__}"  # type: ignore[union-attr]
            if V.graph.cpp_wrapper:
                self.use_runtime_dispatch = True
                self.set_cpp_kernel(kernel)

        if self.use_runtime_dispatch:
            self.codegen_comment(wrapper)

            exported_args = None
            args = None
            if config.abi_compatible:
                exported_args = self.export_extern_kernel_node()
            else:
                args = [*self.codegen_args(), *self.codegen_kwargs()]

            wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
                self.get_name(),
                self.python_kernel_name,
                self.cpp_kernel_name,
                args,
                self.cpp_op_schema,
                self.cpp_kernel_key,
                self.cpp_kernel_overload_name,
                self.op_overload,
                exported_args,
                self.outputs,
            )
        else:
            self.codegen_comment(wrapper)
            args = [*self.codegen_args(), *self.codegen_kwargs()]
            V.graph.wrapper_code.generate_fallback_kernel(self, args)
            if isinstance(self.layout, Layout):
                self.codegen_size_asserts(wrapper)

        self.codegen_unbacked_symbol_defs(wrapper)

    @staticmethod
    def tensor_to_layout(output: torch.Tensor):
        return FixedLayout(
            output.device,
            output.dtype,
            convert_shape_to_inductor(output.size()),
            convert_shape_to_inductor(output.stride()),
        )

    @classmethod
    def create(cls, kernel, *args, **kwargs):
        fake_incorrect_kernels = (aten._fused_moving_avg_obs_fq_helper_functional,)
        context: ContextManager[None] = (
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

        device = cls.find_device(tensor_args, example_output)
        if example_output is None:
            packed = cls(
                NoneLayout(device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings=unbacked_bindings,
            )

        else:
            assert device, "Not sure where to find device info"
            packed = cls(
                MultiOutputLayout(device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings=unbacked_bindings,
            )

        def generate_output(output, indices):
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
                return MultiOutput(
                    cls.tensor_to_layout(output),
                    packed,
                    indices,
                )
            elif isinstance(output, int):
                return output
            elif isinstance(output, torch.SymInt):
                return output.node.expr
            else:
                assert (
                    output is None
                ), f"FallbackKernel output type {type(output)} is not supported"
                return None

        outputs = generate_output(example_output, [])
        if isinstance(outputs, (list, tuple, dict)):
            packed.outputs = outputs  # type: ignore[assignment]
        else:
            packed.outputs = [outputs]
        return outputs

    def apply_constraint(self):
        return super().apply_constraint()


@dataclasses.dataclass
class ComplexView(FallbackKernel):
    """View a complex number as two dtyped numbers or vice versa"""

    def should_allocate(self):
        return False

    def get_inputs_that_alias_output(self):
        # Signal to codegen that our output buffer isn't safe to reuse
        return [self.inputs[0].get_name()]

    def __init__(
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        *,
        unbacked_bindings=None,
    ):
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            unbacked_bindings=unbacked_bindings,
        )


@dataclasses.dataclass
class MultiOutputLayout(IRNode):
    device: torch.device


class MultiOutput(ExternKernel):
    # Given an input MultiOutputLayout buffer, indexes out an actual buffer
    # from that result.  This doesn't actually produce multiple outputs,
    # that's MultiOutputLayout!
    def codegen_list_tuple_access(self, basename, indices):
        if len(indices) > 0:
            itype, i = indices[0]
            if issubclass(itype, list):
                return self.codegen_list_tuple_access(f"{basename}[{i}]", indices[1:])
            elif issubclass(itype, tuple):
                # cpp wrapper code needs to use std::get<> to access a tuple
                tuple_access = V.graph.wrapper_code.codegen_tuple_access(
                    basename, self.get_name(), str(i)
                )
                return self.codegen_list_tuple_access(tuple_access, indices[1:])
            elif issubclass(itype, dict):
                return self.codegen_list_tuple_access(f"{basename}['{i}']", indices[1:])
            else:
                raise AssertionError("non supported index type: ", itype)
        else:
            return basename

    def codegen(self, wrapper):
        wrapper.codegen_multi_output(
            self.get_name(),
            self.codegen_list_tuple_access(self.inputs[0].get_name(), self.indices),
        )

    def __init__(self, layout, input, indices: List[Tuple[Any, ...]]):
        super().__init__(None, layout, [input], ())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        self.indices = indices

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return self.inputs[0].get_unbacked_symbol_uses()

    def should_allocate(self):
        return False

    def get_inputs_that_alias_output(self):
        return [
            inp.get_name()
            for inp in self.inputs
            if isinstance(inp, FallbackKernel)
            and len(inp.get_inputs_that_alias_output()) > 0
        ]


@dataclasses.dataclass
class MutableBox(IRNode):
    """
    TensorBox / StorageBox allow in-place mutation of Tensors
    """

    data: IRNode

    def __getattr__(self, name):
        fn = getattr(self.data, name)
        if callable(fn):
            return fn
        raise AttributeError(f"{type(self.data).__name__}.{name} not callable")

    def realize(self):
        return self.data.realize()

    def get_unbacked_symbol_uses(self) -> Set[sympy.Symbol]:
        return self.data.get_unbacked_symbol_uses()

    def get_read_names(self) -> Set[str]:
        return self.data.get_read_names()

    def get_defining_op(self):
        return self.data.get_defining_op()

    def codegen_reference(self, writer=None):
        return self.data.codegen_reference(writer)

    @property
    def layout(self):
        return self.data.get_layout()

    def get_layout(self):
        return self.layout

    def get_size(self):
        return self.data.get_size()

    @property
    def dtype(self):
        return self.data.dtype

    def __str__(self):
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
    def create(data):
        return TensorBox(StorageBox(data))


class StorageBox(MutableBox):
    def is_input_buffer(self):
        if isinstance(self.data, (InputBuffer, ReinterpretView)):
            return self.data.get_name() in V.graph.graph_inputs
        return False

    def is_module_buffer(self):
        return (
            isinstance(self.data, (ConstantBuffer))
            and self.data.get_name() in V.graph.constants
        )

    def realize(self):
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

    def realize_hint(self):
        """
        Called on buffers we expect to be forced to realize later.
        """
        if (
            isinstance(self.data, (Pointwise, Reduction))
            and self.num_reads() > 1
            and self.is_pointwise_non_scalar_tensor_num_reads_larger_than_one()
        ):
            self.realize()

    def has_exceeded_max_reads(self):
        return isinstance(self.data, Pointwise) and (
            self.num_reads() > config.realize_acc_reads_threshold
            or self.has_large_inner_fn()
        )

    def mark_reuse(self, users):
        """
        A heuristic to decide if we should realize a tensor
        that is used multiple times.
        """

        def should_realize_on_cpu(loops: Union[Pointwise, Reduction]):
            """
            The heuristic for realizing reused result of heavy ops on cpu
            """
            heavy_ops = ["exp", "sigmoid"]  # a list of heavy ops
            fn_str = loops.inner_fn_str()
            return any((op + "(") in fn_str for op in heavy_ops)

        if (
            users > 1
            and isinstance(self.data, (Pointwise, Reduction))
            and (
                self.num_reads() > config.realize_reads_threshold
                or self.has_large_inner_fn()
                or (is_cpu(self.data) and should_realize_on_cpu(self.data))
            )
        ):
            self.realize()

    @cache_on_self
    def num_reads(self):
        data = self.data
        if isinstance(data, (InputsKernel, InputBuffer, ReinterpretView)):
            return 1
        if isinstance(data, ComputedBuffer):
            read_writes = data.get_read_writes()
        else:
            assert isinstance(data, (Pointwise, Reduction)), type(data)
            read_writes = ComputedBuffer(
                name=None,
                layout=FlexibleLayout(
                    device=data.get_device(),
                    dtype=data.get_dtype(),
                    size=data.get_size(),
                ),
                data=data,
            ).get_read_writes()
        return len(read_writes.reads)

    @cache_on_self
    def is_pointwise_non_scalar_tensor_num_reads_larger_than_one(self):
        # Skip the check for non Pointwise instances
        return (
            (sum(read.index != 0 for read in self.data.get_reads()) > 1)
            if isinstance(self.data, Pointwise)
            and all(
                not isinstance(read, dependencies.StarDep)
                for read in self.data.get_reads()
            )
            else True
        )


@dataclasses.dataclass
class Subgraph(IRNode):
    name: str
    graph_module: torch.fx.GraphModule
    graph: Optional[GraphLowering] = None


def _has_aliased_buffers(buffers):
    buffers = [
        buffer.unwrap_view() if isinstance(buffer, ReinterpretView) else buffer
        for buffer in buffers
    ]
    # assuming the same buffer is represented by the same IRNode object
    return len({id(buffer) for buffer in buffers}) < len(buffers)


@dataclasses.dataclass
class Conditional(ExternKernel):
    predicate: Optional[IRNode] = None
    operands: Optional[List[TensorBox]] = None
    true_subgraph: Optional[Subgraph] = None
    false_subgraph: Optional[Subgraph] = None
    outputs: Optional[List[MultiOutput]] = None

    def __init__(
        self,
        predicate: IRNode,
        operands: List[TensorBox],
        true_subgraph: Subgraph,
        false_subgraph: Subgraph,
        layout: MultiOutputLayout,
    ):
        self.predicate = predicate
        self.operands = operands
        self.true_subgraph = true_subgraph
        self.false_subgraph = false_subgraph

        inputs = []
        if not isinstance(predicate, ShapeAsConstantBuffer):
            inputs.append(predicate)
        inputs.extend(operands)

        super().__init__(
            name=None,
            layout=layout,  # type: ignore[arg-type]
            inputs=inputs,  # type: ignore[list-item]
        )

        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(
        cls,
        predicate: TensorBox,
        true_fn: Subgraph,
        false_fn: Subgraph,
        operands: List[TensorBox],
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
        false_outputs = true_fn.graph.graph_outputs  # type: ignore[union-attr]

        for name, outputs in (("true_fn", true_outputs), ("false_fn", false_outputs)):
            if _has_aliased_buffers(true_outputs):
                raise AssertionError(
                    "Output aliasing is currently not supported in compiled torch.cond. "
                    f"The outputs of the {name} subgraph of torch.cond are aliased: {outputs}"
                )

        # make sure true and false outputs are structurally equivalent
        assert len(true_outputs) == len(false_outputs), (true_outputs, false_outputs)
        for i, (to, fo) in enumerate(zip(true_outputs, false_outputs)):
            assert to.get_size() == fo.get_size(), (i, to, fo)
            assert to.get_stride() == fo.get_stride(), (i, to, fo)
            assert to.get_device() == fo.get_device(), (i, to, fo)
            assert to.get_dtype() == fo.get_dtype(), (i, to, fo)
            assert to.get_layout().offset == fo.get_layout().offset, (i, to, fo)

        if not isinstance(predicate, ShapeAsConstantBuffer):
            # use predicate device for consistent codegen-ing
            device = predicate.get_device()
        else:
            # predicate is not a Tensor: use first operand's device
            assert (
                len(operands) > 0
            ), "When predicate is not a Tensor, there must be at least one operand in torch.cond."
            device = operands[0].get_device()

        conditional = Conditional(
            predicate=predicate,
            operands=operands,
            true_subgraph=true_fn,
            false_subgraph=false_fn,
            layout=MultiOutputLayout(device),
        )

        outputs = [
            MultiOutput(
                FixedLayout(
                    device=output.get_device(),
                    dtype=output.get_dtype(),
                    size=output.get_size(),
                    stride=output.get_stride(),
                    offset=output.get_layout().offset,
                ),
                conditional,
                [(list, i)],
            )
            # as the true and false outputs are equivalent,
            # we can use either of them here as a "template"
            for i, output in enumerate(true_outputs)
        ]

        conditional.outputs = outputs
        return outputs

    def codegen(self, wrapper):
        wrapper.codegen_conditional(self)


@dataclasses.dataclass
class WhileLoop(ExternKernel):
    carried_inputs: Optional[List[TensorBox]] = None
    additional_inputs: Optional[List[TensorBox]] = None
    cond_subgraph: Optional[Subgraph] = None
    body_subgraph: Optional[Subgraph] = None
    outputs: Optional[List[MultiOutput]] = None

    def __init__(
        self,
        carried_inputs: List[TensorBox],
        additional_inputs: List[TensorBox],
        cond_subgraph: Subgraph,
        body_subgraph: Subgraph,
        layout: MultiOutputLayout,
    ):
        self.carried_inputs = carried_inputs
        self.additional_inputs = additional_inputs
        self.cond_subgraph = cond_subgraph
        self.body_subgraph = body_subgraph

        super().__init__(
            name=None,
            layout=layout,  # type: ignore[arg-type]
            inputs=carried_inputs + additional_inputs,  # type: ignore[list-item]
        )

        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(
        cls,
        cond_fn: Subgraph,
        body_fn: Subgraph,
        carried_inputs: List[TensorBox],
        additional_inputs: List[TensorBox],
    ):
        carried_inputs = [cls.realize_input(x) for x in carried_inputs]
        additional_inputs = [cls.realize_input(x) for x in additional_inputs]
        all_inputs = carried_inputs + additional_inputs

        fx_all_inputs = V.graph.current_node.args[-2] + V.graph.current_node.args[-1]  # type: ignore[operator]
        fake_all_inputs = [x.meta["val"] for x in fx_all_inputs]  # type: ignore[union-attr]

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

        cond_outputs = cond_fn.graph.graph_outputs  # type: ignore[union-attr]
        body_outputs = body_fn.graph.graph_outputs  # type: ignore[union-attr]

        if _has_aliased_buffers(body_outputs):
            raise AssertionError(
                "Output aliasing is currently not supported in compiled torch.while_loop. "
                f"The outputs of the body_fn subgraph of torch.while_loop are aliased: {body_outputs}"
            )

        # make sure cond_fn returns a boolean scalar Tensor
        assert len(cond_outputs) == 1, cond_outputs
        assert cond_outputs[0].get_dtype() == torch.bool, cond_outputs
        assert len(cond_outputs[0].get_size()) == 0, cond_outputs

        assert (
            len(all_inputs) > 0
        ), "torch.while_loop is assumed to have at least one operand."

        device = all_inputs[0].get_device()

        # make sure carried_inputs and body outputs are structurally equivalent
        assert len(carried_inputs) == len(body_outputs), (carried_inputs, body_outputs)
        for i, (op, bo) in enumerate(zip(carried_inputs, body_outputs)):
            assert op.get_size() == bo.get_size(), (i, op, bo)
            assert op.get_stride() == bo.get_stride(), (i, op, bo)
            # assume all carried_inputs and outputs are on the same device
            # as the MultiOutputLayout below requires single device
            assert op.get_device() == bo.get_device() == device, (i, op, bo, device)
            assert op.get_dtype() == bo.get_dtype(), (i, op, bo)
            assert op.get_layout().offset == bo.get_layout().offset, (i, op, bo)

        while_loop = WhileLoop(
            carried_inputs=carried_inputs,
            additional_inputs=additional_inputs,
            cond_subgraph=cond_fn,
            body_subgraph=body_fn,
            # asserted above that there is at least one operand
            layout=MultiOutputLayout(device),
        )

        outputs = [
            MultiOutput(
                FixedLayout(
                    device=output.get_device(),
                    dtype=output.get_dtype(),
                    size=output.get_size(),
                    stride=output.get_stride(),
                    offset=output.get_layout().offset,
                ),
                while_loop,
                [(list, i)],
            )
            for i, output in enumerate(body_outputs)
        ]

        for inp, out in zip(carried_inputs, outputs):
            if inp.get_name() in V.graph.graph_inputs:
                # if a carried input of the while_loop is a graph input,
                # it can be returned as is when the number of iterations
                # is zero. due to this, we can't (generally) reuse the
                # output buffers corresponding to the graph inputs, as
                # the inputs may end up being mutated.
                V.graph.never_reuse_buffers.add(out.get_name())

        while_loop.outputs = outputs
        return outputs

    def codegen(self, wrapper):
        wrapper.codegen_while_loop(self)


class EffectfulKernel(FallbackKernel):
    def __init__(
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs=None,
        *,
        unbacked_bindings=None,
    ):
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

        effect_type = get_effect_key(kernel, (*nontensor_args, *tensor_args), kwargs)
        assert effect_type is not None
        self.effect_type = effect_type
        self.prev_effect_buffer = V.graph.effectful_ops.get(effect_type, None)
        V.graph.effectful_ops[effect_type] = self

    def get_read_writes(self):
        read_writes = super().get_read_writes()

        if self.prev_effect_buffer is not None:
            read_writes.reads.add(
                dependencies.StarDep(self.prev_effect_buffer.get_name())
            )

        return read_writes

    def has_side_effects(self):
        return True


@dataclasses.dataclass
class TorchBindObject(IRNode):
    name: str
    value: torch._C.ScriptObject

    def get_name(self):
        return self.name

    def get_device(self):
        return None  # is there a device??

    def codegen_reference(self, writer=None):
        return self.name


class InterpreterShim(torch.fx.Interpreter):
    @staticmethod
    @functools.lru_cache(None)
    def _dummy_gm():
        return torch.fx.symbolic_trace(identity)

    def __init__(self, graph, submodules):
        # call super() with a placeholder to avoid constructing a
        # GraphModule which is very expensive (it does codegen).
        super().__init__(self._dummy_gm(), garbage_collect_values=False)
        self.module = self  # type: ignore[assignment]
        self.graph = graph
        self.submodules = submodules
        self.extra_traceback = False
        self.fetch_attr = submodules.__getitem__  # type: ignore[method-assign]
        self.current_node = None

    def run_node(self, n: torch.fx.Node) -> Any:
        self.current_node = n
        return super().run_node(n)

    def run(self, *args, **kwargs):
        with V.set_interpreter_handler(self):
            return super().run(*args, **kwargs)


class LoopBody:
    """
    Captures the body of a Loops subclass into an FX graph.  Persists any
    indexing simplifications and makes it easier to analyze loop bodies.
    """

    def __init__(self, fn, args, var_ranges):
        super().__init__()
        self.var_ranges = var_ranges
        self.indexing_exprs = {}
        self.indexing_exprs_name = {}
        self.reads = []
        self.writes = []
        self.reads_name2expr = {}
        self.writes_name2expr = {}
        self.other = []
        self.submodules = {"get_index": self.get_index}
        self.subblocks = {}
        self.indirect_vars = []
        self.indirect_var_ranges: Dict[sympy.Symbol, sympy.Expr] = {}
        self.root_block = LoopBodyBlock(self, fn, args)
        self.indexing = None

    @cache_on_self
    def get_nodes(self):
        all_graphs = itertools.chain(
            (self.root_block.graph,),
            (block.graph for block in self.subblocks.values()),
        )
        return [node for graph in all_graphs for node in graph.nodes]

    @cache_on_self
    def bounds(self):
        # Doing a local import to avoid dumping all the code here
        from .bounds import BoundVars

        return BoundVars(self)

    def debug_str(self):
        lines = [f"var_ranges = {dict(self.var_ranges)}"]
        lines.extend([f"{name} = {val}" for name, val in self.indexing_exprs.items()])
        lines.extend(
            [
                block.debug_str(name)
                for name, block in itertools.chain(
                    [("body", self.root_block)], self.subblocks.items()
                )
            ]
        )
        return "\n".join(lines)

    def add_index_expr(self, expr: sympy.Expr, category, buf_name):
        getattr(self, category).append(expr)
        if buf_name is not None:
            getattr(self, f"{category}_name2expr")[buf_name] = expr
        if expr not in self.indexing_exprs_name:
            name = f"index{len(self.indexing_exprs)}"
            self.indexing_exprs_name[expr] = name
            self.indexing_exprs[name] = expr
        return self.indexing_exprs_name[expr]

    def add_submodule(self, block, prefix):
        """Not actually for nn.Modules, but subblocks in generated code are mapped to FX call_module opcodes"""
        if prefix[-1].isnumeric() and prefix not in self.submodules:
            name = prefix
        else:
            name = f"{prefix}{len(self.submodules)}"
        self.submodules[name] = block
        return name

    def add_indirect(self, size):
        var = sympy_index_symbol_with_prefix(SymT.INDIRECT, len(self.indirect_vars))
        assert var not in self.indirect_var_ranges
        self.indirect_vars.append(var)
        self.indirect_var_ranges[var] = size
        return var

    def replace_indirect(self, old, new):
        """Swap in a variable used in indirect indexing"""
        if str(old) == str(new):
            return
        assert self.indexing is not None
        self.indexing = {k: sympy_subs(v, {old: new}) for k, v in self.indexing.items()}

    def get_index(self, name):
        assert self.indexing is not None
        return self.indexing[name]

    def indexing_from_args(self, indices):
        index = [*itertools.chain.from_iterable(indices)]
        assert len(index) == len(self.var_ranges), (index, self.var_ranges)
        assert all(v not in self.var_ranges for v in index)
        replacements = dict(zip(self.var_ranges.keys(), index))
        return {
            name: sympy_subs(expr, replacements)
            for name, expr in self.indexing_exprs.items()
        }

    def __call__(self, *indices):
        self.indexing = self.indexing_from_args(indices)
        result = self.root_block()
        self.indexing = None
        return result


class LoopBodyBlock:
    """
    Captures the body of a Loops subclass into an FX graph.
    In normal cases there will be a 1:1 mapping between LoopBody and
    LoopBodyBlock, hower in the case of ops.masked() the masked out
    operations will manifest as an extra LoopBodyBlock.
    """

    def __init__(self, body: LoopBody, fn: Callable[..., Any], args: List[Any]):
        self.body = body

        def add_index(expr, category, buf_name=None):
            return tracer.create_proxy(
                "call_module",
                "get_index",
                (self.body.add_index_expr(expr, category, buf_name),),
                {},
            )

        class CaptureIndexing(V.WrapperHandler):  # type: ignore[name-defined]
            self.name = "CaptureIndexing"

            def load(self, name: str, index: sympy.Expr):
                index = add_index(index, "reads", name)
                return self._inner.load(name, index)

            def store(self, name, index, value, mode=None):
                index = add_index(index, "writes", name)
                return self._inner.store(name, index, value, mode)

            def store_reduction(self, name, index, value):
                index = add_index(index, "writes", name)
                return self._inner.store_reduction(name, index, value)

            def reduction(self, dtype, src_dtype, reduction_type, value):
                result = self._inner.reduction(dtype, src_dtype, reduction_type, value)
                if "welford" in reduction_type:
                    return tuple(result[i] for i in range(3))
                return result

            def index_expr(self, index, dtype):
                if isinstance(index, (int, sympy.Integer)):
                    return self._inner.constant(int(index), dtype)
                index = add_index(index, "other")
                return self._inner.index_expr(index, dtype)

            def check_bounds(self, index, size, lower, upper):
                index = add_index(index, "other")
                size = add_index(size, "other")
                return self._inner.check_bounds(index, size, lower, upper)

            def bucketize(
                self,
                values,
                offsets_name: str,
                offsets_size: sympy.Expr,
                indexing_dtype: torch.dtype,
                right: bool,
            ):
                offsets_size = add_index(offsets_size, "other")
                return self._inner.bucketize(
                    values, offsets_name, offsets_size, indexing_dtype, right
                )

            @staticmethod
            def masked(mask_proxy, masked_body: Callable[..., Any], other_proxy):
                """
                Recursively capture the masked out body in another LoopBodyBlock
                """

                subblock: LoopBodyBlock

                def shim(mask, other):
                    return V.ops.masked(mask, subblock, other)

                name = self.body.add_submodule(shim, "masked_subblock")
                subblock = LoopBodyBlock(self.body, masked_body, [])
                self.body.subblocks[name] = subblock
                return tracer.create_proxy(
                    "call_module", name, (mask_proxy, other_proxy), {}
                )

            @staticmethod
            def scan(
                dtype_proxy,
                combine_fn: Callable[
                    [Tuple[Any, ...], Tuple[Any, ...]], Tuple[Any, ...]
                ],
                value_proxy,
            ):
                def shim(dtypes, values):
                    return V.ops.scan(dtypes, combine_fn, values)

                name = self.body.add_submodule(shim, "scan")
                result = tracer.create_proxy(
                    "call_module",
                    name,
                    (dtype_proxy, value_proxy),
                    {},
                )
                # Proxies are iterable, but some methods expect tuples/lists
                return tuple(result[i] for i in range(len(value_proxy)))

            def sort(self, dtypes, values, stable, descending):
                result = self._inner.sort(dtypes, values, stable, descending)
                # Proxies are iterable, but some methods expect tuples/lists
                return tuple(result[i] for i in range(len(values)))

            def frexp(self, value_proxy):
                result = self._inner.frexp(value_proxy)
                # Proxies are iterable, but some methods expect tuples/lists
                return (result[0], result[1])

            @staticmethod
            def indirect_indexing(index_proxy, size, check=True, wrap_neg=True):
                """
                Flow data from tensors into indexing formulas.
                Introduce a call_module to update the indexing.
                """

                var = self.body.add_indirect(size)

                def set_indirect(new_var):
                    self.body.replace_indirect(
                        var, V.ops.indirect_indexing(new_var, size, check, wrap_neg)
                    )

                tracer.create_proxy(
                    "call_module",
                    self.body.add_submodule(set_indirect, f"set_{var}"),
                    (index_proxy,),
                    {},
                )
                return var

            @staticmethod
            def output(result):
                tracer.create_proxy("output", "output", (result,), {})

        tracer = torch.fx.Tracer()
        tracer.graph = torch.fx.Graph(tracer_cls=tracer.__class__)
        proxy_ops = tracer.create_proxy("placeholder", "ops", (), {})

        from .index_propagation import IndexPropagation
        from .sizevars import SimplifyIndexing

        handler: Any = SimplifyIndexing(
            CaptureIndexing(proxy_ops), self.body.var_ranges
        )
        if config.constant_and_index_propagation:
            handler = IndexPropagation(
                handler, self.body.var_ranges, self.body.indirect_var_ranges
            )

        with V.set_ops_handler(handler):
            # This indirection is just a cute way to get IndexPropagation to
            # unwrap the return value.
            ops.output(fn(*args))
        self.graph = tracer.graph

    def __call__(self):
        graph = self.graph
        submodules = self.body.submodules

        return InterpreterShim(graph, submodules).run(V.get_ops_handler())

    def debug_str(self, name="block"):
        code = torch.fx.GraphModule(self.body.submodules, self.graph).code
        return re.sub(
            # strip `; del var0` suffixes to make output prettier
            r";[^\n]*",
            "",
            code.strip().replace("def forward(", f"def {name}("),
        )


class _CollectiveKernel(FallbackKernel):
    def should_allocate(self):
        return False

    def has_side_effects(self):
        return True

    # This is identical to FallbackKernel.set_cpp_kernel(), minus the
    # part that checks against input aliasing and mutation.
    def set_cpp_kernel(self, kernel):
        from .codegen.wrapper import get_cpp_op_schema

        self.cpp_kernel_name = kernel._schema.name
        self.cpp_kernel_overload_name = kernel._schema.overload_name
        self.cpp_kernel_key = f"{self.cpp_kernel_name.replace('::', '_')}_{self.cpp_kernel_overload_name}"  # type: ignore[union-attr]

        self.cpp_op_schema = get_cpp_op_schema(kernel)
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
    def create_inplace(
        cls, kernel, inputs: Union[TensorBox, List[TensorBox]], *args, **kwargs
    ) -> None:
        cpp_kernel_name = kernel._name
        python_kernel_name = cpp_kernel_name.replace("::", ".")
        with V.graph.fake_mode:
            (
                example_output,
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
            NoneLayout(device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )
        packed.cpp_kernel_name = cpp_kernel_name
        packed.python_kernel_name = python_kernel_name

        inps = pytree.tree_leaves(inputs)
        packed.mutation_outputs.extend(
            [MutationOutput(NoneLayout(device), buf, packed) for buf in inps]
        )

        # For inplace collective ops, the input is guaranteed to be alias of the returned value of op.
        packed.alias_names.extend([inp.get_name() for inp in inps])
        if "out" in kwargs:
            packed.mutation_outputs.append(
                MutationOutput(NoneLayout(device), kwargs["out"], packed)
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
    def create_out_of_place(
        cls, kernel, inputs: Union[TensorBox, List[TensorBox]], *args, **kwargs
    ):
        cpp_kernel_name = kernel._name
        python_kernel_name = cpp_kernel_name.replace("::", ".")
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
                MultiOutputLayout(device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            packed.cpp_kernel_name = cpp_kernel_name
            packed.python_kernel_name = python_kernel_name
            packed.outputs = [
                MultiOutput(
                    cls.tensor_to_layout(tensor),
                    packed,
                    [(list, i)],
                )
                for i, tensor in enumerate(example_output)
            ]
            return packed.outputs
        else:
            packed = cls(
                cls.tensor_to_layout(example_output),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            packed.cpp_kernel_name = cpp_kernel_name
            packed.python_kernel_name = python_kernel_name
            packed.outputs = [packed]
            return packed


class _WaitKernel(_CollectiveKernel):
    def get_volatile_reads(self):
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
    def create_wait(cls, kernel, inp: TensorBox) -> None:
        with V.graph.fake_mode:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, inp)
        assert not unbacked_bindings, f"{kernel} {unbacked_bindings}"
        packed = cls(
            NoneLayout(inp.get_device()),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )
        packed.mutation_outputs.append(
            MutationOutput(NoneLayout(inp.get_device()), inp, packed)
        )

    def get_read_writes(self):
        read_writes = super().get_read_writes()
        # See [Out-of-Place Collective Safety].
        volatile_reads = self.get_volatile_reads()
        for vr in volatile_reads:
            read_writes.reads.add(dependencies.StarDep(vr.get_name()))
        return read_writes


# NB: recursive structure here reflects val_to_arg_str, avoid
# calling free_unbacked_symbols on "exotic" types that don't get pexpr
# treatment
def maybe_free_unbacked_symbols(s):
    if isinstance(s, (SymTypes, sympy.Expr)):
        # This branch should be impossible in return position
        return free_unbacked_symbols(s)
    elif isinstance(s, (tuple, list)):
        r = set()
        for t in s:
            r |= maybe_free_unbacked_symbols(t)
        return r
    elif isinstance(s, torch.Tensor):
        # This branch is impossible in constant-args position
        return free_unbacked_symbols(s)
    else:
        return set()
