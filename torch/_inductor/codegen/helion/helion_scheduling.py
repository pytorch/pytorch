from __future__ import annotations

from typing import Any, cast, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...ir import ComputedBuffer, Reduction
from ...loop_body import MemoryUsageType
from ...ops_handler import OpsHandler
from ...scheduler import SchedulerNode
from ...utils import get_fused_kernel_name
from ...virtualized import V
from ..triton import TritonScheduling
from .helion_kernel import HelionKernel, REDUCTIONS
from .helion_op_overrides import HelionOverrides


if TYPE_CHECKING:
    from ...scheduler import BaseSchedulerNode


# Python operators HelionOverrides inherits from OpOverrides (a + b, a < b, ~a),
# which mean the same on torch tensors. Not every inherited spelling does:
# OpOverrides.mod writes a % b, which floors on torch tensors (-7 % 2 == 1)
# while Inductor's mod truncates (-1).
INHERITED_OPS = OrderedSet(
    [
        "add",
        "sub",
        "mul",
        "truediv",
        "neg",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and_",
        "or_",
        "xor",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_not",
    ]
)
# Every op a Helion kernel emits: those HelionKernel (load, store, dot, ...) or
# HelionOverrides (exp, where, ...) implements, the inherited operators,
# SIMDKernel.store_reduction (which calls HelionKernel.store) and the body's output.
HELION_OPS = (
    OrderedSet(
        name
        for cls in (HelionKernel, HelionOverrides)
        for name in vars(cls)
        if not name.startswith("_") and name in vars(OpsHandler)
    )
    | INHERITED_OPS
    | OrderedSet(["store_reduction", "output"])
)
MATMUL_DTYPES = OrderedSet([torch.float16, torch.bfloat16, torch.float32])


def python_wrapper() -> bool:
    return not (V.graph.cpp_wrapper or V.graph.fx_wrapper)


def reads_dot_operands_by_tile(node: SchedulerNode) -> bool:
    """Whether each load of a native matmul reads a [M, K] or [K, N] tile,
    i.e. is affine in (*B, M, K) or (*B, K, N) with non-negative strides."""
    body = node._body
    *_, m, n = body.iter_vars
    variables = OrderedSet(body.var_ranges)
    for entry in body.memory_usage[MemoryUsageType.LOAD]:
        index = body.indexing_exprs[entry.index_name]
        if not index.free_symbols <= variables or not index.is_polynomial(*variables):
            return False
        poly = sympy.Poly(index, *variables)
        if poly.total_degree() > 1 or any(c < 0 for c in poly.coeffs()):
            return False
        if (m in index.free_symbols) == (n in index.free_symbols):
            return False
    return True


def is_helion_matmul(node: BaseSchedulerNode) -> bool:
    """Whether `node` holds a native matmul (from triton.native_matmul) that a
    Helion kernel emits."""
    if not config.helion.native_matmul or node.is_template() or not python_wrapper():
        return False
    for snode in node.get_nodes():
        if snode.is_native_matmul():
            dtype = cast(Reduction, cast(ComputedBuffer, snode.node).data).src_dtype
            if dtype not in MATMUL_DTYPES:
                return False
            # Dot operands have no flat fallback: without this check,
            # x @ v.expand(K, N) fails with "Cannot read a Helion dot operand at r0_index".
            return reads_dot_operands_by_tile(cast(SchedulerNode, snode))
    return False


def helion_supports(node: SchedulerNode) -> bool:
    # Helion kernels do not implement these numerics modes yet, so they stay Triton.
    if (
        config.strict_reduction
        or config.batch_invariant
        or config.deterministic
        or config.eager_numerics.division_rounding
        or config.eager_numerics.disable_ftz
        or config.emulate_precision_casts
    ):
        return False
    reduction_type = cast(ComputedBuffer, node.node).get_reduction_type()
    body = node._body
    return (
        # sign(x): HelionOverrides has no sign, so codegen raises NotImplementedError.
        # Scatters (index_add_, index_put_(accumulate=True)) stop here on their
        # indirect_indexing.
        not OrderedSet(body.op_counts) - HELION_OPS
        # x.argmax(-1): "Helion kernels do not support argmax".
        and (
            reduction_type in (None, *REDUCTIONS)
            or (reduction_type == "dot" and is_helion_matmul(node))
        )
        # x * torch.tensor(2.0): the kernel gets a CPU tensor, "Pointer argument
        # cannot be accessed from Triton (cpu tensor?)".
        and not any(V.graph.is_unspec_arg(dep.name) for dep in node.read_writes.reads)
        # x[:, :50] * 2 with a dynamic dim 1: the index holds that dim's stride,
        # "Unsupported Helion index: s27*((xindex//50)) + ...".
        and all(
            index.free_symbols <= OrderedSet(body.var_ranges)
            for index in body.indexing_exprs.values()
        )
    )


class HelionScheduling(TritonScheduling):
    """Emits the schedules HelionKernel supports as Helion kernels and the rest
    as Triton."""

    def __init__(self, scheduler: Any) -> None:
        super().__init__(scheduler)
        # Only codegen_node_schedule() emits Helion kernels; fusion benchmarks
        # and combo kernels stay Triton.
        self.use_helion = False

    @staticmethod
    def format_kernel_definition(name: str, source: str) -> str:
        escaped = source.replace("\\", "\\\\").replace('"""', '\\"""')
        return (
            f'{name}_source = """\\\n{escaped}"""\n'
            f"{name} = PyCodeCache.load({name}_source).kernel"
        )

    def can_fuse(self, node1, node2):
        return super().can_fuse(node1, node2)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    def generate_node_schedule(
        self, nodes, numel, rnumel, *, required_post_reduction_index=None
    ):
        # Dots first, so that they share one K loop. No dot reads another node
        # of its kernel, so this keeps every producer before its use.
        if required_post_reduction_index is None and any(map(is_helion_matmul, nodes)):
            nodes = sorted(nodes, key=lambda node: not is_helion_matmul(node))
        return super().generate_node_schedule(
            nodes,
            numel,
            rnumel,
            required_post_reduction_index=required_post_reduction_index,
        )

    def codegen_node_schedule(self, kernel_features):
        self.use_helion = python_wrapper() and all(
            map(helion_supports, kernel_features.scheduler_nodes())
        )
        try:
            return super().codegen_node_schedule(kernel_features)
        finally:
            self.use_helion = False

    def create_kernel_choices(self, kernel_features, kernel_args, kernel_kwargs):
        (tiling,) = kernel_args
        if self.use_helion and HelionKernel.supports_tiling(tiling):
            return [HelionKernel(*kernel_args, **kernel_kwargs)]
        return super().create_kernel_choices(
            kernel_features, kernel_args, kernel_kwargs
        )

    def define_kernel(self, src_code, node_schedule, kernel):
        if not isinstance(kernel, HelionKernel):
            return super().define_kernel(src_code, node_schedule, kernel)
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]
        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        name = "_".join(
            part
            for part in ("helion", fused_name, wrapper.next_kernel_suffix())
            if part
        )
        wrapper.src_to_kernel[src_code] = name
        wrapper.add_import_once("from torch._inductor.codecache import PyCodeCache")
        # gpu=False keeps it out of the compile-time autotune block, which has
        # no PyCodeCache import.
        wrapper.define_kernel(
            name,
            self.format_kernel_definition(name, src_code),
            gpu=False,
            standalone=True,
        )
        return name
