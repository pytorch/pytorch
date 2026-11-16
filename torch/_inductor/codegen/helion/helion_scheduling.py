from __future__ import annotations

from typing import Any, cast, TYPE_CHECKING

import sympy

import torch
from torch.utils._helion import has_helion
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...ir import ComputedBuffer, Reduction
from ...ops_handler import OpsHandler
from ...utils import get_fused_kernel_name
from ...virtualized import V
from ..triton import TritonScheduling
from .helion_kernel import HelionKernel, REDUCTIONS
from .helion_op_overrides import HelionOverrides


if TYPE_CHECKING:
    from ...ir import IRNode
    from ...scheduler import BaseSchedulerNode, SchedulerNode


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


def numerics_supported() -> bool:
    # Helion kernels do not implement these numerics modes yet, so they stay Triton.
    return not (
        config.strict_reduction
        or config.batch_invariant
        or config.deterministic
        or config.eager_numerics.division_rounding
        or config.eager_numerics.disable_ftz
        or config.emulate_precision_casts
    )


def lowers_helion_matmul(mat1: IRNode, mat2: IRNode) -> bool:
    """Whether use_native_matmul() lowers mat1 @ mat2 to a dot for a Helion kernel.

    A dot that helion_supports() declines becomes a Triton native matmul kernel,
    so this declines what is known at lowering time not to become a Helion
    kernel, which keeps its extern kernel or template, as without Helion: e.g.
    relu(a @ b) with a dynamic dim of a keeps extern_kernels.mm. What only
    scheduling shows still becomes a Triton native matmul kernel: an operand with
    an indirect load, as in a[i] @ b, or a fused epilogue that Helion does not
    emit, as in a @ b + torch.tensor(2.0) with the scalar on the CPU. These go
    away as HelionKernel supports more."""
    sizes = (*mat1.get_size(), *mat2.get_size())
    return (
        config.helion.enabled
        and config.helion.native_matmul
        and mat1.get_device_or_error().type == "cuda"
        and python_wrapper()
        and numerics_supported()
        and mat1.get_dtype() in MATMUL_DTYPES
        # HelionKernel.supports_tiling() needs static tiles.
        and all(isinstance(size, (int, sympy.Integer)) for size in sizes)
    )


def is_helion_matmul(node: BaseSchedulerNode) -> bool:
    """Whether `node` holds a native matmul that a Helion kernel emits."""
    if not config.helion.native_matmul or node.is_template() or not python_wrapper():
        return False
    for snode in node.get_nodes():
        if snode.is_native_matmul():
            dtype = cast(Reduction, cast(ComputedBuffer, snode.node).data).src_dtype
            return dtype in MATMUL_DTYPES
    return False


def helion_supports(node: SchedulerNode) -> bool:
    if not numerics_supported():
        return False
    reduction_type = cast(ComputedBuffer, node.node).get_reduction_type()
    body = node._body
    return (
        not OrderedSet(body.op_counts) - HELION_OPS
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
        # Fail here rather than with a ModuleNotFoundError when the output code loads.
        if not has_helion():
            raise RuntimeError("helion.enabled is set, but Helion is not installed")
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
