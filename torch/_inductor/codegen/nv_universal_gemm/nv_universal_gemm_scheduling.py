# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM scheduling for PyTorch Inductor.
"""

import hashlib
import logging
import math
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch._inductor.utils import (
    get_fused_kernel_name,
    get_kernel_metadata,
    Placeholder,
)
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...codecache import code_hash, get_path
from ...ir import (
    Buffer,
    ComputedBuffer,
    Layout,
    MultiOutputReduction,
    MultiTemplateBuffer,
    NVUniversalGemmBuffer,
    Pointwise,
    Reduction,
)
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
)
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer
from ..cutlass.python_evt import _ACCUMULATOR_ARG_NAME, CutlassEVTCodegen
from .nv_universal_gemm import NVUniversalGemmCaller


log = logging.getLogger(__name__)

MAIN_SUFFIX = "main"
_BENCHMARK_KERNEL_PREFIX = "nv_gemm_"
EPILOGUE_FN_NAME = "_epilogue_fn"


class NVUniversalGemmScheduling(BaseScheduling):
    """
    Scheduling implementation for NVIDIA Universal GEMM kernels.

    This class is intended to be used in combination with other schedulers,
    and delegated to by CUDACombinedScheduling.
    """

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def _is_nvgemm_ir_buffer(ir_node: Any) -> bool:
        """Return True if `ir_node` is an NVGEMM buffer or an MTB resolving to one.

        Honors finalize_as_*_caller's swap (a MultiTemplateBuffer whose render
        kind is "triton" is no longer NVGEMM, even if the autotune winner was).
        Falls back to the autotune winner only when no swap has happened.
        """
        if isinstance(ir_node, NVUniversalGemmBuffer):
            return True
        if not isinstance(ir_node, MultiTemplateBuffer):
            return False
        if ir_node._render_kind == "triton":
            return False
        if ir_node._render_kind == "nvgemm":
            return True
        # Fast path: avoid forcing autotune just to answer this query.
        if not any(isinstance(c, NVUniversalGemmCaller) for c in ir_node._choices):
            return False
        try:
            min_choice, _ = ir_node.get_min_choice()
            return isinstance(min_choice, NVUniversalGemmCaller)
        except (RuntimeError, ValueError):
            return False

    @staticmethod
    def _has_nvgemm_choice(ir_node: Any) -> bool:
        return isinstance(ir_node, NVUniversalGemmBuffer) or (
            isinstance(ir_node, MultiTemplateBuffer)
            and any(
                isinstance(choice, NVUniversalGemmCaller) for choice in ir_node._choices
            )
        )

    @staticmethod
    def is_nv_universal_gemm_template(node: BaseSchedulerNode) -> bool:
        """Check if a node is an NVGEMM template SchedulerNode."""
        if not isinstance(node, SchedulerNode):
            return False
        return NVUniversalGemmScheduling._is_nvgemm_ir_buffer(node.node)

    @staticmethod
    def _best_nvgemm_choice(
        ir_node: MultiTemplateBuffer,
        require_epilogue_fusion: bool = False,
        min_tile_n: int = 0,
    ) -> NVUniversalGemmCaller:
        """Find the best NVUniversalGemmCaller from an MTB's choice timings."""
        choice_timings = ir_node.choice_timings()
        best: NVUniversalGemmCaller | None = None
        best_time = float("inf")
        for choice in ir_node._choices:
            if not isinstance(choice, NVUniversalGemmCaller):
                continue
            if require_epilogue_fusion and not choice.supports_epilogue_fusion:
                continue
            if choice.kernel.metadata.design.tile_shape[1] < min_tile_n:
                continue
            timing = choice_timings.get(choice, float("inf"))
            if best is None or timing < best_time:
                best_time = timing
                best = choice
        if best is None:
            kind = "EFC kernel" if require_epilogue_fusion else "NVUniversalGemmCaller"
            raise RuntimeError(f"No {kind} found in choices")
        return best

    @staticmethod
    def get_nv_gemm_buffer_from_node(
        node: BaseSchedulerNode,
        require_epilogue_fusion: bool = False,
        min_tile_n: int = 0,
    ) -> NVUniversalGemmBuffer:
        """Extract NVUniversalGemmBuffer from node (direct or via MultiTemplateBuffer)."""
        assert isinstance(node, SchedulerNode)  # noqa: S101
        ir_node = node.node

        if isinstance(ir_node, NVUniversalGemmBuffer):
            return ir_node
        elif isinstance(ir_node, MultiTemplateBuffer):
            # Honor an explicit swap/finalize -- the fusion benchmark loop swaps
            # in each EFC choice one at a time and must not re-select from timings.
            if (
                isinstance(ir_node._render_caller, NVUniversalGemmCaller)
                and (
                    not require_epilogue_fusion
                    or ir_node._render_caller.supports_epilogue_fusion
                )
                and ir_node._render_caller.kernel.metadata.design.tile_shape[1]
                >= min_tile_n
            ):
                selected_choice = ir_node._render_caller
            elif require_epilogue_fusion:
                selected_choice = NVUniversalGemmScheduling._best_nvgemm_choice(
                    ir_node,
                    require_epilogue_fusion=True,
                    min_tile_n=min_tile_n,
                )
            else:
                min_choice, _ = ir_node.get_min_choice()
                if isinstance(min_choice, NVUniversalGemmCaller):
                    selected_choice = min_choice
                else:
                    selected_choice = NVUniversalGemmScheduling._best_nvgemm_choice(
                        ir_node
                    )
            tensor_box = selected_choice.output_node()
            # pyrefly: ignore [missing-attribute]
            return cast(NVUniversalGemmBuffer, tensor_box.data.data)

        raise TypeError(
            f"Expected NVUniversalGemmBuffer or MultiTemplateBuffer, got {type(ir_node).__name__}"
        )

    @staticmethod
    def is_nv_universal_gemm_fused_template(node: BaseSchedulerNode) -> bool:
        """Check if a node is a fused NVIDIA Universal GEMM template."""
        if not isinstance(node, FusedSchedulerNode):
            return False
        return NVUniversalGemmScheduling._is_nvgemm_ir_buffer(node.get_template_node())

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if self.is_nv_universal_gemm_template(node1):
            return self._can_fuse_epilogue_impl(
                cast(SchedulerNode, node1),
                [],
                node2,
            )
        elif self.is_nv_universal_gemm_fused_template(node1):
            fnode1 = cast(FusedSchedulerNode, node1)
            template_snode = next(
                (n for n in fnode1.snodes if self.is_nv_universal_gemm_template(n)),
                None,
            )
            if template_snode is None:
                return False
            return self._can_fuse_epilogue_impl(
                cast(SchedulerNode, template_snode),
                self._unwrap_epilogue_nodes(fnode1),
                node2,
            )
        return False

    def _unwrap_epilogue_nodes(
        self, fused_node: FusedSchedulerNode
    ) -> list[BaseSchedulerNode]:
        """Extract epilogue nodes from a fused node."""
        epilogue_nodes = []
        for node in fused_node.snodes:
            if not self.is_nv_universal_gemm_template(node):
                epilogue_nodes.append(node)
        return epilogue_nodes

    @classmethod
    def _grouped_reduce_config(
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> tuple[str, int, int, str, str] | None:
        nodes = scheduler_node.get_nodes()
        if len(nodes) not in (1, 2):
            return None
        buffers = [snode.node for snode in nodes]
        if not all(isinstance(buffer, ComputedBuffer) for buffer in buffers):
            return None
        buffers = [cast(ComputedBuffer, buffer) for buffer in buffers]
        origin_targets = OrderedSet(
            str(origin.target)
            for buffer in buffers
            for origin in buffer.get_origins()
            if hasattr(origin, "target")
        )
        reduction_targets = {
            "aten.sum.dim_IntList": "sum",
            "aten.mean.dim": "mean",
            "aten.prod.dim_int": "prod",
            "aten.amax.default": "max",
            "aten.amin.default": "min",
        }
        source_targets = OrderedSet(["aten.abs.default", "aten.pow.Tensor_Scalar"])
        matched_reductions = origin_targets & reduction_targets.keys()
        absmax_finalize_targets = OrderedSet(
            ("aten.clamp_min.default", "aten.div.Tensor")
        )
        allowed_targets = (
            OrderedSet(
                (
                    "aten.reshape.default",
                    "prims.convert_element_type.default",
                    "prims.inductor_force_stride_order.default",
                )
            )
            | OrderedSet(reduction_targets)
            | source_targets
            | absmax_finalize_targets
        )
        flex_gemm_origins = OrderedSet(
            target
            for target in origin_targets
            if target == "flex_gemm" or target.startswith("flex_gemm_body_graph_")
        )
        if len(matched_reductions) != 1 or not origin_targets.issubset(
            allowed_targets | flex_gemm_origins
        ):
            return None
        reduction_type = reduction_targets[next(iter(matched_reductions))]
        is_absmax_scale = OrderedSet(
            (
                "aten.abs.default",
                "aten.amax.default",
                "aten.clamp_min.default",
                "aten.div.Tensor",
            )
        ).issubset(origin_targets)
        if origin_targets & absmax_finalize_targets and not is_absmax_scale:
            return None
        source_origins = OrderedSet(
            origin
            for buffer in buffers
            for origin in buffer.get_origins()
            if str(getattr(origin, "target", "")) in source_targets
        )
        if not source_origins:
            source_type = "identity"
        elif len(source_origins) == 1:
            source_origin = next(iter(source_origins))
            if str(source_origin.target) == "aten.abs.default":
                source_type = "abs"
            elif source_origin.args[1] == 2:
                source_type = "square"
            else:
                return None
        else:
            return None
        if is_absmax_scale:
            source_type = "abs_scale"
        node = buffers[0]
        access_node = scheduler_node
        output_name = node.get_name()
        if len(buffers) == 2:
            finalizer = buffers[1]
            if reduction_type == "mean" and cls._is_mean_finalizer(nodes[0], nodes[1]):
                pass
            elif is_absmax_scale and cls._is_absmax_scale_finalizer(nodes[0], nodes[1]):
                pass
            elif cls._is_layout_finalizer(nodes[0], nodes[1]):
                pass
            else:
                return None
            access_node = nodes[0]
            output_name = finalizer.get_name()
        if len(node.data.ranges) not in (2, 3) or len(gemm_node.get_size()) != 2:
            return None
        m, n = gemm_node.get_size()
        out_size = tuple(node.data.ranges)
        if len(out_size) == 2:
            out_m, out_n = out_size
        elif V.graph.sizevars.statically_known_equals(out_size[-1], 1):
            out_m, out_n, _ = out_size
        elif V.graph.sizevars.statically_known_equals(out_size[1], 1):
            out_m, _, out_n = out_size
        else:
            return None

        def known_equals(left, right) -> bool:
            return (
                V.graph.sizevars.statically_known_equals(left, right)
                or V.graph.sizevars.simplify(left - right) == 0
            )

        if isinstance(node.data, Reduction):
            reduction = node.data
            if (
                reduction.reduction_type
                != ("sum" if reduction_type == "mean" else reduction_type)
                or len(reduction.reduction_ranges) != 1
            ):
                return None
            group_extent = reduction.reduction_ranges[0]
            group = V.graph.sizevars.optimization_hint(group_extent)
            if (known_equals(m, out_m) and known_equals(n, out_n * group_extent)) or (
                known_equals(out_m, m) and known_equals(out_n, n // group_extent)
            ):
                axis = 1
                expected_strides = [n, group_extent, 1]
                max_group = None
            elif (known_equals(m, out_m * group_extent) and known_equals(n, out_n)) or (
                known_equals(out_m, m // group_extent) and known_equals(out_n, n)
            ):
                axis = 0
                expected_strides = [group_extent * n, 1, n]
                max_group = 128
            else:
                return None
        elif isinstance(node.data, Pointwise):
            try:
                m = V.graph.sizevars.optimization_hint(m)
                n = V.graph.sizevars.optimization_hint(n)
                out_m = V.graph.sizevars.optimization_hint(out_m)
                out_n = V.graph.sizevars.optimization_hint(out_n)
            except Exception:
                return None
            if m == out_m and out_n > 0 and n % out_n == 0:
                group = n // out_n
                axis = 1
                max_group = n
                expected_strides = [n, group]
            elif n == out_n and out_m > 0 and m % out_m == 0:
                group = m // out_m
                axis = 0
                max_group = 128
                expected_strides = [group * n, 1]
            else:
                return None
        else:
            return None
        if group <= 1 or (max_group is not None and group > max_group):
            return None

        reads = list(access_node.read_writes.reads)
        if not reads or any(read.name != gemm_node.get_name() for read in reads):
            return None
        range_vars = access_node.read_writes.range_vars
        if range_vars is None:
            return None
        if not range_vars:
            return output_name, group, axis, reduction_type, source_type
        if isinstance(node.data, Reduction):
            if len(reads) != 1:
                return None
            strides = V.graph.sizevars.stride_vars(reads[0].index, range_vars)
            expected_stride_options = [expected_strides]
            if axis == 1:
                expected_stride_options.append([n, 1])
            else:
                expected_stride_options.append([1, n])
            if not any(
                len(strides) == len(expected)
                and all(
                    known_equals(stride, expected_stride)
                    for stride, expected_stride in zip(strides, expected)
                )
                for expected in expected_stride_options
            ):
                return None
        else:
            if len(reads) != group:
                return None
            expected_base = (
                n * range_vars[0] + group * range_vars[1]
                if axis == 1
                else group * n * range_vars[0] + range_vars[1]
            )
            offsets = []
            for read in reads:
                strides = V.graph.sizevars.stride_vars(read.index, range_vars)
                if list(strides) != expected_strides:
                    return None
                offsets.append(V.graph.sizevars.simplify(read.index - expected_base))
            expected_offsets = OrderedSet(
                offset if axis == 1 else offset * n for offset in range(group)
            )
            if OrderedSet(offsets) != expected_offsets:
                return None
        return output_name, group, axis, reduction_type, source_type

    @classmethod
    def _grouped_variance_config(
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> tuple[str, int, int, str, str] | None:
        nodes = scheduler_node.get_nodes()
        if len(nodes) != 1 or not isinstance(nodes[0].node, ComputedBuffer):
            return None
        buffer = cast(ComputedBuffer, nodes[0].node)
        if not isinstance(buffer.data, Pointwise):
            return None
        origins = OrderedSet(buffer.get_origins())
        targets = OrderedSet(str(getattr(origin, "target", "")) for origin in origins)
        allowed = OrderedSet(
            (
                "aten.add.Tensor",
                "aten.mean.dim",
                "aten.mul.Tensor",
                "aten.pow.Tensor_Scalar",
                "aten.reshape.default",
                "aten.sub.Tensor",
                "prims.convert_element_type.default",
            )
        )
        if not targets or not targets.issubset(allowed):
            return None

        def target(node: object) -> str:
            return str(getattr(node, "target", ""))

        def tensor_and_scalar(node: object, expected_target: str):
            if target(node) != expected_target:
                return None
            args = getattr(node, "args", ())
            if len(args) < 2:
                return None
            if isinstance(args[0], (int, float)):
                return args[1], float(args[0])
            if isinstance(args[1], (int, float)):
                return args[0], float(args[1])
            return None

        matches: list[tuple[float, float]] = []
        for add in origins:
            add_parts = tensor_and_scalar(add, "aten.add.Tensor")
            if add_parts is None:
                continue
            mul, bias = add_parts
            mul_parts = tensor_and_scalar(mul, "aten.mul.Tensor")
            if mul_parts is None:
                continue
            mean, scale = mul_parts
            if target(mean) != "aten.mean.dim":
                continue
            mean_args = getattr(mean, "args", ())
            if len(mean_args) < 2 or tuple(mean_args[1]) != (-1,):
                continue
            square = mean_args[0]
            square_parts = tensor_and_scalar(square, "aten.pow.Tensor_Scalar")
            if square_parts is None or square_parts[1] != 2.0:
                continue
            centered = square_parts[0]
            if target(centered) != "aten.sub.Tensor":
                continue
            centered_args = getattr(centered, "args", ())
            if len(centered_args) < 2:
                continue
            grouped, group_mean = centered_args[:2]
            if target(group_mean) != "aten.mean.dim":
                continue
            group_mean_args = getattr(group_mean, "args", ())
            if (
                len(group_mean_args) < 3
                or group_mean_args[0] is not grouped
                or tuple(group_mean_args[1]) != (-1,)
                or group_mean_args[2] is not True
            ):
                continue
            matches.append((scale, bias))
        if len(matches) != 1:
            return None

        try:
            m, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
            out_m, out_n = map(V.graph.sizevars.optimization_hint, buffer.get_size())
        except Exception:
            return None
        if out_m != m or out_n <= 0 or n % out_n != 0:
            return None
        group = n // out_n
        if group <= 1 or group > 32:
            return None
        reads = list(scheduler_node.read_writes.reads)
        range_vars = scheduler_node.read_writes.range_vars
        if not reads or range_vars is None:
            return None
        if not range_vars:
            range_vars = tuple(
                sorted(
                    OrderedSet(
                        symbol for read in reads for symbol in read.index.free_symbols
                    ),
                    key=str,
                )
            )
        range_vars = cast(tuple[Any, ...], range_vars)
        if len(range_vars) not in (1, 2):
            return None
        if any(read.name != gemm_node.get_name() for read in reads):
            return None
        if len(range_vars) == 2:
            base = n * range_vars[0] + group * range_vars[1]
            expected_strides = [n, group]
        else:
            base = group * range_vars[0]
            expected_strides = [group]
        offsets = OrderedSet()
        for read in reads:
            strides = V.graph.sizevars.stride_vars(read.index, range_vars)
            if list(strides) != expected_strides:
                return None
            offsets.add(V.graph.sizevars.simplify(read.index - base))
        if offsets != OrderedSet(range(group)):
            return None
        scale, bias = matches[0]
        reduce_type = "variance_affine:" + ":".join(
            format(value, ".17g") for value in (scale, bias)
        )
        return buffer.get_name(), group, 1, reduce_type, "identity"

    @classmethod
    def _direct_bool_mask_config(
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> tuple[str, int, int, str, str] | None:
        nodes = scheduler_node.get_nodes()
        if len(nodes) != 1 or not isinstance(nodes[0].node, ComputedBuffer):
            return None
        buffer = cast(ComputedBuffer, nodes[0].node)
        if (
            not isinstance(buffer.data, Pointwise)
            or buffer.get_dtype() != torch.bool
            or not V.graph.sizevars.statically_known_list_equals(
                buffer.get_size(), gemm_node.get_size()
            )
        ):
            return None
        origins = list(buffer.get_origins())
        comparisons = [
            origin
            for origin in origins
            if str(getattr(origin, "target", ""))
            in ("aten.gt.Scalar", "aten.gt.Tensor")
        ]
        if len(comparisons) != 1:
            return None
        args = comparisons[0].args
        if len(args) < 2 or args[1] != 0:
            return None
        allowed = OrderedSet(
            (
                "aten.gt.Scalar",
                "aten.gt.Tensor",
                "prims.convert_element_type.default",
            )
        )
        if any(str(getattr(origin, "target", "")) not in allowed for origin in origins):
            return None
        reads = list(scheduler_node.read_writes.reads)
        if len(reads) != 1 or reads[0].name != gemm_node.get_name():
            return None
        range_vars = scheduler_node.read_writes.range_vars
        if range_vars is None:
            return None
        if not range_vars:
            range_vars = tuple(sorted(reads[0].index.free_symbols, key=str))
        strides = V.graph.sizevars.stride_vars(reads[0].index, range_vars)
        expected = [1] if len(range_vars) == 1 else list(gemm_node.get_stride())
        if len(strides) != len(expected) or not all(
            V.graph.sizevars.statically_known_equals(actual, wanted)
            for actual, wanted in zip(strides, expected)
        ):
            return None
        return buffer.get_name(), 2, 1, "direct_bool_gt_zero", "identity"

    @classmethod
    def _grouped_logsumexp_config(
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> tuple[str, int, int, str, str] | None:
        nodes = scheduler_node.get_nodes()
        if len(nodes) != 1 or not isinstance(nodes[0].node, ComputedBuffer):
            return None
        buffer = cast(ComputedBuffer, nodes[0].node)
        if not isinstance(buffer.data, Pointwise):
            return None
        origins = OrderedSet(buffer.get_origins())
        targets = OrderedSet(str(getattr(origin, "target", "")) for origin in origins)
        guarded_required = OrderedSet(
            (
                "aten.abs.default",
                "aten.add.Tensor",
                "aten.amax.default",
                "aten.eq.Scalar",
                "aten.exp.default",
                "aten.full.default",
                "aten.log.default",
                "aten.reshape.default",
                "aten.squeeze.dims",
                "aten.sub.Tensor",
                "aten.sum.dim_IntList",
                "aten.where.self",
                "prims.convert_element_type.default",
            )
        )
        stable_required = OrderedSet(
            (
                "aten.add.Tensor",
                "aten.amax.default",
                "aten.exp.default",
                "aten.log.default",
                "aten.reshape.default",
                "aten.squeeze.dim",
                "aten.sub.Tensor",
                "aten.sum.dim_IntList",
                "prims.convert_element_type.default",
            )
        )
        if targets not in (guarded_required, stable_required):
            return None

        def target(node: object) -> str:
            return str(getattr(node, "target", ""))

        matched = False
        for add in origins:
            if target(add) != "aten.add.Tensor":
                continue
            add_args = getattr(add, "args", ())
            if len(add_args) < 2:
                continue
            log_node, squeeze = add_args[:2]
            if target(log_node) != "aten.log.default":
                log_node, squeeze = squeeze, log_node
            if (
                target(log_node) != "aten.log.default"
                or target(squeeze) != "aten.squeeze.dims"
            ):
                continue
            sum_node = getattr(log_node, "args", (None,))[0]
            if target(sum_node) != "aten.sum.dim_IntList":
                continue
            sum_args = getattr(sum_node, "args", ())
            if len(sum_args) < 2 or tuple(sum_args[1]) != (-1,):
                continue
            exp = sum_args[0]
            sub = getattr(exp, "args", (None,))[0]
            if target(exp) != "aten.exp.default" or target(sub) != "aten.sub.Tensor":
                continue
            sub_args = getattr(sub, "args", ())
            squeeze_args = getattr(squeeze, "args", ())
            if len(sub_args) < 2 or len(squeeze_args) < 2:
                continue
            grouped, shift = sub_args[:2]
            if squeeze_args[0] is not shift or tuple(squeeze_args[1]) != (-1,):
                continue
            if target(shift) != "aten.where.self":
                continue
            where_args = getattr(shift, "args", ())
            if len(where_args) < 3:
                continue
            condition, zero, maximum = where_args[:3]
            if target(maximum) != "aten.amax.default":
                continue
            maximum_args = getattr(maximum, "args", ())
            if (
                len(maximum_args) < 3
                or maximum_args[0] is not grouped
                or tuple(maximum_args[1]) != (-1,)
                or maximum_args[2] is not True
            ):
                continue
            if target(condition) != "aten.eq.Scalar":
                continue
            condition_args = getattr(condition, "args", ())
            if (
                len(condition_args) < 2
                or target(condition_args[0]) != "aten.abs.default"
                or getattr(condition_args[0], "args", (None,))[0] is not maximum
                or condition_args[1] != float("inf")
            ):
                continue
            if target(zero) != "aten.full.default":
                continue
            zero_args = getattr(zero, "args", ())
            if len(zero_args) < 2 or zero_args[1] != 0.0:
                continue
            matched = True
            break
        if not matched and targets == stable_required:
            for add in origins:
                if target(add) != "aten.add.Tensor":
                    continue
                add_args = getattr(add, "args", ())
                if len(add_args) < 2:
                    continue
                log_node, squeeze = add_args[:2]
                if target(log_node) != "aten.log.default":
                    log_node, squeeze = squeeze, log_node
                if (
                    target(log_node) != "aten.log.default"
                    or target(squeeze) != "aten.squeeze.dim"
                ):
                    continue
                sum_node = getattr(log_node, "args", (None,))[0]
                if target(sum_node) != "aten.sum.dim_IntList":
                    continue
                sum_args = getattr(sum_node, "args", ())
                if len(sum_args) < 2 or tuple(sum_args[1]) != (-1,):
                    continue
                exp = sum_args[0]
                sub = getattr(exp, "args", (None,))[0]
                if (
                    target(exp) != "aten.exp.default"
                    or target(sub) != "aten.sub.Tensor"
                ):
                    continue
                sub_args = getattr(sub, "args", ())
                squeeze_args = getattr(squeeze, "args", ())
                if len(sub_args) < 2 or len(squeeze_args) < 2:
                    continue
                grouped, maximum = sub_args[:2]
                if (
                    squeeze_args[0] is not maximum
                    or squeeze_args[1] != -1
                    or target(maximum) != "aten.amax.default"
                ):
                    continue
                maximum_args = getattr(maximum, "args", ())
                if (
                    len(maximum_args) < 3
                    or maximum_args[0] is not grouped
                    or tuple(maximum_args[1]) != (-1,)
                    or maximum_args[2] is not True
                ):
                    continue
                matched = True
                break
        if not matched:
            return None

        try:
            m, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
            out_m, out_n = map(V.graph.sizevars.optimization_hint, buffer.get_size())
        except Exception:
            return None
        if out_m != m or out_n <= 0 or n % out_n != 0:
            return None
        group = n // out_n
        if group <= 1 or group > 32:
            return None
        reads = list(scheduler_node.read_writes.reads)
        range_vars = scheduler_node.read_writes.range_vars
        if not reads or range_vars is None:
            return None
        if not range_vars:
            range_vars = tuple(
                sorted(
                    OrderedSet(
                        symbol for read in reads for symbol in read.index.free_symbols
                    ),
                    key=str,
                )
            )
        range_vars = cast(tuple[Any, ...], range_vars)
        if len(range_vars) == 2:
            base = n * range_vars[0] + group * range_vars[1]
            expected_strides = [n, group]
        elif len(range_vars) == 1:
            base = group * range_vars[0]
            expected_strides = [group]
        else:
            return None
        offsets = OrderedSet()
        for read in reads:
            if read.name != gemm_node.get_name():
                return None
            strides = V.graph.sizevars.stride_vars(read.index, range_vars)
            if list(strides) != expected_strides:
                return None
            offsets.add(V.graph.sizevars.simplify(read.index - base))
        if offsets != OrderedSet(range(group)):
            return None
        return buffer.get_name(), group, 1, "logsumexp", "identity"

    @classmethod
    def _grouped_reduce_feeds_main_config_from_nodes(
        cls, gemm_node: Buffer, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[str, int, int, str, str] | None:
        if len(gemm_node.get_size()) != 2:
            return None
        allowed_targets = OrderedSet(
            (
                "aten.sub.Tensor",
                "aten.add.Tensor",
                "aten.mul.Tensor",
                "aten.reshape.default",
                "prims.convert_element_type.default",
                "aten.mean.dim",
                "aten.expand.default",
                "aten.clone.default",
                "aten._unsafe_view.default",
            )
        )
        if len(nodes) > 1:
            candidate_nodes = [
                node
                for node in nodes
                if isinstance(node.node, ComputedBuffer)
                and any(
                    str(getattr(origin, "target", ""))
                    in ("aten.mean.dim", "aten.add.Tensor", "aten.sub.Tensor")
                    for origin in node.node.get_origins()
                )
            ]
            buffers = [node.node for node in candidate_nodes]
            if not all(isinstance(buffer, ComputedBuffer) for buffer in buffers):
                return None
            targets = OrderedSet(
                str(origin.target)
                for buffer in cast(list[ComputedBuffer], buffers)
                for origin in buffer.get_origins()
                if hasattr(origin, "target")
            )
            binary_targets = targets & OrderedSet(
                ("aten.add.Tensor", "aten.sub.Tensor")
            )
            if (
                "aten.mean.dim" not in targets
                or not binary_targets
                or not targets.issubset(allowed_targets)
            ):
                return None
            reductions = [
                config
                for node in candidate_nodes
                if (config := cls._grouped_reduce_config(gemm_node, node)) is not None
                and config[2:] == (0, "mean", "identity")
            ]
            if len(reductions) != 1 or reductions[0][1] > 64:
                return None
            group = reductions[0][1]
            m, n = gemm_node.get_size()

            def is_full_output_size(size) -> bool:
                return V.graph.sizevars.statically_known_list_equals(size, (m, n)) or (
                    len(size) == 3
                    and V.graph.sizevars.statically_known_equals(size[0] * group, m)
                    and V.graph.sizevars.statically_known_equals(size[1], group)
                    and V.graph.sizevars.statically_known_equals(size[2], n)
                )

            finalizers = [
                cast(ComputedBuffer, buffer)
                for buffer in buffers
                if isinstance(cast(ComputedBuffer, buffer).data, Pointwise)
                and is_full_output_size(cast(ComputedBuffer, buffer).get_size())
            ]
            if not finalizers:
                return None
            finalizer = finalizers[-1]
            consumer_type = cls._centered_mean_consumer_type(finalizer)
            if consumer_type is None:
                return None
            return (
                finalizer.get_name(),
                reductions[0][1],
                0,
                consumer_type,
                "identity",
            )
        if len(nodes) != 1 or not isinstance(nodes[0].node, ComputedBuffer):
            return None
        node = cast(ComputedBuffer, nodes[0].node)
        if not isinstance(node.data, Pointwise):
            return None
        targets = OrderedSet(
            str(origin.target)
            for origin in node.get_origins()
            if hasattr(origin, "target")
        )
        binary_targets = targets & OrderedSet(("aten.add.Tensor", "aten.sub.Tensor"))
        required_targets: OrderedSet[str] = OrderedSet(("aten.mean.dim",))
        if (
            not required_targets.issubset(targets)
            or not targets.issubset(allowed_targets)
            or not binary_targets
        ):
            return None
        consumer_type = cls._centered_mean_consumer_type(node)
        if consumer_type is None:
            return None
        reads = list(nodes[0].read_writes.reads)
        range_vars = nodes[0].read_writes.range_vars
        if range_vars is None:
            return None
        if len(reads) == 2:
            reduction_reads = [
                read for read in reads if read.name != gemm_node.get_name()
            ]
            if len(reduction_reads) != 1:
                return None
            reduction = V.graph.get_buffer(reduction_reads[0].name)
            if not (
                isinstance(reduction, ComputedBuffer)
                and isinstance(reduction.data, Reduction)
                and reduction.data.reduction_type == "sum"
                and len(reduction.data.reduction_ranges) == 1
                and len(reduction.data.ranges) == 3
            ):
                return None
            try:
                group = V.graph.sizevars.optimization_hint(
                    reduction.data.reduction_ranges[0]
                )
            except Exception:
                return None
            m, n = gemm_node.get_size()
            out_m, singleton, out_n = reduction.data.ranges
            if (
                group <= 1
                or group > 64
                or not V.graph.sizevars.statically_known_list_equals(
                    (m, n, singleton), (out_m * group, out_n, 1)
                )
                or not V.graph.sizevars.statically_known_list_equals(
                    node.get_size(), gemm_node.get_size()
                )
            ):
                return None
            return node.get_name(), group, 0, consumer_type, "identity"
        if len(reads) <= 2:
            return None
        try:
            _, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
        except Exception:
            return None
        group = len(reads) - 1
        if group <= 1 or group > 4:
            return None
        if any(read.name != gemm_node.get_name() for read in reads):
            return None
        grouped_base = reads[1].index
        if any(
            V.graph.sizevars.simplify(read.index - grouped_base) != i * n
            for i, read in enumerate(reads[1:])
        ):
            return None
        return node.get_name(), group, 0, consumer_type, "identity"

    @staticmethod
    def _centered_mean_consumer_type(node: ComputedBuffer) -> str | None:
        origins = OrderedSet(node.get_origins())
        arithmetic_targets = OrderedSet(
            ("aten.add.Tensor", "aten.mul.Tensor", "aten.sub.Tensor")
        )
        arithmetic = [
            origin
            for origin in origins
            if str(getattr(origin, "target", "")) in arithmetic_targets
        ]
        referenced = OrderedSet(
            arg
            for origin in arithmetic
            for arg in origin.args
            if hasattr(arg, "target") and arg in origins
        )
        roots = [origin for origin in arithmetic if origin not in referenced]

        def scale(value, factor):
            return tuple(component * factor for component in value)

        def combine(lhs, rhs, rhs_scale=1.0):
            return tuple(a + rhs_scale * b for a, b in zip(lhs, rhs))

        def lower(value):
            if isinstance(value, (int, float)):
                return 0.0, 0.0, float(value)
            target = str(getattr(value, "target", ""))
            if target == "aten.mean.dim":
                return 0.0, 1.0, 0.0
            if target == "prims.convert_element_type.default":
                nested = lower(value.args[0]) if value.args else None
                return nested if nested is not None else (1.0, 0.0, 0.0)
            if target in (
                "aten.clone.default",
                "aten.expand.default",
                "aten.reshape.default",
                "aten.view.default",
                "aten._unsafe_view.default",
            ):
                return lower(value.args[0])
            if target in ("aten.add.Tensor", "aten.sub.Tensor"):
                lhs = lower(value.args[0])
                rhs = lower(value.args[1])
                if lhs is None or rhs is None:
                    return None
                alpha = float(value.kwargs.get("alpha", 1.0))
                return combine(
                    lhs,
                    rhs,
                    alpha if target == "aten.add.Tensor" else -alpha,
                )
            if target == "aten.mul.Tensor":
                lhs = lower(value.args[0])
                rhs = lower(value.args[1])
                if lhs is None or rhs is None:
                    return None
                if lhs[:2] == (0.0, 0.0):
                    return scale(rhs, lhs[2])
                if rhs[:2] == (0.0, 0.0):
                    return scale(lhs, rhs[2])
            return None

        for root in roots:
            coefficients = lower(root)
            if (
                coefficients is not None
                and coefficients[0] != 0.0
                and coefficients[1] != 0.0
                and all(math.isfinite(value) for value in coefficients)
            ):
                return "mean_linear:" + ":".join(
                    format(value, ".17g") for value in coefficients
                )
        return None

    @classmethod
    def _grouped_reduce_feeds_main_config(
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> tuple[str, int, int, str, str] | None:
        return cls._grouped_reduce_feeds_main_config_from_nodes(
            gemm_node, scheduler_node.get_nodes()
        )

    @classmethod
    def _grouped_softmax_config_from_nodes(
        cls, gemm_node: Buffer, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[str, int, int, str, str] | None:
        if len(gemm_node.get_size()) != 2:
            return None
        if len(nodes) == 1 and isinstance(nodes[0].node, ComputedBuffer):
            buffer = cast(ComputedBuffer, nodes[0].node)
            if isinstance(buffer.data, Pointwise):
                try:
                    group = V.graph.sizevars.optimization_hint(buffer.get_size()[2])
                except Exception:
                    return None
                m, n = gemm_node.get_size()
                out_m, out_groups, _ = buffer.get_size()
                targets = OrderedSet(
                    str(origin.target)
                    for origin in buffer.get_origins()
                    if hasattr(origin, "target")
                )
                required: OrderedSet[str] = OrderedSet(
                    (
                        "prims.prepare_softmax_online.default",
                        "aten.sub.Tensor",
                        "aten.exp.default",
                        "aten.div.Tensor",
                    )
                )
                allowed = required | OrderedSet(
                    (
                        "prims.convert_element_type.default",
                        "aten._to_copy.default",
                        "aten.reshape.default",
                        "aten.view.default",
                    )
                )
                if (
                    1 < group <= 32
                    and V.graph.sizevars.statically_known_equals(out_m, m)
                    and V.graph.sizevars.statically_known_equals(out_groups * group, n)
                    and required.issubset(targets)
                    and targets.issubset(allowed)
                    and nodes[0].read_writes.reads
                    and all(
                        read.name == gemm_node.get_name()
                        for read in nodes[0].read_writes.reads
                    )
                ):
                    return buffer.get_name(), group, 1, "online_softmax", "identity"
            return None
        if len(nodes) != 3:
            return None
        buffers = [node.node for node in nodes]
        if not (
            all(isinstance(buffer, ComputedBuffer) for buffer in buffers)
            and all(
                isinstance(cast(ComputedBuffer, buffer).data, MultiOutputReduction)
                for buffer in buffers[:2]
            )
            and isinstance(cast(ComputedBuffer, buffers[2]).data, Pointwise)
        ):
            return None
        reductions: list[MultiOutputReduction] = [
            cast(MultiOutputReduction, cast(ComputedBuffer, buffer).data)
            for buffer in buffers[:2]
        ]
        if any(
            reduction.reduction_type != "online_softmax_reduce"
            or len(reduction.reduction_ranges) != 1
            for reduction in reductions
        ):
            return None
        group = V.graph.sizevars.optimization_hint(reductions[0].reduction_ranges[0])
        if group <= 1 or group > 32:
            return None
        try:
            m, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
            reduction_sizes = [
                tuple(map(V.graph.sizevars.optimization_hint, reduction.ranges))
                for reduction in reductions
            ]
            output_size = tuple(
                map(
                    V.graph.sizevars.optimization_hint,
                    cast(ComputedBuffer, buffers[2]).get_size(),
                )
            )
        except Exception:
            return None
        if (
            n % group != 0
            or any(size != (m, n // group, 1) for size in reduction_sizes)
            or output_size != (m, n // group, group)
            or reductions[0].reduction_ranges != reductions[1].reduction_ranges
        ):
            return None
        targets = OrderedSet(
            str(origin.target)
            for buffer in cast(list[ComputedBuffer], buffers)
            for origin in buffer.get_origins()
            if hasattr(origin, "target")
        )
        required: OrderedSet[str] = OrderedSet(
            (
                "prims.prepare_softmax_online.default",
                "aten.sub.Tensor",
                "aten.exp.default",
                "aten.div.Tensor",
            )
        )
        allowed = required | OrderedSet(
            (
                "prims.convert_element_type.default",
                "aten._to_copy.default",
                "aten.reshape.default",
                "aten.view.default",
            )
        )
        if not required.issubset(targets) or not targets.issubset(allowed):
            return None
        if not any(
            read.name == gemm_node.get_name()
            for node in nodes
            for read in node.read_writes.reads
        ):
            return None
        return (
            cast(ComputedBuffer, buffers[2]).get_name(),
            group,
            1,
            "online_softmax",
            "identity",
        )

    @classmethod
    def _grouped_softmax_config(
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> tuple[str, int, int, str, str] | None:
        return cls._grouped_softmax_config_from_nodes(
            gemm_node, scheduler_node.get_nodes()
        )

    @classmethod
    def _grouped_sum_normalize_config_from_nodes(
        cls, gemm_node: Buffer, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[str, int, int, str, str] | None:
        if len(gemm_node.get_size()) != 2:
            return None
        buffers = [node.node for node in nodes]
        if not buffers or not all(
            isinstance(buffer, ComputedBuffer) for buffer in buffers
        ):
            return None
        targets = OrderedSet(
            str(origin.target)
            for buffer in cast(list[ComputedBuffer], buffers)
            for origin in buffer.get_origins()
            if hasattr(origin, "target")
        )
        forward_required: OrderedSet[str] = OrderedSet(
            ("aten.mul.Tensor", "aten.reciprocal.default", "aten.sum.dim_IntList")
        )
        divide_required: OrderedSet[str] = OrderedSet(
            ("aten.div.Tensor", "aten.sum.dim_IntList")
        )
        allowed = (
            forward_required
            | divide_required
            | OrderedSet(
                (
                    "aten.add.Tensor",
                    "aten.sub.Tensor",
                    "prims.convert_element_type.default",
                    "aten._to_copy.default",
                    "aten.reshape.default",
                    "aten.view.default",
                )
            )
        )
        flex_gemm_origins = OrderedSet(
            target
            for target in targets
            if target == "flex_gemm" or target.startswith("flex_gemm_body_graph_")
        )
        if not (
            forward_required.issubset(targets) or divide_required.issubset(targets)
        ) or not targets.issubset(allowed | flex_gemm_origins):
            return None
        m, n = gemm_node.get_size()
        reductions = [
            config
            for node in nodes
            if (config := cls._grouped_reduce_config(gemm_node, node)) is not None
        ]
        finalizers = []
        for buffer in cast(list[ComputedBuffer], buffers):
            if not isinstance(buffer.data, Pointwise):
                continue
            out_size = buffer.get_size()
            consumer_type = cls._sum_normalize_consumer_type(buffer)
            if consumer_type is None:
                continue
            if len(out_size) == 2 and len(reductions) == 1:
                _, group, axis, reduction_type, source_type = reductions[0]
                if (
                    reduction_type == "sum"
                    and source_type == "identity"
                    and V.graph.sizevars.statically_known_list_equals(out_size, (m, n))
                ):
                    reads = list(buffer.get_read_writes().reads)
                    if len(reads) == 2 and OrderedSet(
                        read.name for read in reads
                    ) == OrderedSet((gemm_node.get_name(), reductions[0][0])):
                        finalizers.append((buffer, group, axis, consumer_type))
                continue
            if len(out_size) != 3:
                continue
            out0, out1, out2 = out_size
            try:
                n_group = V.graph.sizevars.optimization_hint(out2)
            except Exception:
                n_group = 0
            if (
                1 < n_group <= 32
                and V.graph.sizevars.statically_known_equals(out0, m)
                and V.graph.sizevars.statically_known_equals(out1 * n_group, n)
            ):
                finalizers.append((buffer, n_group, 1, consumer_type))
                continue
            try:
                m_group = V.graph.sizevars.optimization_hint(out1)
            except Exception:
                m_group = 0
            if (
                1 < m_group <= 64
                and V.graph.sizevars.statically_known_equals(out0 * m_group, m)
                and V.graph.sizevars.statically_known_equals(out2, n)
            ):
                finalizers.append((buffer, m_group, 0, consumer_type))
        if not finalizers or not any(
            read.name == gemm_node.get_name()
            for node in nodes
            for read in node.read_writes.reads
        ):
            return None
        contracts = OrderedSet(
            (group, axis, consumer_type) for _, group, axis, consumer_type in finalizers
        )
        if len(contracts) != 1:
            return None
        group, axis, consumer_type = next(iter(contracts))
        if reductions:
            if len(reductions) != 1 or reductions[0][1:] != (
                group,
                axis,
                "sum",
                "identity",
            ):
                return None
        else:
            reads = list(nodes[0].read_writes.reads) if len(nodes) == 1 else []
            if (
                group > 4
                or len(reads) != group + 1
                or any(read.name != gemm_node.get_name() for read in reads)
            ):
                return None
        return finalizers[0][0].get_name(), group, axis, consumer_type, "identity"

    @staticmethod
    def _sum_normalize_consumer_type(node: ComputedBuffer) -> str | None:
        origins = OrderedSet(node.get_origins())

        def contains(value, target):
            if str(getattr(value, "target", "")) == target:
                return True
            return any(
                contains(arg, target)
                for arg in getattr(value, "args", ())
                if hasattr(arg, "target")
            )

        def sum_affine(value):
            if isinstance(value, (int, float)):
                return 0.0, float(value)
            target = str(getattr(value, "target", ""))
            if target == "aten.sum.dim_IntList":
                return 1.0, 0.0
            if target in (
                "aten.clone.default",
                "aten.expand.default",
                "aten.reshape.default",
                "aten.view.default",
                "aten._unsafe_view.default",
            ):
                return sum_affine(value.args[0])
            if target in ("aten.add.Tensor", "aten.sub.Tensor"):
                lhs = sum_affine(value.args[0])
                rhs = sum_affine(value.args[1])
                if lhs is None or rhs is None:
                    return None
                alpha = float(value.kwargs.get("alpha", 1.0))
                rhs_scale = alpha if target == "aten.add.Tensor" else -alpha
                return lhs[0] + rhs_scale * rhs[0], lhs[1] + rhs_scale * rhs[1]
            if target == "aten.mul.Tensor":
                lhs = sum_affine(value.args[0])
                rhs = sum_affine(value.args[1])
                if lhs is None or rhs is None:
                    return None
                if lhs[0] == 0.0:
                    return rhs[0] * lhs[1], rhs[1] * lhs[1]
                if rhs[0] == 0.0:
                    return lhs[0] * rhs[1], lhs[1] * rhs[1]
            return None

        def normalize_kind(value):
            target = str(getattr(value, "target", ""))
            if target == "aten.mul.Tensor":
                lhs, rhs = value.args[:2]
                reciprocal = None
                if contains(lhs, "aten.reciprocal.default") and contains(
                    rhs, "prims.convert_element_type.default"
                ):
                    reciprocal = lhs
                elif contains(rhs, "aten.reciprocal.default") and contains(
                    lhs, "prims.convert_element_type.default"
                ):
                    reciprocal = rhs
                if reciprocal is not None:
                    while str(getattr(reciprocal, "target", "")) != (
                        "aten.reciprocal.default"
                    ):
                        reciprocal = reciprocal.args[0]
                    affine = sum_affine(reciprocal.args[0])
                    if affine is not None:
                        return "forward", *affine
            if target == "aten.div.Tensor":
                lhs, rhs = value.args[:2]
                lhs_sum = contains(lhs, "aten.sum.dim_IntList")
                rhs_sum = contains(rhs, "aten.sum.dim_IntList")
                lhs_input = contains(lhs, "prims.convert_element_type.default")
                rhs_input = contains(rhs, "prims.convert_element_type.default")
                if lhs_input and rhs_sum:
                    affine = sum_affine(rhs)
                    if affine is not None:
                        return "forward", *affine
                if lhs_sum and rhs_input:
                    affine = sum_affine(lhs)
                    if affine is not None:
                        return "reverse", *affine
            return None

        arithmetic = [
            origin
            for origin in origins
            if str(getattr(origin, "target", ""))
            in (
                "aten.add.Tensor",
                "aten.div.Tensor",
                "aten.mul.Tensor",
                "aten.sub.Tensor",
            )
        ]
        referenced = OrderedSet(
            arg
            for origin in arithmetic
            for arg in origin.args
            if hasattr(arg, "target") and arg in origins
        )
        roots = [origin for origin in arithmetic if origin not in referenced]
        normalization_parameters = None

        def scale(value, factor):
            return value[0] * factor, value[1] * factor, value[2] * factor

        def lower(value):
            if isinstance(value, (int, float)):
                return 0.0, 0.0, float(value)
            kind = normalize_kind(value)
            if kind is not None:
                nonlocal normalization_parameters
                normalization_parameters = kind[1:]
            if kind is not None and kind[0] == "forward":
                return 1.0, 0.0, 0.0
            if kind is not None and kind[0] == "reverse":
                return 0.0, 1.0, 0.0
            target = str(getattr(value, "target", ""))
            if target in (
                "aten.clone.default",
                "aten.reshape.default",
                "aten.view.default",
                "aten._unsafe_view.default",
            ):
                return lower(value.args[0])
            if target in ("aten.add.Tensor", "aten.sub.Tensor"):
                lhs = lower(value.args[0])
                rhs = lower(value.args[1])
                if lhs is None or rhs is None:
                    return None
                alpha = float(value.kwargs.get("alpha", 1.0))
                rhs_scale = alpha if target == "aten.add.Tensor" else -alpha
                return tuple(a + rhs_scale * b for a, b in zip(lhs, rhs))
            if target == "aten.mul.Tensor":
                lhs = lower(value.args[0])
                rhs = lower(value.args[1])
                if lhs is None or rhs is None:
                    return None
                if lhs[:2] == (0.0, 0.0):
                    return scale(rhs, lhs[2])
                if rhs[:2] == (0.0, 0.0):
                    return scale(lhs, rhs[2])
            return None

        for root in roots:
            coefficients = lower(root)
            if (
                coefficients is not None
                and normalization_parameters is not None
                and (coefficients[0] != 0.0) != (coefficients[1] != 0.0)
                and all(math.isfinite(value) for value in coefficients)
            ):
                kind = (
                    "normalize_sum_affine"
                    if coefficients[0]
                    else "normalize_sum_reverse_affine"
                )
                return (
                    kind
                    + ":"
                    + ":".join(
                        format(value, ".17g")
                        for value in (
                            coefficients[0] or coefficients[1],
                            coefficients[2],
                            *normalization_parameters,
                        )
                    )
                )
        return None

    @staticmethod
    def _sum_multiply_consumer_type(node: ComputedBuffer) -> str | None:
        origins = OrderedSet(node.get_origins())

        def contains(value, target):
            if str(getattr(value, "target", "")) == target:
                return True
            return any(
                contains(arg, target)
                for arg in getattr(value, "args", ())
                if hasattr(arg, "target")
            )

        def sum_affine(value):
            if isinstance(value, (int, float)):
                return 0.0, float(value)
            target = str(getattr(value, "target", ""))
            if target == "aten.sum.dim_IntList":
                return 1.0, 0.0
            if target in (
                "aten.clone.default",
                "aten.expand.default",
                "aten.reshape.default",
                "aten.view.default",
                "aten._unsafe_view.default",
            ):
                return sum_affine(value.args[0])
            if target in ("aten.add.Tensor", "aten.sub.Tensor"):
                lhs = sum_affine(value.args[0])
                rhs = sum_affine(value.args[1])
                if lhs is None or rhs is None:
                    return None
                alpha = float(value.kwargs.get("alpha", 1.0))
                rhs_scale = alpha if target == "aten.add.Tensor" else -alpha
                return lhs[0] + rhs_scale * rhs[0], lhs[1] + rhs_scale * rhs[1]
            if target == "aten.mul.Tensor":
                lhs = sum_affine(value.args[0])
                rhs = sum_affine(value.args[1])
                if lhs is None or rhs is None:
                    return None
                if lhs[0] == 0.0:
                    return rhs[0] * lhs[1], rhs[1] * lhs[1]
                if rhs[0] == 0.0:
                    return lhs[0] * rhs[1], lhs[1] * rhs[1]
            return None

        for origin in origins:
            if str(getattr(origin, "target", "")) != "aten.mul.Tensor":
                continue
            lhs, rhs = origin.args[:2]
            for input_value, reduction_value in ((lhs, rhs), (rhs, lhs)):
                if not contains(
                    input_value, "prims.convert_element_type.default"
                ) or not contains(reduction_value, "aten.sum.dim_IntList"):
                    continue
                affine = sum_affine(reduction_value)
                if affine is not None and all(math.isfinite(value) for value in affine):
                    return "sum_mul_affine:" + ":".join(
                        format(value, ".17g") for value in affine
                    )
        return None

    @classmethod
    def _grouped_absmax_normalize_config_from_nodes(
        cls, gemm_node: Buffer, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[str, int, int, str, str] | None:
        if len(gemm_node.get_size()) != 2:
            return None
        buffers = [node.node for node in nodes]
        if not buffers or not all(
            isinstance(buffer, ComputedBuffer) for buffer in buffers
        ):
            return None
        required: OrderedSet[str] = OrderedSet(
            (
                "aten.abs.default",
                "aten.amax.default",
                "aten.clamp_min.default",
                "aten.div.Tensor",
                "aten.mul.Tensor",
                "aten.reciprocal.default",
            )
        )
        allowed = required | OrderedSet(
            (
                "prims.convert_element_type.default",
                "aten._to_copy.default",
                "aten.reshape.default",
                "aten.view.default",
            )
        )
        targets = OrderedSet(
            str(origin.target)
            for buffer in cast(list[ComputedBuffer], buffers)
            for origin in buffer.get_origins()
            if hasattr(origin, "target")
        )
        if not required.issubset(targets) or not targets.issubset(allowed):
            return None
        try:
            m, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
        except Exception:
            return None
        finalizers = []
        for buffer in cast(list[ComputedBuffer], buffers):
            if not isinstance(buffer.data, Pointwise):
                continue
            try:
                out_m, out_groups, group = map(
                    V.graph.sizevars.optimization_hint, buffer.get_size()
                )
            except Exception:
                continue
            if (
                1 < group <= 32
                and out_m == m
                and out_groups * group == n
                and OrderedSet(("aten.mul.Tensor", "aten.reciprocal.default")).issubset(
                    OrderedSet(
                        str(origin.target)
                        for origin in buffer.get_origins()
                        if hasattr(origin, "target")
                    )
                )
            ):
                finalizers.append((buffer, group))
        if len(finalizers) != 1:
            return None
        finalizer, group = finalizers[0]
        reductions = cls._partition_local_reductions(gemm_node, nodes)[0]
        if len(reductions) != 1 or reductions[0][1:] != (
            group,
            1,
            "max",
            "abs_scale",
        ):
            return None
        return finalizer.get_name(), group, 1, "normalize_absmax", "abs_scale"

    @staticmethod
    def _is_mean_finalizer(
        reduction_node: BaseSchedulerNode, finalizer_node: BaseSchedulerNode
    ) -> bool:
        reduction_nodes = reduction_node.get_nodes()
        finalizer_nodes = finalizer_node.get_nodes()
        if len(reduction_nodes) != 1 or len(finalizer_nodes) != 1:
            return False
        reduction = reduction_nodes[0].node
        finalizer = finalizer_nodes[0].node
        if not (
            isinstance(reduction, ComputedBuffer)
            and isinstance(reduction.data, Reduction)
            and isinstance(finalizer, ComputedBuffer)
            and isinstance(finalizer.data, Pointwise)
        ):
            return False
        reads = list(finalizer_node.read_writes.reads)
        return (
            bool(reads)
            and all(read.name == reduction.get_name() for read in reads)
            and V.graph.sizevars.statically_known_list_equals(
                reduction.get_size(), finalizer.get_size()
            )
        )

    @staticmethod
    def _is_layout_finalizer(
        reduction_node: BaseSchedulerNode, finalizer_node: BaseSchedulerNode
    ) -> bool:
        reduction_nodes = reduction_node.get_nodes()
        finalizer_nodes = finalizer_node.get_nodes()
        if len(reduction_nodes) != 1 or len(finalizer_nodes) != 1:
            return False
        reduction = reduction_nodes[0].node
        finalizer = finalizer_nodes[0].node
        if not (
            isinstance(reduction, ComputedBuffer)
            and isinstance(finalizer, ComputedBuffer)
            and isinstance(finalizer.data, Pointwise)
        ):
            return False
        targets = OrderedSet(
            str(origin.target)
            for origin in finalizer.get_origins()
            if hasattr(origin, "target")
        )
        reads = list(finalizer_nodes[0].read_writes.reads)
        return (
            "prims.inductor_force_stride_order.default" in targets
            and bool(reads)
            and all(read.name == reduction.get_name() for read in reads)
            and V.graph.sizevars.statically_known_list_equals(
                reduction.get_size(), finalizer.get_size()
            )
        )

    @staticmethod
    def _is_absmax_scale_finalizer(
        reduction_node: BaseSchedulerNode, finalizer_node: BaseSchedulerNode
    ) -> bool:
        reduction_nodes = reduction_node.get_nodes()
        finalizer_nodes = finalizer_node.get_nodes()
        if len(reduction_nodes) != 1 or len(finalizer_nodes) != 1:
            return False
        reduction = reduction_nodes[0].node
        finalizer = finalizer_nodes[0].node
        if not (
            isinstance(reduction, ComputedBuffer)
            and isinstance(finalizer, ComputedBuffer)
            and isinstance(finalizer.data, Pointwise)
        ):
            return False
        targets = OrderedSet(
            str(origin.target)
            for origin in finalizer.get_origins()
            if hasattr(origin, "target")
        )
        reads = list(finalizer_node.read_writes.reads)
        return (
            OrderedSet(("aten.clamp_min.default", "aten.div.Tensor")).issubset(targets)
            and bool(reads)
            and all(read.name == reduction.get_name() for read in reads)
            and V.graph.sizevars.statically_known_list_equals(
                reduction.get_size(), finalizer.get_size()
            )
        )

    @classmethod
    def _partition_local_reductions(
        cls, gemm_node: Buffer, epilogue_nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[list[tuple[str, int, int, str, str]], OrderedSet[BaseSchedulerNode]]:
        reductions = []
        reduction_nodes: OrderedSet[BaseSchedulerNode] = OrderedSet()
        index = 0
        while index < len(epilogue_nodes):
            node = epilogue_nodes[index]
            config = (
                cls._grouped_reduce_config(gemm_node, node)
                or cls._grouped_variance_config(gemm_node, node)
                or cls._grouped_logsumexp_config(gemm_node, node)
                or cls._direct_bool_mask_config(gemm_node, node)
            )
            if config is None:
                index += 1
                continue
            if (
                config[3] == "mean"
                and index + 2 < len(epilogue_nodes)
                and cls._is_layout_finalizer(node, epilogue_nodes[index + 1])
                and cls._is_mean_finalizer(
                    epilogue_nodes[index + 1], epilogue_nodes[index + 2]
                )
            ):
                layout_finalizer = epilogue_nodes[index + 1]
                mean_finalizer = epilogue_nodes[index + 2]
                mean_buffer = cast(ComputedBuffer, mean_finalizer.get_nodes()[0].node)
                config = (
                    mean_buffer.get_name(),
                    *config[1:],
                )
                reduction_nodes.add(layout_finalizer)
                reduction_nodes.add(mean_finalizer)
                index += 2
            elif (
                config[3] == "mean"
                and index + 1 < len(epilogue_nodes)
                and cls._is_mean_finalizer(node, epilogue_nodes[index + 1])
            ):
                finalizer = epilogue_nodes[index + 1]
                finalizer_buffer = cast(ComputedBuffer, finalizer.get_nodes()[0].node)
                config = (finalizer_buffer.get_name(), *config[1:])
                reduction_nodes.add(finalizer)
                index += 1
            elif index + 1 < len(epilogue_nodes) and cls._is_layout_finalizer(
                node, epilogue_nodes[index + 1]
            ):
                finalizer = epilogue_nodes[index + 1]
                finalizer_buffer = cast(ComputedBuffer, finalizer.get_nodes()[0].node)
                config = (finalizer_buffer.get_name(), *config[1:])
                reduction_nodes.add(finalizer)
                index += 1
            elif config[3:] == ("max", "abs"):
                finalizer = next(
                    (
                        candidate
                        for candidate in epilogue_nodes[index + 1 :]
                        if cls._is_absmax_scale_finalizer(node, candidate)
                    ),
                    None,
                )
                if finalizer is not None:
                    finalizer_buffer = finalizer.get_nodes()[0].node
                    if not isinstance(finalizer_buffer, Buffer):
                        index += 1
                        continue
                    config = (
                        finalizer_buffer.get_name(),
                        *config[1:4],
                        "abs_scale",
                    )
                    reduction_nodes.add(finalizer)
            reductions.append(config)
            reduction_nodes.add(node)
            index += 1
        return reductions, reduction_nodes

    def _can_fuse_epilogue_impl(
        self,
        gemm_template_node: SchedulerNode,
        existing_epilogue_nodes: list[BaseSchedulerNode],
        node_to_fuse: BaseSchedulerNode,
    ) -> bool:
        from .nv_universal_gemm import GemmVariant

        if not config.epilogue_fusion:
            return False

        ir_node = gemm_template_node.node
        if not isinstance(ir_node, (NVUniversalGemmBuffer, MultiTemplateBuffer)):
            return False

        if isinstance(ir_node, NVUniversalGemmBuffer):
            if ir_node.variant not in (GemmVariant.GEMM, GemmVariant.SCALED_GEMM):
                log.debug(
                    "NVGEMM epilogue fusion: not supported for %s variant",
                    ir_node.variant.op_name,
                )
                return False
            if not ir_node.supports_epilogue_fusion:
                log.debug(
                    "NVGEMM epilogue fusion: kernel %s does not support epilogue fusion",
                    ir_node.kernel_metadata.get("kernel_name", "unknown"),
                )
                return False
        elif isinstance(ir_node, MultiTemplateBuffer):
            # Use _choices, not choice_timings() — the latter forces autotune sync.
            has_efc_choice = False
            for choice in ir_node._choices:
                if not (
                    isinstance(choice, NVUniversalGemmCaller)
                    and choice.supports_epilogue_fusion
                ):
                    continue
                has_efc_choice = True
                if choice.variant not in (
                    GemmVariant.GEMM,
                    GemmVariant.SCALED_GEMM,
                ):
                    log.debug(
                        "NVGEMM epilogue fusion: MultiTemplateBuffer has unsupported EFC choices"
                    )
                    return False
            if not has_efc_choice:
                log.debug("NVGEMM epilogue fusion: no EFC kernel available in choices")
                return False

        all_scheduler_nodes = [
            node
            for epilogue_node in (*existing_epilogue_nodes, node_to_fuse)
            for node in epilogue_node.get_nodes()
        ]
        if (
            sum(
                isinstance(scheduler_node.node.data, Reduction)
                for scheduler_node in all_scheduler_nodes
                if isinstance(scheduler_node.node, ComputedBuffer)
            )
            > 1
        ):
            log.debug("NVGEMM supports one grouped local reduction")
            return False
        feed_main = self._grouped_reduce_feeds_main_config_from_nodes(
            ir_node, all_scheduler_nodes
        )
        if feed_main is None:
            feed_main = self._grouped_softmax_config(ir_node, node_to_fuse)
        if feed_main is None:
            feed_main = self._grouped_sum_normalize_config_from_nodes(
                ir_node, all_scheduler_nodes
            )
        if feed_main is None:
            feed_main = self._grouped_absmax_normalize_config_from_nodes(
                ir_node, all_scheduler_nodes
            )
        if feed_main is not None:
            fused_names = OrderedSet(
                scheduler_node.get_name() for scheduler_node in all_scheduler_nodes
            )
            fused_names.add(gemm_template_node.get_name())
            if not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                ir_node.get_name(), fused_names
            ):
                return False
        _, local_reduce_nodes = self._partition_local_reductions(
            ir_node, all_scheduler_nodes
        )
        local_reduce = (
            self._grouped_reduce_config(ir_node, node_to_fuse)
            or self._grouped_variance_config(ir_node, node_to_fuse)
            or self._grouped_logsumexp_config(ir_node, node_to_fuse)
            or self._direct_bool_mask_config(ir_node, node_to_fuse)
        )
        variants = (
            (ir_node.variant,)
            if isinstance(ir_node, NVUniversalGemmBuffer)
            else tuple(
                choice.variant
                for choice in ir_node._choices
                if isinstance(choice, NVUniversalGemmCaller)
                and choice.supports_epilogue_fusion
            )
        )
        scaled_epilogue = bool(variants) and all(
            variant == GemmVariant.SCALED_GEMM for variant in variants
        )
        if local_reduce is not None:
            standard_aux_sum = (
                bool(variants)
                and all(variant == GemmVariant.GEMM for variant in variants)
                and local_reduce[2] in (0, 1)
                and (
                    local_reduce[3]
                    in (
                        "sum",
                        "mean",
                        "prod",
                        "max",
                        "min",
                    )
                    or local_reduce[3].startswith("variance_affine:")
                    or local_reduce[3] == "logsumexp"
                    or local_reduce[3] == "direct_bool_gt_zero"
                )
                and local_reduce[4] in ("identity", "square", "abs")
            )
            if not scaled_epilogue and not standard_aux_sum:
                return False

        for s_node in all_scheduler_nodes:
            node = s_node.node
            if not isinstance(node, ComputedBuffer):
                log.debug("NVGEMM epilogue fusion: %s is not a ComputedBuffer", node)
                return False
            if (
                feed_main is None
                and not isinstance(node.data, Pointwise)
                and s_node not in local_reduce_nodes
            ):
                log.debug("NVGEMM epilogue fusion: %s is not a Pointwise op", node)
                return False

            if (
                feed_main is None
                and s_node not in local_reduce_nodes
                and not V.graph.sizevars.statically_known_list_equals(
                    node.get_size(), ir_node.get_size()
                )
            ):
                log.debug(
                    "NVGEMM epilogue fusion: size mismatch %s vs %s",
                    node.get_size(),
                    ir_node.get_size(),
                )
                return False
            if (
                feed_main is None
                and s_node not in local_reduce_nodes
                and isinstance(node.data, Pointwise)
                and not V.graph.sizevars.statically_known_list_equals(
                    node.data.ranges, ir_node.get_size()
                )
            ):
                log.debug(
                    "NVGEMM epilogue fusion: iteration-domain mismatch %s vs %s",
                    node.data.ranges,
                    ir_node.get_size(),
                )
                return False
        # cutlass.operators' EVT supports matrix, row, and column loads here.
        # Reject unresolved inputs and shapes outside those broadcast patterns;
        # the trial EVT trace below remains the final capability check.
        gemm_size = ir_node.get_size()
        name_to_buf = V.graph.name_to_buffer | V.graph.graph_inputs
        internal_names = OrderedSet([ir_node.get_name()]) | OrderedSet(
            [s_node.get_name() for s_node in all_scheduler_nodes]
        )
        epilogue_inputs: OrderedSet[str] = OrderedSet()
        for s_node in all_scheduler_nodes:
            for rd in s_node.read_writes.reads:
                if feed_main is not None:
                    continue
                if rd.name in internal_names:
                    continue
                epilogue_inputs.add(rd.name)
                read_buf = name_to_buf.get(rd.name)
                if read_buf is None:
                    log.debug(
                        "NVGEMM epilogue fusion: read %s not in name_to_buffer/graph_inputs, refusing to fuse",
                        rd.name,
                    )
                    return False
                read_size = read_buf.get_size()
                if not read_size or len(read_size) > len(gemm_size):
                    log.debug(
                        "NVGEMM epilogue fusion: read buffer %s has unsupported rank",
                        rd.name,
                    )
                    return False
                padded_size = [1] * (len(gemm_size) - len(read_size)) + list(read_size)
                supported_shapes = (
                    gemm_size,
                    [1, gemm_size[1]],
                    [gemm_size[0], 1],
                )
                if not any(
                    V.graph.sizevars.statically_known_list_equals(padded_size, shape)
                    for shape in supported_shapes
                ):
                    log.debug(
                        "NVGEMM epilogue fusion: read buffer %s size %s is not broadcastable to GEMM size %s",
                        rd.name,
                        read_size,
                        gemm_size,
                    )
                    return False
        if scaled_epilogue and len(epilogue_inputs) > 4:
            log.debug(
                "NVGEMM scaled epilogue has %d captured tensors; at most four are supported",
                len(epilogue_inputs),
            )
            return False

        if not existing_epilogue_nodes:
            reads = OrderedSet(rd.name for rd in node_to_fuse.read_writes.reads)
            if ir_node.get_name() not in reads:
                log.debug(
                    "NVGEMM epilogue fusion: first epilogue node doesn't read from GEMM output"
                )
                return False

        if node_to_fuse.has_aliasing_or_mutation():
            log.debug("NVGEMM epilogue fusion: node has aliasing or mutation")
            return False
        elif node_to_fuse.is_reduction() and local_reduce is None and feed_main is None:
            log.debug("NVGEMM epilogue fusion: reductions not supported")
            return False

        all_epilogue_nodes = list(existing_epilogue_nodes) + list(
            node_to_fuse.get_nodes()
        )
        fused_buffer_names = OrderedSet(
            n.get_name() for n in [gemm_template_node, *all_epilogue_nodes]
        )
        preserve_gemm_output = (
            not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                ir_node.get_name(), fused_buffer_names
            )
        )
        # Multi-store epilogues wire each output to its own destination tensor.
        trial_removed_buffers = V.graph.removed_buffers.copy()
        if not preserve_gemm_output:
            trial_removed_buffers.add(ir_node.get_name())
        try:
            trial_reads: list[str] = []
            trial_writes: list[str] = []
            evt_nodes = [
                node
                for node in all_epilogue_nodes
                if node not in local_reduce_nodes and feed_main is None
            ]
            if evt_nodes:
                trial_reads, trial_writes, _, _ = (
                    CutlassEVTCodegen.ir_to_evt_python_code(
                        ir_node.get_name(),
                        evt_nodes,
                        trial_removed_buffers,
                    )
                )
            if scaled_epilogue:
                if len(trial_reads) > 4 or len(trial_writes) > 4:
                    return False
                for read_name in trial_reads:
                    read_buf = name_to_buf.get(read_name)
                    if read_buf is None:
                        log.debug(
                            "NVGEMM scaled EVT input %s cannot be resolved",
                            read_name,
                        )
                        return False
        except (NotImplementedError, AssertionError) as e:
            log.debug("NVGEMM epilogue fusion: trial EVT codegen failed: %s", e)
            return False

        return True

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # NVIDIA Universal GEMM templates don't support horizontal fusion yet
        return False

    @staticmethod
    def has_bool_output(node: BaseSchedulerNode) -> bool:
        return any(
            isinstance(scheduler_node.node, ComputedBuffer)
            and scheduler_node.node.get_dtype() == torch.bool
            for scheduler_node in node.get_nodes()
        )

    def can_fuse_reduction_chain(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if not any(
            isinstance(node.node, ComputedBuffer)
            and isinstance(node.node.data, Reduction)
            for node in node1.get_nodes()
        ):
            return False
        for read in node1.read_writes.reads:
            producer = V.graph.try_get_buffer(read.name)
            if not isinstance(producer, Buffer) or not self._has_nvgemm_choice(
                producer
            ):
                continue
            combined_nodes = [*node1.get_nodes(), *node2.get_nodes()]
            return bool(
                self._grouped_reduce_feeds_main_config_from_nodes(
                    producer, combined_nodes
                )
                or self._grouped_sum_normalize_config_from_nodes(
                    producer, combined_nodes
                )
                or self._grouped_absmax_normalize_config_from_nodes(
                    producer, combined_nodes
                )
            )
        return False

    def get_fusion_pair_priority(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        has_reduction = any(
            isinstance(node.node, ComputedBuffer)
            and isinstance(node.node.data, Reduction)
            for node in node1.get_nodes()
        )
        if has_reduction and self.can_fuse_reduction_chain(node1, node2):
            return 0
        if not has_reduction:
            node1_ir = node1.node if isinstance(node1, SchedulerNode) else None
            if self._has_nvgemm_choice(node1_ir) and any(
                isinstance(node.node, ComputedBuffer)
                and isinstance(node.node.data, Reduction)
                for node in node2.get_nodes()
            ):
                return 1
            return 2
        node1_outputs = node1.get_buffer_names()
        if not any(read.name in node1_outputs for read in node2.read_writes.reads):
            return 2
        scheduler = self.scheduler
        if scheduler is None:
            return 2
        for read in node1.read_writes.reads:
            producer = scheduler.name_to_fused_node.get(read.name)
            try:
                producer_buffer = V.graph.get_buffer(read.name)
            except RuntimeError:
                producer_buffer = None
            has_nvgemm_producer = (
                producer is not None
                and (
                    self.is_nv_universal_gemm_template(producer)
                    or self.is_nv_universal_gemm_fused_template(producer)
                )
            ) or self._has_nvgemm_choice(producer_buffer)
            if has_nvgemm_producer and isinstance(producer_buffer, Buffer):
                combined_nodes = [*node1.get_nodes(), *node2.get_nodes()]
                if (
                    self._grouped_reduce_feeds_main_config_from_nodes(
                        producer_buffer, combined_nodes
                    )
                    or self._grouped_sum_normalize_config_from_nodes(
                        producer_buffer, combined_nodes
                    )
                    or self._grouped_absmax_normalize_config_from_nodes(
                        producer_buffer, combined_nodes
                    )
                ):
                    return 0
                combined_reductions, combined_reduction_nodes = (
                    self._partition_local_reductions(producer_buffer, combined_nodes)
                )
                if combined_reductions and all(
                    node in combined_reduction_nodes for node in node2.get_nodes()
                ):
                    log.debug(
                        "prioritizing NVGEMM reduction chain %s -> %s",
                        node1.get_name(),
                        node2.get_name(),
                    )
                    return 0
                reductions, reduction_nodes = self._partition_local_reductions(
                    producer_buffer, node1.get_nodes()
                )
                if reductions and all(
                    node in reduction_nodes for node in node1.get_nodes()
                ):
                    return 2
                return 0
        return 2

    def can_fuse_reduction_epilogue(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        template = node1.get_template_node()
        if not isinstance(template, Buffer):
            return False
        epilogue_nodes = [
            node
            for node in node1.get_nodes()
            if not self.is_nv_universal_gemm_template(node)
        ]
        combined_nodes = [*epilogue_nodes, *node2.get_nodes()]
        if (
            sum(
                isinstance(node.node.data, Reduction)
                for node in combined_nodes
                if isinstance(node.node, ComputedBuffer)
            )
            > 1
        ):
            return False
        combined_feed_main = self._grouped_reduce_feeds_main_config_from_nodes(
            template, combined_nodes
        )
        feed_main_ordered = combined_feed_main is not None and (
            bool(epilogue_nodes)
            or all(
                read.name == template.get_name()
                for node in node2.get_nodes()
                for read in node.read_writes.reads
            )
        )
        candidate_reductions, candidate_reduction_nodes = (
            self._partition_local_reductions(template, node2.get_nodes())
            if isinstance(template, Buffer)
            else ([], OrderedSet())
        )
        exact_reduction_candidate = bool(candidate_reductions) and all(
            node in candidate_reduction_nodes for node in node2.get_nodes()
        )
        eligible = isinstance(template, Buffer) and (
            bool(candidate_reductions)
            or feed_main_ordered
            or self._grouped_softmax_config(template, node2) is not None
            or self._grouped_sum_normalize_config_from_nodes(template, combined_nodes)
            is not None
            or self._grouped_absmax_normalize_config_from_nodes(
                template, combined_nodes
            )
            is not None
        )
        if not eligible:
            return False
        if exact_reduction_candidate:
            return True
        template_snode = next(
            node
            for node in node1.get_nodes()
            if self.is_nv_universal_gemm_template(node)
        )
        return self._can_fuse_epilogue_impl(
            cast(SchedulerNode, template_snode), epilogue_nodes, node2
        )

    def define_kernel(
        self, src_code: str, node_schedule, precompile_metadata=None
    ) -> str:
        """
        Define a NVIDIA Universal GEMM kernel by writing source code and generating wrapper.

        Based on CuteDSLScheduling.define_kernel.
        """
        wrapper = V.graph.wrapper_code

        # Use the string as the key for caching
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )

        kernel_hash = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
        if fused_name == "fused":
            kernel_name = f"nv_universal_gemm_{kernel_hash}"
        else:
            kernel_name = f"nv_universal_gemm_{fused_name}_{kernel_hash}"

        wrapper.src_to_kernel[src_code] = kernel_name

        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        _, _, kernel_path = get_path(code_hash(src_code), "py")

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(
            f"async_compile.nv_universal_gemm({kernel_name!r}, r'''"
        )
        compile_wrapper.splice(src_code, strip=True)
        if precompile_metadata is not None:
            compile_wrapper.writeline(
                f"''', precompile_metadata={precompile_metadata!r})"
            )
        else:
            compile_wrapper.writeline("''')")

        metadata_comment = f"# kernel path: {kernel_path}"
        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment += "\n" + origins + "\n" + detailed_origins
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)

        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
        *,
        only_gen_src_code: bool = False,
    ) -> str | None:
        """
        Codegen a NVIDIA Universal GEMM template with optional epilogue fusion.

        If `only_gen_src_code=True` the src code will be returned instead of being
        codegenned into the wrapper (used for benchmarking).
        """
        log.debug(
            "NVGEMM codegen_template: template_node=%s, epilogue_nodes=%s, prologue_nodes=%s",
            template_node,
            [n.get_name() for n in epilogue_nodes] if epilogue_nodes else [],
            [n.get_name() for n in prologue_nodes] if prologue_nodes else [],
        )
        assert self.is_nv_universal_gemm_template(template_node), (  # noqa: S101
            "Template node passed to NVUniversalGemmScheduling.codegen_template must be a "
            "SchedulerNode that wraps a NVUniversalGemmBuffer or MultiTemplateBuffer with NVGEMM choice"
        )
        assert not prologue_nodes, (  # noqa: S101
            "NVIDIA Universal GEMM doesn't support prologue fusion yet"
        )

        template_node = cast(SchedulerNode, template_node)

        original_ir_node = template_node.node
        assert isinstance(original_ir_node, Buffer)  # noqa: S101
        original_buffer_name = original_ir_node.get_name()

        min_tile_n = 0
        reductions = []
        if epilogue_nodes:
            reductions = self._partition_local_reductions(
                original_ir_node, epilogue_nodes
            )[0]
            min_tile_n = max(
                (group for _, group, axis, _, _ in reductions if axis == 1),
                default=0,
            )
            for matcher in (
                self._grouped_reduce_feeds_main_config_from_nodes,
                self._grouped_softmax_config_from_nodes,
                self._grouped_sum_normalize_config_from_nodes,
                self._grouped_absmax_normalize_config_from_nodes,
            ):
                feed_main = matcher(original_ir_node, epilogue_nodes)
                if feed_main is not None and feed_main[2] == 1:
                    min_tile_n = max(min_tile_n, feed_main[1])
                    break
        ctb: NVUniversalGemmBuffer = self.get_nv_gemm_buffer_from_node(
            template_node,
            require_epilogue_fusion=bool(epilogue_nodes),
            min_tile_n=min_tile_n,
        )

        epilogue_fn_code: str | None = None
        epilogue_reads: list[str] = []
        epilogue_writes: list[str] = []
        epilogue_var_renames: dict[str, Any] = {}
        local_reduce: (
            tuple[
                str | None,
                int,
                int,
                str,
                str,
                str,
                bool,
                str | None,
                str | None,
                str | None,
            ]
            | None
        ) = None

        if epilogue_nodes:
            scheduler = V.graph.scheduler
            try:
                local_reductions, local_reduce_nodes = self._partition_local_reductions(
                    original_ir_node, epilogue_nodes
                )
                if len(local_reductions) > 1:
                    raise NotImplementedError(
                        "NVGEMM supports one grouped local reduction"
                    )
                if local_reductions and local_reductions[0][3] == "direct_bool_gt_zero":
                    direct_output, group, axis, reduce_type, source = local_reductions[
                        0
                    ]
                    local_reduce = (
                        None,
                        group,
                        axis,
                        reduce_type,
                        source,
                        original_buffer_name,
                        False,
                        None,
                        direct_output,
                        reduce_type,
                    )
                else:
                    local_reduce = (
                        (
                            *local_reductions[0],
                            original_buffer_name,
                            False,
                            None,
                            None,
                            None,
                        )
                        if local_reductions
                        else None
                    )
                feed_main = self._grouped_reduce_feeds_main_config_from_nodes(
                    original_ir_node, epilogue_nodes
                )
                if feed_main is None:
                    feed_main = self._grouped_softmax_config_from_nodes(
                        original_ir_node, epilogue_nodes
                    )
                if feed_main is None:
                    feed_main = self._grouped_sum_normalize_config_from_nodes(
                        original_ir_node, epilogue_nodes
                    )
                if feed_main is None:
                    feed_main = self._grouped_absmax_normalize_config_from_nodes(
                        original_ir_node, epilogue_nodes
                    )
                if feed_main is not None:
                    output_name, group, axis, reduce_type, source = feed_main
                    reduce_output = (
                        local_reduce[0]
                        if local_reduce is not None
                        and local_reduce[1:3] == (group, axis)
                        else None
                    )
                    feed_fused_names = OrderedSet(
                        node.get_name() for node in epilogue_nodes
                    )
                    feed_fused_names.add(original_buffer_name)
                    if (
                        reduce_output is not None
                        and scheduler.can_buffer_be_removed_through_fusion(
                            reduce_output, feed_fused_names
                        )
                    ):
                        reduce_output = None
                    typed_feed_output = (
                        output_name
                        if V.graph.get_dtype(output_name)
                        != original_ir_node.get_dtype()
                        else None
                    )
                    feed_reads = next(
                        (
                            OrderedSet(
                                read.name
                                for read in scheduler_node.read_writes.reads
                                if read.name != original_buffer_name
                            )
                            for epilogue_node in epilogue_nodes
                            for scheduler_node in epilogue_node.get_nodes()
                            if isinstance(scheduler_node.node, Buffer)
                            and scheduler_node.node.get_name() == output_name
                        ),
                        OrderedSet(),
                    )
                    equivalent_feed_outputs = []
                    secondary_feed_output = None
                    secondary_feed_type = None
                    m, n = original_ir_node.get_size()

                    def is_feed_output_size(size) -> bool:
                        return V.graph.sizevars.statically_known_list_equals(
                            size, (m, n)
                        ) or (
                            len(size) == 3
                            and (
                                (
                                    axis == 0
                                    and V.graph.sizevars.statically_known_equals(
                                        size[0] * group, m
                                    )
                                    and V.graph.sizevars.statically_known_equals(
                                        size[1], group
                                    )
                                    and V.graph.sizevars.statically_known_equals(
                                        size[2], n
                                    )
                                )
                                or (
                                    axis == 1
                                    and V.graph.sizevars.statically_known_equals(
                                        size[0], m
                                    )
                                    and V.graph.sizevars.statically_known_equals(
                                        size[1] * group, n
                                    )
                                    and V.graph.sizevars.statically_known_equals(
                                        size[2], group
                                    )
                                )
                            )
                        )

                    for epilogue_node in epilogue_nodes:
                        for scheduler_node in epilogue_node.get_nodes():
                            buffer = scheduler_node.node
                            if not (
                                isinstance(buffer, ComputedBuffer)
                                and isinstance(buffer.data, Pointwise)
                                and buffer.get_name() != output_name
                                and is_feed_output_size(buffer.get_size())
                            ):
                                continue
                            candidate_reads = OrderedSet(
                                read.name
                                for read in scheduler_node.read_writes.reads
                                if read.name != original_buffer_name
                            )
                            if not feed_reads or candidate_reads != feed_reads:
                                continue
                            consumer_type = self._sum_normalize_consumer_type(
                                buffer
                            ) or self._centered_mean_consumer_type(buffer)
                            if consumer_type == reduce_type:
                                equivalent_feed_outputs.append(buffer.get_name())
                            elif secondary_feed_output is None:
                                secondary_type = self._sum_multiply_consumer_type(
                                    buffer
                                )
                                if secondary_type is not None:
                                    secondary_feed_output = buffer.get_name()
                                    secondary_feed_type = secondary_type
                    for equivalent_output in equivalent_feed_outputs:
                        if typed_feed_output is None:
                            typed_feed_output = equivalent_output
                        elif (
                            equivalent_output != typed_feed_output
                            and secondary_feed_output is None
                        ):
                            secondary_feed_output = equivalent_output
                            secondary_feed_type = reduce_type
                    primary_output = (
                        original_buffer_name
                        if typed_feed_output is not None
                        else output_name
                    )
                    local_reduce = (
                        reduce_output,
                        group,
                        axis,
                        reduce_type,
                        source,
                        primary_output,
                        True,
                        typed_feed_output,
                        secondary_feed_output,
                        secondary_feed_type,
                    )
                    local_reduce_nodes |= OrderedSet(
                        epilogue_node
                        for epilogue_node in epilogue_nodes
                        if any(
                            scheduler_node.node.get_name()
                            in (
                                output_name,
                                typed_feed_output,
                                secondary_feed_output,
                            )
                            for scheduler_node in epilogue_node.get_nodes()
                            if isinstance(scheduler_node.node, Buffer)
                        )
                    )
                    epilogue_fn_code = (
                        f"def {EPILOGUE_FN_NAME}(accum):\n    D = accum\n    return D"
                    )
                    epilogue_writes = [primary_output]
                    epilogue_var_renames = {
                        _ACCUMULATOR_ARG_NAME: original_buffer_name,
                        "D": primary_output,
                    }
                evt_nodes = [
                    node for node in epilogue_nodes if node not in local_reduce_nodes
                ]
                fused_buffer_names: OrderedSet[str] = OrderedSet(
                    n.get_name() for n in epilogue_nodes
                )
                fused_buffer_names.add(original_buffer_name)
                removed_buffers_with_gemm = V.graph.removed_buffers.copy()
                if scheduler.can_buffer_be_removed_through_fusion(
                    original_buffer_name, fused_buffer_names
                ):
                    removed_buffers_with_gemm.add(original_buffer_name)

                if evt_nodes:
                    reads, writes, var_renames, evt_code = (
                        CutlassEVTCodegen.ir_to_evt_python_code(
                            original_buffer_name,
                            evt_nodes,
                            removed_buffers_with_gemm,
                            fn_name=EPILOGUE_FN_NAME,
                            as_standalone_function=True,
                        )
                    )
                    epilogue_fn_code = evt_code
                    epilogue_reads = reads
                    epilogue_writes = writes
                    epilogue_var_renames = var_renames
                    if feed_main is not None and local_reduce is not None:
                        d_buf = var_renames.get("D")
                        if isinstance(d_buf, str):
                            local_reduce = (
                                local_reduce[0],
                                local_reduce[1],
                                local_reduce[2],
                                local_reduce[3],
                                local_reduce[4],
                                d_buf,
                                local_reduce[6],
                                local_reduce[7],
                                local_reduce[8],
                                local_reduce[9],
                            )

                if not only_gen_src_code:
                    write_bufs = OrderedSet(epilogue_writes)
                    if local_reduce is not None:
                        if local_reduce[0] is not None:
                            write_bufs.add(local_reduce[0])
                        if local_reduce[7] is not None:
                            write_bufs.add(local_reduce[7])
                        if local_reduce[8] is not None:
                            write_bufs.add(local_reduce[8])
                    for node in epilogue_nodes:
                        node_name = node.get_name()
                        if node_name in write_bufs:
                            continue
                        if scheduler.can_buffer_be_removed_through_fusion(
                            node_name, fused_buffer_names
                        ):
                            V.graph.removed_buffers.add(node_name)
                    if (
                        original_buffer_name not in write_bufs
                        and scheduler.can_buffer_be_removed_through_fusion(
                            original_buffer_name, fused_buffer_names
                        )
                    ):
                        V.graph.removed_buffers.add(original_buffer_name)
                    for node in epilogue_nodes:
                        node.mark_run()

                log.debug(
                    "NVGEMM epilogue fusion: %d nodes, reads=%s, writes=%s, local_reduce=%s",
                    len(epilogue_nodes),
                    epilogue_reads,
                    epilogue_writes,
                    local_reduce,
                )
            except (NotImplementedError, AssertionError) as e:
                log.warning("NVGEMM epilogue codegen failed unexpectedly: %s", e)
                raise

        assert ctb.make_kernel_render is not None  # noqa: S101 # noqa: S101
        kernel, render = ctb.make_kernel_render(
            ctb,
            epilogue_fn_code=epilogue_fn_code,
            epilogue_reads=epilogue_reads,
            epilogue_writes=epilogue_writes,
            epilogue_var_renames=epilogue_var_renames,
            local_reduce=local_reduce,
        )

        if not only_gen_src_code:
            template_node.mark_run()

        src_code = render()

        if only_gen_src_code:
            return src_code

        # Precompile only base (non-EFC) kernels. EFC kernels produce
        # closure-wrapped artifacts that can't be serialized to disk cache.
        if epilogue_nodes:
            precompile_metadata = None
        else:
            precompile_metadata = self._build_precompile_metadata(kernel, ctb)

        with V.set_kernel_handler(kernel):
            node_schedule: list[BaseSchedulerNode] = [template_node]
            if epilogue_nodes:
                node_schedule.extend(epilogue_nodes)
            kernel_name = self.define_kernel(
                src_code, node_schedule, precompile_metadata
            )

        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
        self.free_buffers_in_scheduler()
        return None

    def _build_precompile_metadata(self, kernel, ctb):
        """Extract shapes and dtypes from kernel inputs/output for subprocess precompilation.

        Only called for base (non-epilogue) kernels. Returns None if shapes are
        symbolic (dynamic shapes), in which case the subprocess will skip
        precompilation and the kernel compiles lazily on first call.
        """
        if not hasattr(kernel, "_template_input_args"):
            return None

        precompile_shapes = {}
        precompile_strides = {}
        precompile_dtypes = {}

        try:
            for param_name, input_node in kernel._template_input_args:
                size = input_node.get_size()
                precompile_shapes[param_name] = [int(s) for s in size]
                stride = input_node.get_stride()
                precompile_strides[param_name] = [int(s) for s in stride]
                precompile_dtypes[param_name] = str(
                    input_node.get_dtype()
                ).removeprefix("torch.")

            out_layout = cast(Layout, ctb.layout)
            precompile_shapes["output"] = [int(s) for s in out_layout.size]
            precompile_strides["output"] = [int(s) for s in out_layout.stride]
            precompile_dtypes["output"] = str(out_layout.dtype).removeprefix("torch.")
        except (TypeError, RuntimeError, ValueError):
            log.debug(
                "Skipping NV Universal GEMM precompile metadata: symbolic sizes "
                "cannot be resolved to concrete values"
            )
            return None

        device = ctb.layout.device
        device_index = device.index if device.index is not None else 0

        import torch

        device_capability = None
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability(device_index)

        max_active_clusters = None
        kernel_name = ctb.kernel_metadata.get("kernel_name")
        try:
            if kernel_name and torch.cuda.is_available():
                from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
                    get_kernel_by_name,
                )

                k = get_kernel_by_name(kernel_name)
                if k is not None and hasattr(k, "impl"):
                    from cutlass.operators.providers.cutedsl.integration_utils.mma import (
                        get_max_active_clusters,
                    )

                    max_active_clusters = get_max_active_clusters(
                        k.impl.cluster_shape_mn
                    )
        except Exception:
            log.debug(
                "Failed to resolve max_active_clusters for precompile", exc_info=True
            )

        return {
            "precompile_shapes": precompile_shapes,
            "precompile_strides": precompile_strides,
            "precompile_dtypes": precompile_dtypes,
            "device_index": device_index,
            "device_capability": device_capability,
            "max_active_clusters": max_active_clusters,
        }

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[BaseSchedulerNode],
        benchmark_kernel: bool = False,
        hint_override: int | None = None,
    ) -> str:
        """Generate benchmark source for an NVGEMM template and its epilogue."""
        prologue, template, epilogue = nodes[0].get_prologue_template_epilogue(
            list(nodes)
        )

        epilogue_reads: list[str] = []
        output_bufs: list[str] = []
        if epilogue:
            template_sn = cast(SchedulerNode, template)
            assert isinstance(template_sn.node, Buffer)  # noqa: S101
            original_buffer_name = template_sn.node.get_name()
            local_reductions, local_reduce_nodes = self._partition_local_reductions(
                template_sn.node, epilogue
            )
            feed_main = self._grouped_reduce_feeds_main_config_from_nodes(
                template_sn.node, epilogue
            )
            if feed_main is None:
                feed_main = self._grouped_softmax_config_from_nodes(
                    template_sn.node, epilogue
                )
            if feed_main is None:
                feed_main = self._grouped_sum_normalize_config_from_nodes(
                    template_sn.node, epilogue
                )
            if feed_main is None:
                feed_main = self._grouped_absmax_normalize_config_from_nodes(
                    template_sn.node, epilogue
                )
            evt_nodes = [
                node
                for node in epilogue
                if node not in local_reduce_nodes and feed_main is None
            ]
            removed_buffers_with_gemm = V.graph.removed_buffers.copy()
            if not local_reductions:
                removed_buffers_with_gemm.add(original_buffer_name)
            try:
                if evt_nodes:
                    reads, writes, var_renames, _ = (
                        CutlassEVTCodegen.ir_to_evt_python_code(
                            original_buffer_name,
                            evt_nodes,
                            removed_buffers_with_gemm,
                        )
                    )
                    epilogue_reads = reads
                    d_buf = var_renames.get("D")
                    output_bufs = ([d_buf] if d_buf else []) + [
                        w for w in writes if w != d_buf
                    ]
                if local_reductions:
                    output_bufs = [
                        original_buffer_name,
                        *output_bufs,
                        local_reductions[0][0],
                    ]
                elif feed_main is not None:
                    output_bufs = [feed_main[0]]
            except (NotImplementedError, AssertionError) as e:
                log.warning("NVGEMM benchmark epilogue codegen failed: %s", e)

        with config.patch("benchmark_kernel", benchmark_kernel):
            src_code = self.codegen_template(
                template,
                epilogue,
                prologue,
                only_gen_src_code=True,
            )

        assert src_code is not None  # noqa: S101 # noqa: S101
        src_code = src_code.replace(
            str(Placeholder.KERNEL_NAME), _BENCHMARK_KERNEL_PREFIX
        )

        if benchmark_kernel:
            src_code = self._add_benchmark_helpers(
                src_code, template, epilogue, epilogue_reads, output_bufs
            )

        return src_code

    def _add_benchmark_helpers(
        self,
        src_code: str,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        epilogue_reads: list[str],
        output_bufs: list[str] | None = None,
    ) -> str:
        template_node = cast(SchedulerNode, template_node)
        ctb: NVUniversalGemmBuffer = self.get_nv_gemm_buffer_from_node(
            template_node, require_epilogue_fusion=bool(epilogue_nodes)
        )

        input_nodes = cast(list[Buffer], ctb.inputs)
        # Output store layouts in out_ptr order. A multi-store epilogue has one
        # per graph output; otherwise a single output (the fused final node, or
        # the plain GEMM layout).
        output_layouts: list[Layout] = []
        if output_bufs:
            for b in output_bufs:
                buf = V.graph.get_buffer(b)
                # pyrefly: ignore [missing-attribute]
                output_layouts.append(cast(Layout, buf.get_layout()))
        elif epilogue_nodes:
            final_node = cast(SchedulerNode, epilogue_nodes[-1])
            # pyrefly: ignore [missing-attribute]
            output_layouts.append(cast(Layout, final_node.node.get_layout()))
        else:
            output_layouts.append(cast(Layout, ctb.layout))

        args_code = IndentedBuffer()
        args_code.writeline("")
        args_code.writeline("is_nvgemm = True")
        args_code.writeline("")
        args_code.writeline("def get_args():")
        with args_code.indent():
            args_code.writeline("import torch")
            args_code.writeline("from torch._dynamo.testing import rand_strided")
            args_code.writeline("args = []")

            for inp in input_nodes:
                size = V.graph.sizevars.optimization_hints(inp.get_size())
                stride = V.graph.sizevars.optimization_hints(inp.get_stride())
                dtype = inp.get_dtype()
                device = inp.get_device()
                args_code.writeline(
                    f"args.append(rand_strided({size}, {stride}, device='{device}', dtype={dtype}))"
                )

            for ol in output_layouts:
                out_size = V.graph.sizevars.optimization_hints(ol.size)
                out_stride = V.graph.sizevars.optimization_hints(ol.stride)
                args_code.writeline(
                    f"args.append(rand_strided({out_size}, {out_stride}, device='{ol.device}', dtype={ol.dtype}))"
                )

            for read_name in epilogue_reads:
                buf = V.graph.get_buffer(read_name)
                size = V.graph.sizevars.optimization_hints(buf.get_size())
                stride = V.graph.sizevars.optimization_hints(buf.get_stride())
                dtype = buf.get_dtype()
                device = buf.get_device()
                args_code.writeline(
                    f"args.append(rand_strided({size}, {stride}, device='{device}', dtype={dtype}))"
                )

            if ctb.workspace_size > 0:
                args_code.writeline(
                    f"args.append(torch.empty({ctb.workspace_size}, "
                    f"device='{output_layouts[0].device}', dtype=torch.int8))"
                )

            args_code.writeline("return args")

        args_code.writeline("")
        args_code.writeline("def call(args):")
        with args_code.indent():
            args_code.writeline("import torch")
            num_inputs = len(input_nodes)
            n_fixed = num_inputs + len(output_layouts)
            param_list = [f"args[{i}]" for i in range(n_fixed)]

            for j in range(len(epilogue_reads)):
                param_list.append(f"args[{n_fixed + j}]")

            if ctb.workspace_size > 0:
                param_list.append(f"args[{n_fixed + len(epilogue_reads)}]")

            params_str = ", ".join(param_list)
            args_code.writeline("stream = torch.cuda.current_stream().cuda_stream")
            bench_fn_name = f"{_BENCHMARK_KERNEL_PREFIX}_{MAIN_SUFFIX}"
            args_code.writeline(f"{bench_fn_name}({params_str}, stream=stream)")

        return src_code + args_code.getvalue()
