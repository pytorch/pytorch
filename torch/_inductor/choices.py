from __future__ import annotations

import typing
from typing import Any, Dict, List, Type, TYPE_CHECKING

from . import config
from .codecache import write_text
from .metrics import get_metric_table, is_metric_table_enabled
from .runtime.hints import ReductionHint
from .scheduler import BaseSchedulerNode, Scheduler, WhyNoFuse
from .virtualized import V


if TYPE_CHECKING:
    import sympy

    from .codegen.simd_kernel_features import SIMDKernelFeatures
    from .codegen.triton import TritonKernel


class Sortable(typing.Protocol):
    """Anything that can be used as a list.sort() key (int/tuple/etc)"""

    def __lt__(self, other: typing.Self) -> bool:
        ...


class InductorChoices:
    """
    This class contains a collection of default heuristics that effect performance of our generated
    code.  We try to not put correctness requirements in this file.

    You can override the choices made here by doing:

            class MyHeuristics(InductorChoices):
                ...

            torch._inductor.virtualized.V.set_choices_handler(MyHeuristics())
    """

    def triton_kernel_kwargs(
        self,
        kernel_cls: Type[TritonKernel],
        features: SIMDKernelFeatures,
        groups: List[sympy.Expr],
        kernel_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Hook to change the kwargs passed to TritonKernel, used to apply fixed configurations"""
        return kernel_kwargs

    @staticmethod
    def should_use_cooperative_reduction(features: SIMDKernelFeatures) -> bool:
        """Heuristic to decide if a cooperative reduction should be used."""
        if config.triton.force_cooperative_reductions:
            return True
        if (
            not config.triton.cooperative_reductions
            or V.graph.get_current_device_or_throw().type == "cpu"
        ):
            return False

        xhint = V.graph.sizevars.size_hint(features.numel, fallback=2)
        if xhint <= 8:
            threshold = 32768 * xhint
        elif xhint <= 16:
            threshold = 2097152
        else:
            return False
        # TODO(jansel): should this default on for dynamic shapes?
        return V.graph.sizevars.statically_known_geq(
            features.reduction_numel, threshold
        )

    @staticmethod
    def should_use_persistent_reduction(
        features: SIMDKernelFeatures, cooperative_reduction: bool
    ) -> bool:
        """
        Heuristic to decide if a persistent reduction should be used.
        """
        if not config.triton.persistent_reductions:
            return False
        threshold = {
            ReductionHint.INNER: 1024,
        }.get(features.get_reduction_hint(), 64)

        if cooperative_reduction:
            # The RSPLIT of cooperative reductions means each thread block is operating on fewer elements
            try:
                threshold *= 32 // min(V.graph.sizevars.size_hint(features.numel), 32)
            except ValueError:
                pass  # unbacked symint

        # If multi_kernel is enabled, we do more aggressive persistent reduction.
        # This may result in some persistent reductions slower than the
        # corresponding non-persistent reductions. MultiKernel will do benchmarking
        # to pick the faster one.
        if config.triton.multi_kernel:
            threshold *= 16
        return V.graph.sizevars.statically_known_leq(features.reduction_numel, threshold)  # type: ignore[arg-types]

    @staticmethod
    def want_no_x_dim(features: SIMDKernelFeatures) -> bool:
        """
        Heuristic to decide if we should drop the X dimension from a persistent reduction kernel.
        So the [XBLOCK, RBLOCK] block becomes a [RBLOCK] block and XBLOCK is forced to be always 1.
        Strangely this is faster than a [1, RBLOCK] block in some cases.
        """
        return (
            features.get_reduction_hint() == ReductionHint.INNER
            and V.graph.sizevars.statically_known_geq(features.reduction_numel, 256)
        )

    @staticmethod
    def can_fuse(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """
        Heuristics to prevent fusion applied to both horizontal and vertical fusions.  Heuristics here should not
        be needed for correctness and tweaking them may yield additional performance.

        See also some related heuristics that can be changed via config:
            - config.triton.tiling_prevents_pointwise_fusion
            - config.triton.tiling_prevents_reduction_fusion
            - config.aggressive_fusion (will cause this function to be called more times)
        """
        if shared_data_score == 0 and (
            not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
        ):
            if is_metric_table_enabled("fusion_failure_due_to_indexing_mismatch"):
                common_buf_names = (
                    node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
                )
                if len(common_buf_names) > 0:
                    get_metric_table("fusion_failure_due_to_indexing_mismatch").add_row(
                        lambda: {
                            "pre_grad_graph_id": V.graph.graph_id,
                            "post_grad_graph_id": V.graph.post_grad_graph_id,
                            "node1_name": node1.get_name(),
                            "node2_name": node2.get_name(),
                            "node1_debug_str": write_text(node1.debug_str()),
                            "node2_debug_str": write_text(node2.debug_str()),
                            "common_buffer_names": list(common_buf_names),
                            "failure_reason": scheduler.decide_fusion_fail_reason(
                                node1, node2, common_buf_names
                            ),
                        }
                    )

                    WhyNoFuse(node1, node2)("no shared data due to indexing mismatch")
                    return False
            WhyNoFuse(node1, node2)("no shared data")
            return False  # heuristic not needed for correctness

        if (
            not node1.is_foreach()
            and not node2.is_foreach()
            and len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size
        ):
            WhyNoFuse(node1, node2)("exceeds max fusion")
            return False  # heuristic not needed for correctness

        if scheduler.can_fusion_increase_peak_memory(node1, node2):
            WhyNoFuse(node1, node2)("Fusion will increase peak memory")
            return False

        return True

    @staticmethod
    def can_fuse_vertical(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """Hook for heuristics to prevent vertical (producer/consumer) fusions"""
        return True

    @staticmethod
    def can_fuse_horizontal(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """Hook for heuristics to prevent horizontal (consumer/consumer) fusions"""
        if shared_data_score < config.score_fusion_memory_threshold:
            WhyNoFuse(node1, node2)("score_fusion_memory_threshold")
            return False
        if scheduler.are_long_distant_nodes(node1, node2):
            WhyNoFuse(node1, node2)(
                "Nodes are too far away. Fusing them may increase peak memory."
            )
            return False
        return True

    @staticmethod
    def score_fusion(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
    ) -> Sortable:
        """
        Assign a score (higher comes first) to the fusion of node1 and node2.
        When different fusions conflict with each other, this is the way we
        decide what order to run them in.

        Our current score is based on:
        - The type of fusion (template/reduction/etc)
        - Estimate of the saved memory operations
        - Fusions closer together in original graph order
        """
        memory_score = scheduler.score_fusion_memory(node1, node2)
        proximity_score = -max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )
        return (
            node1.is_template() == config.epilogue_fusion_first and memory_score > 0,
            node1.is_reduction() == node2.is_reduction() and memory_score > 0,
            memory_score,
            proximity_score,
        )
