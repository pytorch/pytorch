from __future__ import annotations

import typing
from typing import Any, Dict, List, Type, TYPE_CHECKING

import sympy

from . import config
from .codecache import write_text
from .codegen.triton import FixedTritonConfig, TritonKernel
from .metrics import get_metric_table, is_metric_table_enabled
from .runtime.hints import DeviceProperties, ReductionHint
from .runtime.runtime_utils import next_power_of_2
from .runtime.triton_heuristics import _num_warps
from .scheduler import BaseSchedulerNode, Scheduler, WhyNoFuse
from .virtualized import V


if TYPE_CHECKING:
    import torch

    from .codegen.simd_kernel_features import MemoryStats, SIMDKernelFeatures


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
        kernel_args: List[Dict[str, sympy.Expr]],
        kernel_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Hook to change the kwargs passed to TritonKernel, used to apply fixed configurations"""
        (groups,) = kernel_args
        device = V.graph.get_current_device_or_throw()
        if not (
            (
                config.max_autotune
                or config.max_autotune_pointwise
                or config.coordinate_descent_tuning
                or config.triton.multi_kernel
            )
            and kernel_cls is TritonKernel
            and features.is_reduction()
            and len(groups) == 2
            and device.type != "cpu"
        ):
            return kernel_kwargs

        mstats = features.memory_stats(groups)
        if mstats.persistent.reads.dim[1].count_per_thread_non_contiguous > 0:
            # TODO(jansel): still need to tune heuristics for non-contiguous
            return kernel_kwargs

        return self._contiguous_reduction_fixed_config(
            features, kernel_kwargs, mstats, device
        )

    def _contiguous_reduction_fixed_config(
        self,
        features: SIMDKernelFeatures,
        kernel_kwargs: Dict[str, Any],
        mstats: MemoryStats,
        device: torch.device,
    ) -> Dict[str, Any]:
        pstats = mstats.persistent
        lstats = mstats.looped
        xhint = next_power_of_2(
            V.graph.sizevars.size_hint(features.numel, fallback=8192)
        )
        rhint = next_power_of_2(
            V.graph.sizevars.size_hint(features.reduction_numel, fallback=8192)
        )

        # need to respect existing overrides
        cooperative = config.triton.cooperative_reductions and kernel_kwargs.get(
            "override_cooperative_reduction"
        )
        if cooperative is not False:
            rsplit = next_power_of_2(
                self.reduction_split_factor(
                    device, xhint, rhint, pstats.reads.dim[1].contiguous_score >= 0.5
                )
            )
            cooperative = rsplit > 1
        else:
            cooperative = False
            rsplit = 1

        props = DeviceProperties.create(device)
        safety_factor = 0.9
        assert props.regs_per_multiprocessor is not None
        register_bytes_per_sm_threshold = int(
            4 * props.regs_per_multiprocessor * safety_factor
        )
        xblock = 1

        target = 16384 // next_power_of_2(max(pstats.reads.bytes_per_thread // 2, 1))
        target = max(512, min(target, 8192))

        persistent = config.triton.persistent_reductions and kernel_kwargs.get(
            "override_persistent_reduction"
        )
        if persistent is None:
            if (
                pstats.memory.bytes_per_thread < lstats.memory.bytes_per_thread
                and pstats.reads.dim[0].count_per_thread_contiguous == 0
            ):
                # using a persistent reduction will save memory lets try to make it happen
                threshold = register_bytes_per_sm_threshold
            elif pstats.reads.dim[0].count_per_thread_contiguous == 0:
                threshold = register_bytes_per_sm_threshold // 4
            else:
                threshold = 8192
            threshold = threshold * rsplit // xblock

            persistent = next_power_of_2(
                rhint
            ) < threshold and V.graph.sizevars.statically_known_leq(
                pstats.memory.bytes_per_thread * features.reduction_numel, threshold
            )

        if persistent:
            rblock = rhint // rsplit
        else:
            rblock = min(rhint, max(target // xblock, 8))

        if xblock * rblock < target // 2:
            xblock = target // rblock

        cfg = {"XBLOCK": xblock}
        if not persistent:
            cfg["R0_BLOCK"] = rblock
        if cooperative:
            cfg["RSPLIT"] = rsplit
        if pstats.reads.bytes_per_thread > 16:
            cfg["num_warps"] = _num_warps(xblock * rblock // 64, 32)
        else:
            cfg["num_warps"] = _num_warps(xblock * rblock // 128, 16)
        cfg["num_stages"] = 1
        return {
            **kernel_kwargs,
            "override_cooperative_reduction": cooperative,
            "override_persistent_reduction": persistent,
            "fixed_config": FixedTritonConfig(cfg),
        }

    def should_use_cooperative_reduction(self, features: SIMDKernelFeatures) -> bool:
        """Heuristic to decide if a cooperative reduction should be used."""
        if config.triton.force_cooperative_reductions:
            return True
        device = V.graph.get_current_device_or_throw()
        if not config.triton.cooperative_reductions or device.type == "cpu":
            return False

        split = self.reduction_split_factor(
            device,
            V.graph.sizevars.size_hint(features.reduction_numel, fallback=8192),
            V.graph.sizevars.size_hint(features.numel, fallback=8192),
            features.memory_stats().persistent.reads.dim[1].contiguous_score >= 0.5,
        )
        return split > 1

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
    def reduction_split_factor(
        device: torch.device,
        reduction_numel_hint: int,
        numel_hint: int,
        inner_reduction: bool,
    ) -> int:
        """Heuristic to decide the RSPLIT used for split reductions.
        When a reduction has a small number of outputs there is not enough parallelism,
        so we will do the reduction in two phases."""
        props = DeviceProperties.create(device)
        num_sm = props.multi_processor_count
        min_elements_per_thread = 32
        max_elements_per_thread = 512
        threads_per_sm = 2048
        min_elements_per_device = min_elements_per_thread * num_sm * threads_per_sm
        max_elements_per_device = max_elements_per_thread * num_sm * threads_per_sm
        num_warps = 8
        num_threads = 32 * num_warps

        if reduction_numel_hint <= 32 or numel_hint >= num_sm * 64:
            return 1

        if inner_reduction:
            # do heuristics that's close to eager mode for split inner reduction
            # we leak reduction autotune configs here, and will need to refactor to avoid this later
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
        else:
            # TODO the best heuristic currently has XBLOCK (corresponding to numel_hint) 128
            # extend to even smaller number of outputs
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

        # prologue fusion always last
        if node2.is_template():
            template_score = 0
        else:
            template_score = 1 + (
                (node1.is_template() == config.epilogue_fusion_first)
                and memory_score > 0
            )

        return (
            template_score,
            node1.is_reduction() == node2.is_reduction() and memory_score > 0,
            memory_score,
            proximity_score,
        )
