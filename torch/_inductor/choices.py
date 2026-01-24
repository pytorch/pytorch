from __future__ import annotations

import dataclasses
import typing
from typing import Any, Optional, TYPE_CHECKING, Union

import sympy

import torch
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.scheduler import MixOrderReduction
from torch.utils._sympy.value_ranges import bound_sympy

from . import config
from .codecache import write_text
from .kernel_inputs import KernelInputs  # noqa: TC001
from .kernel_template_choice import make_ktc_generator
from .metrics import get_metric_table, is_metric_table_enabled
from .runtime.hints import DeviceProperties, ReductionHint
from .scheduler import BaseSchedulerNode, Scheduler, WhyNoFuse
from .select_algorithm import ExternKernelChoice
from .template_heuristics import get_template_heuristic
from .template_heuristics.triton import (
    BaseConfigHeuristic,
    CPUConfigHeuristic,
    CUDAConfigHeuristic,
    MTIAConfigHeuristic,
    ROCmConfigHeuristic,
    XPUConfigHeuristic,
)
from .utils import _use_autotune_backend
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Generator
    from functools import partial

    from triton import Config as TritonConfig

    from .codegen.common import KernelTemplate
    from .codegen.simd_kernel_features import SIMDKernelFeatures
    from .codegen.triton import TritonKernel
    from .ir import ChoiceCaller
    from .kernel_template_choice import KernelTemplateChoice

    from torch.utils._ordered_set import OrderedSet  # isort: skip


class Sortable(typing.Protocol):
    """Anything that can be used as a list.sort() key (int/tuple/etc)"""

    def __lt__(self, other: typing.Self) -> bool: ...


@dataclasses.dataclass
class FusionScore:
    template_score: int
    node_type_score: bool
    memory_score: int
    proximity_score: int

    def __lt__(self, other):
        """
        node_type_score has higher priority than memory_score unless
        the memory_score differs too much
        """
        threshold = 16
        if self.template_score != other.template_score:
            return self.template_score < other.template_score

        if (
            max(self.memory_score, other.memory_score)
            > min(self.memory_score, other.memory_score) * threshold
        ):
            return self.memory_score < other.memory_score

        return (self.node_type_score, self.memory_score, self.proximity_score) < (
            other.node_type_score,
            other.memory_score,
            other.proximity_score,
        )


class InductorChoices:
    """
    This class contains a collection of default heuristics that effect performance of our generated
    code.  We try to not put correctness requirements in this file.

    You can override the choices made here by doing:

            class MyHeuristics(InductorChoices):
                ...

            torch._inductor.virtualized.V.set_choices_handler(MyHeuristics())
    """

    def get_config_heuristics(
        self, device_type: Optional[str] = "cuda"
    ) -> BaseConfigHeuristic:
        if device_type == "cuda":
            if torch.version.hip is None:
                return CUDAConfigHeuristic()
            else:
                return ROCmConfigHeuristic()
        elif device_type == "xpu":
            return XPUConfigHeuristic()
        elif device_type == "cpu":
            return CPUConfigHeuristic()
        elif device_type == "mtia":
            return MTIAConfigHeuristic()
        else:
            return BaseConfigHeuristic()

    # Conv configs
    def get_conv_configs(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        conv_heuristics = self.get_config_heuristics(device_type)
        return conv_heuristics.get_conv_configs()

    # Flex attention configs
    # TODO(coconutruben): break out flexattention/decode configs into the new retrieval mechanism
    def get_flex_attention_fwd_configs(
        self, head_dim: int, dtype: torch.dtype, device_type: Optional[str] = "cuda"
    ) -> list[Any]:
        flex_heuristics = self.get_config_heuristics(device_type)
        return flex_heuristics.get_flex_attn_fwd_configs(head_dim, dtype)

    def get_flex_attention_bwd_configs(
        self, head_dim: int, dtype: torch.dtype, device_type: Optional[str] = "cuda"
    ) -> list[Any]:
        flex_heuristics = self.get_config_heuristics(device_type)
        return flex_heuristics.get_flex_attn_bwd_configs(head_dim, dtype)

    def get_flex_decode_configs(
        self, head_dim: int, dtype: torch.dtype, device_type: Optional[str] = "cuda"
    ) -> list[Any]:
        flex_heuristics = self.get_config_heuristics(device_type)
        return flex_heuristics.get_flex_decode_configs(head_dim, dtype)

    def _finalize_template_configs(
        self,
        template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        templates: list[Union[KernelTemplate, ExternKernelChoice]],
        op_name: str,
        kwarg_overrides: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[KernelTemplateChoice]:
        """
        This method can be subclassed to perform any override/modification of the choices.
        The incoming parameters are cheap (generators), so you can do any overrides without
        incurring too much cost. Override this method to customize the kernel template choices
        before they are converted to ChoiceCaller objects, which is expensive on template codegen.

        The full list of arguments are here to facilitate any overrides you may want to do,
        as they can be used to start from scratch for each template if so desired.

        Args:
            template_choices: Dictionary mapping template UIDs to generators of KernelTemplateChoice objects
            kernel_inputs: MMKernelInputs containing input tensor nodes and matrix indices
            templates: List of template objects (KernelTemplate or ExternKernelChoice) in use
            op_name: Operation name (e.g., "bmm", "baddbmm", "addmm")
            kwarg_overrides: Optional dict of kwargs to override for each template heuristic

        Returns:
            Flattened list of KernelTemplateChoice objects across all templates
        """
        choices: list[KernelTemplateChoice] = []
        for choice_gen in template_choices.values():
            choices.extend(choice_gen)
        return choices

    def get_ktc(
        self,
        kernel_inputs: KernelInputs,
        template: Union[KernelTemplate, ExternKernelChoice],
        op_name: str,
        kwarg_overrides: Optional[dict[str, Any]] = None,
    ) -> Generator[KernelTemplateChoice, None, None]:
        """
        Utility to get the KernelTemplateChoice generator for a specific input.

        This is a per template/op call, whereas get_template_configs is an op wide call (all templates).
        Consider when overriding/using at which level you need to make decisions
        """
        # Extract device_type from kernel_inputs
        device_type = kernel_inputs.device_type
        assert device_type is not None, "get_ktc requires a valid device type"
        # Extract template_name from the template object
        template_name = template.uid

        # Get the appropriate template-specific heuristic
        heuristic = get_template_heuristic(template_name, device_type, op_name)
        cs = heuristic.get_template_configs(
            kernel_inputs,
            op_name,
        )
        # adjust the kernel inputs to the template-specific heuristic, if needed
        # default here is to just return the kernel_inputs as is
        inputs_val = heuristic.adjust_kernel_inputs(kernel_inputs, op_name)
        extra_kwargs = heuristic.get_extra_kwargs(kernel_inputs, op_name)
        # Create KernelTemplateChoice generator using the moved function
        overrides = kwarg_overrides or {}
        return make_ktc_generator(
            template=template,
            cs=cs,
            extra_kwargs=extra_kwargs,
            overrides=overrides,
            layout=kernel_inputs.output_layout(),
            inputs=inputs_val,
        )

    def _need_to_fix_layout(
        self,
        adjusted_choices: list[KernelTemplateChoice],
        op_name: str,
    ) -> bool:
        """
        Check if we need to fix the layout instead of keeping it flexible

        Args:
            ktc: KernelTemplateChoice object

        Returns:
            True if we need to fix the layout, False otherwise
        """
        # TODO: debug and fix
        # NOTE: on mps, we see issues with flexible layouts on baddmm. This check just makes sure
        # that for mps, everything stays as it was before this optimization
        if len(adjusted_choices) > 0:
            if adjusted_choices[0].inputs.device_type == "mps" and op_name not in [
                "mm",
                "addmm",
            ]:
                return True

        # Since the following backends are not using get_mm_configs yet through the singular call,
        if not (config.max_autotune or config.max_autotune_gemm):
            # no danger of using other backends than ATEN
            if not config.max_autotune_allow_flexible_layouts and op_name not in [
                # The historical implementation for mm and addmm allowed had flexible layouts in the
                # not max-autotune world
                "mm",
                "addmm",
            ]:
                # TODO: deprecate this by migrating users to the new behavior
                return True
            return False

        if not config.max_autotune_allow_flexible_layouts:
            # we always need to fix the layout
            return True

        # Since the following backends are not using get_template_configs yet through the singular call,
        # we don't know if they are a valid choice or not. Instead, just skip the optimization
        # defensively.
        # TODO(coconutruben): remove this once CPP,CK,CUTLASS are supported
        if _use_autotune_backend("CUTLASS"):
            return True
        if _use_autotune_backend("CK") or _use_autotune_backend("CKTILE"):
            return True
        if _use_autotune_backend("CPP"):
            return True
        return any(
            not isinstance(ktc.template, ExternKernelChoice) for ktc in adjusted_choices
        )

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        templates: list[Union[KernelTemplate, ExternKernelChoice]],
        op_name: str,
        kwarg_overrides: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[ChoiceCaller]:
        """
        Get list of ChoiceCallers for MM templates using template-specific heuristics.

        Args:
            kernel_inputs: MMKernelInputs containing input tensor nodes and matrix indices
            layout: Output layout
            templates: List of template objects (KernelTemplate or ExternKernelChoice)
            op_name: Operation name (e.g., "bmm", "baddbmm", "addmm", "mm_plus_mm")
            kwarg_overrides: Optional dict of kwargs to override for each template heuristic,
                             indexed by template.uid. These only override the per config kwargs, not the extra kwargs
        Returns:
            List of ChoiceCaller objects from the templates
        """
        if kwarg_overrides is None:
            kwarg_overrides = {}
        input_tensors = kernel_inputs.nodes()
        if len(input_tensors) < 2:
            raise ValueError(f"Need at least 2 input tensors, got {len(input_tensors)}")
        layout = kernel_inputs.output_layout()
        # First pass: Create dict of template.uid to generator of KernelTemplateChoice objects
        template_choices = {}
        for template in templates:
            template_choices[template.uid] = self.get_ktc(
                kernel_inputs,
                template,
                op_name,
                kwarg_overrides.get(template.uid, {}),
            )

        # Second pass: Adjust the template choices
        adjusted_choices = self._finalize_template_configs(
            template_choices,
            kernel_inputs,
            templates,
            op_name,
            kwarg_overrides,
        )
        # Layout optimization: if all choices are ExternKernelChoice and layout is FixedLayout, convert to FlexibleLayout
        if self._need_to_fix_layout(adjusted_choices, op_name):
            layout = kernel_inputs.output_layout(flexible=False)
            for ktc in adjusted_choices:
                ktc.layout = layout
                # for good measure, delete the cached ChoiceCaller from the ktc if it existed.
                # ExternKernelChoice are cheap to generate
                if hasattr(ktc, "_choice"):
                    del ktc._choice
        # Third pass: Convert to ChoiceCaller objects
        return [ktc.choice for ktc in adjusted_choices if ktc.choice is not None]

    def triton_kernel_kwargs(
        self,
        kernel_cls: type[TritonKernel],
        features: SIMDKernelFeatures,
        groups: list[sympy.Expr],
        kernel_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
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

        xhint = V.graph.sizevars.optimization_hint(features.numel, fallback=2)
        if xhint <= 8:
            threshold = 32768 * xhint
        elif xhint <= 16:
            threshold = 2097152
        else:
            return False
        # TODO(jansel): should this default on for dynamic shapes?
        # TODO(laith) What if hint(features.reduction_numel) >= threshold ?
        # shall we compare hints instead
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

        if features.get_reduction_hint() not in (
            ReductionHint.INNER,
            ReductionHint.OUTER_TINY,
        ):
            bounds = bound_sympy(features.reduction_numel)
            lower = bounds.lower
            upper = bounds.upper

            if not all(
                (
                    (isinstance(bound, int) or bound.is_constant())
                    and bound != torch.utils._sympy.numbers.IntInfinity()
                )
                for bound in (lower, upper)
            ):
                return False

            lower = next_power_of_2(int(lower))
            upper = next_power_of_2(int(upper))

            # If we are are coalescing on xblock (not ReductionHint.INNER) and this is not a tiny kernel
            # (not ReductionHint.OUTER_TINY), do not use persistent reduction if it induces tile
            # quantization. Persistent reduction forces rblock == rnumel, if the bounds between lower
            # and upper are large, for the lower values we will be masking off large % of read/writes,
            # when we could expand the coalescing xblock instead.
            if lower != upper:
                return False

        if cooperative_reduction:
            # The RSPLIT of cooperative reductions means each thread block is operating on fewer elements
            try:
                threshold *= 32 // min(
                    V.graph.sizevars.size_hint_or_throw(features.numel), 32
                )
            except ValueError:
                pass  # unbacked symint

        # If multi_kernel is enabled, we do more aggressive persistent reduction.
        # This may result in some persistent reductions slower than the
        # corresponding non-persistent reductions. MultiKernel will do benchmarking
        # to pick the faster one.
        if config.triton.multi_kernel:
            threshold *= 16

        return V.graph.sizevars.statically_known_leq(
            features.reduction_numel, threshold
        )  # type: ignore[arg-types]

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
        warp_size = props.warp_size if props.warp_size is not None else 32
        max_threads_per_sm = (
            props.max_threads_per_multi_processor
            if props.max_threads_per_multi_processor is not None
            else 2048
        )
        min_elements_per_thread = warp_size
        max_elements_per_thread = 512
        threads_per_sm = max_threads_per_sm
        min_elements_per_device = min_elements_per_thread * num_sm * threads_per_sm
        max_elements_per_device = max_elements_per_thread * num_sm * threads_per_sm
        num_warps = 8
        num_threads = warp_size * num_warps

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
                common_buf_names: OrderedSet[str] = (
                    node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
                )
                if len(common_buf_names) > 0:
                    get_metric_table("fusion_failure_due_to_indexing_mismatch").add_row(
                        # pyrefly: ignore [bad-argument-type]
                        lambda: {
                            "pre_grad_graph_id": V.graph.graph_id,
                            "post_grad_graph_id": V.graph.post_grad_graph_id,
                            "node1_name": node1.get_name(),
                            "node2_name": node2.get_name(),
                            "node1_debug_str": write_text(node1.debug_str()),
                            "node2_debug_str": write_text(node2.debug_str()),
                            "common_buffer_names": list(common_buf_names),  # type: ignore[dict-item]
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

        if (
            config.max_fusion_unique_io_buffers is not None
            and scheduler.fusion_prevent_too_many_reads_and_writes(
                node1,
                node2,
                config.max_fusion_unique_io_buffers,
            )
        ):
            WhyNoFuse(node1, node2)("fusion_prevent_too_many_reads_and_writes")
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
        if MixOrderReduction.can_fuse(node1, node2):
            # For mix order reduction, we disregard shared data or
            # distance.
            return True
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

        memory_score, is_mix_order_reduction = typing.cast(
            tuple[int, bool],
            scheduler.score_fusion_memory(
                node1, node2, return_is_mix_order_reduction=True
            ),
        )
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

        type_score = node1.is_reduction() == node2.is_reduction() and memory_score > 0

        return FusionScore(
            template_score,
            type_score,
            memory_score,
            proximity_score,
        )
