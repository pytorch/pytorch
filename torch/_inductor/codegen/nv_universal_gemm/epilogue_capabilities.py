"""Semantic reduction capabilities shared by NVGEMM scheduling and providers."""

import dataclasses

from torch._inductor.kernel.gemm_epilogue import (
    GEMM_REDUCTION_FRAGMENT_WIDTH,
    GemmReductionArguments,
    GemmReductionDescriptor,
    GemmReductionPlan,
)


def _reduction_descriptor(
    value: str | None,
) -> GemmReductionDescriptor | None:
    try:
        return GemmReductionDescriptor.parse(value or "")
    except ValueError:
        return None


@dataclasses.dataclass(frozen=True, kw_only=True)
class NVGemmReductionCapabilities:
    reduction_kinds: frozenset[str]
    feed_main_only_kinds: frozenset[str]
    source_types: frozenset[str]
    secondary_kinds: frozenset[str] | None
    m_axis_feed_main_secondary_kinds: frozenset[str] = frozenset()
    max_n_axis_consumer_group: int | None = None

    def supports(
        self,
        reduction: str,
        source_type: str,
        *,
        feeds_main: bool = False,
    ) -> bool:
        descriptor = _reduction_descriptor(reduction)
        if descriptor is None:
            return False
        extra_kinds = self.feed_main_only_kinds if feeds_main else frozenset()
        kinds = self.reduction_kinds | extra_kinds
        return (
            descriptor.kind in kinds
            and descriptor.has_valid_parameters
            and source_type in self.source_types
        )

    def supports_contract(
        self, contract: GemmReductionPlan | GemmReductionArguments
    ) -> bool:
        if contract.group <= 1 or contract.group & (contract.group - 1):
            return False
        if (
            self.max_n_axis_consumer_group is not None
            and contract.axis == 1
            and contract.group > self.max_n_axis_consumer_group
            and (
                contract.feeds_main
                or contract.feed_output is not None
                or contract.secondary_feed_output is not None
                or contract.consumer_fn is not None
                or contract.secondary_consumer_fn is not None
            )
        ):
            return False
        if not self.supports(
            contract.reduction_type,
            contract.source_type,
            feeds_main=contract.feeds_main,
        ):
            return False
        if contract.secondary_feed_output is None:
            return True
        if self.secondary_kinds is None:
            return False
        if contract.secondary_consumer_fn is not None:
            return True
        descriptor = _reduction_descriptor(contract.secondary_feed_type)
        if descriptor is None:
            return False
        kinds = self.secondary_kinds
        if contract.feeds_main and contract.axis == 0:
            kinds |= self.m_axis_feed_main_secondary_kinds
        return descriptor.kind in kinds and descriptor.has_valid_parameters


DENSE_GEMM_REDUCTION_CAPABILITIES = NVGemmReductionCapabilities(
    reduction_kinds=frozenset(
        (
            "sum",
            "mean",
            "prod",
            "max",
            "min",
            "logsumexp",
            "direct_bool_gt_zero",
            "variance_affine",
        )
    ),
    feed_main_only_kinds=frozenset(
        ("mean_linear", "normalize_sum_affine", "normalize_sum_reverse_affine")
    ),
    source_types=frozenset(("identity", "square", "abs", "abs_scale")),
    secondary_kinds=frozenset(("direct_bool_gt_zero",)),
    m_axis_feed_main_secondary_kinds=frozenset(
        ("normalize_sum_affine", "normalize_sum_reverse_affine", "sum_mul_affine")
    ),
    max_n_axis_consumer_group=GEMM_REDUCTION_FRAGMENT_WIDTH,
)


BLOCK_SCALED_GEMM_REDUCTION_CAPABILITIES = NVGemmReductionCapabilities(
    reduction_kinds=(
        DENSE_GEMM_REDUCTION_CAPABILITIES.reduction_kinds
        - frozenset(("direct_bool_gt_zero", "logsumexp", "variance_affine"))
    ),
    feed_main_only_kinds=DENSE_GEMM_REDUCTION_CAPABILITIES.feed_main_only_kinds,
    source_types=DENSE_GEMM_REDUCTION_CAPABILITIES.source_types,
    secondary_kinds=None,
)
