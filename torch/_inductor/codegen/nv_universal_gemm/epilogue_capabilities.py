"""Semantic reduction capabilities shared by NVGEMM scheduling and providers."""

import dataclasses

from torch._inductor.kernel.gemm_epilogue import (
    GEMM_REDUCTION_FRAGMENT_WIDTH,
    GemmReductionArguments,
    GemmReductionPlan,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class NVGemmReductionCapabilities:
    reduction_types: frozenset[str]
    supports_secondary: bool = False
    max_n_axis_consumer_group: int | None = None

    def supports(
        self,
        reduction_type: str,
    ) -> bool:
        return reduction_type in self.reduction_types

    def supports_contract(
        self, contract: GemmReductionPlan | GemmReductionArguments
    ) -> bool:
        if contract.group <= 1 or contract.group & (contract.group - 1):
            return False
        if contract.tensor_epilogue_returns_local_reduce:
            return (
                contract.reduction_type is None
                and contract.source_fn is None
                and contract.geometry.needs_physical_callbacks
                == (contract.combine_fn is not None)
                and not contract.feeds_main
                and contract.feed_output is None
                and contract.secondary_feed_output is None
            )
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
        if contract.reduction_type is None or contract.source_fn is None:
            return False
        if not self.supports(contract.reduction_type):
            return False
        if contract.secondary_feed_output is None:
            return True
        return self.supports_secondary and contract.secondary_consumer_fn is not None


DENSE_GEMM_REDUCTION_CAPABILITIES = NVGemmReductionCapabilities(
    reduction_types=frozenset(("sum", "mean", "prod", "max", "min")),
    supports_secondary=True,
    max_n_axis_consumer_group=GEMM_REDUCTION_FRAGMENT_WIDTH,
)


BLOCK_SCALED_GEMM_REDUCTION_CAPABILITIES = NVGemmReductionCapabilities(
    reduction_types=DENSE_GEMM_REDUCTION_CAPABILITIES.reduction_types,
    max_n_axis_consumer_group=GEMM_REDUCTION_FRAGMENT_WIDTH,
)
