"""Semantic reduction capabilities shared by NVGEMM scheduling and providers."""

import dataclasses

from torch._inductor.kernel.gemm_epilogue import (
    GEMM_REDUCTION_FRAGMENT_WIDTH,
    GemmReductionArguments,
    GemmReductionPlan,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class NVGemmReductionCapabilities:
    reduction_programs: frozenset[tuple[str, str]]
    source_types: frozenset[str]
    supports_secondary: bool = False
    max_n_axis_consumer_group: int | None = None

    def supports(
        self,
        reduction_type: str,
        source_type: str,
        reduction_algorithm: str = "default",
    ) -> bool:
        return (
            reduction_type,
            reduction_algorithm,
        ) in self.reduction_programs and source_type in self.source_types

    def supports_contract(
        self, contract: GemmReductionPlan | GemmReductionArguments
    ) -> bool:
        if contract.reduction_type is None:
            return False
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
            contract.reduction_algorithm,
        ):
            return False
        if contract.secondary_feed_output is None:
            return True
        return self.supports_secondary and contract.secondary_consumer_fn is not None


DENSE_GEMM_REDUCTION_CAPABILITIES = NVGemmReductionCapabilities(
    reduction_programs=frozenset(
        (
            ("sum", "default"),
            ("mean", "default"),
            ("prod", "default"),
            ("max", "default"),
            ("min", "default"),
            ("max", "logsumexp"),
            ("sum", "online_softmax"),
            ("sum", "variance"),
        )
    ),
    source_types=frozenset(("identity", "square", "abs", "abs_scale")),
    supports_secondary=True,
    max_n_axis_consumer_group=GEMM_REDUCTION_FRAGMENT_WIDTH,
)


BLOCK_SCALED_GEMM_REDUCTION_CAPABILITIES = NVGemmReductionCapabilities(
    reduction_programs=(
        DENSE_GEMM_REDUCTION_CAPABILITIES.reduction_programs
        - frozenset(
            (
                ("max", "logsumexp"),
                ("sum", "online_softmax"),
                ("sum", "variance"),
            )
        )
    ),
    source_types=DENSE_GEMM_REDUCTION_CAPABILITIES.source_types,
    max_n_axis_consumer_group=GEMM_REDUCTION_FRAGMENT_WIDTH,
)
