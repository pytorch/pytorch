# flake8: noqa: B950
# fmt: off
from typing import List, Optional, Tuple

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import (
    LearnedHeuristicDecision,
)

class MMRankingA100(LearnedHeuristicDecision):

    def __init__(self) -> None:
        self.choices: List[Choice] = []
        self.fill_choices()

    def check_precondition(self, metadata: AHMetadata, context: AHContext,) -> bool:
        return (
            metadata.name == self.get_name()
            and metadata.shared_memory == 166912
            and str(metadata.device_capa) == "(8, 0)"
        )

    def get_confidence_threshold(self) -> float:
        return 0.0

    def get_choice(self, idx: int) -> Optional[str]:
        if idx < len(self.choices):
            return self.choices[idx]
        return None

    def fill_choices(self) -> None:
        self.choices.append('extern_mm')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=128_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=128_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=1')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=1_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=1')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=16_numstages=1_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=32_numstages=1_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=4')

    def get_name(self) -> str:
        return 'mm'

    def get_best_choices(self, context: AHContext) -> Optional[List[Tuple[float, int]]]:
        if context.get_value('arith_intensity') <= 52.6245059967041:
            if context.get_value('n') <= 34.0:
                if context.get_value('n') <= 18.0:
                    if context.get_value('k') <= 19.5:
                        return [(0.093, 12), (0.081, 16), (0.081, 150), (0.070, 10), (0.070, 17), (0.070, 151), (0.070, 153), (0.070, 152), (0.070, 14), (0.058, 11), (0.058, 15), (0.058, 13), (0.058, 124), (0.047, 123), (0.035, 125), (0.012, 93)]
                    else:
                        if context.get_value('k') <= 40.0:
                            return [(0.083, 43), (0.083, 47), (0.083, 45), (0.083, 41), (0.083, 130), (0.067, 46), (0.067, 44), (0.067, 42), (0.067, 171), (0.067, 173), (0.067, 170), (0.067, 131), (0.067, 172), (0.033, 104), (0.017, 123)]
                        else:
                            return [(0.105, 139), (0.098, 138), (0.094, 0), (0.079, 137), (0.076, 1), (0.072, 189), (0.065, 68), (0.054, 42), (0.047, 72), (0.043, 69), (0.043, 71), (0.033, 170), (0.029, 45), (0.025, 44), (0.025, 172), (0.018, 191), (0.018, 190), (0.014, 171), (0.014, 173), (0.011, 40), (0.011, 117), (0.011, 70), (0.011, 41), (0.004, 104)]
                else:
                    if context.get_value('k') <= 40.0:
                        if context.get_value('n') <= 27.5:
                            return [(0.064, 48), (0.064, 22), (0.064, 50), (0.063, 52), (0.043, 159), (0.043, 54), (0.043, 156), (0.043, 25), (0.043, 158), (0.043, 21), (0.043, 157), (0.043, 154), (0.043, 20), (0.043, 155), (0.043, 19), (0.043, 18), (0.043, 160), (0.043, 23), (0.042, 24), (0.042, 126), (0.021, 127), (0.021, 128), (0.021, 51)]
                        else:
                            return [(0.041, 48), (0.041, 159), (0.041, 155), (0.041, 52), (0.041, 157), (0.041, 177), (0.041, 174), (0.041, 50), (0.041, 54), (0.041, 23), (0.041, 19), (0.041, 25), (0.040, 176), (0.021, 18), (0.021, 22), (0.021, 175), (0.021, 98), (0.021, 21), (0.021, 179), (0.021, 160), (0.021, 49), (0.021, 53), (0.020, 127), (0.020, 51), (0.020, 55), (0.020, 126), (0.020, 154), (0.020, 20), (0.020, 24), (0.020, 96), (0.020, 97), (0.020, 156), (0.020, 158), (0.020, 180), (0.020, 128), (0.020, 178)]
                    else:
                        if context.get_value('k') <= 68.0:
                            return [(0.134, 74), (0.122, 75), (0.122, 77), (0.110, 73), (0.110, 194), (0.098, 76), (0.097, 192), (0.086, 195), (0.073, 193), (0.024, 135), (0.012, 0), (0.012, 49)]
                        else:
                            return [(0.153, 73), (0.136, 53), (0.112, 76), (0.093, 74), (0.077, 50), (0.076, 52), (0.051, 0), (0.051, 178), (0.045, 2), (0.042, 192), (0.034, 193), (0.034, 180), (0.026, 175), (0.025, 77), (0.022, 140), (0.011, 141), (0.011, 142)]
            else:
                if context.get_value('k') <= 35.0:
                    if context.get_value('k') <= 18.0:
                        if context.get_value('m*n') <= 19505152.0:
                            return [(0.155, 162), (0.145, 166), (0.144, 161), (0.053, 129), (0.049, 29), (0.042, 163), (0.042, 149), (0.039, 148), (0.038, 31), (0.035, 147), (0.025, 28), (0.021, 91), (0.021, 94), (0.021, 95), (0.021, 101), (0.021, 127), (0.021, 160), (0.021, 159), (0.011, 88), (0.011, 89), (0.011, 90), (0.011, 92), (0.011, 96), (0.011, 97), (0.011, 99), (0.011, 100)]
                        else:
                            return [(0.069, 7), (0.069, 5), (0.067, 149), (0.066, 8), (0.061, 147), (0.058, 148), (0.052, 126), (0.049, 29), (0.049, 161), (0.046, 31), (0.043, 159), (0.041, 9), (0.041, 4), (0.040, 6), (0.035, 166), (0.035, 162), (0.026, 160), (0.017, 127), (0.017, 28), (0.017, 32), (0.017, 164), (0.017, 27), (0.017, 30), (0.017, 163), (0.009, 33), (0.009, 26), (0.009, 165), (0.006, 0)]
                    else:
                        if context.get_value('n') <= 68.0:
                            return [(0.101, 184), (0.101, 60), (0.088, 58), (0.076, 186), (0.076, 62), (0.076, 181), (0.076, 63), (0.076, 59), (0.063, 182), (0.063, 61), (0.051, 57), (0.050, 183), (0.025, 132), (0.025, 179), (0.025, 185), (0.013, 180), (0.013, 56)]
                        else:
                            return [(0.089, 182), (0.079, 61), (0.066, 35), (0.066, 183), (0.066, 38), (0.066, 59), (0.066, 181), (0.066, 58), (0.062, 186), (0.053, 37), (0.044, 168), (0.040, 56), (0.040, 39), (0.040, 36), (0.040, 167), (0.040, 169), (0.027, 179), (0.027, 34), (0.022, 161)]
                else:
                    if context.get_value('m*n') <= 309760.0:
                        if context.get_value('arith_intensity') <= 27.09415912628174:
                            return [(0.328, 0), (0.127, 87), (0.121, 84), (0.121, 85), (0.085, 86), (0.054, 111), (0.054, 83), (0.030, 110), (0.012, 107), (0.012, 108), (0.012, 113), (0.012, 109), (0.006, 102), (0.006, 103), (0.006, 112), (0.006, 114), (0.006, 115)]
                        else:
                            return [(0.178, 142), (0.148, 0), (0.111, 118), (0.067, 180), (0.067, 119), (0.067, 106), (0.067, 132), (0.059, 122), (0.052, 121), (0.044, 143), (0.037, 120), (0.022, 96), (0.022, 105), (0.022, 146), (0.015, 116), (0.015, 134), (0.007, 133)]
                    else:
                        if context.get_value('n') <= 72.0:
                            return [(0.227, 78), (0.118, 79), (0.102, 196), (0.086, 81), (0.059, 58), (0.054, 82), (0.049, 198), (0.048, 199), (0.048, 60), (0.043, 80), (0.032, 197), (0.027, 182), (0.022, 3), (0.021, 143), (0.016, 61), (0.016, 144), (0.011, 185), (0.011, 0), (0.011, 146)]
                        else:
                            return [(0.140, 188), (0.132, 187), (0.109, 64), (0.085, 66), (0.078, 37), (0.077, 35), (0.062, 199), (0.047, 196), (0.046, 167), (0.046, 58), (0.039, 79), (0.039, 80), (0.039, 67), (0.039, 65), (0.016, 197), (0.008, 161)]
        else:
            if str(context.get_value('using_tf32')) != 'False':
                if context.get_value('m*n') <= 815360.0:
                    if context.get_value('k') <= 1012.0:
                        return [(0.229, 142), (0.217, 0), (0.132, 146), (0.084, 143), (0.048, 187), (0.048, 106), (0.036, 79), (0.036, 169), (0.036, 118), (0.024, 167), (0.024, 132), (0.024, 58), (0.024, 180), (0.012, 197), (0.012, 188), (0.012, 61)]
                    else:
                        return [(0.872, 0), (0.037, 146), (0.030, 3), (0.026, 136), (0.016, 143), (0.005, 79), (0.005, 78), (0.002, 58), (0.002, 196), (0.002, 60), (0.002, 61), (0.002, 145)]
                else:
                    if context.get_value('arith_intensity') <= 187.23922729492188:
                        if context.get_value('k') <= 198.0:
                            return [(0.263, 64), (0.166, 35), (0.150, 37), (0.138, 58), (0.113, 167), (0.055, 187), (0.028, 65), (0.014, 0), (0.014, 61), (0.014, 79), (0.009, 56), (0.005, 34), (0.005, 136), (0.005, 169), (0.005, 181), (0.005, 67), (0.005, 188), (0.005, 196), (0.003, 66)]
                        else:
                            return [(0.283, 64), (0.234, 0), (0.123, 65), (0.087, 37), (0.065, 79), (0.057, 187), (0.048, 35), (0.028, 58), (0.019, 78), (0.015, 196), (0.013, 67), (0.009, 167), (0.007, 66), (0.003, 3), (0.003, 136), (0.003, 143), (0.002, 168)]
                    else:
                        if context.get_value('m*n') <= 3096960.0:
                            return [(0.563, 0), (0.167, 37), (0.056, 187), (0.047, 58), (0.043, 67), (0.034, 167), (0.027, 64), (0.022, 66), (0.016, 65), (0.013, 35), (0.005, 56), (0.005, 79)]
                        else:
                            return [(0.386, 0), (0.257, 37), (0.186, 64), (0.160, 35), (0.004, 65), (0.003, 66), (0.002, 187), (0.001, 58), (0.001, 167), (0.001, 79)]
            else:
                return [(0.386, 0), (0.100, 167), (0.097, 181), (0.091, 58), (0.078, 65), (0.069, 169), (0.065, 161), (0.063, 61), (0.029, 35), (0.008, 197), (0.002, 105), (0.002, 182), (0.002, 106), (0.002, 126), (0.002, 196), (0.002, 199), (0.001, 34), (0.001, 168), (0.001, 79)]
