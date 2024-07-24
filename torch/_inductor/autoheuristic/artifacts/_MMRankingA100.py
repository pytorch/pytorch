# flake8: noqa: B950
# fmt: off
from typing import Any, List, Optional, Tuple

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
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=8')
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
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=1')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=1')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=8')
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
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=32_numstages=1_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=4')

    def get_name(self) -> str:
        return 'mm'

    def get_best_choices(self, context: AHContext) -> Optional[List[Tuple[float, int]]]:
        if str(context.get_value('using_tf32')) != 'False':
            if context.get_value('arith_intensity') <= 44.70475769042969:
                if context.get_value('k') <= 42.0:
                    if context.get_value('m*n') <= 54550528.0:
                        return [(0.124, 105), (0.118, 106), (0.110, 108), (0.093, 103), (0.077, 13), (0.070, 12), (0.054, 107), (0.054, 102), (0.039, 101), (0.039, 11), (0.038, 9), (0.038, 6), (0.030, 4), (0.023, 121), (0.016, 123), (0.016, 120), (0.016, 111), (0.015, 8), (0.008, 112), (0.008, 5), (0.008, 7), (0.008, 27)]
                    else:
                        if context.get_value('m*k') <= 193792.0:
                            return [(0.104, 4), (0.091, 102), (0.090, 103), (0.082, 5), (0.076, 101), (0.074, 9), (0.074, 7), (0.067, 105), (0.060, 8), (0.059, 6), (0.052, 108), (0.052, 106), (0.029, 107), (0.029, 13), (0.022, 12), (0.022, 11), (0.014, 104)]
                        else:
                            return [(0.075, 11), (0.070, 6), (0.069, 106), (0.064, 103), (0.064, 7), (0.063, 9), (0.063, 12), (0.062, 13), (0.062, 10), (0.058, 102), (0.058, 101), (0.058, 8), (0.058, 5), (0.057, 105), (0.057, 4), (0.044, 108), (0.019, 107)]
                else:
                    if context.get_value('m') <= 1008.0:
                        if context.get_value('m') <= 12.0:
                            return [(0.144, 69), (0.129, 52), (0.123, 53), (0.114, 68), (0.100, 54), (0.092, 49), (0.076, 0), (0.038, 56), (0.031, 51), (0.023, 63), (0.023, 71), (0.022, 67), (0.015, 65), (0.015, 66), (0.015, 58), (0.008, 70), (0.008, 73), (0.008, 72), (0.008, 57), (0.007, 64)]
                        else:
                            return [(0.129, 76), (0.073, 85), (0.073, 0), (0.064, 80), (0.064, 79), (0.064, 62), (0.053, 77), (0.052, 78), (0.043, 52), (0.032, 98), (0.031, 87), (0.031, 49), (0.031, 61), (0.031, 53), (0.021, 51), (0.021, 74), (0.021, 68), (0.021, 67), (0.021, 66), (0.020, 69), (0.020, 88), (0.011, 86), (0.010, 54), (0.010, 71), (0.010, 65), (0.010, 90), (0.010, 82), (0.010, 84), (0.010, 50)]
                    else:
                        if context.get_value('n') <= 24.0:
                            return [(0.126, 94), (0.110, 38), (0.101, 93), (0.096, 37), (0.083, 1), (0.082, 92), (0.048, 129), (0.043, 34), (0.039, 113), (0.038, 35), (0.034, 19), (0.028, 0), (0.028, 75), (0.024, 20), (0.024, 21), (0.019, 115), (0.014, 18), (0.014, 116), (0.014, 36), (0.010, 130), (0.010, 131), (0.010, 114), (0.005, 17)]
                        else:
                            return [(0.106, 43), (0.080, 40), (0.080, 132), (0.078, 133), (0.060, 42), (0.060, 39), (0.052, 2), (0.040, 134), (0.040, 135), (0.040, 41), (0.039, 89), (0.039, 96), (0.039, 22), (0.039, 25), (0.039, 23), (0.039, 24), (0.032, 97), (0.026, 0), (0.020, 117), (0.019, 83), (0.019, 118), (0.013, 95)]
            else:
                if context.get_value('m*n') <= 4554752.0:
                    if context.get_value('m*n') <= 966656.0:
                        return [(0.393, 0), (0.083, 91), (0.080, 100), (0.075, 3), (0.064, 98), (0.035, 27), (0.030, 97), (0.030, 112), (0.030, 126), (0.029, 46), (0.025, 29), (0.020, 45), (0.020, 48), (0.015, 44), (0.015, 99), (0.015, 136), (0.015, 124), (0.015, 28), (0.010, 47)]
                    else:
                        return [(0.285, 0), (0.173, 16), (0.094, 110), (0.087, 124), (0.078, 27), (0.075, 15), (0.057, 30), (0.040, 33), (0.030, 31), (0.017, 112), (0.013, 45), (0.009, 91), (0.009, 127), (0.009, 32), (0.007, 29), (0.004, 125), (0.004, 98), (0.004, 100), (0.002, 26), (0.002, 128), (0.002, 137)]
                else:
                    return [(0.313, 0), (0.222, 16), (0.217, 30), (0.213, 15), (0.014, 31), (0.012, 32), (0.003, 27), (0.003, 33), (0.003, 110), (0.000, 124), (0.000, 126)]
        else:
            if context.get_value('mat2_stride_0') <= 192.0:
                return [(0.913, 0), (0.046, 15), (0.009, 17), (0.006, 93), (0.003, 60), (0.003, 83), (0.003, 94), (0.003, 120), (0.003, 122), (0.003, 92), (0.003, 59), (0.003, 76)]
            else:
                if context.get_value('arith_intensity') <= 108.96703338623047:
                    return [(0.125, 0), (0.103, 105), (0.077, 119), (0.058, 61), (0.057, 62), (0.057, 55), (0.047, 53), (0.039, 137), (0.038, 52), (0.029, 108), (0.029, 106), (0.029, 85), (0.029, 81), (0.029, 136), (0.029, 138), (0.028, 45), (0.028, 46), (0.028, 107), (0.019, 54), (0.019, 120), (0.019, 111), (0.019, 110), (0.019, 122), (0.019, 112), (0.009, 109), (0.009, 102), (0.009, 101)]
                else:
                    if context.get_value('mat1_stride_1') <= 896.0:
                        if context.get_value('m*n') <= 3686400.0:
                            return [(0.477, 0), (0.069, 112), (0.057, 119), (0.057, 27), (0.056, 110), (0.047, 31), (0.046, 29), (0.045, 137), (0.045, 76), (0.034, 45), (0.034, 105), (0.033, 61)]
                        else:
                            return [(0.134, 119), (0.129, 0), (0.125, 27), (0.120, 29), (0.118, 110), (0.114, 105), (0.113, 112), (0.076, 31), (0.039, 137), (0.030, 45)]
                    else:
                        if context.get_value('m*n') <= 17399808.0:
                            return [(0.195, 0), (0.118, 119), (0.106, 27), (0.106, 105), (0.093, 29), (0.090, 110), (0.090, 112), (0.085, 137), (0.069, 31), (0.041, 45), (0.004, 14), (0.004, 15)]
                        else:
                            return [(0.119, 27), (0.111, 0), (0.111, 105), (0.106, 119), (0.103, 110), (0.099, 15), (0.098, 31), (0.089, 29), (0.076, 112), (0.060, 137), (0.027, 45)]
