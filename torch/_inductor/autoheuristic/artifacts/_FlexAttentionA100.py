# flake8: noqa: B950
from typing import List, Optional, Tuple

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import (
    LearnedHeuristicDecision,
)


class FlexAttentionA100(LearnedHeuristicDecision):
    def __init__(self) -> None:
        self.choices: List[Choice] = []
        self.fill_choices()

    def check_precondition(
        self,
        metadata: AHMetadata,
        context: AHContext,
    ) -> bool:
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
        self.choices.append(
            "type=triton_BLOCK-M=128_BLOCK-K=-1_BLOCK-N=128_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=128_BLOCK-K=-1_BLOCK-N=32_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=128_BLOCK-K=-1_BLOCK-N=64_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=128_BLOCK-K=-1_BLOCK-N=64_numstages=3_numwarps=8"
        )
        self.choices.append(
            "type=triton_BLOCK-M=32_BLOCK-K=-1_BLOCK-N=64_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=64_BLOCK-K=-1_BLOCK-N=128_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=64_BLOCK-K=-1_BLOCK-N=64_numstages=3_numwarps=4"
        )

    def get_name(self) -> str:
        return "flex_attention"

    def get_best_choices(self, context: AHContext) -> Optional[List[Tuple[float, int]]]:
        if context.get_value("n") <= 192.0:
            if context.get_value("b*h") <= 21.5:
                if context.get_value("b*m") <= 2304.0:
                    if context.get_value("q_stride_1") <= 48.0:
                        if context.get_value("h*m") <= 1536.0:
                            if context.get_value("h*m") <= 704.0:
                                return [
                                    (0.397, 6),
                                    (0.315, 4),
                                    (0.096, 5),
                                    (0.082, 3),
                                    (0.068, 1),
                                    (0.027, 0),
                                    (0.014, 2),
                                ]
                            else:
                                return [(0.518, 6), (0.286, 4), (0.143, 5), (0.054, 3)]
                        else:
                            if context.get_value("q_stride_0") <= 114688.0:
                                return None
                            else:
                                return [
                                    (0.308, 6),
                                    (0.231, 1),
                                    (0.212, 2),
                                    (0.115, 5),
                                    (0.096, 3),
                                    (0.038, 4),
                                ]
                    else:
                        if context.get_value("h*m") <= 2304.0:
                            if context.get_value("q_stride_1") <= 96.0:
                                return None
                            else:
                                return [(0.849, 4), (0.085, 6), (0.038, 1), (0.028, 5)]
                        else:
                            return [
                                (0.491, 1),
                                (0.302, 4),
                                (0.132, 6),
                                (0.038, 2),
                                (0.038, 3),
                            ]
                else:
                    if context.get_value("q_stride_0") <= 290816.0:
                        if context.get_value("b*n") <= 208.0:
                            if context.get_value("b*m") <= 6272.0:
                                return [
                                    (0.271, 6),
                                    (0.254, 1),
                                    (0.169, 5),
                                    (0.136, 3),
                                    (0.085, 0),
                                    (0.085, 2),
                                ]
                            else:
                                return [
                                    (0.327, 6),
                                    (0.255, 1),
                                    (0.236, 2),
                                    (0.109, 3),
                                    (0.036, 5),
                                    (0.018, 0),
                                    (0.018, 4),
                                ]
                        else:
                            if context.get_value("b*m") <= 10496.0:
                                return None
                            else:
                                return [
                                    (0.716, 1),
                                    (0.119, 6),
                                    (0.075, 2),
                                    (0.060, 3),
                                    (0.015, 0),
                                    (0.015, 4),
                                ]
                    else:
                        if str(context.get_value("dtype")) != "torch.float32":
                            if context.get_value("m*n") <= 344064.0:
                                return [
                                    (0.560, 1),
                                    (0.262, 6),
                                    (0.155, 2),
                                    (0.012, 0),
                                    (0.012, 3),
                                ]
                            else:
                                return [(0.768, 1), (0.232, 6)]
                        else:
                            if context.get_value("m") <= 8000.0:
                                return [(0.966, 1), (0.034, 4)]
                            else:
                                return [(0.902, 1), (0.059, 6), (0.039, 2)]
            else:
                if context.get_value("q_stride_0") <= 176128.0:
                    if context.get_value("b*h") <= 89.0:
                        if context.get_value("m*n") <= 12288.0:
                            if context.get_value("b*m") <= 3712.0:
                                return [
                                    (0.494, 6),
                                    (0.136, 2),
                                    (0.111, 1),
                                    (0.074, 0),
                                    (0.062, 3),
                                    (0.062, 4),
                                    (0.062, 5),
                                ]
                            else:
                                return None
                        else:
                            if str(context.get_value("dtype")) != "torch.float16":
                                return None
                            else:
                                return [
                                    (0.411, 1),
                                    (0.339, 6),
                                    (0.143, 2),
                                    (0.054, 0),
                                    (0.054, 3),
                                ]
                    else:
                        if context.get_value("m*n") <= 3072.0:
                            return None
                        else:
                            if context.get_value("q_stride_0") <= 22528.0:
                                return None
                            else:
                                return None
                else:
                    if context.get_value("b*n") <= 736.0:
                        if context.get_value("q_stride_0") <= 3506176.0:
                            if str(context.get_value("dtype")) != "torch.float32":
                                return None
                            else:
                                return [(0.972, 1), (0.011, 2), (0.011, 3), (0.006, 4)]
                        else:
                            return [(1.000, 1)]
                    else:
                        if context.get_value("b*h") <= 36.0:
                            return [(0.868, 1), (0.118, 6), (0.015, 2)]
                        else:
                            if context.get_value("subgraph_num_args_used") <= 2.5:
                                return [(1.000, 1)]
                            else:
                                return [(0.985, 1), (0.012, 6), (0.003, 2)]
        else:
            if str(context.get_value("dtype")) != "torch.float32":
                return [(1.000, 4)]
            else:
                return [(1.000, 0)]
