# flake8: noqa: B950
from typing import Any, Tuple

from torch._inductor.autoheuristic.autoheuristic_utils import Choice, ContextDictT
from torch._inductor.autoheuristic.learnedheuristic_interface import LearnedHeuristic


class PadMMA100(LearnedHeuristic):
    def __init__(self) -> None:
        pass

    def check_precondition(
        self,
        name: str,
        context_dict: ContextDictT,
        shared_memory: Any,
        device_capa: Tuple[int, int],
    ) -> bool:
        if (
            context_dict["m"] < 512
            or context_dict["k"] < 512
            or context_dict["n"] < 512
        ):
            return False
        return (
            name == "pad_mm"
            and shared_memory == 166912
            and str(device_capa) == "(8, 0)"
        )

    def get_feedback(self, context_dict: ContextDictT, choice: Choice) -> float:
        return self.predict(context_dict)

    def get_speedup_threshold(self) -> float:
        return 1.7025303314066003

    def get_name(self) -> str:
        return "pad_mm"

    def predict(self, context_dict: ContextDictT) -> float:
        if str(context_dict["choice"]) != "pad":
            if str(context_dict["using_tf32"]) != "False":
                if context_dict["m*n"] <= 4171264.0:
                    if context_dict["m*k"] <= 3999308.0:
                        return 1.875146976407118
                    else:
                        if str(context_dict["n_multiple_32"]) != "False":
                            return 1.1607689608873863
                        else:
                            return 0.9117231355626346
                else:
                    if str(context_dict["n_multiple_2"]) != "True":
                        if context_dict["mat2_align_size"] <= 6.0:
                            return 0.8531269794448672
                        else:
                            return 0.7430382200435998
                    else:
                        if str(context_dict["k_multiple_2"]) != "True":
                            return 0.7577181972719915
                        else:
                            return 0.8977349440424212
            else:
                if context_dict["m*n"] <= 1299712.0:
                    return 1.1669723418995592
                else:
                    if context_dict["mat2_stride_1"] <= 45217.5:
                        if context_dict["m*n"] <= 55884158.0:
                            return 1.0262769936909606
                        else:
                            return 1.0022677428470865
                    else:
                        if context_dict["m"] <= 18478.0:
                            return 1.1127066261894314
                        else:
                            return 1.0337740659894263
        else:
            if str(context_dict["mat2_dtype"]) != "torch.float32":
                if str(context_dict["n_multiple_2"]) != "True":
                    if context_dict["k"] <= 28238.5:
                        if context_dict["k/(m*n)"] <= 0.00026227018679492176:
                            return 1.6770542505397155
                        else:
                            return 1.3974785435105923
                    else:
                        if str(context_dict["mat1_dtype"]) != "torch.bfloat16":
                            return 1.3952699800111992
                        else:
                            return 1.5759286511628325
                else:
                    if str(context_dict["k_multiple_2"]) != "True":
                        if context_dict["mat1_stride_0"] <= 561.0:
                            return 1.2900382135142954
                        else:
                            return 1.576173761605788
                    else:
                        if context_dict["num_dims_needs_padding"] <= 1.5:
                            return 1.0472263310239425
                        else:
                            return 1.1727673465762498
            else:
                if str(context_dict["using_tf32"]) != "True":
                    if context_dict["arith_intensity"] <= 396.8774871826172:
                        return 0.8994016186955102
                    else:
                        if context_dict["mat2_stride_1"] <= 45217.5:
                            return 0.9964328169353532
                        else:
                            return 0.9493479238294825
                else:
                    if context_dict["m*n"] <= 14119424.0:
                        return 0.8875772670422479
                    else:
                        if str(context_dict["mat2_innermost_needs_padding"]) != "False":
                            return 1.2158429635329966
                        else:
                            return 1.1467728924377265
