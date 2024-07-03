# flake8: noqa: B950

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import LearnedHeuristic


class PadMMA100(LearnedHeuristic):
    def __init__(self) -> None:
        pass

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

    def get_feedback(self, context: AHContext, choice: Choice) -> float:
        context.context_dict["choice"] = choice
        return self.predict(context)

    def get_speedup_threshold(self) -> float:
        return 1.7025303314066003

    def get_name(self) -> str:
        return "pad_mm"

    def predict(self, context: AHContext) -> float:
        if str(context.get_value("choice")) != "pad":
            if str(context.get_value("using_tf32")) != "False":
                if context.get_value("m*n") <= 4171264.0:
                    if context.get_value("m*k") <= 3999308.0:
                        return 1.875146976407118
                    else:
                        if str(context.get_value("n_multiple_32")) != "False":
                            return 1.1607689608873863
                        else:
                            return 0.9117231355626346
                else:
                    if str(context.get_value("n_multiple_2")) != "True":
                        if context.get_value("mat2_align_size") <= 6.0:
                            return 0.8531269794448672
                        else:
                            return 0.7430382200435998
                    else:
                        if str(context.get_value("k_multiple_2")) != "True":
                            return 0.7577181972719915
                        else:
                            return 0.8977349440424212
            else:
                if context.get_value("m*n") <= 1299712.0:
                    return 1.1669723418995592
                else:
                    if context.get_value("mat2_stride_1") <= 45217.5:
                        if context.get_value("m*n") <= 55884158.0:
                            return 1.0262769936909606
                        else:
                            return 1.0022677428470865
                    else:
                        if context.get_value("m") <= 18478.0:
                            return 1.1127066261894314
                        else:
                            return 1.0337740659894263
        else:
            if str(context.get_value("mat2_dtype")) != "torch.float32":
                if str(context.get_value("n_multiple_2")) != "True":
                    if context.get_value("k") <= 28238.5:
                        if context.get_value("k/(m*n)") <= 0.00026227018679492176:
                            return 1.6770542505397155
                        else:
                            return 1.3974785435105923
                    else:
                        if str(context.get_value("mat1_dtype")) != "torch.bfloat16":
                            return 1.3952699800111992
                        else:
                            return 1.5759286511628325
                else:
                    if str(context.get_value("k_multiple_2")) != "True":
                        if context.get_value("mat1_stride_0") <= 561.0:
                            return 1.2900382135142954
                        else:
                            return 1.576173761605788
                    else:
                        if context.get_value("num_dims_needs_padding") <= 1.5:
                            return 1.0472263310239425
                        else:
                            return 1.1727673465762498
            else:
                if str(context.get_value("using_tf32")) != "True":
                    if context.get_value("arith_intensity") <= 396.8774871826172:
                        return 0.8994016186955102
                    else:
                        if context.get_value("mat2_stride_1") <= 45217.5:
                            return 0.9964328169353532
                        else:
                            return 0.9493479238294825
                else:
                    if context.get_value("m*n") <= 14119424.0:
                        return 0.8875772670422479
                    else:
                        if (
                            str(context.get_value("mat2_innermost_needs_padding"))
                            != "False"
                        ):
                            return 1.2158429635329966
                        else:
                            return 1.1467728924377265
