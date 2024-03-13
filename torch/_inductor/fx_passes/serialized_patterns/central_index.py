# mypy: ignore-errors

# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python
# torchgen/fuse_attention_patterns/gen_attention_patterns.py
from ._sfdp_pattern_1 import (_sfdp_pattern_1_training, _sfdp_pattern_1_inference, _sfdp_pattern_1_half_training, _sfdp_pattern_1_half_inference)
from ._sfdp_pattern_2 import (_sfdp_pattern_2_training, _sfdp_pattern_2_inference, _sfdp_pattern_2_half_training, _sfdp_pattern_2_half_inference)
from ._sfdp_pattern_3 import (_sfdp_pattern_3_training, _sfdp_pattern_3_inference, _sfdp_pattern_3_half_training, _sfdp_pattern_3_half_inference)
from ._sfdp_pattern_4 import (_sfdp_pattern_4_training, _sfdp_pattern_4_inference, _sfdp_pattern_4_half_training, _sfdp_pattern_4_half_inference)
from ._sfdp_pattern_5 import (_sfdp_pattern_5_training, _sfdp_pattern_5_inference, _sfdp_pattern_5_half_training, _sfdp_pattern_5_half_inference)
from ._sfdp_pattern_6 import (_sfdp_pattern_6_training, _sfdp_pattern_6_inference, _sfdp_pattern_6_half_training, _sfdp_pattern_6_half_inference)
from ._sfdp_pattern_7 import (_sfdp_pattern_7_training, _sfdp_pattern_7_inference, _sfdp_pattern_7_half_training, _sfdp_pattern_7_half_inference)
from ._sfdp_pattern_8 import (_sfdp_pattern_8_training, _sfdp_pattern_8_inference, _sfdp_pattern_8_half_training, _sfdp_pattern_8_half_inference)
from ._sfdp_pattern_9 import (_sfdp_pattern_9_training, _sfdp_pattern_9_inference, _sfdp_pattern_9_half_training, _sfdp_pattern_9_half_inference)
from ._sfdp_pattern_10 import (_sfdp_pattern_10_training, _sfdp_pattern_10_inference, _sfdp_pattern_10_half_training, _sfdp_pattern_10_half_inference)
from ._sfdp_pattern_11 import (_sfdp_pattern_11_training, _sfdp_pattern_11_inference, _sfdp_pattern_11_half_training, _sfdp_pattern_11_half_inference)
from ._sfdp_pattern_12 import (_sfdp_pattern_12_training, _sfdp_pattern_12_inference, _sfdp_pattern_12_half_training, _sfdp_pattern_12_half_inference)
from ._sfdp_pattern_13 import (_sfdp_pattern_13_training, _sfdp_pattern_13_inference, _sfdp_pattern_13_half_training, _sfdp_pattern_13_half_inference)
from ._sfdp_pattern_14 import (_sfdp_pattern_14_training, _sfdp_pattern_14_inference, _sfdp_pattern_14_half_training, _sfdp_pattern_14_half_inference)
from ._sfdp_pattern_15 import (_sfdp_pattern_15_training, _sfdp_pattern_15_inference, _sfdp_pattern_15_half_training, _sfdp_pattern_15_half_inference)
from ._sfdp_pattern_16 import (_sfdp_pattern_16_training, _sfdp_pattern_16_inference, _sfdp_pattern_16_bs1_training, _sfdp_pattern_16_bs1_inference, _sfdp_pattern_16_half_training, _sfdp_pattern_16_half_inference, _sfdp_pattern_16_half_bs1_training, _sfdp_pattern_16_half_bs1_inference, _sfdp_pattern_16_half_mask_fp32_training, _sfdp_pattern_16_half_mask_fp32_inference, _sfdp_pattern_16_half_mask_fp32_bs1_training, _sfdp_pattern_16_half_mask_fp32_bs1_inference)
from ._sfdp_pattern_17 import (_sfdp_pattern_17_training, _sfdp_pattern_17_inference, _sfdp_pattern_17_half_training, _sfdp_pattern_17_half_inference)

central_index = {
    '_sfdp_pattern_1_training': _sfdp_pattern_1_training,
    '_sfdp_pattern_1_inference': _sfdp_pattern_1_inference,
    '_sfdp_pattern_2_training': _sfdp_pattern_2_training,
    '_sfdp_pattern_2_inference': _sfdp_pattern_2_inference,
    '_sfdp_pattern_3_training': _sfdp_pattern_3_training,
    '_sfdp_pattern_3_inference': _sfdp_pattern_3_inference,
    '_sfdp_pattern_4_training': _sfdp_pattern_4_training,
    '_sfdp_pattern_4_inference': _sfdp_pattern_4_inference,
    '_sfdp_pattern_5_training': _sfdp_pattern_5_training,
    '_sfdp_pattern_5_inference': _sfdp_pattern_5_inference,
    '_sfdp_pattern_6_training': _sfdp_pattern_6_training,
    '_sfdp_pattern_6_inference': _sfdp_pattern_6_inference,
    '_sfdp_pattern_7_training': _sfdp_pattern_7_training,
    '_sfdp_pattern_7_inference': _sfdp_pattern_7_inference,
    '_sfdp_pattern_8_training': _sfdp_pattern_8_training,
    '_sfdp_pattern_8_inference': _sfdp_pattern_8_inference,
    '_sfdp_pattern_9_training': _sfdp_pattern_9_training,
    '_sfdp_pattern_9_inference': _sfdp_pattern_9_inference,
    '_sfdp_pattern_10_training': _sfdp_pattern_10_training,
    '_sfdp_pattern_10_inference': _sfdp_pattern_10_inference,
    '_sfdp_pattern_11_training': _sfdp_pattern_11_training,
    '_sfdp_pattern_11_inference': _sfdp_pattern_11_inference,
    '_sfdp_pattern_12_training': _sfdp_pattern_12_training,
    '_sfdp_pattern_12_inference': _sfdp_pattern_12_inference,
    '_sfdp_pattern_13_training': _sfdp_pattern_13_training,
    '_sfdp_pattern_13_inference': _sfdp_pattern_13_inference,
    '_sfdp_pattern_14_training': _sfdp_pattern_14_training,
    '_sfdp_pattern_14_inference': _sfdp_pattern_14_inference,
    '_sfdp_pattern_15_training': _sfdp_pattern_15_training,
    '_sfdp_pattern_15_inference': _sfdp_pattern_15_inference,
    '_sfdp_pattern_16_training': _sfdp_pattern_16_training,
    '_sfdp_pattern_16_inference': _sfdp_pattern_16_inference,
    '_sfdp_pattern_16_bs1_training': _sfdp_pattern_16_bs1_training,
    '_sfdp_pattern_16_bs1_inference': _sfdp_pattern_16_bs1_inference,
    '_sfdp_pattern_17_training': _sfdp_pattern_17_training,
    '_sfdp_pattern_17_inference': _sfdp_pattern_17_inference,
    '_sfdp_pattern_1_half_training': _sfdp_pattern_1_half_training,
    '_sfdp_pattern_1_half_inference': _sfdp_pattern_1_half_inference,
    '_sfdp_pattern_2_half_training': _sfdp_pattern_2_half_training,
    '_sfdp_pattern_2_half_inference': _sfdp_pattern_2_half_inference,
    '_sfdp_pattern_3_half_training': _sfdp_pattern_3_half_training,
    '_sfdp_pattern_3_half_inference': _sfdp_pattern_3_half_inference,
    '_sfdp_pattern_4_half_training': _sfdp_pattern_4_half_training,
    '_sfdp_pattern_4_half_inference': _sfdp_pattern_4_half_inference,
    '_sfdp_pattern_5_half_training': _sfdp_pattern_5_half_training,
    '_sfdp_pattern_5_half_inference': _sfdp_pattern_5_half_inference,
    '_sfdp_pattern_6_half_training': _sfdp_pattern_6_half_training,
    '_sfdp_pattern_6_half_inference': _sfdp_pattern_6_half_inference,
    '_sfdp_pattern_7_half_training': _sfdp_pattern_7_half_training,
    '_sfdp_pattern_7_half_inference': _sfdp_pattern_7_half_inference,
    '_sfdp_pattern_8_half_training': _sfdp_pattern_8_half_training,
    '_sfdp_pattern_8_half_inference': _sfdp_pattern_8_half_inference,
    '_sfdp_pattern_9_half_training': _sfdp_pattern_9_half_training,
    '_sfdp_pattern_9_half_inference': _sfdp_pattern_9_half_inference,
    '_sfdp_pattern_10_half_training': _sfdp_pattern_10_half_training,
    '_sfdp_pattern_10_half_inference': _sfdp_pattern_10_half_inference,
    '_sfdp_pattern_11_half_training': _sfdp_pattern_11_half_training,
    '_sfdp_pattern_11_half_inference': _sfdp_pattern_11_half_inference,
    '_sfdp_pattern_12_half_training': _sfdp_pattern_12_half_training,
    '_sfdp_pattern_12_half_inference': _sfdp_pattern_12_half_inference,
    '_sfdp_pattern_13_half_training': _sfdp_pattern_13_half_training,
    '_sfdp_pattern_13_half_inference': _sfdp_pattern_13_half_inference,
    '_sfdp_pattern_14_half_training': _sfdp_pattern_14_half_training,
    '_sfdp_pattern_14_half_inference': _sfdp_pattern_14_half_inference,
    '_sfdp_pattern_15_half_training': _sfdp_pattern_15_half_training,
    '_sfdp_pattern_15_half_inference': _sfdp_pattern_15_half_inference,
    '_sfdp_pattern_16_half_training': _sfdp_pattern_16_half_training,
    '_sfdp_pattern_16_half_inference': _sfdp_pattern_16_half_inference,
    '_sfdp_pattern_16_half_bs1_training': _sfdp_pattern_16_half_bs1_training,
    '_sfdp_pattern_16_half_bs1_inference': _sfdp_pattern_16_half_bs1_inference,
    '_sfdp_pattern_17_half_training': _sfdp_pattern_17_half_training,
    '_sfdp_pattern_17_half_inference': _sfdp_pattern_17_half_inference,
    '_sfdp_pattern_16_half_mask_fp32_training': _sfdp_pattern_16_half_mask_fp32_training,
    '_sfdp_pattern_16_half_mask_fp32_inference': _sfdp_pattern_16_half_mask_fp32_inference,
    '_sfdp_pattern_16_half_mask_fp32_bs1_training': _sfdp_pattern_16_half_mask_fp32_bs1_training,
    '_sfdp_pattern_16_half_mask_fp32_bs1_inference': _sfdp_pattern_16_half_mask_fp32_bs1_inference,
}


def get_serialized_pattern(key):
    import torch._inductor  # noqa: F401
    from torch._inductor import config
    if config.fallback_random:
        return None

    # TODO - could add more validation that the same set of decomps used when
    # tracing SDPA are also used in current context. softmax, dropout, etc
    # decomp use is stable so not an issue in practice.
    return central_index.get(key)
