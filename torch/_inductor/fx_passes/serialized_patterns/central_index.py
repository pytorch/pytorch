# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python
# torchgen/fuse_attention_patterns/gen_attention_patterns.py
from ._sfdp_pattern_1 import (_sfdp_pattern_1_training, _sfdp_pattern_1_inference)
from ._sfdp_pattern_2 import (_sfdp_pattern_2_training, _sfdp_pattern_2_inference)
from ._sfdp_pattern_3 import (_sfdp_pattern_3_training, _sfdp_pattern_3_inference)
from ._sfdp_pattern_4 import (_sfdp_pattern_4_training, _sfdp_pattern_4_inference)
from ._sfdp_pattern_5 import (_sfdp_pattern_5_training, _sfdp_pattern_5_inference)
from ._sfdp_pattern_6 import (_sfdp_pattern_6_training, _sfdp_pattern_6_inference)
from ._sfdp_pattern_7 import (_sfdp_pattern_7_training, _sfdp_pattern_7_inference)
from ._sfdp_pattern_8 import (_sfdp_pattern_8_training, _sfdp_pattern_8_inference)
from ._sfdp_pattern_9 import (_sfdp_pattern_9_training, _sfdp_pattern_9_inference)

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
