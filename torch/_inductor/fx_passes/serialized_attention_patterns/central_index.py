# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python
# torchgen/fuse_attention_patterns/gen_attention_patterns.py
from ._sfdp_pattern_1 import (_sfdp_pattern_1_training, _sfdp_pattern_1_inference)

central_index = {
    '_sfdp_pattern_1_training': _sfdp_pattern_1_training,
    '_sfdp_pattern_1_inference': _sfdp_pattern_1_inference,
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
