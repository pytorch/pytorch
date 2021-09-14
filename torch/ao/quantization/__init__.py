from .fake_quantize import *  # noqa: F403

# TODO(future PR): fix the typo, should be `__all__`
_all__ = [
    # FakeQuantize (for qat)
    'default_fake_quant', 'default_weight_fake_quant',
    'default_symmetric_fixed_qparams_fake_quant',
    'default_affine_fixed_qparams_fake_quant',
    'default_per_channel_weight_fake_quant',
    'default_histogram_fake_quant',
]
