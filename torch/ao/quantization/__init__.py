from .fake_quantize import (
    # FakeQuantize (for qat)
    default_fake_quant,
    default_weight_fake_quant,
    default_symmetric_fixed_qparams_fake_quant,
    default_affine_fixed_qparams_fake_quant,
    default_per_channel_weight_fake_quant,
    default_histogram_fake_quant,
)
from .quantize import (
    add_observer_,
    add_quant_dequant,
    convert,
    get_observer_dict,
    prepare,
    prepare_qat,
    propagate_qconfig_,
    quantize,
    quantize_dynamic,
    quantize_qat,
    register_activation_post_process_hook,
    swap_module,
)  # noqa: F401
