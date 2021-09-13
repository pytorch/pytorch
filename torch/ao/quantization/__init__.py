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

# TODO: Only a few of these functions are used outside the `fuse_modules.py`
#       Keeping here for compatibility, need to remove them later.
from .fuse_modules import (
    _fuse_modules,
    _get_module,
    _set_module,
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_known_modules,
    fuse_modules,
    get_fuser_method,
)  # noqa: F401

from .quant_type import (
    QuantType,
    quant_type_to_str,
)
