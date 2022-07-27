
from . import default_hooks as default

LOW_PRECISION_HOOKS = [
    default.fp16_compress_hook,
    default.bf16_compress_hook,
]
