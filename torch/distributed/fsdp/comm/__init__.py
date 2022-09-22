
from . import default_hooks

LOW_PRECISION_HOOKS = [
    default_hooks.fp16_compress_hook,
    default_hooks.bf16_compress_hook,
]
