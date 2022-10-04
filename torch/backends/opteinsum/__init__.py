from typing import Any
import torch
import sys
from functools import lru_cache as _lru_cache
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

try:
    import opt_einsum as _opt_einsum  # type: ignore[import]
except ImportError:
    _opt_einsum = None

# Write:
#
#   torch.backends.opteinsum.enabled = False
#
# to globally disable usage of opt_einsum in einsum for path calculation.

@_lru_cache()
def is_available() -> bool:
    r"""Returns a bool indicating if opt_einsum is currently available."""
    return _opt_einsum is not None


def get_opt_einsum() -> Any:
    r"""Returns the opt_einsum package if opt_einsum is currently available, else None."""
    if is_available():
        return _opt_einsum
    return None

def set_flags(_enabled=None, _strategy=None):
    orig_flags = (torch._C._get_opt_einsum_enabled(),  # type: ignore[attr-defined]
                  None if not is_available() else torch._C._get_opt_einsum_strategy())  # type: ignore[attr-defined]
    if _enabled is not None:
        torch._C._set_opt_einsum_enabled(_enabled)  # type: ignore[attr-defined]
    if _strategy is not None:
        if not is_available():
            raise ValueError('opt_einsum is not available, so `strategy` cannot be set. Please install opt-einsum or '
                             'unset `strategy`.')
        if _strategy not in ['auto', 'greedy', 'optimal']:
            raise ValueError(f'`strategy` must be one of the following: [auto, greedy, optimal] but is {_strategy}')
        torch._C._set_opt_einsum_strategy(_strategy)  # type: ignore[attr-defined]

    return orig_flags


@contextmanager
def flags(enabled=False, strategy='auto'):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, strategy)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.opteinsum.enabled = True

class OptEinsumModule(PropModule):
    def __init__(self, m, name):
        super(OptEinsumModule, self).__init__(m, name)

    enabled = ContextProp(torch._C._get_opt_einsum_enabled, torch._C._set_opt_einsum_enabled)  # type: ignore[attr-defined]
    strategy = None
    if is_available():
        strategy = ContextProp(torch._C._get_opt_einsum_strategy, torch._C._set_opt_einsum_strategy)  # type: ignore[attr-defined]

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = OptEinsumModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
enabled: bool
strategy: str
