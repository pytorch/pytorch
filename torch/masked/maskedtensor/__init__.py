from .core import is_masked_tensor, MaskedTensor
from .creation import as_masked_tensor, masked_tensor

try:
    from .version import __version__  # type: ignore[import] # noqa: F401
except ImportError:
    pass
