import warnings
from typing_extensions import deprecated

from torch._C import _return_types as return_types


__all__ = ["pytree_register_structseq", "all_return_types"]


all_return_types = []


@deprecated(
    "torch.return_types.pytree_register_structseq is now a no-op "
    "and will be removed in a future release.",
    category=FutureWarning,
)
def pytree_register_structseq(cls):
    from torch.utils._pytree import is_structseq_class

    if is_structseq_class(cls):
        return

    raise TypeError(f"Class {cls!r} is not a PyStructSequence class.")


_name, _attr = "", None
for _name in dir(return_types):
    if _name.startswith("__"):
        continue

    _attr = getattr(return_types, _name)
    globals()[_name] = _attr

    if not _name.startswith("_"):
        __all__.append(_name)
        all_return_types.append(_attr)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=__name__,
        append=False,
    )
    # Today everything in torch.return_types is a structseq, aka a "namedtuple"-like
    # thing defined by the Python C-API. We're going to need to modify this when that
    # is no longer the case.
    for _attr in all_return_types:
        if isinstance(_attr, type) and issubclass(_attr, tuple):
            pytree_register_structseq(_attr)

del _name, _attr, warnings, deprecated
