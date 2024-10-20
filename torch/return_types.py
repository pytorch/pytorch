from torch._C import _return_types as return_types


__all__ = ["pytree_register_structseq", "all_return_types"]


all_return_types = []


def pytree_register_structseq(cls):
    from torch.utils._pytree import is_structseq_class

    if is_structseq_class(cls):
        return

    raise TypeError(f"Class {cls!r} is not a PyStructSequence class.")


_name = ""
for _name in dir(return_types):
    if _name.startswith("__"):
        continue

    _attr = getattr(return_types, _name)
    globals()[_name] = _attr

    if not _name.startswith("_"):
        __all__.append(_name)
        all_return_types.append(_attr)

    # Today everything in torch.return_types is a structseq, aka a "namedtuple"-like
    # thing defined by the Python C-API. We're going to need to modify this when that
    # is no longer the case.
    # NB: I don't know how to check that something is a "structseq" so we do a fuzzy
    # check for tuple
    if isinstance(_attr, type) and issubclass(_attr, tuple):
        pytree_register_structseq(_attr)

    del _attr

del _name
