import torch
import inspect

__all__ = []

# error: Module has no attribute "_return_types"
return_types = torch._C._return_types  # type: ignore[attr-defined]

def pytree_register_structseq(cls):
    def structseq_flatten(structseq):
        return list(structseq), None

    def structseq_unflatten(values, context):
        return cls(values)

    torch.utils._pytree._register_pytree_node(cls, structseq_flatten, structseq_unflatten)

for name in dir(return_types):
    if name.startswith('__'):
        continue
    globals()[name] = getattr(return_types, name)
    __all__.append(name)

    # Today everything in torch.return_types is a structseq, aka a "namedtuple"-like
    # thing defined by the Python C-API. We're going to need to modify this when that
    # is no longer the case.
    # NB: I don't know how to check that something is a "structseq" so we do a fuzzy
    # check for tuple
    attr = globals()[name]
    if inspect.isclass(attr) and issubclass(attr, tuple):
        pytree_register_structseq(attr)
