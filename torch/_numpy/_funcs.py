

import inspect
import itertools

from . import _funcs_impl, _reductions_impl
from ._normalizations import normalizer


# _funcs_impl.py contains functions which mimic NumPy's eponymous equivalents,
# and consume/return PyTorch tensors/dtypes.
# They are also type annotated.
# Pull these functions from _funcs_impl and decorate them with @normalizer, which
# - Converts any input `np.ndarray`, `torch._numpy.ndarray`, list of lists, Python scalars, etc into a `torch.Tensor`.
# - Maps NumPy dtypes to PyTorch dtypes
# - If the input to the `axis` kwarg is an ndarray, it maps it into a tuple
# - Implements the semantics for the `out=` arg
# - Wraps back the outputs into `torch._numpy.ndarrays`


def _public_functions(mod):
    def is_public_function(f):
        return inspect.isfunction(f) and not f.__name__.startswith("_")

    return inspect.getmembers(mod, is_public_function)


# We fill in __all__ in the loop below
__all__ = []

# decorate implementer functions with argument normalizers and export to the top namespace
for name, func in itertools.chain(
    _public_functions(_funcs_impl), _public_functions(_reductions_impl)
):
    if name in ["percentile", "quantile", "median"]:
        decorated = normalizer(func, promote_scalar_result=True)
    elif name == "einsum":
        # normalized manually
        decorated = func
    else:
        decorated = normalizer(func)

    decorated.__qualname__ = name
    decorated.__name__ = name
    vars()[name] = decorated
    __all__.append(name)


"""
Vendored objects from numpy.lib.index_tricks
"""


class IndexExpression:
    """
    Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
    last revision: 1999-7-23

    Cosmetic changes by T. Oliphant 2001
    """

    def __init__(self, maketuple):
        self.maketuple = maketuple

    def __getitem__(self, item):
        if self.maketuple and not isinstance(item, tuple):
            return (item,)
        else:
            return item


index_exp = IndexExpression(maketuple=True)
s_ = IndexExpression(maketuple=False)


__all__ += ["index_exp", "s_"]
