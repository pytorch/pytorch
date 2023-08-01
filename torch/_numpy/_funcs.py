import inspect

from . import _funcs_impl
from ._normalizations import normalizer

# _funcs_impl.py contains functions which mimic NumPy's eponimous equivalents,
# and consume/return PyTorch tensors/dtypes.
# They are also type annotated.
# Pull these functions from _funcs_impl and decorate them with @normalizer, which
# - Converts any input `np.ndarray`, `torch_np.ndarray`, list of lists, Python scalars, etc into a `torch.Tensor`.
# - Maps NumPy dtypes to PyTorch dtypes
# - If the input to the `axis` kwarg is an ndarray, it maps it into a tuple
# - Implements the semantics for the `out=` arg
# - Wraps back the outputs into `torch_np.ndarrays`

__all__ = [
    x
    for x in dir(_funcs_impl)
    if inspect.isfunction(getattr(_funcs_impl, x)) and not x.startswith("_")
]

# decorate implementer functions with argument normalizers and export to the top namespace
for name in __all__:
    func = getattr(_funcs_impl, name)
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
