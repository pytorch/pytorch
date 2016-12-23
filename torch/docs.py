"""Adds docstrings to functions defined in the torch._C"""

import torch._C
from torch._C import _add_docstr as add_docstr

add_docstr(torch._C.abs,
"""abs([result], tensor) -> tensor

Computes the element-wise absolute value of a tensor.

Example:
    >>> torch.abs(torch.FloatTensor([-1, -2, 3]))
    FloatTensor([1, 2, 3])
""")

add_docstr(torch._C.acos,
"""
acos([result], tensor) -> tensor

Computes the element-wise inverse cosine of a tensor.

Example:
    >>> torch.acos(torch.FloatTensor([1, -1]))
    FloatTensor([0.0000, 3.1416])
""")

add_docstr(torch._C.add,
"""
""")

add_docstr(torch._C.addbmm,
"""
""")

add_docstr(torch._C.addcdiv,
"""
""")

add_docstr(torch._C.addcmul,
"""
""")

add_docstr(torch._C.addmm,
"""
""")

add_docstr(torch._C.addmv,
"""
""")

add_docstr(torch._C.addr,
"""
""")

add_docstr(torch._C.all,
"""
""")

add_docstr(torch._C.any,
"""
""")

add_docstr(torch._C.asin,
"""
""")

add_docstr(torch._C.atan,
"""
""")

add_docstr(torch._C.atan2,
"""
""")

add_docstr(torch._C.baddbmm,
"""
""")

add_docstr(torch._C.bernoulli,
"""
""")

add_docstr(torch._C.bmm,
"""
""")

add_docstr(torch._C.cat,
"""
""")

add_docstr(torch._C.cauchy,
"""
""")

add_docstr(torch._C.cdiv,
"""
""")

add_docstr(torch._C.ceil,
"""
""")

add_docstr(torch._C.cfmod,
"""
""")

add_docstr(torch._C.cinv,
"""
""")

add_docstr(torch._C.clamp,
"""
""")

add_docstr(torch._C.cmax,
"""
""")

add_docstr(torch._C.cmin,
"""
""")

add_docstr(torch._C.cmod,
"""
""")

add_docstr(torch._C.cmul,
"""
""")

add_docstr(torch._C.cos,
"""
""")

add_docstr(torch._C.cosh,
"""
""")

add_docstr(torch._C.cpow,
"""
""")

add_docstr(torch._C.cremainder,
"""
""")

add_docstr(torch._C.cross,
"""
""")

add_docstr(torch._C.csub,
"""
""")

add_docstr(torch._C.cumprod,
"""
""")

add_docstr(torch._C.cumsum,
"""
""")

add_docstr(torch._C.diag,
"""
""")

add_docstr(torch._C.dist,
"""
""")

add_docstr(torch._C.div,
"""
""")

add_docstr(torch._C.dot,
"""
""")

add_docstr(torch._C.eig,
"""
""")

add_docstr(torch._C.eq,
"""
""")

add_docstr(torch._C.equal,
"""
""")

add_docstr(torch._C.exp,
"""
""")

add_docstr(torch._C.exponential,
"""
""")

add_docstr(torch._C.eye,
"""
""")

add_docstr(torch._C.fill,
"""
""")

add_docstr(torch._C.floor,
"""
""")

add_docstr(torch._C.fmod,
"""
""")

add_docstr(torch._C.frac,
"""
""")

add_docstr(torch._C.from_numpy,
"""
""")

add_docstr(torch._C.gather,
"""
""")

add_docstr(torch._C.ge,
"""
""")

add_docstr(torch._C.gels,
"""
""")

add_docstr(torch._C.geometric,
"""
""")

add_docstr(torch._C.geqrf,
"""
""")

add_docstr(torch._C.ger,
"""
""")

add_docstr(torch._C.gesv,
"""
""")

add_docstr(torch._C.get_num_threads,
"""
get_num_threads() -> int

Gets the number of OpenMP threads used for parallelizing CPU operations
""")

add_docstr(torch._C.gt,
"""
""")

add_docstr(torch._C.histc,
"""
histc([result], tensor, bins=100, min=0, max=0) -> tensor

Computes the histogram of a tensor.

The elements are sorted into equal width bins between `min` and `max`. If `min`
and `max` are both zero, the minimum and maximum values of the data are used.

Args:
    result: (tensor) optional result tensor
    tensor: (tensor) input data
    bins: (int) number of histogram bins
    min: (int) lower end of the range (inclusive)
    max: (int) upper end of the range (inclusive)

Returns:
    tensor: the histogram

Example:
    >>> torch.histc(torch.FloatTensor([1, 2, 1]), bins=4, min=0, max=3)
    FloatTensor([0, 2, 1, 0])

""")

add_docstr(torch._C.index_select,
"""
""")

add_docstr(torch._C.inverse,
"""
""")

add_docstr(torch._C.kthvalue,
"""
""")

add_docstr(torch._C.le,
"""
""")

add_docstr(torch._C.lerp,
"""
""")

add_docstr(torch._C.linspace,
"""
""")

add_docstr(torch._C.log,
"""
""")

add_docstr(torch._C.log1p,
"""
""")

add_docstr(torch._C.log_normal,
"""
""")

add_docstr(torch._C.logspace,
"""
""")

add_docstr(torch._C.lt,
"""
""")

add_docstr(torch._C.masked_select,
"""
""")

add_docstr(torch._C.max,
"""
""")

add_docstr(torch._C.mean,
"""
""")

add_docstr(torch._C.median,
"""
""")

add_docstr(torch._C.min,
"""
""")

add_docstr(torch._C.mm,
"""
""")

add_docstr(torch._C.mod,
"""
""")

add_docstr(torch._C.mode,
"""
""")

add_docstr(torch._C.mul,
"""
""")

add_docstr(torch._C.multinomial,
"""
""")

add_docstr(torch._C.mv,
"""
""")

add_docstr(torch._C.ne,
"""
""")

add_docstr(torch._C.neg,
"""
""")

add_docstr(torch._C.nonzero,
"""
""")

add_docstr(torch._C.norm,
"""
""")

add_docstr(torch._C.normal,
"""
""")

add_docstr(torch._C.numel,
"""
""")

add_docstr(torch._C.ones,
"""
""")

add_docstr(torch._C.orgqr,
"""
""")

add_docstr(torch._C.ormqr,
"""
""")

add_docstr(torch._C.potrf,
"""
""")

add_docstr(torch._C.potri,
"""
""")

add_docstr(torch._C.potrs,
"""
""")

add_docstr(torch._C.pow,
"""
""")

add_docstr(torch._C.prod,
"""
""")

add_docstr(torch._C.pstrf,
"""
""")

add_docstr(torch._C.qr,
"""
""")

add_docstr(torch._C.rand,
"""
""")

add_docstr(torch._C.randn,
"""
""")

add_docstr(torch._C.random,
"""
""")

add_docstr(torch._C.randperm,
"""
""")

add_docstr(torch._C.range,
"""
""")

add_docstr(torch._C.remainder,
"""
""")

add_docstr(torch._C.renorm,
"""
""")

add_docstr(torch._C.reshape,
"""
""")

add_docstr(torch._C.round,
"""
""")

add_docstr(torch._C.rsqrt,
"""
""")

add_docstr(torch._C.scatter,
"""
""")

add_docstr(torch._C.set_num_threads,
"""
set_num_threads(int)

Sets the number of OpenMP threads used for parallelizing CPU operations
""")

add_docstr(torch._C.sigmoid,
"""
""")

add_docstr(torch._C.sign,
"""
""")

add_docstr(torch._C.sin,
"""
""")

add_docstr(torch._C.sinh,
"""
""")

add_docstr(torch._C.sort,
"""
""")

add_docstr(torch._C.sqrt,
"""
""")

add_docstr(torch._C.squeeze,
"""
""")

add_docstr(torch._C.std,
"""
""")

add_docstr(torch._C.sum,
"""
""")

add_docstr(torch._C.svd,
"""
""")

add_docstr(torch._C.symeig,
"""
""")

add_docstr(torch._C.t,
"""
""")

add_docstr(torch._C.tan,
"""
""")

add_docstr(torch._C.tanh,
"""
""")

add_docstr(torch._C.topk,
"""
""")

add_docstr(torch._C.trace,
"""
""")

add_docstr(torch._C.transpose,
"""
""")

add_docstr(torch._C.tril,
"""
""")

add_docstr(torch._C.triu,
"""
""")

add_docstr(torch._C.trtrs,
"""
""")

add_docstr(torch._C.trunc,
"""
""")

add_docstr(torch._C.unfold,
"""
""")

add_docstr(torch._C.uniform,
"""
""")

add_docstr(torch._C.var,
"""
""")

add_docstr(torch._C.zero,
"""
""")

add_docstr(torch._C.zeros,
"""
""")
