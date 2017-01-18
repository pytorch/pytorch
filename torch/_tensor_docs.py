"""Adds docstrings to Tensor functions"""

import torch._C
from torch._C import _add_docstr as add_docstr

add_docstr(torch._C.FloatTensorBase.abs,
"""
abs(out=None) -> Tensor

See :func:`torch.abs`
""")

add_docstr(torch._C.FloatTensorBase.abs_,
"""
abs_(out=None) -> Tensor

In-place version of :meth:`~Tensor.abs`
""")

add_docstr(torch._C.FloatTensorBase.acos,
"""
acos(out=None) -> Tensor

See :func:`torch.acos`
""")

add_docstr(torch._C.FloatTensorBase.acos_,
"""
acos_(out=None) -> Tensor

In-place version of :meth:`~Tensor.acos`
""")

add_docstr(torch._C.FloatTensorBase.add,
"""
add(value, out=None)

See :func:`torch.add`
""")

add_docstr(torch._C.FloatTensorBase.add_,
"""
add_(value, out=None)

In-place version of :meth:`~Tensor.add`
""")

add_docstr(torch._C.FloatTensorBase.addbmm,
"""
addbmm(beta=1, mat, alpha=1, batch1, batch2, out=None) -> Tensor

See :func:`torch.addbmm`
""")

add_docstr(torch._C.FloatTensorBase.addbmm_,
"""
addbmm_(beta=1, mat, alpha=1, batch1, batch2, out=None) -> Tensor

In-place version of :meth:`~Tensor.addbmm`
""")

add_docstr(torch._C.FloatTensorBase.addcdiv,
"""
addcdiv(value=1, tensor1, tensor2, out=None) -> Tensor

See :func:`torch.addcdiv`
""")

add_docstr(torch._C.FloatTensorBase.addcdiv_,
"""
addcdiv_(value=1, tensor1, tensor2, out=None) -> Tensor

In-place version of :meth:`~Tensor.addcdiv`
""")

add_docstr(torch._C.FloatTensorBase.addcmul,
"""
addcmul(value=1, tensor1, tensor2, out=None) -> Tensor

See :func:`torch.addcmul`
""")

add_docstr(torch._C.FloatTensorBase.addcmul_,
"""
addcmul_(value=1, tensor1, tensor2, out=None) -> Tensor

In-place version of :meth:`~Tensor.addcmul`
""")

add_docstr(torch._C.FloatTensorBase.addmm,
"""
addmm(beta=1, mat, alpha=1, mat1, mat2, out=None) -> Tensor

See :func:`torch.addmm`
""")

add_docstr(torch._C.FloatTensorBase.addmm_,
"""
addmm_(beta=1, mat, alpha=1, mat1, mat2, out=None) -> Tensor

In-place version of :meth:`~Tensor.addmm`
""")

add_docstr(torch._C.FloatTensorBase.addmv,
"""
addmv(beta=1, tensor, alpha=1, mat, vec, out=None) -> Tensor

See :func:`torch.addmv`
""")

add_docstr(torch._C.FloatTensorBase.addmv_,
"""
addmv_(beta=1, tensor, alpha=1, mat, vec, out=None) -> Tensor

In-place version of :meth:`~Tensor.addmv`
""")

add_docstr(torch._C.FloatTensorBase.addr,
"""
addr(beta=1, alpha=1, vec1, vec2, out=None) -> Tensor

See :func:`torch.addr`
""")

add_docstr(torch._C.FloatTensorBase.addr_,
"""
addr_(beta=1, alpha=1, vec1, vec2, out=None) -> Tensor

In-place version of :meth:`~Tensor.addr`
""")

add_docstr(torch._C.FloatTensorBase.apply_,
"""
""")

add_docstr(torch._C.FloatTensorBase.asin,
"""
asin(out=None) -> Tensor

See :func:`torch.asin`
""")

add_docstr(torch._C.FloatTensorBase.asin_,
"""
asin_(out=None) -> Tensor

In-place version of :meth:`~Tensor.asin`
""")

add_docstr(torch._C.FloatTensorBase.atan,
"""
atan(out=None) -> Tensor

See :func:`torch.atan`
""")

add_docstr(torch._C.FloatTensorBase.atan2,
"""
atan2(other, out=None) -> Tensor

See :func:`torch.atan2`
""")

add_docstr(torch._C.FloatTensorBase.atan2_,
"""
atan2_(other, out=None) -> Tensor

In-place version of :meth:`~Tensor.atan2`
""")

add_docstr(torch._C.FloatTensorBase.atan_,
"""
atan_(out=None) -> Tensor

In-place version of :meth:`~Tensor.atan`
""")

add_docstr(torch._C.FloatTensorBase.baddbmm,
"""
baddbmm(beta=1, alpha=1, batch1, batch2, out=None) -> Tensor

See :func:`torch.baddbmm`
""")

add_docstr(torch._C.FloatTensorBase.baddbmm_,
"""
baddbmm_(beta=1, alpha=1, batch1, batch2, out=None) -> Tensor

In-place version of :meth:`~Tensor.baddbmm`
""")

add_docstr(torch._C.FloatTensorBase.bernoulli,
"""
bernoulli(out=None) -> Tensor

See :func:`torch.bernoulli`
""")

add_docstr(torch._C.FloatTensorBase.bernoulli_,
"""
bernoulli_(out=None) -> Tensor

In-place version of :meth:`~Tensor.bernoulli`
""")

add_docstr(torch._C.FloatTensorBase.bmm,
"""
bmm(batch2, out=None) -> Tensor

See :func:`torch.bmm`
""")

add_docstr(torch._C.FloatTensorBase.cauchy_,
"""
""")

add_docstr(torch._C.FloatTensorBase.ceil,
"""
ceil(out=None) -> Tensor

See :func:`torch.ceil`
""")

add_docstr(torch._C.FloatTensorBase.ceil_,
"""
ceil_(out=None) -> Tensor

In-place version of :meth:`~Tensor.ceil`
""")

add_docstr(torch._C.FloatTensorBase.clamp,
"""
clamp(min, max, out=None) -> Tensor

See :func:`torch.clamp`
""")

add_docstr(torch._C.FloatTensorBase.clamp_,
"""
clamp_(min, max, out=None) -> Tensor

In-place version of :meth:`~Tensor.clamp`
""")

add_docstr(torch._C.FloatTensorBase.clone,
"""
""")

add_docstr(torch._C.FloatTensorBase.contiguous,
"""
""")

add_docstr(torch._C.FloatTensorBase.copy_,
"""
""")

add_docstr(torch._C.FloatTensorBase.cos,
"""
cos(out=None) -> Tensor

See :func:`torch.cos`
""")

add_docstr(torch._C.FloatTensorBase.cos_,
"""
cos_(out=None) -> Tensor

In-place version of :meth:`~Tensor.cos`
""")

add_docstr(torch._C.FloatTensorBase.cosh,
"""
cosh(out=None) -> Tensor

See :func:`torch.cosh`
""")

add_docstr(torch._C.FloatTensorBase.cosh_,
"""
cosh_(out=None) -> Tensor

In-place version of :meth:`~Tensor.cosh`
""")

add_docstr(torch._C.FloatTensorBase.cross,
"""
cross(other, dim=-1, out=None) -> Tensor

See :func:`torch.cross`
""")

add_docstr(torch._C.FloatTensorBase.cumprod,
"""
cumprod(dim, out=None) -> Tensor

See :func:`torch.cumprod`
""")

add_docstr(torch._C.FloatTensorBase.cumsum,
"""
cumsum(dim, out=None) -> Tensor

See :func:`torch.cumsum`
""")

add_docstr(torch._C.FloatTensorBase.data_ptr,
"""
""")

add_docstr(torch._C.FloatTensorBase.diag,
"""
diag(diagonal=0, out=None) -> Tensor

See :func:`torch.diag`
""")

add_docstr(torch._C.FloatTensorBase.dim,
"""
""")

add_docstr(torch._C.FloatTensorBase.dist,
"""
dist(other, p=2, out=None) -> Tensor

See :func:`torch.dist`
""")

add_docstr(torch._C.FloatTensorBase.div,
"""
div(value, out=None)

See :func:`torch.div`
""")

add_docstr(torch._C.FloatTensorBase.div_,
"""
div_(value, out=None)

In-place version of :meth:`~Tensor.div`
""")

add_docstr(torch._C.FloatTensorBase.dot,
"""
dot(tensor2) -> float

See :func:`torch.dot`
""")

add_docstr(torch._C.FloatTensorBase.eig,
"""
eig(eigenvectors=False, out=None) -> (Tensor, Tensor)

See :func:`torch.eig`
""")

add_docstr(torch._C.FloatTensorBase.element_size,
"""
""")

add_docstr(torch._C.FloatTensorBase.eq,
"""
eq(other, out=None) -> Tensor

See :func:`torch.eq`
""")

add_docstr(torch._C.FloatTensorBase.eq_,
"""
eq_(other, out=None) -> Tensor

In-place version of :meth:`~Tensor.eq`
""")

add_docstr(torch._C.FloatTensorBase.equal,
"""
equal(other) -> bool

See :func:`torch.equal`
""")

add_docstr(torch._C.FloatTensorBase.exp,
"""
exp(out=None) -> Tensor

See :func:`torch.exp`
""")

add_docstr(torch._C.FloatTensorBase.exp_,
"""
exp_(out=None) -> Tensor

In-place version of :meth:`~Tensor.exp`
""")

add_docstr(torch._C.FloatTensorBase.exponential_,
"""
""")

add_docstr(torch._C.FloatTensorBase.fill_,
"""
""")

add_docstr(torch._C.FloatTensorBase.floor,
"""
floor(out=None) -> Tensor

See :func:`torch.floor`
""")

add_docstr(torch._C.FloatTensorBase.floor_,
"""
floor_(out=None) -> Tensor

In-place version of :meth:`~Tensor.floor`
""")

add_docstr(torch._C.FloatTensorBase.fmod,
"""
fmod(divisor, out=None) -> Tensor

See :func:`torch.fmod`
""")

add_docstr(torch._C.FloatTensorBase.fmod_,
"""
fmod_(divisor, out=None) -> Tensor

In-place version of :meth:`~Tensor.fmod`
""")

add_docstr(torch._C.FloatTensorBase.frac,
"""
frac(out=None) -> Tensor

See :func:`torch.frac`
""")

add_docstr(torch._C.FloatTensorBase.frac_,
"""
frac_(out=None) -> Tensor

In-place version of :meth:`~Tensor.frac`
""")

add_docstr(torch._C.FloatTensorBase.gather,
"""
gather(dim, index, out=None) -> Tensor

See :func:`torch.gather`
""")

add_docstr(torch._C.FloatTensorBase.ge,
"""
ge(other, out=None) -> Tensor

See :func:`torch.ge`
""")

add_docstr(torch._C.FloatTensorBase.ge_,
"""
ge_(other, out=None) -> Tensor

In-place version of :meth:`~Tensor.ge`
""")

add_docstr(torch._C.FloatTensorBase.gels,
"""
gels(A, out=None) -> Tensor

See :func:`torch.gels`
""")

add_docstr(torch._C.FloatTensorBase.geometric_,
"""
""")

add_docstr(torch._C.FloatTensorBase.geqrf,
"""
geqrf() -> (Tensor, Tensor)

TODO: fix signature
See :func:`torch.geqrf`
""")

add_docstr(torch._C.FloatTensorBase.ger,
"""


See :func:`torch.ger`
""")

add_docstr(torch._C.FloatTensorBase.gesv,
"""


See :func:`torch.gesv`
""")

add_docstr(torch._C.FloatTensorBase.gt,
"""
gt(other, out=None) -> Tensor

See :func:`torch.gt`
""")

add_docstr(torch._C.FloatTensorBase.gt_,
"""
gt_(other, out=None) -> Tensor

In-place version of :meth:`~Tensor.gt`
""")

add_docstr(torch._C.FloatTensorBase.histc,
"""
histc(bins=100, min=0, max=0, out=None) -> Tensor

See :func:`torch.histc`
""")

add_docstr(torch._C.FloatTensorBase.index,
"""
""")

add_docstr(torch._C.FloatTensorBase.index_add_,
"""
""")

add_docstr(torch._C.FloatTensorBase.index_copy_,
"""
""")

add_docstr(torch._C.FloatTensorBase.index_fill_,
"""
""")

add_docstr(torch._C.FloatTensorBase.index_select,
"""
index_select(dim, index, out=None) -> Tensor

See :func:`torch.index_select`
""")

add_docstr(torch._C.FloatTensorBase.inverse,
"""


See :func:`torch.inverse`
""")

add_docstr(torch._C.FloatTensorBase.is_contiguous,
"""
""")

add_docstr(torch._C.FloatTensorBase.is_same_size,
"""
""")

add_docstr(torch._C.FloatTensorBase.is_set_to,
"""
""")

add_docstr(torch._C.FloatTensorBase.kthvalue,
"""
kthvalue(k, dim=None, out=None) -> (Tensor, LongTensor)

See :func:`torch.kthvalue`
""")

add_docstr(torch._C.FloatTensorBase.le,
"""
le(other, out=None) -> Tensor

See :func:`torch.le`
""")

add_docstr(torch._C.FloatTensorBase.le_,
"""
le_(other, out=None) -> Tensor

In-place version of :meth:`~Tensor.le`
""")

add_docstr(torch._C.FloatTensorBase.lerp,
"""
lerp(start, end, weight, out=None)

See :func:`torch.lerp`
""")

add_docstr(torch._C.FloatTensorBase.lerp_,
"""
lerp_(start, end, weight, out=None)

In-place version of :meth:`~Tensor.lerp`
""")

add_docstr(torch._C.FloatTensorBase.log,
"""
log(out=None) -> Tensor

See :func:`torch.log`
""")

add_docstr(torch._C.FloatTensorBase.log1p,
"""
log1p(out=None) -> Tensor

See :func:`torch.log1p`
""")

add_docstr(torch._C.FloatTensorBase.log1p_,
"""
log1p_(out=None) -> Tensor

In-place version of :meth:`~Tensor.log1p`
""")

add_docstr(torch._C.FloatTensorBase.log_,
"""
log_(out=None) -> Tensor

In-place version of :meth:`~Tensor.log`
""")

add_docstr(torch._C.FloatTensorBase.log_normal_,
"""
""")

add_docstr(torch._C.FloatTensorBase.lt,
"""
lt(other, out=None) -> Tensor

See :func:`torch.lt`
""")

add_docstr(torch._C.FloatTensorBase.lt_,
"""
lt_(other, out=None) -> Tensor

In-place version of :meth:`~Tensor.lt`
""")

add_docstr(torch._C.FloatTensorBase.map2_,
"""
""")

add_docstr(torch._C.FloatTensorBase.map_,
"""
""")

add_docstr(torch._C.FloatTensorBase.masked_copy_,
"""
""")

add_docstr(torch._C.FloatTensorBase.masked_fill_,
"""
""")

add_docstr(torch._C.FloatTensorBase.masked_select,
"""
masked_select(mask, out=None) -> Tensor

See :func:`torch.masked_select`
""")

add_docstr(torch._C.FloatTensorBase.max,
"""
max() -> float

See :func:`torch.max`
""")

add_docstr(torch._C.FloatTensorBase.mean,
"""
mean() -> float

See :func:`torch.mean`
""")

add_docstr(torch._C.FloatTensorBase.median,
"""
median(dim=-1, values=None, indices=None) -> (Tensor, LongTensor)

See :func:`torch.median`
""")

add_docstr(torch._C.FloatTensorBase.min,
"""
min() -> float

See :func:`torch.min`
""")

add_docstr(torch._C.FloatTensorBase.mm,
"""
mm(mat2, out=None) -> Tensor

See :func:`torch.mm`
""")

add_docstr(torch._C.FloatTensorBase.mode,
"""
mode(dim=-1, values=None, indices=None) -> (Tensor, LongTensor)

See :func:`torch.mode`
""")

add_docstr(torch._C.FloatTensorBase.mul,
"""
mul(value, out=None)

See :func:`torch.mul`
""")

add_docstr(torch._C.FloatTensorBase.mul_,
"""
mul_(value, out=None)

In-place version of :meth:`~Tensor.mul`
""")

add_docstr(torch._C.FloatTensorBase.multinomial,
"""


See :func:`torch.multinomial`
""")

add_docstr(torch._C.FloatTensorBase.mv,
"""
mv(vec, out=None) -> Tensor

See :func:`torch.mv`
""")

add_docstr(torch._C.FloatTensorBase.narrow,
"""
""")

add_docstr(torch._C.FloatTensorBase.ndimension,
"""
""")

add_docstr(torch._C.FloatTensorBase.ne,
"""
ne(other, out=None) -> Tensor

See :func:`torch.ne`
""")

add_docstr(torch._C.FloatTensorBase.ne_,
"""
ne_(other, out=None) -> Tensor

In-place version of :meth:`~Tensor.ne`
""")

add_docstr(torch._C.FloatTensorBase.neg,
"""
neg(out=None) -> Tensor

See :func:`torch.neg`
""")

add_docstr(torch._C.FloatTensorBase.neg_,
"""
neg_(out=None) -> Tensor

In-place version of :meth:`~Tensor.neg`
""")

add_docstr(torch._C.FloatTensorBase.nelement,
"""
""")

add_docstr(torch._C.FloatTensorBase.nonzero,
"""
nonzero(out=None) -> LongTensor

See :func:`torch.nonzero`
""")

add_docstr(torch._C.FloatTensorBase.norm,
"""
norm(p=2) -> float

See :func:`torch.norm`
""")

add_docstr(torch._C.FloatTensorBase.normal_,
"""
""")

add_docstr(torch._C.FloatTensorBase.numel,
"""
numel() -> int

See :func:`torch.numel`
""")

add_docstr(torch._C.FloatTensorBase.numpy,
"""
""")

add_docstr(torch._C.FloatTensorBase.ones_,
"""
""")

add_docstr(torch._C.FloatTensorBase.orgqr,
"""


See :func:`torch.orgqr`
""")

add_docstr(torch._C.FloatTensorBase.ormqr,
"""


See :func:`torch.ormqr`
""")

add_docstr(torch._C.FloatTensorBase.potrf,
"""


See :func:`torch.potrf`
""")

add_docstr(torch._C.FloatTensorBase.potri,
"""


See :func:`torch.potri`
""")

add_docstr(torch._C.FloatTensorBase.potrs,
"""


See :func:`torch.potrs`
""")

add_docstr(torch._C.FloatTensorBase.pow,
"""
pow(exponent, out=None)

See :func:`torch.pow`
""")

add_docstr(torch._C.FloatTensorBase.pow_,
"""
pow_(exponent, out=None)

In-place version of :meth:`~Tensor.pow`
""")

add_docstr(torch._C.FloatTensorBase.prod,
"""
prod() -> float

See :func:`torch.prod`
""")

add_docstr(torch._C.FloatTensorBase.pstrf,
"""


See :func:`torch.pstrf`
""")

add_docstr(torch._C.FloatTensorBase.qr,
"""


See :func:`torch.qr`
""")

add_docstr(torch._C.FloatTensorBase.random_,
"""
""")

add_docstr(torch._C.FloatTensorBase.reciprocal,
"""
reciprocal(out=None) -> Tensor

See :func:`torch.reciprocal`
""")

add_docstr(torch._C.FloatTensorBase.reciprocal_,
"""
reciprocal_(out=None) -> Tensor

In-place version of :meth:`~Tensor.reciprocal`
""")

add_docstr(torch._C.FloatTensorBase.remainder,
"""
remainder(divisor, out=None) -> Tensor

See :func:`torch.remainder`
""")

add_docstr(torch._C.FloatTensorBase.remainder_,
"""
remainder_(divisor, out=None) -> Tensor

In-place version of :meth:`~Tensor.remainder`
""")

add_docstr(torch._C.FloatTensorBase.renorm,
"""
renorm(p, dim, maxnorm, out=None) -> Tensor

See :func:`torch.renorm`
""")

add_docstr(torch._C.FloatTensorBase.renorm_,
"""
renorm_(p, dim, maxnorm, out=None) -> Tensor

In-place version of :meth:`~Tensor.renorm`
""")

add_docstr(torch._C.FloatTensorBase.resize_,
"""
""")

add_docstr(torch._C.FloatTensorBase.resize_as_,
"""
""")

add_docstr(torch._C.FloatTensorBase.round,
"""
round(out=None) -> Tensor

See :func:`torch.round`
""")

add_docstr(torch._C.FloatTensorBase.round_,
"""
round_(out=None) -> Tensor

In-place version of :meth:`~Tensor.round`
""")

add_docstr(torch._C.FloatTensorBase.rsqrt,
"""
rsqrt(out=None) -> Tensor

See :func:`torch.rsqrt`
""")

add_docstr(torch._C.FloatTensorBase.rsqrt_,
"""
rsqrt_(out=None) -> Tensor

In-place version of :meth:`~Tensor.rsqrt`
""")

add_docstr(torch._C.FloatTensorBase.scatter_,
"""
""")

add_docstr(torch._C.FloatTensorBase.select,
"""
""")

add_docstr(torch._C.FloatTensorBase.set_,
"""
""")

add_docstr(torch._C.FloatTensorBase.set_index,
"""
""")

add_docstr(torch._C.FloatTensorBase.sigmoid,
"""
sigmoid(out=None) -> Tensor

See :func:`torch.sigmoid`
""")

add_docstr(torch._C.FloatTensorBase.sigmoid_,
"""
sigmoid_(out=None) -> Tensor

In-place version of :meth:`~Tensor.sigmoid`
""")

add_docstr(torch._C.FloatTensorBase.sign,
"""
sign(out=None) -> Tensor

See :func:`torch.sign`
""")

add_docstr(torch._C.FloatTensorBase.sign_,
"""
sign_(out=None) -> Tensor

In-place version of :meth:`~Tensor.sign`
""")

add_docstr(torch._C.FloatTensorBase.sin,
"""
sin(out=None) -> Tensor

See :func:`torch.sin`
""")

add_docstr(torch._C.FloatTensorBase.sin_,
"""
sin_(out=None) -> Tensor

In-place version of :meth:`~Tensor.sin`
""")

add_docstr(torch._C.FloatTensorBase.sinh,
"""
sinh(out=None) -> Tensor

See :func:`torch.sinh`
""")

add_docstr(torch._C.FloatTensorBase.sinh_,
"""
sinh_(out=None) -> Tensor

In-place version of :meth:`~Tensor.sinh`
""")

add_docstr(torch._C.FloatTensorBase.size,
"""
""")

add_docstr(torch._C.FloatTensorBase.sort,
"""
sort(dim=None, descending=False, out=None) -> (Tensor, LongTensor)

See :func:`torch.sort`
""")

add_docstr(torch._C.FloatTensorBase.sqrt,
"""
sqrt(out=None) -> Tensor

See :func:`torch.sqrt`
""")

add_docstr(torch._C.FloatTensorBase.sqrt_,
"""
sqrt_(out=None) -> Tensor

In-place version of :meth:`~Tensor.sqrt`
""")

add_docstr(torch._C.FloatTensorBase.squeeze,
"""
squeeze(dim=None, out=None)

See :func:`torch.squeeze`
""")

add_docstr(torch._C.FloatTensorBase.squeeze_,
"""
squeeze_(dim=None, out=None)

In-place version of :meth:`~Tensor.squeeze`
""")

add_docstr(torch._C.FloatTensorBase.std,
"""
std() -> float

See :func:`torch.std`
""")

add_docstr(torch._C.FloatTensorBase.storage,
"""
""")

add_docstr(torch._C.FloatTensorBase.storage_offset,
"""
""")

add_docstr(torch._C.FloatTensorBase.stride,
"""
stride() -> tuple

Returns the stride of the tensor.
""")

add_docstr(torch._C.FloatTensorBase.sub,
"""
""")

add_docstr(torch._C.FloatTensorBase.sub_,
"""
sub_(x) -> Tensor

In-place version of :meth:`~Tensor.sub`
""")

add_docstr(torch._C.FloatTensorBase.sum,
"""
sum(dim=None) -> float

See :func:`torch.sum`
""")

add_docstr(torch._C.FloatTensorBase.svd,
"""


See :func:`torch.svd`
""")

add_docstr(torch._C.FloatTensorBase.symeig,
"""


See :func:`torch.symeig`
""")

add_docstr(torch._C.FloatTensorBase.t,
"""
t(out=None) -> Tensor

See :func:`torch.t`
""")

add_docstr(torch._C.FloatTensorBase.t_,
"""
t_(out=None) -> Tensor

In-place version of :meth:`~Tensor.t`
""")

add_docstr(torch._C.FloatTensorBase.tan,
"""
tan(out=None) -> Tensor

See :func:`torch.tan`
""")

add_docstr(torch._C.FloatTensorBase.tan_,
"""
tan_(out=None) -> Tensor

In-place version of :meth:`~Tensor.tan`
""")

add_docstr(torch._C.FloatTensorBase.tanh,
"""
tanh(out=None) -> Tensor

See :func:`torch.tanh`
""")

add_docstr(torch._C.FloatTensorBase.tanh_,
"""
tanh_(out=None) -> Tensor

In-place version of :meth:`~Tensor.tanh`
""")

add_docstr(torch._C.FloatTensorBase.topk,
"""
topk(k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)

See :func:`torch.topk`
""")

add_docstr(torch._C.FloatTensorBase.trace,
"""
trace() -> float

See :func:`torch.trace`
""")

add_docstr(torch._C.FloatTensorBase.transpose,
"""
transpose(dim0, dim1, out=None) -> Tensor

See :func:`torch.transpose`
""")

add_docstr(torch._C.FloatTensorBase.transpose_,
"""
transpose_(dim0, dim1, out=None) -> Tensor

In-place version of :meth:`~Tensor.transpose`
""")

add_docstr(torch._C.FloatTensorBase.tril,
"""
tril(k=0, out=None) -> Tensor

See :func:`torch.tril`
""")

add_docstr(torch._C.FloatTensorBase.tril_,
"""
tril_(k=0, out=None) -> Tensor

In-place version of :meth:`~Tensor.tril`
""")

add_docstr(torch._C.FloatTensorBase.triu,
"""
triu(k=0, out=None) -> Tensor

See :func:`torch.triu`
""")

add_docstr(torch._C.FloatTensorBase.triu_,
"""
triu_(k=0, out=None) -> Tensor

In-place version of :meth:`~Tensor.triu`
""")

add_docstr(torch._C.FloatTensorBase.trtrs,
"""


See :func:`torch.trtrs`
""")

add_docstr(torch._C.FloatTensorBase.trunc,
"""
trunc(out=None) -> Tensor

See :func:`torch.trunc`
""")

add_docstr(torch._C.FloatTensorBase.trunc_,
"""
trunc_(out=None) -> Tensor

In-place version of :meth:`~Tensor.trunc`
""")

add_docstr(torch._C.FloatTensorBase.unfold,
"""


See :func:`torch.unfold`
""")

add_docstr(torch._C.FloatTensorBase.uniform_,
"""
""")

add_docstr(torch._C.FloatTensorBase.var,
"""
var() -> float

See :func:`torch.var`
""")

add_docstr(torch._C.FloatTensorBase.zero_,
"""
""")

add_docstr(torch._C.FloatTensorBase.zeros_,
"""
""")
