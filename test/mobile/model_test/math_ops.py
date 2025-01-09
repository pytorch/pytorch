# https://pytorch.org/docs/stable/torch.html#math-operations

import math

import torch


class PointwiseOpsModule(torch.nn.Module):
    def forward(self):
        return self.pointwise_ops()

    def pointwise_ops(self):
        a = torch.randn(4)
        b = torch.randn(4)
        t = torch.tensor([-1, -2, 3], dtype=torch.int8)
        r = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
        t = torch.tensor([-1, -2, 3], dtype=torch.int8)
        s = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
        f = torch.zeros(3)
        g = torch.tensor([-1, 0, 1])
        w = torch.tensor([0.3810, 1.2774, -0.2972, -0.3719, 0.4637])
        return len(
            torch.abs(torch.tensor([-1, -2, 3])),
            torch.absolute(torch.tensor([-1, -2, 3])),
            torch.acos(a),
            torch.arccos(a),
            torch.acosh(a.uniform_(1.0, 2.0)),
            torch.add(a, 20),
            torch.add(a, b, out=a),
            b.add(a),
            b.add(a, out=b),
            b.add_(a),
            b.add(1),
            torch.add(a, torch.randn(4, 1), alpha=10),
            torch.addcdiv(
                torch.randn(1, 3), torch.randn(3, 1), torch.randn(1, 3), value=0.1
            ),
            torch.addcmul(
                torch.randn(1, 3), torch.randn(3, 1), torch.randn(1, 3), value=0.1
            ),
            torch.angle(a),
            torch.asin(a),
            torch.arcsin(a),
            torch.asinh(a),
            torch.arcsinh(a),
            torch.atan(a),
            torch.arctan(a),
            torch.atanh(a.uniform_(-1.0, 1.0)),
            torch.arctanh(a.uniform_(-1.0, 1.0)),
            torch.atan2(a, a),
            torch.bitwise_not(t),
            torch.bitwise_and(t, torch.tensor([1, 0, 3], dtype=torch.int8)),
            torch.bitwise_or(t, torch.tensor([1, 0, 3], dtype=torch.int8)),
            torch.bitwise_xor(t, torch.tensor([1, 0, 3], dtype=torch.int8)),
            torch.ceil(a),
            torch.ceil(float(torch.tensor(0.5))),
            torch.ceil(torch.tensor(0.5).item()),
            torch.clamp(a, min=-0.5, max=0.5),
            torch.clamp(a, min=0.5),
            torch.clamp(a, max=0.5),
            torch.clip(a, min=-0.5, max=0.5),
            torch.conj(a),
            torch.copysign(a, 1),
            torch.copysign(a, b),
            torch.cos(a),
            torch.cosh(a),
            torch.deg2rad(
                torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])
            ),
            torch.div(a, b),
            a.div(b),
            a.div(1),
            a.div_(b),
            torch.divide(a, b, rounding_mode="trunc"),
            torch.divide(a, b, rounding_mode="floor"),
            torch.digamma(torch.tensor([1.0, 0.5])),
            torch.erf(torch.tensor([0.0, -1.0, 10.0])),
            torch.erfc(torch.tensor([0.0, -1.0, 10.0])),
            torch.erfinv(torch.tensor([0.0, 0.5, -1.0])),
            torch.exp(torch.tensor([0.0, math.log(2.0)])),
            torch.exp(float(torch.tensor(1))),
            torch.exp2(torch.tensor([0.0, math.log(2.0), 3.0, 4.0])),
            torch.expm1(torch.tensor([0.0, math.log(2.0)])),
            torch.fake_quantize_per_channel_affine(
                torch.randn(2, 2, 2),
                (torch.randn(2) + 1) * 0.05,
                torch.zeros(2),
                1,
                0,
                255,
            ),
            torch.fake_quantize_per_tensor_affine(a, 0.1, 0, 0, 255),
            torch.float_power(torch.randint(10, (4,)), 2),
            torch.float_power(torch.arange(1, 5), torch.tensor([2, -3, 4, -5])),
            torch.floor(a),
            torch.floor(float(torch.tensor(1))),
            torch.floor_divide(torch.tensor([4.0, 3.0]), torch.tensor([2.0, 2.0])),
            torch.floor_divide(torch.tensor([4.0, 3.0]), 1.4),
            torch.fmod(torch.tensor([-3, -2, -1, 1, 2, 3]), 2),
            torch.fmod(torch.tensor([1, 2, 3, 4, 5]), 1.5),
            torch.frac(torch.tensor([1.0, 2.5, -3.2])),
            torch.randn(4, dtype=torch.cfloat).imag,
            torch.ldexp(torch.tensor([1.0]), torch.tensor([1])),
            torch.ldexp(torch.tensor([1.0]), torch.tensor([1, 2, 3, 4])),
            torch.lerp(torch.arange(1.0, 5.0), torch.empty(4).fill_(10), 0.5),
            torch.lerp(
                torch.arange(1.0, 5.0),
                torch.empty(4).fill_(10),
                torch.full_like(torch.arange(1.0, 5.0), 0.5),
            ),
            torch.lgamma(torch.arange(0.5, 2, 0.5)),
            torch.log(torch.arange(5) + 10),
            torch.log10(torch.rand(5)),
            torch.log1p(torch.randn(5)),
            torch.log2(torch.rand(5)),
            torch.logaddexp(torch.tensor([-1.0]), torch.tensor([-1, -2, -3])),
            torch.logaddexp(
                torch.tensor([-100.0, -200.0, -300.0]), torch.tensor([-1, -2, -3])
            ),
            torch.logaddexp(
                torch.tensor([1.0, 2000.0, 30000.0]), torch.tensor([-1, -2, -3])
            ),
            torch.logaddexp2(torch.tensor([-1.0]), torch.tensor([-1, -2, -3])),
            torch.logaddexp2(
                torch.tensor([-100.0, -200.0, -300.0]), torch.tensor([-1, -2, -3])
            ),
            torch.logaddexp2(
                torch.tensor([1.0, 2000.0, 30000.0]), torch.tensor([-1, -2, -3])
            ),
            torch.logical_and(r, s),
            torch.logical_and(r.double(), s.double()),
            torch.logical_and(r.double(), s),
            torch.logical_and(r, s, out=torch.empty(4, dtype=torch.bool)),
            torch.logical_not(torch.tensor([0, 1, -10], dtype=torch.int8)),
            torch.logical_not(torch.tensor([0.0, 1.5, -10.0], dtype=torch.double)),
            torch.logical_not(
                torch.tensor([0.0, 1.0, -10.0], dtype=torch.double),
                out=torch.empty(3, dtype=torch.int16),
            ),
            torch.logical_or(r, s),
            torch.logical_or(r.double(), s.double()),
            torch.logical_or(r.double(), s),
            torch.logical_or(r, s, out=torch.empty(4, dtype=torch.bool)),
            torch.logical_xor(r, s),
            torch.logical_xor(r.double(), s.double()),
            torch.logical_xor(r.double(), s),
            torch.logical_xor(r, s, out=torch.empty(4, dtype=torch.bool)),
            torch.logit(torch.rand(5), eps=1e-6),
            torch.hypot(torch.tensor([4.0]), torch.tensor([3.0, 4.0, 5.0])),
            torch.i0(torch.arange(5, dtype=torch.float32)),
            torch.igamma(a, b),
            torch.igammac(a, b),
            torch.mul(torch.randn(3), 100),
            b.mul(a),
            b.mul(5),
            b.mul(a, out=b),
            b.mul_(a),
            b.mul_(5),
            torch.multiply(torch.randn(4, 1), torch.randn(1, 4)),
            torch.mvlgamma(torch.empty(2, 3).uniform_(1.0, 2.0), 2),
            torch.tensor([float("nan"), float("inf"), -float("inf"), 3.14]),
            torch.nan_to_num(w),
            torch.nan_to_num_(w),
            torch.nan_to_num(w, nan=2.0),
            torch.nan_to_num(w, nan=2.0, posinf=1.0),
            torch.neg(torch.randn(5)),
            # torch.nextafter(torch.tensor([1, 2]), torch.tensor([2, 1])) == torch.tensor([eps + 1, 2 - eps]),
            torch.polygamma(1, torch.tensor([1.0, 0.5])),
            torch.polygamma(2, torch.tensor([1.0, 0.5])),
            torch.polygamma(3, torch.tensor([1.0, 0.5])),
            torch.polygamma(4, torch.tensor([1.0, 0.5])),
            torch.pow(a, 2),
            torch.pow(2, float(torch.tensor(0.5))),
            torch.pow(torch.arange(1.0, 5.0), torch.arange(1.0, 5.0)),
            torch.rad2deg(
                torch.tensor([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]])
            ),
            torch.randn(4, dtype=torch.cfloat).real,
            torch.reciprocal(a),
            torch.remainder(torch.tensor([-3.0, -2.0]), 2),
            torch.remainder(torch.tensor([1, 2, 3, 4, 5]), 1.5),
            torch.round(a),
            torch.round(torch.tensor(0.5).item()),
            torch.rsqrt(a),
            torch.sigmoid(a),
            torch.sign(torch.tensor([0.7, -1.2, 0.0, 2.3])),
            torch.sgn(a),
            torch.signbit(torch.tensor([0.7, -1.2, 0.0, 2.3])),
            torch.sin(a),
            torch.sinc(a),
            torch.sinh(a),
            torch.sqrt(a),
            torch.square(a),
            torch.sub(torch.tensor((1, 2)), torch.tensor((0, 1)), alpha=2),
            b.sub(a),
            b.sub_(a),
            b.sub(5),
            torch.sum(5),
            torch.tan(a),
            torch.tanh(a),
            torch.true_divide(a, a),
            torch.trunc(a),
            torch.trunc_(a),
            torch.xlogy(f, g),
            torch.xlogy(f, g),
            torch.xlogy(f, 4),
            torch.xlogy(2, g),
        )


class ReductionOpsModule(torch.nn.Module):
    def forward(self):
        return self.reduction_ops()

    def reduction_ops(self):
        a = torch.randn(4)
        b = torch.randn(4)
        c = torch.tensor(0.5)
        return len(
            torch.argmax(a),
            torch.argmin(a),
            torch.amax(a),
            torch.amin(a),
            torch.aminmax(a),
            torch.all(a),
            torch.any(a),
            torch.max(a),
            a.max(a),
            torch.max(a, 0),
            torch.min(a),
            a.min(a),
            torch.min(a, 0),
            torch.dist(a, b),
            torch.logsumexp(a, 0),
            torch.mean(a),
            torch.mean(a, 0),
            torch.nanmean(a),
            torch.median(a),
            torch.nanmedian(a),
            torch.mode(a),
            torch.norm(a),
            a.norm(2),
            torch.norm(a, dim=0),
            torch.norm(c, torch.tensor(2)),
            torch.nansum(a),
            torch.prod(a),
            torch.quantile(a, torch.tensor([0.25, 0.5, 0.75])),
            torch.quantile(a, 0.5),
            torch.nanquantile(a, torch.tensor([0.25, 0.5, 0.75])),
            torch.std(a),
            torch.std_mean(a),
            torch.sum(a),
            torch.unique(a),
            torch.unique_consecutive(a),
            torch.var(a),
            torch.var_mean(a),
            torch.count_nonzero(a),
        )


class ComparisonOpsModule(torch.nn.Module):
    def forward(self):
        a = torch.tensor(0)
        b = torch.tensor(1)
        return len(
            torch.allclose(a, b),
            torch.argsort(a),
            torch.eq(a, b),
            torch.eq(a, 1),
            torch.equal(a, b),
            torch.ge(a, b),
            torch.ge(a, 1),
            torch.greater_equal(a, b),
            torch.greater_equal(a, 1),
            torch.gt(a, b),
            torch.gt(a, 1),
            torch.greater(a, b),
            torch.isclose(a, b),
            torch.isfinite(a),
            torch.isin(a, b),
            torch.isinf(a),
            torch.isposinf(a),
            torch.isneginf(a),
            torch.isnan(a),
            torch.isreal(a),
            torch.kthvalue(a, 1),
            torch.le(a, b),
            torch.le(a, 1),
            torch.less_equal(a, b),
            torch.lt(a, b),
            torch.lt(a, 1),
            torch.less(a, b),
            torch.maximum(a, b),
            torch.minimum(a, b),
            torch.fmax(a, b),
            torch.fmin(a, b),
            torch.ne(a, b),
            torch.ne(a, 1),
            torch.not_equal(a, b),
            torch.sort(a),
            torch.topk(a, 1),
            torch.msort(a),
        )


class OtherMathOpsModule(torch.nn.Module):
    def forward(self):
        return self.other_ops()

    def other_ops(self):
        a = torch.randn(4)
        b = torch.randn(4)
        c = torch.randint(0, 8, (5,), dtype=torch.int64)
        e = torch.randn(4, 3)
        f = torch.randn(4, 4, 4)
        dims = [0, 1]
        return len(
            torch.atleast_1d(a),
            torch.atleast_2d(a),
            torch.atleast_3d(a),
            torch.bincount(c),
            torch.block_diag(a),
            torch.broadcast_tensors(a),
            torch.broadcast_to(a, (4)),
            # torch.broadcast_shapes(a),
            torch.bucketize(a, b),
            torch.cartesian_prod(a),
            torch.cdist(e, e),
            torch.clone(a),
            torch.combinations(a),
            torch.corrcoef(a),
            # torch.cov(a),
            torch.cross(e, e),
            torch.cummax(a, 0),
            torch.cummin(a, 0),
            torch.cumprod(a, 0),
            torch.cumsum(a, 0),
            torch.diag(a),
            torch.diag_embed(a),
            torch.diagflat(a),
            torch.diagonal(e),
            torch.diff(a),
            torch.einsum("iii", f),
            torch.flatten(a),
            torch.flip(e, dims),
            torch.fliplr(e),
            torch.flipud(e),
            torch.kron(a, b),
            torch.rot90(e),
            torch.gcd(c, c),
            torch.histc(a),
            torch.histogram(a),
            torch.meshgrid(a),
            torch.meshgrid(a, indexing="xy"),
            torch.lcm(c, c),
            torch.logcumsumexp(a, 0),
            torch.ravel(a),
            torch.renorm(e, 1, 0, 5),
            torch.repeat_interleave(c),
            torch.roll(a, 1, 0),
            torch.searchsorted(a, b),
            torch.tensordot(e, e),
            torch.trace(e),
            torch.tril(e),
            torch.tril_indices(3, 3),
            torch.triu(e),
            torch.triu_indices(3, 3),
            torch.vander(a),
            torch.view_as_real(torch.randn(4, dtype=torch.cfloat)),
            torch.view_as_complex(torch.randn(4, 2)).real,
            torch.resolve_conj(a),
            torch.resolve_neg(a),
        )


class SpectralOpsModule(torch.nn.Module):
    def forward(self):
        return self.spectral_ops()

    def spectral_ops(self):
        a = torch.randn(10)
        b = torch.randn(10, 8, 4, 2)
        return len(
            torch.stft(a, 8),
            torch.stft(a, torch.tensor(8)),
            torch.istft(b, 8),
            torch.bartlett_window(2, dtype=torch.float),
            torch.blackman_window(2, dtype=torch.float),
            torch.hamming_window(4, dtype=torch.float),
            torch.hann_window(4, dtype=torch.float),
            torch.kaiser_window(4, dtype=torch.float),
        )


class BlasLapackOpsModule(torch.nn.Module):
    def forward(self):
        return self.blas_lapack_ops()

    def blas_lapack_ops(self):
        m = torch.randn(3, 3)
        a = torch.randn(10, 3, 4)
        b = torch.randn(10, 4, 3)
        v = torch.randn(3)
        return len(
            torch.addbmm(m, a, b),
            torch.addmm(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3)),
            torch.addmv(torch.randn(2), torch.randn(2, 3), torch.randn(3)),
            torch.addr(torch.zeros(3, 3), v, v),
            torch.baddbmm(m, a, b),
            torch.bmm(a, b),
            torch.chain_matmul(torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3)),
            # torch.cholesky(a), # deprecated
            # torch.cholesky_inverse(torch.randn(3, 3)), # had some error
            # torch.cholesky_solve(torch.randn(3, 3), torch.randn(3, 3)),
            torch.dot(v, v),
            # torch.linalg.eig(m), # not build with lapack
            # torch.geqrf(a),
            torch.ger(v, v),
            torch.inner(m, m),
            # torch.inverse(m),
            # torch.det(m),
            # torch.logdet(m),
            # torch.slogdet(m),
            # torch.lstsq(m, m),
            # torch.linalg.lu_factor(m),
            # torch.lu_solve(m, *torch.linalg.lu_factor(m)),
            # torch.lu_unpack(*torch.linalg.lu_factor(m)),
            torch.matmul(m, m),
            torch.matrix_power(m, 2),
            # torch.matrix_rank(m),
            torch.matrix_exp(m),
            torch.mm(m, m),
            torch.mv(m, v),
            # torch.orgqr(a, m),
            # torch.ormqr(a, m, v),
            torch.outer(v, v),
            # torch.pinverse(m),
            # torch.qr(a),
            # torch.solve(m, m),
            # torch.svd(a),
            # torch.svd_lowrank(a),
            # torch.pca_lowrank(a),
            # torch.symeig(a), # deprecated
            # torch.lobpcg(a, b), # not supported
            torch.trapz(m, m),
            torch.trapezoid(m, m),
            torch.cumulative_trapezoid(m, m),
            # torch.triangular_solve(m, m),
            torch.vdot(v, v),
        )
