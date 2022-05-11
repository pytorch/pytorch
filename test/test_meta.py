# Owner(s): ["module: primTorch"]

import torch
from torch.utils._pytree import tree_map, tree_flatten
from torch.testing._internal.common_utils import (
    TestCase,
    skipIfCrossRef,
    suppress_warnings,
    TEST_WITH_ASAN,
    run_tests,
)
from torch.overrides import push_torch_function_mode
from torch.testing._internal.common_device_type import (
    onlyNativeDeviceTypes,
    ops,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_methods_invocations import op_db

import functools
import re
from functools import partial
import unittest
import warnings

RE_NOT_IMPLEMENTED_MSG = re.compile(r"Could not run '([^']+)' with arguments ")

# These just need an implementation of meta tensors, once you
# implement them remove from this set.  When doing comprehensive
# testing, we will verify that these raise errors when meta is run under
# OpInfo
meta_exclude_set = {
    torch.Tensor.__lshift__,  # MISSING aten::__lshift__.Scalar
    torch.Tensor.__lshift__,  # MISSING aten::__lshift__.Tensor
    torch.Tensor.__reversed__,  # MISSING aten::flip
    torch.Tensor.__rmatmul__,  # MISSING aten::dot
    torch.Tensor.__rshift__,  # MISSING aten::__rshift__.Scalar
    torch.Tensor.__rshift__,  # MISSING aten::__rshift__.Tensor
    torch.Tensor.addbmm,  # MISSING aten::addbmm
    torch.Tensor.addcmul,  # MISSING aten::_local_scalar_dense
    torch.Tensor.angle,  # MISSING aten::angle
    torch.Tensor.argsort,  # MISSING aten::sort
    torch.Tensor.bincount,  # MISSING aten::bincount
    torch.Tensor.cholesky,  # MISSING aten::cholesky
    torch.Tensor.cholesky_inverse,  # MISSING aten::cholesky_inverse
    torch.Tensor.cholesky_solve,  # MISSING aten::_cholesky_solve_helper
    torch.Tensor.clamp,  # MISSING aten::clamp.Tensor
    torch.Tensor.clamp_,  # MISSING aten::clamp.Tensor_out
    torch.Tensor.clip,  # MISSING aten::clamp.Tensor
    torch.Tensor.clip_,  # MISSING aten::clamp.Tensor_out
    torch.Tensor.conj_physical,  # MISSING aten::conj_physical.out
    torch.Tensor.corrcoef,  # MISSING aten::_local_scalar_dense
    torch.Tensor.count_nonzero,  # MISSING aten::count_nonzero.dim_IntList
    torch.Tensor.cov,  # MISSING aten::_local_scalar_dense
    torch.Tensor.cummax,  # MISSING aten::_cummax_helper
    torch.Tensor.cummin,  # MISSING aten::_cummin_helper
    torch.Tensor.cumprod_,  # MISSING aten::logical_and.out
    torch.Tensor.dequantize,  # MISSING aten::dequantize.self
    torch.Tensor.det,  # MISSING aten::_det_lu_based_helper
    torch.Tensor.diag,  # MISSING aten::diag.out
    torch.Tensor.diagflat,  # MISSING aten::diag.out
    torch.Tensor.dot,  # MISSING aten::dot
    torch.Tensor.eig,  # MISSING aten::_local_scalar_dense
    torch.Tensor.equal,  # MISSING aten::equal
    torch.Tensor.flip,  # MISSING aten::flip
    torch.Tensor.fliplr,  # MISSING aten::flip
    torch.Tensor.flipud,  # MISSING aten::flip
    torch.Tensor.floor_divide,  # MISSING aten::floor_divide
    torch.Tensor.frexp,  # MISSING aten::frexp.Tensor_out
    torch.Tensor.geqrf,  # MISSING aten::geqrf
    torch.Tensor.histc,  # MISSING aten::histc
    torch.Tensor.histogram,  # MISSING aten::histogram.bin_ct
    torch.Tensor.inverse,  # MISSING aten::_local_scalar_dense
    torch.Tensor.is_set_to,  # MISSING aten::is_set_to
    torch.Tensor.isnan,  # MISSING aten::isnan
    torch.Tensor.istft,  # MISSING aten::view_as_complex
    torch.Tensor.kthvalue,  # MISSING aten::kthvalue.values
    torch.Tensor.logcumsumexp,  # MISSING aten::_logcumsumexp
    torch.Tensor.logdet,  # MISSING aten::_local_scalar_dense
    torch.Tensor.logical_and,  # MISSING aten::logical_and.out
    torch.Tensor.logical_and_,  # MISSING aten::logical_and.out
    torch.Tensor.logical_not,  # MISSING aten::logical_not.out
    torch.Tensor.logical_or,  # MISSING aten::logical_or.out
    torch.Tensor.logical_or_,  # MISSING aten::logical_or.out
    torch.Tensor.logical_xor,  # MISSING aten::logical_xor.out
    torch.Tensor.logical_xor_,  # MISSING aten::logical_xor.out
    torch.Tensor.logit,  # MISSING aten::logit
    torch.Tensor.logsumexp,  # MISSING aten::abs
    torch.Tensor.lstsq,  # MISSING aten::lstsq
    torch.Tensor.masked_select,  # MISSING aten::masked_select
    torch.Tensor.matmul,  # MISSING aten::dot
    torch.Tensor.matrix_exp,  # MISSING aten::linalg_matrix_exp
    torch.Tensor.matrix_power,  # MISSING aten::eye.m_out
    torch.Tensor.median,  # MISSING aten::median
    torch.Tensor.median,  # MISSING aten::median.dim_values
    torch.Tensor.mode,  # MISSING aten::mode
    torch.Tensor.msort,  # MISSING aten::sort
    torch.Tensor.multinomial,  # MISSING aten::multinomial
    torch.Tensor.mvlgamma,  # MISSING aten::_local_scalar_dense
    torch.Tensor.mvlgamma_,  # MISSING aten::_local_scalar_dense
    torch.Tensor.nan_to_num,  # MISSING aten::nan_to_num.out
    torch.Tensor.nan_to_num_,  # MISSING aten::nan_to_num.out
    torch.Tensor.nanmean,  # MISSING aten::logical_not.out
    torch.Tensor.nanmedian,  # MISSING aten::nanmedian
    torch.Tensor.nanmedian,  # MISSING aten::nanmedian.dim_values
    torch.Tensor.nanquantile,  # MISSING aten::sort
    torch.Tensor.nansum,  # MISSING aten::nansum
    torch.Tensor.narrow,  # MISSING aten::_local_scalar_dense
    torch.Tensor.nonzero,  # MISSING aten::nonzero
    torch.Tensor.orgqr,  # MISSING aten::linalg_householder_product
    torch.Tensor.ormqr,  # MISSING aten::ormqr
    torch.Tensor.pinverse,  # MISSING aten::where.self
    torch.Tensor.prod,  # MISSING aten::prod
    torch.Tensor.qr,  # MISSING aten::_linalg_qr_helper
    torch.Tensor.quantile,  # MISSING aten::sort
    torch.Tensor.relu,  # MISSING aten::relu
    torch.Tensor.renorm_,  # MISSING aten::_local_scalar_dense
    torch.Tensor.repeat_interleave,  # MISSING aten::repeat_interleave.Tensor
    torch.Tensor.roll,  # MISSING aten::roll
    torch.Tensor.rot90,  # MISSING aten::flip
    torch.Tensor.slogdet,  # MISSING aten::linalg_slogdet
    torch.Tensor.solve,  # MISSING aten::_solve_helper
    torch.Tensor.sort,  # MISSING aten::sort
    torch.Tensor.std,  # MISSING aten::std.correction
    torch.Tensor.stft,  # MISSING aten::_fft_r2c
    torch.Tensor.symeig,  # MISSING aten::_symeig_helper
    torch.Tensor.take,  # MISSING aten::take
    torch.Tensor.to_mkldnn,  # MISSING aten::to_mkldnn
    torch.Tensor.to_sparse,  # MISSING aten::to_sparse
    torch.Tensor.to_sparse_csr,  # MISSING aten::to_sparse_csr
    torch.Tensor.topk,  # MISSING aten::_local_scalar_dense
    torch.Tensor.trace,  # MISSING aten::trace
    torch.Tensor.unique,  # MISSING aten::_unique2
    torch.Tensor.unique_consecutive,  # MISSING aten::unique_consecutive
    torch.Tensor.unsqueeze,  # MISSING aten::_local_scalar_dense
    torch.Tensor.var,  # MISSING aten::var.correction
    torch.Tensor.vdot,  # MISSING aten::vdot
    torch.Tensor.where,  # MISSING aten::where.self
    torch._add_relu,  # MISSING aten::_add_relu.Tensor
    torch._aminmax,  # MISSING aten::_aminmax
    torch._assert_async,  # MISSING aten::_assert_async
    torch._compute_linear_combination,  # MISSING aten::_compute_linear_combination
    torch._det_lu_based_helper,  # MISSING aten::_det_lu_based_helper
    torch._dirichlet_grad,  # MISSING aten::_dirichlet_grad
    torch._fake_quantize_learnable_per_channel_affine,  # MISSING aten::_fake_quantize_learnable_per_channel_affine
    torch._fake_quantize_learnable_per_tensor_affine,  # MISSING aten::_fake_quantize_learnable_per_tensor_affine
    torch._fake_quantize_per_tensor_affine_cachemask_tensor_qparams,  # MISSING aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams  # noqa: E501
    torch._foreach_abs,  # MISSING aten::_foreach_abs
    torch._foreach_abs_,  # MISSING aten::_foreach_abs_
    torch._foreach_acos,  # MISSING aten::_foreach_acos
    torch._foreach_acos_,  # MISSING aten::_foreach_acos_
    torch._foreach_add,  # MISSING aten::_foreach_add.Scalar
    torch._foreach_add_,  # MISSING aten::_foreach_add_.Scalar
    torch._foreach_addcdiv,  # MISSING aten::_foreach_addcdiv.Scalar
    torch._foreach_addcdiv_,  # MISSING aten::_foreach_addcdiv_.Scalar
    torch._foreach_addcmul,  # MISSING aten::_foreach_addcmul.Scalar
    torch._foreach_addcmul_,  # MISSING aten::_foreach_addcmul_.Scalar
    torch._foreach_asin,  # MISSING aten::_foreach_asin
    torch._foreach_asin_,  # MISSING aten::_foreach_asin_
    torch._foreach_atan,  # MISSING aten::_foreach_atan
    torch._foreach_atan_,  # MISSING aten::_foreach_atan_
    torch._foreach_ceil,  # MISSING aten::_foreach_ceil
    torch._foreach_ceil_,  # MISSING aten::_foreach_ceil_
    torch._foreach_cos,  # MISSING aten::_foreach_cos
    torch._foreach_cos_,  # MISSING aten::_foreach_cos_
    torch._foreach_cosh,  # MISSING aten::_foreach_cosh
    torch._foreach_cosh_,  # MISSING aten::_foreach_cosh_
    torch._foreach_div,  # MISSING aten::_foreach_div.Scalar
    torch._foreach_div_,  # MISSING aten::_foreach_div_.ScalarList
    torch._foreach_erf,  # MISSING aten::_foreach_erf
    torch._foreach_erf_,  # MISSING aten::_foreach_erf_
    torch._foreach_erfc,  # MISSING aten::_foreach_erfc
    torch._foreach_erfc_,  # MISSING aten::_foreach_erfc_
    torch._foreach_exp,  # MISSING aten::_foreach_exp
    torch._foreach_exp_,  # MISSING aten::_foreach_exp_
    torch._foreach_expm1,  # MISSING aten::_foreach_expm1
    torch._foreach_expm1_,  # MISSING aten::_foreach_expm1_
    torch._foreach_floor,  # MISSING aten::_foreach_floor
    torch._foreach_floor_,  # MISSING aten::_foreach_floor_
    torch._foreach_frac,  # MISSING aten::_foreach_frac
    torch._foreach_frac_,  # MISSING aten::_foreach_frac_
    torch._foreach_log,  # MISSING aten::_foreach_log
    torch._foreach_log10,  # MISSING aten::_foreach_log10
    torch._foreach_log10_,  # MISSING aten::_foreach_log10_
    torch._foreach_log1p,  # MISSING aten::_foreach_log1p
    torch._foreach_log1p_,  # MISSING aten::_foreach_log1p_
    torch._foreach_log2,  # MISSING aten::_foreach_log2
    torch._foreach_log2_,  # MISSING aten::_foreach_log2_
    torch._foreach_log_,  # MISSING aten::_foreach_log_
    torch._foreach_maximum,  # MISSING aten::_foreach_maximum.List
    torch._foreach_minimum,  # MISSING aten::_foreach_minimum.List
    torch._foreach_mul,  # MISSING aten::_foreach_mul.Scalar
    torch._foreach_mul_,  # MISSING aten::_foreach_mul_.ScalarList
    torch._foreach_neg,  # MISSING aten::_foreach_neg
    torch._foreach_neg_,  # MISSING aten::_foreach_neg_
    torch._foreach_norm,  # MISSING aten::_foreach_norm.Scalar
    torch._foreach_reciprocal,  # MISSING aten::_foreach_reciprocal
    torch._foreach_reciprocal_,  # MISSING aten::_foreach_reciprocal_
    torch._foreach_round,  # MISSING aten::_foreach_round
    torch._foreach_round_,  # MISSING aten::_foreach_round_
    torch._foreach_sigmoid,  # MISSING aten::_foreach_sigmoid
    torch._foreach_sigmoid_,  # MISSING aten::_foreach_sigmoid_
    torch._foreach_sin,  # MISSING aten::_foreach_sin
    torch._foreach_sin_,  # MISSING aten::_foreach_sin_
    torch._foreach_sinh,  # MISSING aten::_foreach_sinh
    torch._foreach_sinh_,  # MISSING aten::_foreach_sinh_
    torch._foreach_sqrt,  # MISSING aten::_foreach_sqrt
    torch._foreach_sqrt_,  # MISSING aten::_foreach_sqrt_
    torch._foreach_sub,  # MISSING aten::_foreach_sub.Scalar
    torch._foreach_sub_,  # MISSING aten::_foreach_sub_.ScalarList
    torch._foreach_tan,  # MISSING aten::_foreach_tan
    torch._foreach_tan_,  # MISSING aten::_foreach_tan_
    torch._foreach_tanh,  # MISSING aten::_foreach_tanh
    torch._foreach_tanh_,  # MISSING aten::_foreach_tanh_
    torch._foreach_trunc,  # MISSING aten::_foreach_trunc
    torch._foreach_trunc_,  # MISSING aten::_foreach_trunc_
    torch._foreach_zero_,  # MISSING aten::_foreach_zero_
    torch._fused_moving_avg_obs_fq_helper,  # MISSING aten::_fused_moving_avg_obs_fq_helper
    torch._make_per_tensor_quantized_tensor,  # MISSING aten::_make_per_tensor_quantized_tensor
    torch._masked_softmax,  # MISSING aten::_masked_softmax
    torch._sample_dirichlet,  # MISSING aten::_sample_dirichlet
    torch._standard_gamma,  # MISSING aten::_standard_gamma
    torch._unique,  # MISSING aten::_unique
    torch._unique2,  # MISSING aten::_unique2
    torch.addbmm,  # MISSING aten::addbmm
    torch.angle,  # MISSING aten::angle
    torch.batch_norm,  # MISSING aten::native_batch_norm
    torch.bernoulli,  # MISSING aten::bernoulli.out
    torch.bincount,  # MISSING aten::bincount
    torch.binomial,  # MISSING aten::binomial
    torch.bucketize,  # MISSING aten::bucketize.Tensor
    torch.cholesky,  # MISSING aten::cholesky
    torch.cholesky_inverse,  # MISSING aten::cholesky_inverse
    torch.cholesky_solve,  # MISSING aten::_cholesky_solve_helper
    torch.clip,  # MISSING aten::clamp.Tensor
    torch.combinations,  # MISSING aten::masked_select
    torch.complex,  # MISSING aten::complex.out
    torch.conj_physical,  # MISSING aten::conj_physical.out
    torch.corrcoef,  # MISSING aten::_local_scalar_dense
    torch.count_nonzero,  # MISSING aten::count_nonzero.dim_IntList
    torch.cov,  # MISSING aten::_local_scalar_dense
    torch.cummax,  # MISSING aten::_cummax_helper
    torch.cummin,  # MISSING aten::_cummin_helper
    torch.det,  # MISSING aten::_det_lu_based_helper
    torch.diag,  # MISSING aten::diag.out
    torch.diagflat,  # MISSING aten::diag.out
    torch.dot,  # MISSING aten::dot
    torch.eig,  # MISSING aten::_local_scalar_dense
    torch.equal,  # MISSING aten::equal
    torch.eye,  # MISSING aten::eye.m_out
    torch.fake_quantize_per_channel_affine,  # MISSING aten::fake_quantize_per_channel_affine_cachemask
    torch.fake_quantize_per_tensor_affine,  # MISSING aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams
    torch.fft.fft,  # MISSING aten::_fft_r2c
    torch.fft.fft2,  # MISSING aten::_fft_c2c
    torch.fft.fftn,  # MISSING aten::_fft_c2c
    torch.fft.fftshift,  # MISSING aten::roll
    torch.fft.hfft2,  # MISSING aten::_fft_c2c
    torch.fft.hfftn,  # MISSING aten::_fft_c2c
    torch.fft.ifft,  # MISSING aten::_fft_r2c
    torch.fft.ifft2,  # MISSING aten::_fft_c2c
    torch.fft.ifftn,  # MISSING aten::_fft_c2c
    torch.fft.ifftshift,  # MISSING aten::roll
    torch.fft.ihfft,  # MISSING aten::_fft_r2c
    torch.fft.ihfft2,  # MISSING aten::_fft_r2c
    torch.fft.ihfftn,  # MISSING aten::_fft_r2c
    torch.fft.irfft,  # MISSING aten::_fft_c2r
    torch.fft.irfft2,  # MISSING aten::_fft_c2r
    torch.fft.irfftn,  # MISSING aten::_fft_c2r
    torch.fft.rfft,  # MISSING aten::_fft_r2c
    torch.fft.rfft2,  # MISSING aten::_fft_r2c
    torch.fft.rfftn,  # MISSING aten::_fft_r2c
    torch.flip,  # MISSING aten::flip
    torch.fliplr,  # MISSING aten::flip
    torch.flipud,  # MISSING aten::flip
    torch.floor_divide,  # MISSING aten::floor_divide
    torch.frexp,  # MISSING aten::frexp.Tensor_out
    torch.functional.cdist,  # MISSING aten::_cdist_forward
    torch.functional.einsum,  # MISSING aten::dot
    torch.functional.istft,  # MISSING aten::view_as_complex
    torch.functional.pca_lowrank,  # MISSING aten::_linalg_qr_helper
    torch.functional.stft,  # MISSING aten::_fft_r2c
    torch.functional.svd_lowrank,  # MISSING aten::_linalg_qr_helper
    torch.functional.tensordot,  # MISSING aten::tensordot.out
    torch.functional.unique,  # MISSING aten::_unique2
    torch.functional.unique_consecutive,  # MISSING aten::unique_consecutive
    torch.fused_moving_avg_obs_fake_quant,  # MISSING aten::_fused_moving_avg_obs_fq_helper
    torch.geqrf,  # MISSING aten::geqrf
    torch.group_norm,  # MISSING aten::native_batch_norm
    torch.histc,  # MISSING aten::histc.out
    torch.histogram,  # MISSING aten::histogram.bin_ct
    torch.histogramdd,  # MISSING aten::_histogramdd_bin_edges
    torch.inner,  # MISSING aten::tensordot.out
    torch.inverse,  # MISSING aten::_local_scalar_dense
    torch.isnan,  # MISSING aten::isnan
    torch.kthvalue,  # MISSING aten::kthvalue.values
    torch.layer_norm,  # MISSING aten::native_batch_norm
    torch.linalg.cholesky,  # MISSING aten::linalg_cholesky_ex
    torch.linalg.cholesky_ex,  # MISSING aten::linalg_cholesky_ex
    torch.linalg.det,  # MISSING aten::_det_lu_based_helper
    torch.linalg.eig,  # MISSING aten::linalg_eig
    torch.linalg.eig,  # MISSING aten::linalg_eig.out
    torch.linalg.eigh,  # MISSING aten::linalg_eigh
    torch.linalg.eigvals,  # MISSING aten::linalg_eig
    torch.linalg.eigvalsh,  # MISSING aten::linalg_eigh
    torch.linalg.eigvalsh,  # MISSING aten::linalg_eigvalsh.out
    torch.linalg.householder_product,  # MISSING aten::linalg_householder_product
    torch.linalg.inv,  # MISSING aten::_local_scalar_dense
    torch.linalg.lstsq,  # MISSING aten::linalg_lstsq.out
    torch.linalg.lu_factor,  # MISSING aten::_local_scalar_dense
    torch.linalg.matmul,  # MISSING aten::dot
    torch.linalg.matrix_exp,  # MISSING aten::linalg_matrix_exp
    torch.linalg.matrix_power,  # MISSING aten::_local_scalar_dense
    torch.linalg.matrix_power,  # MISSING aten::eye.m_out
    torch.linalg.pinv,  # MISSING aten::where.self
    torch.linalg.qr,  # MISSING aten::_linalg_qr_helper
    torch.linalg.slogdet,  # MISSING aten::linalg_slogdet
    torch.linalg.solve,  # MISSING aten::linalg_solve
    torch.linalg.solve_triangular,  # MISSING aten::linalg_solve_triangular
    torch.linalg.tensorinv,  # MISSING aten::_local_scalar_dense
    torch.linalg.tensorsolve,  # MISSING aten::linalg_solve
    torch.logcumsumexp,  # MISSING aten::_logcumsumexp
    torch.logdet,  # MISSING aten::_local_scalar_dense
    torch.logical_and,  # MISSING aten::logical_and.out
    torch.logical_not,  # MISSING aten::logical_not.out
    torch.logical_or,  # MISSING aten::logical_or.out
    torch.logical_xor,  # MISSING aten::logical_xor.out
    torch.logit,  # MISSING aten::logit
    torch.lstsq,  # MISSING aten::lstsq
    torch.lu_solve,  # MISSING aten::lu_solve
    torch.masked_select,  # MISSING aten::masked_select
    torch.matmul,  # MISSING aten::dot
    torch.matrix_exp,  # MISSING aten::linalg_matrix_exp
    torch.matrix_power,  # MISSING aten::eye.m_out
    torch.matrix_rank,  # MISSING aten::linalg_eigvalsh.out
    torch.median,  # MISSING aten::median
    torch.median,  # MISSING aten::median.dim_values
    torch.mode,  # MISSING aten::mode
    torch.multinomial,  # MISSING aten::multinomial
    torch.mvlgamma,  # MISSING aten::_local_scalar_dense
    torch.nan_to_num,  # MISSING aten::nan_to_num.out
    torch.nanmean,  # MISSING aten::logical_not.out
    torch.nanmedian,  # MISSING aten::nanmedian
    torch.nanmedian,  # MISSING aten::nanmedian.dim_values
    torch.nansum,  # MISSING aten::nansum
    torch.nn.functional.adaptive_avg_pool1d,  # MISSING aten::_adaptive_avg_pool2d
    torch.nn.functional.adaptive_avg_pool2d,  # MISSING aten::_adaptive_avg_pool2d
    torch.nn.functional.adaptive_avg_pool3d,  # MISSING aten::_adaptive_avg_pool3d
    torch.nn.functional.batch_norm,  # MISSING aten::native_batch_norm
    torch.nn.functional.binary_cross_entropy,  # MISSING aten::binary_cross_entropy
    torch.nn.functional.channel_shuffle,  # MISSING aten::channel_shuffle
    torch.nn.functional.cosine_embedding_loss,  # MISSING aten::clamp_min.out
    torch.nn.functional.cross_entropy,  # MISSING aten::_local_scalar_dense
    torch.nn.functional.cross_entropy,  # MISSING aten::nll_loss2d_forward
    torch.nn.functional.ctc_loss,  # MISSING aten::_ctc_loss
    torch.nn.functional.embedding_bag,  # MISSING aten::_embedding_bag
    torch.nn.functional.fold,  # MISSING aten::col2im
    torch.nn.functional.gaussian_nll_loss,  # MISSING aten::_local_scalar_dense
    torch.nn.functional.grid_sample,  # MISSING aten::grid_sampler_2d
    torch.nn.functional.group_norm,  # MISSING aten::native_batch_norm
    torch.nn.functional.hardswish,  # MISSING aten::hardswish
    torch.nn.functional.hardtanh,  # MISSING aten::hardtanh
    torch.nn.functional.hinge_embedding_loss,  # MISSING aten::clamp_min.out
    torch.nn.functional.huber_loss,  # MISSING aten::huber_loss
    torch.nn.functional.instance_norm,  # MISSING aten::native_batch_norm
    torch.nn.functional.kl_div,  # MISSING aten::where.self
    torch.nn.functional.layer_norm,  # MISSING aten::native_batch_norm
    torch.nn.functional.logsigmoid,  # MISSING aten::log_sigmoid_forward
    torch.nn.functional.max_pool3d,  # MISSING aten::max_pool3d_with_indices
    torch.nn.functional.max_pool3d_with_indices,  # MISSING aten::max_pool3d_with_indices
    torch.nn.functional.max_unpool1d,  # MISSING aten::max_unpool2d
    torch.nn.functional.max_unpool2d,  # MISSING aten::max_unpool2d
    torch.nn.functional.max_unpool3d,  # MISSING aten::max_unpool3d
    torch.nn.functional.multi_head_attention_forward,  # MISSING aten::logical_or.out
    torch.nn.functional.multi_margin_loss,  # MISSING aten::multi_margin_loss
    torch.nn.functional.multilabel_margin_loss,  # MISSING aten::multilabel_margin_loss_forward
    torch.nn.functional.multilabel_soft_margin_loss,  # MISSING aten::log_sigmoid_forward
    torch.nn.functional.nll_loss,  # MISSING aten::nll_loss2d_forward
    torch.nn.functional.one_hot,  # MISSING aten::_local_scalar_dense
    torch.nn.functional.pdist,  # MISSING aten::_pdist_forward
    torch.nn.functional.prelu,  # MISSING aten::prelu
    torch.nn.functional.relu,  # MISSING aten::relu
    torch.nn.functional.relu6,  # MISSING aten::hardtanh
    torch.nn.functional.rrelu,  # MISSING aten::rrelu_with_noise
    torch.nn.functional.unfold,  # MISSING aten::im2col
    torch.nonzero,  # MISSING aten::nonzero
    torch.normal,  # MISSING aten::_local_scalar_dense
    torch.orgqr,  # MISSING aten::linalg_householder_product
    torch.ormqr,  # MISSING aten::ormqr
    torch.pinverse,  # MISSING aten::where.self
    torch.poisson,  # MISSING aten::poisson
    torch.polar,  # MISSING aten::polar.out
    torch.prod,  # MISSING aten::prod
    torch.qr,  # MISSING aten::_linalg_qr_helper
    torch.quantize_per_channel,  # MISSING aten::quantize_per_channel
    torch.quantize_per_tensor,  # MISSING aten::quantize_per_tensor
    torch.quantize_per_tensor_dynamic,  # MISSING aten::quantize_per_tensor_dynamic
    torch.relu,  # MISSING aten::relu
    torch.repeat_interleave,  # MISSING aten::repeat_interleave.Tensor
    torch.rnn_relu,  # MISSING aten::relu
    torch.rnn_relu_cell,  # MISSING aten::relu
    torch.roll,  # MISSING aten::roll
    torch.rot90,  # MISSING aten::flip
    torch.rsub,  # MISSING aten::rsub.Tensor
    torch.searchsorted,  # MISSING aten::searchsorted.Tensor
    torch.slogdet,  # MISSING aten::linalg_slogdet
    torch.solve,  # MISSING aten::_solve_helper
    torch.special.logit,  # MISSING aten::logit
    torch.special.logsumexp,  # MISSING aten::abs.out
    torch.special.multigammaln,  # MISSING aten::_local_scalar_dense
    torch.square,  # MISSING aten::square.out
    torch.std,  # MISSING aten::std.correction
    torch.std_mean,  # MISSING aten::std_mean.correction
    torch.symeig,  # MISSING aten::_symeig_helper
    torch.take,  # MISSING aten::take
    torch.threshold,  # MISSING aten::_local_scalar_dense
    torch.trace,  # MISSING aten::trace
    torch.var,  # MISSING aten::var.correction
    torch.var_mean,  # MISSING aten::var_mean.correction
    torch.vdot,  # MISSING aten::vdot
    torch.where,  # MISSING aten::where.self
    torch.quantile,  # MISSING aten::isnan
    torch.nanquantile,  # MISSING aten::isnan
}

# Only some overloads/configurations are covered with meta tensors,
# so we can't use these to toggle expected failure.  Try to prioritize these
overload_exclude_set = {
    torch.clamp,  # MISSING aten::clamp.Tensor
    torch.nn.functional.interpolate,  # MISSING aten::upsample_nearest3d.vec
    torch.nn.functional.upsample_nearest,  # MISSING aten::upsample_nearest3d.vec
    torch.nn.functional.pad,  # MISSING aten::reflection_pad2d
    torch.remainder,  # MISSING aten::remainder.Scalar_Tensor
    torch.linalg.matrix_rank,  # MISSING aten::linalg_eigh
    torch.diff,  # MISSING aten::logical_xor.out
}

# These are fine in OpInfo tests, but triggered errors in full test suite
# crossref testing, which means there is probably not enough coverage from
# OpInfo.  Patch in https://github.com/pytorch/pytorch/pull/75994 and find
# out where these fails come from.
suspicious_exclude_set = {
    torch.add,  # MISSING aten::_local_scalar_dense
    torch.cat,  # MISSING aten::_local_scalar_dense
    torch.cumprod,  # MISSING aten::logical_and.out
    torch.cumsum,  # MISSING aten::_local_scalar_dense
    torch.functional.norm,  # MISSING aten::isnan

    # RuntimeError: Expected 3D or 4D (batch mode) tensor with optional 0 dim
    # batch size for input, but got:[1, 1, 0]
    # in test_nn.py TestNNDeviceTypeCPU.test_max_pool1d_corner_cases_cpu_float64
    torch.nn.functional.max_pool1d,

    # Factory functions need tricky kwarg handling
    torch.zeros_like,
}

# These also are known to not work, but they fail in a more special way
# than the regular "Meta not implemented for aten op" way
meta_exclude_set |= {
    # Convolutions have a special error message
    torch.nn.functional.conv1d,
    torch.nn.functional.conv2d,
    torch.nn.functional.conv3d,
    torch.nn.functional.conv_transpose1d,
    torch.nn.functional.conv_transpose2d,
    torch.nn.functional.conv_transpose3d,
    # complex stuff handle it specially
    torch.view_as_complex,
    torch.view_as_real,
    # These operators happen very frequently, although they should
    # work with meta we intentionally don't test them to speed
    # up the test suite
    torch.Tensor.__getitem__,
    torch.Tensor.__rsub__,
    torch.Tensor.__setitem__,
    torch.Tensor.add,
    torch.Tensor.add_,
    torch.Tensor.clone,
    torch.Tensor.detach,
    torch.Tensor.div,
    torch.Tensor.gt,
    torch.Tensor.lt,
    torch.Tensor.mul,
    torch.Tensor.reshape,
    torch.Tensor.sub,
    torch.Tensor.sum,
    torch.rand,
    # These correctly report NotImplemented but they don't print
    # correctly from resolve_name
    torch.ops.quantized.linear_dynamic,
    torch._VF.unique_dim,
    torch._C._nn.binary_cross_entropy,
    torch._C._nn.adaptive_avg_pool2d,
    torch._C._nn._test_optional_filled_intlist,
    torch._C._nn._test_optional_floatlist,
    torch._C._nn._test_optional_intlist,
    # Meta tensors don't support storage Python bindings at the
    # moment, to be fixed
    torch.Tensor.storage,
    torch.Tensor.storage_type,
    torch.Tensor.share_memory_,
    # Weird stuff that hypothetically should work but it's weird
    torch._make_dual,
    torch._unpack_dual,  # fails because we don't preserve forward ad tangent in test code
    # These functions cannot, even in principle, be implemented on meta
    # tensors (because they involve accessing data somehow), so don't test
    # them.
    torch.Tensor.__bool__,
    torch.Tensor.__float__,
    torch.Tensor.__int__,
    torch.Tensor.__complex__,
    torch.Tensor.__index__,
    torch.Tensor.__contains__,
    torch.Tensor.cpu,
    torch.Tensor.to,
    torch.Tensor.tolist,
    torch.Tensor.unbind,
    torch.Tensor.item,
    torch.Tensor.is_nonzero,
    torch.Tensor.copy_,
    torch.Tensor.numpy,
    torch.Tensor.allclose,
    torch.Tensor.argwhere,
    torch.allclose,
    torch.argwhere,
    torch.Tensor.__array__,  # doesn't raise NotImplementedError
    torch.Tensor.__dlpack_device__,  # doesn't raise NotImplementedError
    torch.Tensor.__dlpack__,  # doesn't raise NotImplementedError
    torch.to_dlpack,  # doesn't raise NotImplementedError
    # Utility functions that get frequently invoked; don't test
    torch.Tensor.__format__,
    torch.Tensor.__repr__,
    # These are getters/setters for properties on tensors; it's not
    # really useful to test meta tensors on them
    torch.Tensor.device.__get__,
    torch.Tensor.dtype.__get__,
    torch.Tensor.grad.__get__,
    torch.Tensor.grad.__set__,
    torch.Tensor.is_sparse.__get__,
    torch.Tensor.layout.__get__,
    torch.Tensor.shape.__get__,
    torch.Tensor.requires_grad.__get__,
    torch.Tensor.requires_grad.__set__,
    torch.Tensor.data.__get__,
    torch.Tensor.data.__set__,
    torch.Tensor._base.__get__,
    torch.Tensor.is_shared,
    torch.Tensor.imag.__get__,
    torch.Tensor.real.__get__,
    torch.Tensor.__setstate__,
    torch.Tensor.is_complex,
    torch.Tensor.is_floating_point,
    torch.Tensor.numel,
    torch.Tensor.requires_grad_,
    torch.Tensor.size,
    # These perturb RNG and can cause tests to fail, so don't run
    # them (TODO: this is not a complete list)
    torch.randint,
    torch.randn,
    # Indirect use of conjugate fallback
    torch.fft.hfft,
    # These don't raise NotImplementedError, which suggests something
    # is wrong with how they're registered with the dispatcher
    torch.fbgemm_pack_gemm_matrix_fp16,
    torch.fbgemm_pack_quantized_matrix,
    torch.fbgemm_linear_fp16_weight,
    torch._empty_per_channel_affine_quantized,
    torch.fbgemm_linear_int8_weight,
    torch._grid_sampler_2d_cpu_fallback,  # WAT
    torch._nnpack_spatial_convolution,
    torch.lstm,
    torch.Tensor.conj_physical_,
    torch.rnn_tanh,
    torch.fbgemm_linear_quantize_weight,
    torch._reshape_from_tensor,
    torch.gru,
    torch.Tensor.unflatten,
    torch._saturate_weight_to_fp16,
    torch.choose_qparams_optimized,
    torch._validate_sparse_coo_tensor_args,
    torch.sparse.mm,
    torch.Tensor.new,
    torch.Tensor.resize,  # WTF is this
    torch._sobol_engine_initialize_state_,
    torch._sobol_engine_draw,
    torch._sobol_engine_scramble_,
    torch._sobol_engine_ff_,
    torch.tensor_split,
    torch.Tensor.tensor_split,
    torch._pack_padded_sequence,
    torch._pad_packed_sequence,
    torch.sparse_coo_tensor,
    torch.linalg.ldl_factor,
    torch._index_reduce,
    # IndexError: select() cannot be applied to a 0-dim tensor.
    # e.g. test_fn_fwgrad_bwgrad_index_add_cpu_complex128 (__main__.TestGradientsCPU)
    torch.index_add,
    torch.Tensor.index_add,
    torch.Tensor.index_add_,
    # Can't copy out of meta tensor
    torch.linalg.eigvals,
    torch.linalg.lu_factor,
    torch.nn.functional.ctc_loss,
    # Our conversion to meta is not accurate enough (doesn't
    # preserve storage_offset, e.g.)
    torch.Tensor.as_strided,
    # This one segfaults when you call it
    torch.Tensor.type,
    # We don't clone autograd history, so this will generally not work
    torch.autograd.grad,
    torch.Tensor.backward,
    torch.Tensor.__deepcopy__,
    # Don't do factories
    torch.ones,
    torch.full,
    torch.empty,
    torch.randperm,
    torch.logspace,
    torch.zeros,
    torch.arange,
    torch.vander,
    torch.as_tensor,
    torch.tensor,
    torch.randn_like,
    torch.sparse_csr_tensor,
    torch._sparse_coo_tensor_unsafe,
    torch._sparse_csr_tensor_unsafe,
    torch._validate_sparse_csr_tensor_args,
}

# This is a __torch_function__ mode that, when enabled, interposes every
# Torch API call and runs the operator as normal, and then reruns it
# with meta inputs, and then checks that everything about the output agrees.
# Most of the logic deals with faithfully replicating the original tensor
# as a meta tensor, which is nontrivial because there are a lot of subsystems
# that may potentially be exercised.
#
# That being said, this class is a little overkill for what it is doing in
# this test file (since I could have just inlined __torch_function__ on the
# OpInfo call, and OpInfos generally have very regular inputs), but it will be
# useful for more comprehensive testing e.g., as seen in
# https://github.com/pytorch/pytorch/pull/75994
class MetaCrossRefMode(torch.overrides.TorchFunctionMode):
    test_case: TestCase
    run_excludes_anyway: bool

    def __init__(self, test_case, *, run_excludes_anyway):
        self.test_case = test_case
        self.run_excludes_anyway = run_excludes_anyway

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        hit = 0
        miss = 0

        # Doesn't actually return a storage
        @functools.lru_cache(None)
        def meta_storage(s):
            return torch.empty(s.size(), dtype=s.dtype, device='meta')

        def safe_is_leaf(t):
            try:
                return t.is_leaf
            except RuntimeError:
                # inference mode can trigger this
                return False

        @functools.lru_cache(None)
        def meta_tensor(t):
            with torch.inference_mode(t.is_inference()):
                s = meta_storage(t.storage())
                is_leaf = safe_is_leaf(t)
                if is_leaf or not t._is_view():
                    r = torch.empty(
                        (0,), dtype=t.dtype, device='meta'
                    )
                    r.set_(s, t.storage_offset(), t.size(), t.stride())
                    r.requires_grad = t.requires_grad
                    if not is_leaf and t.requires_grad:
                        with torch.enable_grad():
                            r = r.clone()
                else:
                    base = torch.empty(
                        (0,), dtype=t.dtype, device='meta'
                    )
                    base.set_(s, 0, s.size(), (1,))
                    base.requires_grad = t.requires_grad
                    with torch.enable_grad():
                        if t._is_view() and not safe_is_leaf(t._base):
                            base = base.clone()
                        r = base.as_strided(t.size(), t.stride(), t.storage_offset())
                torch._C._set_conj(r, t.is_conj())
                torch._C._set_neg(r, t.is_neg())
            return r

        def to_meta(t):
            nonlocal hit, miss
            # TODO: zero tensors?  We appear to have eliminated them by
            # excluding complex for now
            if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
                if any([
                    t.is_sparse_csr, t.is_sparse, t.is_mkldnn, t.is_quantized,
                    t.is_nested, torch._is_functional_tensor(t),
                    # these are supported in meta conversion but the fallbacks
                    # don't work
                    t.is_neg(), t.is_conj(),
                    # conjugate fallback does not support meta tensors
                    t.dtype in (torch.complex128, torch.complex64),
                ]):
                    # TODO: sparse should support meta
                    # NB technically to('meta') does work but our logging
                    # instrumentation will see the meta conversions and the
                    # tests all break so we just exclude this.  In any case
                    # the to conversion isn't really right anyhow.
                    miss += 1
                    return t
                elif any([
                    t.device.type in ("lazy", "meta"), t.is_complex(),
                    # We need a way to test if a tensor is batched but there
                    # is no official APi to do it
                    # torch._C._is_batched(t),
                ]):
                    # TODO: this stuff should support storage
                    # (well, maybe not batched)
                    hit += 1
                    return t.to("meta")
                else:
                    hit += 1
                    r = meta_tensor(t)
                    if type(t) is torch.nn.Parameter:
                        r = torch.nn.Parameter(r, requires_grad=r.requires_grad)
                    return r
            elif torch.overrides.is_tensor_like(t):
                # Blindly converting tensor subclasses to meta can cause
                # unpredictable problems; e.g., FX tests will trace meta
                # tensors into their trace / some subclasses don't correctly
                # support meta.  Trying to YOLO this is more trouble than it's
                # worth.
                miss += 1
                return t
            else:
                # non-Tensor types don't count as hit or miss
                return t

        do_meta = (
            (self.run_excludes_anyway or func not in meta_exclude_set) and
            not torch.jit.is_tracing() and
            not isinstance(func, torch.ScriptMethod)
        )

        if do_meta:
            try:
                meta_args = tree_map(to_meta, args)
                meta_kwargs = tree_map(to_meta, kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"failed to convert args to meta; "
                    f"originally (*{args}, **{kwargs})") from e

        rs = func(*args, **kwargs)

        # TODO: also handle cases where func raise an exception

        # For now, only attempt if we managed to convert all tensor types
        # (if any of them failed, we're in a mixed device situation and
        # this isn't well supported)
        if do_meta and hit > 0 and miss == 0:
            try:
                # suppress warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    meta_rs = func(*meta_args, **meta_kwargs)
            except Exception as e:
                suppress = False
                """
                # This code can be helpful for full crossref test to filter
                # out "pedestrian" omissions
                if isinstance(e, NotImplementedError):
                    m = RE_NOT_IMPLEMENTED_MSG.search(e.args[0])
                    if m and m.group(1) not in ("aten::_efficientzerotensor", "aten::view_as_real"):
                        suppress = True
                """
                if not suppress:
                    raise RuntimeError(f"""\
failed to run: {func}(
    *{meta_args},
    **{meta_kwargs}
  )""") from e
            else:
                def test_assert(cond, msg):
                    if not cond:
                        raise RuntimeError(f"""\
meta disagrees with real impl:
{func}(
    *{meta_args},
    **{meta_kwargs}
) = {meta_r}
{msg}
""")
                flat_meta_rs, _ = tree_flatten(meta_rs)
                flat_rs, _ = tree_flatten(rs)
                self.test_case.assertEqual(len(flat_meta_rs), len(flat_rs))
                for i, meta_r, r in zip(range(len(flat_rs)), flat_meta_rs, flat_rs):
                    if isinstance(r, torch.Tensor):
                        test_assert(isinstance(meta_r, torch.Tensor), f"but real {i}th result is Tensor")
                        test_assert(meta_r.dtype == r.dtype, f"but real dtype was {r.dtype}")
                        test_assert(meta_r.shape == r.shape, f"but real shape was {r.shape}")
                        test_assert(meta_r.stride() == r.stride(), f"but real stride was {r.stride()}")
                        test_assert(
                            meta_r.storage_offset() == r.storage_offset(),
                            f"but real storage_offset was {r.storage_offset()}")
                        test_assert(meta_r.requires_grad == r.requires_grad, f"but real requires_grad was {r.requires_grad}")
                        test_assert(meta_r.is_conj() == r.is_conj(), f"but real is_conj was {r.is_conj()}")
                        test_assert(meta_r.is_neg() == r.is_neg(), f"but real is_neg was {r.is_neg()}")

        return rs

class TestMeta(TestCase):
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(op_db)
    def test_meta(self, device, dtype, op):
        # run the OpInfo sample inputs, cross-referencing them with the
        # meta implementation and check the results are the same.  All
        # the heavy lifting happens in MetaCrossRefMode
        func = op.get_op()

        def do_test(run_excludes_anyway=False):
            samples = op.sample_inputs(device, dtype, requires_grad=False)
            for sample_input in samples:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs
                with push_torch_function_mode(partial(MetaCrossRefMode, self, run_excludes_anyway=run_excludes_anyway)):
                    expected = func(*args, **kwargs)
                    if isinstance(expected, torch.Tensor) and op.supports_out:
                        func(*args, **kwargs, out=expected)

        if func in overload_exclude_set:
            self.skipTest('permanently excluded')
        elif func in meta_exclude_set and dtype not in (torch.complex128, torch.complex64):
            try:
                do_test(run_excludes_anyway=True)
            except Exception:
                pass
            else:
                self.fail('expected failure, but succeeded')
        else:
            do_test()

instantiate_device_type_tests(TestMeta, globals())

if __name__ == "__main__":
    run_tests()
