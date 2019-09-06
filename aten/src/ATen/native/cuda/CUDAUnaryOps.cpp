#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#ifdef BUILD_NAMEDTENSOR
#include <ATen/NamedTensorUtils.h>
#endif

namespace at { namespace native {

Tensor& _clamp__cuda(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  return _clamp_out_cuda(self, self, min, max);
}

Tensor& _clamp_out_cuda(
    Tensor& result,
    const Tensor& self,
    optional<Scalar> min,
    optional<Scalar> max) {
  if (min && max) {
    legacy::cuda::_th_clamp_out(result, self, *min, *max);
  } else if (max) {
    legacy::cuda::_th_clamp_max_out(result, self, *max);
  } else if (min) {
    legacy::cuda::_th_clamp_min_out(result, self, *min);
  } else {
    AT_ERROR("At least one of 'min' or 'max' must not be None");
  }
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(result, self);
#endif
  return result;
}

Tensor& _clamp_max__cuda(Tensor& self, Scalar max) {
  return legacy::cuda::_th_clamp_max_out(self, self, max);
}

Tensor& _clamp_max_out_cuda(Tensor& result, const Tensor& self, Scalar max) {
  legacy::cuda::_th_clamp_max_out(result, self, max);
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(result, self);
#endif
  return result;
}

Tensor& _clamp_min__cuda(Tensor& self, Scalar min) {
  return legacy::cuda::_th_clamp_min_out(self, self, min);
}

Tensor& _clamp_min_out_cuda(Tensor& result, const Tensor& self, Scalar min) {
  legacy::cuda::_th_clamp_min_out(result, self, min);
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(result, self);
#endif
  return result;
}

// These are just forwarding stubs

#define IMPLEMENT_UNARY_OP_PREQUEL(op, _th_op)                   \
  Tensor& _##op##__cuda(Tensor& self) {                          \
    return _th_op##_out(self, self);                             \
  }                                                              \
  Tensor& _##op##_out_cuda(Tensor& result, const Tensor& self) { \
    return _th_##op##_out(result, self);                         \
  }


IMPLEMENT_UNARY_OP_PREQUEL(abs, legacy::cuda::_th_abs_out)
IMPLEMENT_UNARY_OP_PREQUEL(acos, legacy::cuda::_th_acos_out)
IMPLEMENT_UNARY_OP_PREQUEL(asin, legacy::cuda::_th_asin_out)
IMPLEMENT_UNARY_OP_PREQUEL(atan, legacy::cuda::_th_atan_out)
IMPLEMENT_UNARY_OP_PREQUEL(ceil, legacy::cuda::_th_ceil_out)
IMPLEMENT_UNARY_OP_PREQUEL(cos, legacy::cuda::_th_cos_out)
IMPLEMENT_UNARY_OP_PREQUEL(cosh, legacy::cuda::_th_cosh_out)
IMPLEMENT_UNARY_OP_PREQUEL(erf, legacy::cuda::_th_erf_out)
IMPLEMENT_UNARY_OP_PREQUEL(erfc, legacy::cuda::_th_erfc_out)
IMPLEMENT_UNARY_OP_PREQUEL(exp, legacy::cuda::_th_exp_out)
IMPLEMENT_UNARY_OP_PREQUEL(expm1, legacy::cuda::_th_expm1_out)
IMPLEMENT_UNARY_OP_PREQUEL(frac, legacy::cuda::_th_frac_out)
IMPLEMENT_UNARY_OP_PREQUEL(floor, legacy::cuda::_th_floor_out)
IMPLEMENT_UNARY_OP_PREQUEL(log, legacy::cuda::_th_log_out)
IMPLEMENT_UNARY_OP_PREQUEL(log10, legacy::cuda::_th_log10_out)
IMPLEMENT_UNARY_OP_PREQUEL(log1p, legacy::cuda::_th_log1p_out)
IMPLEMENT_UNARY_OP_PREQUEL(log2, legacy::cuda::_th_log2_out)
IMPLEMENT_UNARY_OP_PREQUEL(reciprocal, legacy::cuda::_th_reciprocal_out)
IMPLEMENT_UNARY_OP_PREQUEL(round, legacy::cuda::_th_round_out)
IMPLEMENT_UNARY_OP_PREQUEL(rsqrt, legacy::cuda::_th_rqsrt_out)
IMPLEMENT_UNARY_OP_PREQUEL(sigmoid, legacy::cuda::_th_sigmoid_out)
IMPLEMENT_UNARY_OP_PREQUEL(sin, legacy::cuda::_th_sin_out)
IMPLEMENT_UNARY_OP_PREQUEL(sinh, legacy::cuda::_th_sinh_out)
IMPLEMENT_UNARY_OP_PREQUEL(sqrt, legacy::cuda::_th_sqrt_out)
IMPLEMENT_UNARY_OP_PREQUEL(tan, legacy::cuda::_th_tan_out)
IMPLEMENT_UNARY_OP_PREQUEL(tanh, legacy::cuda::_th_tanh_out)
IMPLEMENT_UNARY_OP_PREQUEL(trunc, legacy::cuda::_th_trunc_out)

}}
