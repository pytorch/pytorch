#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/EnableNamedTensor.h>

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

#define IMPLEMENT_UNARY_OP_PREQUEL(op)                           \
  Tensor& _##op##__cuda(Tensor& self) {                          \
    return legacy::cuda::_th_##op##_out(self, self);         \
  }                                                              \
  Tensor& _##op##_out_cuda(Tensor& result, const Tensor& self) { \
    return legacy::cuda::_th_##op##_out(result, self);       \
  }


IMPLEMENT_UNARY_OP_PREQUEL(acos)
IMPLEMENT_UNARY_OP_PREQUEL(atan)
IMPLEMENT_UNARY_OP_PREQUEL(cos)
IMPLEMENT_UNARY_OP_PREQUEL(cosh)
IMPLEMENT_UNARY_OP_PREQUEL(erf)
IMPLEMENT_UNARY_OP_PREQUEL(erfc)
IMPLEMENT_UNARY_OP_PREQUEL(exp)
IMPLEMENT_UNARY_OP_PREQUEL(reciprocal)
IMPLEMENT_UNARY_OP_PREQUEL(tan)
IMPLEMENT_UNARY_OP_PREQUEL(tanh)

}}
