#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/NamedTensorUtils.h>

namespace at { namespace native {

// These are just forwarding stubs

#define IMPLEMENT_UNARY_OP_PREQUEL(op, _th_op)                   \
  Tensor& _##op##__cuda(Tensor& self) {                          \
    return _th_op##_out(self, self);                             \
  }                                                              \
  Tensor& _##op##_out_cuda(Tensor& result, const Tensor& self) { \
    return _th_op##_out(result, self);                           \
  }

}}
