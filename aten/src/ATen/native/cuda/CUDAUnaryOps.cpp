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

<<<<<<< HEAD

IMPLEMENT_UNARY_OP_PREQUEL(atan)
IMPLEMENT_UNARY_OP_PREQUEL(erfc)
IMPLEMENT_UNARY_OP_PREQUEL(exp)
=======
IMPLEMENT_UNARY_OP_PREQUEL(atan, legacy::cuda::_th_atan)
IMPLEMENT_UNARY_OP_PREQUEL(erf,  legacy::cuda::_th_erf)
IMPLEMENT_UNARY_OP_PREQUEL(erfc, legacy::cuda::_th_erfc)
IMPLEMENT_UNARY_OP_PREQUEL(exp,  legacy::cuda::_th_exp)
>>>>>>> Specify _th_ ops in CUDAUnaryOps macros so they are easier to find.

}}
