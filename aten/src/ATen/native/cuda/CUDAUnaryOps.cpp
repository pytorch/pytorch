#include "ATen/ATen.h"

namespace at { namespace native {

// These are just forwarding stubs

#define IMPLEMENT_UNARY_OP_PREQUEL(op)                           \
  Tensor& _##op##__cuda(Tensor& self) {                          \
    return at::_##op##_out(self, self);                          \
  }                                                              \
  Tensor& _##op##_out_cuda(Tensor& result, const Tensor& self) { \
    return at::_##op##_out(result, self);                        \
  }

IMPLEMENT_UNARY_OP_PREQUEL(abs)
IMPLEMENT_UNARY_OP_PREQUEL(ceil)
IMPLEMENT_UNARY_OP_PREQUEL(cos)
IMPLEMENT_UNARY_OP_PREQUEL(exp)
IMPLEMENT_UNARY_OP_PREQUEL(floor)
IMPLEMENT_UNARY_OP_PREQUEL(log)
IMPLEMENT_UNARY_OP_PREQUEL(round)
IMPLEMENT_UNARY_OP_PREQUEL(sin)
IMPLEMENT_UNARY_OP_PREQUEL(sqrt)
IMPLEMENT_UNARY_OP_PREQUEL(trunc)

}}
