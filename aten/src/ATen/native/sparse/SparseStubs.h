#pragma once

#include <ATen/native/DispatchStub.h>

namespace at {

class Tensor;

namespace native {

using mul_sparse_sparse_out_fn = void (*)(Tensor& res, const Tensor& x, const Tensor& y);
DECLARE_DISPATCH(mul_sparse_sparse_out_fn, mul_sparse_sparse_out_stub);

}

}
