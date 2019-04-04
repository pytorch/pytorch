#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctions.h>
#include <ATen/native/Cross.h>

namespace at { namespace native {

void cross_kernel_impl(Tensor& result, const Tensor& x1, const Tensor& x2, const int64_t dim) {
  at::legacy::th::_th_cross_kernel_out(result, x1, x2, dim);
}

REGISTER_DISPATCH(cross_stub, &cross_kernel_impl);

}}

