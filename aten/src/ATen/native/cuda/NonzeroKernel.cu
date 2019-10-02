#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/native/Nonzero.h>

namespace at { namespace native {

void nonzero_kernel_impl(Tensor& subscript, const Tensor& self) {
  legacy::cuda::_th_nonzero_out(subscript, self);
}

REGISTER_DISPATCH(nonzero_stub, &nonzero_kernel_impl);

}}
