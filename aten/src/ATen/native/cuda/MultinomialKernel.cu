#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/native/UnaryOps.h>

namespace at { namespace native {

void multinomial_kernel_impl(Tensor& result, const Tensor& self, const int64_t num_samples, const bool replacement, Generator* generator) {
  legacy::cuda::_th_multinomial_out(result, self, num_samples, replacement, generator);
}

REGISTER_DISPATCH(multinomial_stub, &multinomial_kernel_impl);

}}