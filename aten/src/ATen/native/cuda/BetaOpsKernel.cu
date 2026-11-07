#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>

#include <ATen/core/Tensor.h>

#include <ATen/native/Math.h>
#include <ATen/native/BetaOps.h>

namespace at::native {

void betainc_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "betainc_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t x) -> scalar_t {
        return calc_betainc(a, b, x);
    });
  });
}

REGISTER_DISPATCH(betainc_stub, &betainc_kernel_cuda)

extern TORCH_API std::tuple<Tensor, Tensor, Tensor> _special_betainc_partials(const Tensor& a, const Tensor& b, const Tensor& x);

std::tuple<Tensor, Tensor, Tensor> _special_betainc_partials_cuda(
    const Tensor& a, const Tensor& b, const Tensor& x) {
    return _special_betainc_partials(a, b, x);
}

extern TORCH_API std::tuple<Tensor, Tensor, Tensor> _special_betaincinv_partials(const Tensor& a, const Tensor& b, const Tensor& y);

std::tuple<Tensor, Tensor, Tensor> _special_betaincinv_partials_cuda(const Tensor& a, const Tensor& b, const Tensor& y) {
  return _special_betaincinv_partials(a, b, y);
}


} // namespace at::native
