#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>


namespace at { namespace native {

void and_kernel_cuda(TensorIterator& iter) {
  gpu_reduce_kernel<uint8_t, uint8_t>(
    iter, func_wrapper<uint8_t> ([]GPU_LAMBDA(uint8_t a, uint8_t b) -> uint8_t {
      return a && b;
    }), true);
}

void or_kernel_cuda(TensorIterator& iter) {
  gpu_reduce_kernel<uint8_t, uint8_t>(
    iter, func_wrapper<uint8_t> ([]GPU_LAMBDA(uint8_t a, uint8_t b) -> uint8_t {
      return a || b;
    }), false);
}

REGISTER_DISPATCH(and_stub, &and_kernel_cuda);
REGISTER_DISPATCH(or_stub, &or_kernel_cuda);

bool cuda_equal(const Tensor& self, const Tensor &src) {
  if (!at::namedinference::are_names_equal(
        self.unsafeGetTensorImpl(), src.unsafeGetTensorImpl())) {
    return false;
  }
  at::NoNamesGuard guard;
  TORCH_CHECK(self.device() == src.device(), "Cannot compare two tensors on "
      "different devices. Got: ", self.device(), " and ", src.device());
  if (self.sizes() != src.sizes()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }
  return at::native::eq(self, src).all().item().to<bool>();
}

}} // namespace at::native
