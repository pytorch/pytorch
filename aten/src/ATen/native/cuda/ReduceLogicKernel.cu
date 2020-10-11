#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>


namespace at { namespace native {

void and_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.dtype(), "and_kernel", [&]() {
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter,
        func_wrapper<scalar_t>([] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<scalar_t>(static_cast<bool>(a) && static_cast<bool>(b));
        }),
        static_cast<scalar_t>(true));
  });
}

void or_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.dtype(), "or_kernel", [&]() {
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter,
        func_wrapper<scalar_t>([] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<scalar_t>(static_cast<bool>(a) || static_cast<bool>(b));
        }),
        static_cast<scalar_t>(false));
  });
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
