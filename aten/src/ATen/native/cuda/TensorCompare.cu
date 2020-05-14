#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/CUDAApplyUtils.cuh>


namespace at { namespace native {

using where_fn = void (*)(TensorIterator &, ScalarType);
DECLARE_DISPATCH(where_fn, where_kernel);

namespace {

void where_kernel_impl(TensorIterator &iter, ScalarType condition_type) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBool, iter.dtype(), "where_cuda", [&] {
    if (condition_type == at::ScalarType::Byte) {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (uint8_t cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
          return cond_val ? self_val : other_val;
        });
    } else {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (bool cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
          return cond_val ? self_val : other_val;
        });
    }
  });
}

} // anonymous namespace


REGISTER_DISPATCH(where_kernel, &where_kernel_impl);

}} // namespace at::native
