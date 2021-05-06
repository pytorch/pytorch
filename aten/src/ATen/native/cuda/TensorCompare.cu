#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/CUDAApplyUtils.cuh>


namespace at { namespace native {

using where_fn = void (*)(TensorIterator &, ScalarType);
DECLARE_DISPATCH(where_fn, where_kernel);

using is_infinity_op_fn = void (*)(TensorIterator &);
DECLARE_DISPATCH(is_infinity_op_fn, isposinf_stub);
DECLARE_DISPATCH(is_infinity_op_fn, isneginf_stub);

namespace {

void where_kernel_impl(TensorIterator &iter, ScalarType condition_type) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBFloat16, kBool, iter.dtype(), "where_cuda", [&] {
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

void isposinf_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isposinf_cuda", [&]() {
    gpu_kernel(
      iter,
      [] GPU_LAMBDA (scalar_t a) -> bool { return a == std::numeric_limits<scalar_t>::infinity(); }
    );
  });
}

void isneginf_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isneginf_cuda", [&]() {
    gpu_kernel(
      iter,
      [] GPU_LAMBDA (scalar_t a) -> bool { return a == -std::numeric_limits<scalar_t>::infinity(); }
    );
  });
}

} // anonymous namespace


REGISTER_DISPATCH(where_kernel, &where_kernel_impl);
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_impl);
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_impl);

template <typename scalar_t>
__global__ void _assert_async_cuda_kernel(scalar_t* input) {
  CUDA_KERNEL_ASSERT(input[0] != 0);
}

__global__ void _assert_async_cuda_kernel(c10::complex<float>* input) {
  CUDA_KERNEL_ASSERT(input[0] != c10::complex<float>(0, 0));
}
__global__ void _assert_async_cuda_kernel(c10::complex<double>* input) {
  CUDA_KERNEL_ASSERT(input[0] != c10::complex<double>(0, 0));
}

void _assert_async_cuda(const Tensor& self) {
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(n < 2, "Boolean value of Tensor with more than one value is ambiguous");
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_assert_async_cuda", [&] {
    _assert_async_cuda_kernel<<<1, 1, 0, stream>>>(self.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

}} // namespace at::native
