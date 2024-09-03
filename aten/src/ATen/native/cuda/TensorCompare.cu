#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/NumericUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/core/Scalar.h>


namespace at::native {

namespace {

void where_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBFloat16, kBool, iter.dtype(), "where_cuda", [&] {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (bool cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
          return cond_val ? self_val : other_val;
        });
  });
}

void isposinf_kernel_impl(TensorIteratorBase &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isposinf_cuda", [&]() {
    gpu_kernel(
      iter,
      [] GPU_LAMBDA (scalar_t a) -> bool { return a == std::numeric_limits<scalar_t>::infinity(); }
    );
  });
}

void isneginf_kernel_impl(TensorIteratorBase &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isneginf_cuda", [&]() {
    gpu_kernel(
      iter,
      [] GPU_LAMBDA (scalar_t a) -> bool { return a == -std::numeric_limits<scalar_t>::infinity(); }
    );
  });
}

void clamp_kernel_impl(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "clamp_cuda", [&] {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t v, scalar_t lower, scalar_t upper) -> scalar_t {
      // Propagate nan, which doesn't propagate automatically for ROCm
      if (at::_isnan(v)) {
        return v;
      } if (at::_isnan(lower)) {
        return lower;
      } if (at::_isnan(upper)) {
        return upper;
      } else {
        return ::min(::max(v, lower), upper);
      }
    });
  });
}

void inline launch_clamp_scalar(TensorIteratorBase& iter, Scalar lim0, Scalar lim1, at::native::detail::ClampLimits minmax){
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "clamp_scalar_cuda", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    auto lim0_val = lim0.to<opmath_t>();
    auto lim1_val = lim1.to<opmath_t>();

    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t v) -> scalar_t {
      // Propagate nan, which doesn't propagate automatically for ROCm
      if (_isnan(static_cast<opmath_t>(v))) {
        return v;
      } else if (minmax==at::native::detail::ClampLimits::Min){
        return ::max(static_cast<opmath_t>(v), lim0_val);
      } else if (minmax==at::native::detail::ClampLimits::Max){
        return ::min(static_cast<opmath_t>(v), lim0_val);
      } else {
        return ::min(::max(static_cast<opmath_t>(v), lim0_val), lim1_val);
      }
    });
  });
}


void clamp_scalar_kernel_impl(TensorIteratorBase& iter, const Scalar& min, const Scalar& max) {
  launch_clamp_scalar(iter, min, max, at::native::detail::ClampLimits::MinMax);
}

void clamp_min_scalar_kernel_impl(TensorIteratorBase& iter, Scalar min) {
  launch_clamp_scalar(iter, min, min, at::native::detail::ClampLimits::Min);
}

void clamp_max_scalar_kernel_impl(TensorIteratorBase& iter, Scalar max) {
  launch_clamp_scalar(iter, max, max, at::native::detail::ClampLimits::Max);
}

} // anonymous namespace


REGISTER_DISPATCH(where_kernel, &where_kernel_impl);
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_impl);
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_impl);
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_impl);
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel_impl);
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_impl);
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel_impl);

template <typename scalar_t>
__global__ void _assert_async_cuda_kernel(const scalar_t* input) {
  CUDA_KERNEL_ASSERT(input[0] != 0);
}

__global__ void _assert_async_cuda_kernel(const c10::complex<float>* input) {
  CUDA_KERNEL_ASSERT(input[0] != c10::complex<float>(0, 0));
}
__global__ void _assert_async_cuda_kernel(const c10::complex<double>* input) {
  CUDA_KERNEL_ASSERT(input[0] != c10::complex<double>(0, 0));
}

void _assert_async_cuda(const Tensor& self_tensor) {
  const TensorBase &self = get_tensor_base(self_tensor);
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(n < 2, "Boolean value of Tensor with more than one value is ambiguous");
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_assert_async_cuda", [&] {
    _assert_async_cuda_kernel<<<1, 1, 0, stream>>>(self.const_data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

// TODO (tmanlaibaatar) Ignore assert msg for now
void _assert_async_msg_cuda(const Tensor& self_tensor, c10::string_view assert_msg) {
  _assert_async_cuda(self_tensor);
}

} // namespace at::native
