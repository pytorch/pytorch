#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/NumericUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/core/Scalar.h>
#include <iostream>

namespace at::native {

namespace {

void where_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_V2(iter.dtype(), "where_cuda", [&] {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (bool cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
          return cond_val ? self_val : other_val;
        });
  },
  kComplexHalf, kHalf, kBFloat16, kBool, AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_FLOAT8_TYPES));
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

template <typename scalar_t>
void isclose_kernel_impl_complex(TensorIteratorBase& iter, double rtol, double atol, bool equal_nan) {
  using opmath_t = at::opmath_type<scalar_t>;
  gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {

    opmath_t cast_a = static_cast<opmath_t>(a);
    opmath_t cast_b = static_cast<opmath_t>(b);
    if (rtol == 0 && atol == 0) {
      return cast_a == cast_b;
  }
    if (equal_nan &&
      (::isnan(cast_a.real()) || ::isnan(cast_a.imag())) &&
      (::isnan(cast_b.real()) || ::isnan(cast_b.imag()))) {
      return true;
    }

    if (cast_a == cast_b) {
      return true;
    }

    if (rtol == 0 && atol == 0) {
      return false;
    }

    bool is_a_finite = ::isfinite(cast_a.real()) && ::isfinite(cast_a.imag());
    bool is_b_finite = ::isfinite(cast_b.real()) && ::isfinite(cast_b.imag());

    if (is_a_finite && is_b_finite) {
      auto abs_b = std::abs(cast_b);
      auto allowed_error = atol + (rtol * abs_b);
      return std::abs(cast_a - cast_b) <= allowed_error;
    }

    return false;
  });
}

template <typename scalar_t>
void isclose_kernel_impl_real(TensorIteratorBase& iter, double rtol, double atol, bool equal_nan) {
  if constexpr (std::is_same_v<scalar_t, bool>) {
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return a == b;
    });
  } else {
    using opmath_t = at::opmath_type<scalar_t>;
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      if (a == b) {
        return true;
      }

      opmath_t cast_a = static_cast<opmath_t>(a);
      opmath_t cast_b = static_cast<opmath_t>(b);

      if (equal_nan && ::isnan(static_cast<double>(cast_a)) && ::isnan(static_cast<double>(cast_b))) {
        return true;
      }

      if (rtol == 0 && atol == 0) {
        return false;
      }

      if (::isfinite(static_cast<double>(cast_a)) && ::isfinite(static_cast<double>(cast_b))) {
        auto allowed_error = static_cast<opmath_t>(atol + rtol * ::abs(cast_b));
        return ::abs(cast_a - cast_b) <= allowed_error;
      }
      return false;
    });
  }
}

void isclose_kernel_impl(TensorIteratorBase& iter, double rtol, double atol, bool equal_nan) {
  if (iter.common_dtype() == kComplexFloat || iter.common_dtype() == kComplexDouble) {
    AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "isclose_cuda", [&]() {
      isclose_kernel_impl_complex<scalar_t>(iter, rtol, atol, equal_nan);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "isclose_cuda", [&]() {
      isclose_kernel_impl_real<scalar_t>(iter, rtol, atol, equal_nan);
    });
  }
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

REGISTER_DISPATCH(isclose_stub, &isclose_kernel_impl);
REGISTER_DISPATCH(where_kernel, &where_kernel_impl)
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_impl)
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_impl)
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_impl)
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel_impl)
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_impl)
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel_impl)

struct Msg {
 static constexpr size_t MAX_MSG_LENGTH = 256;
 char msg[MAX_MSG_LENGTH];
};
template <typename scalar_t>
__global__ void _assert_async_cuda_kernel(const scalar_t* input, Msg msg) {
  CUDA_KERNEL_ASSERT_MSG(input[0] != 0, msg.msg);
}

__global__ void _assert_async_cuda_kernel(const c10::complex<float>* input, Msg msg) {
  CUDA_KERNEL_ASSERT_MSG(input[0] != c10::complex<float>(0, 0), msg.msg);
}
__global__ void _assert_async_cuda_kernel(const c10::complex<double>* input, Msg msg) {
  CUDA_KERNEL_ASSERT_MSG(input[0] != c10::complex<double>(0, 0), msg.msg);
}

void _assert_async_msg_cuda(const Tensor& self_tensor, std::string_view assert_msg) {
  const TensorBase &self = get_tensor_base(self_tensor);
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(n < 2, "Boolean value of Tensor with more than one value is ambiguous");
  auto stream = at::cuda::getCurrentCUDAStream();
  Msg msg;
  size_t copy_length = assert_msg.length();
  TORCH_CHECK(copy_length < Msg::MAX_MSG_LENGTH - 1, "Message length must be smaller than " + std::to_string(Msg::MAX_MSG_LENGTH - 1));
  std::copy_n(assert_msg.data(), copy_length, msg.msg);
  msg.msg[copy_length] = '\0';  // Ensure null-termination
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_assert_async_cuda", [&] {
    _assert_async_cuda_kernel<<<1, 1, 0, stream>>>(self.const_data_ptr<scalar_t>(), msg);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void _assert_async_cuda(const Tensor& self_tensor) {
  _assert_async_msg_cuda(self_tensor, "");
}

} // namespace at::native
