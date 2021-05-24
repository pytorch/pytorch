#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/NumericUtils.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/NumericUtils.h>
#include <c10/util/complex.h>

namespace at {
namespace native {

void exp2_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "exp2_cuda",
      [&]() {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::exp2(a);
        });
      });
}

void i0_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "i0_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return calc_i0(a);
    });
  });
}

void i0e_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "i0e_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return calc_i0e(a);
    });
  });
}

void sigmoid_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "sigmoid_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp(-a));
    });
  });
}

void sinc_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "sinc_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          if (a == scalar_t(0)) {
            return scalar_t(1);
          } else {
            // NVCC says constexpr var is not accessible from device
            scalar_t product = c10::detail::pi<scalar_t>() * a;
            return std::sin(product) / product;
          }
        });
      });
}

void logit_kernel_cuda(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "logit_cuda",
      [&]() {
        using T_ACC = acc_type<scalar_t, true>;
        const T_ACC eps = eps_scalar.to<T_ACC>();
        if (eps < T_ACC(0)) {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
            const T_ACC x_acc = static_cast<T_ACC>(x);
            return c10::cuda::compat::log(x_acc / (T_ACC(1) - x_acc));
          });
        } else {
          const T_ACC lo = eps;
          const T_ACC hi = T_ACC(1) - eps;
          gpu_kernel(
              iter, [lo, hi] GPU_LAMBDA(scalar_t x) -> scalar_t {
                const T_ACC x_acc = static_cast<T_ACC>(x);
                T_ACC z = x_acc < lo ? lo : (x_acc > hi ? hi : x_acc);
                return c10::cuda::compat::log(z / (T_ACC(1) - z));
              });
        }
      });
}

void erf_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "erf_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::erf(a);
    });
  });
}

void erfc_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "erfc_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::erfc(a);
        });
      });
}

void erfinv_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "erfinv_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::erfinv(a);
    });
  });
}

void kaiser_window_kernel_cuda(TensorIteratorBase& iter, int64_t window_length, double beta_){
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "kaiser_window_cuda", [&](){
    using T_ACC = acc_type<scalar_t, true>;
    const T_ACC inv_alpha = static_cast<T_ACC>(2.0 / (window_length - 1));
    const T_ACC beta = static_cast<T_ACC>(beta_);
    const T_ACC inv_i0_beta = 1.0 / calc_i0(beta);
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t a) -> scalar_t {
      T_ACC x = static_cast<T_ACC>(a) * inv_alpha - 1;
      T_ACC y = std::max<T_ACC>(0, 1 - x * x);
      return calc_i0(beta * ::sqrt(y)) * inv_i0_beta;
    });
  });
}

void entr_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "entr_cuda",
      [&]() {
        gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t x) -> scalar_t {
          if (at::_isnan(x)) {
            return x;
          } else if (x > 0) {
            return -x * std::log(x);
          } else if (x == 0) {
            return 0;
          }
          return static_cast<scalar_t>(-INFINITY);
        });
      });
}

REGISTER_DISPATCH(exp2_stub, &exp2_kernel_cuda);
REGISTER_DISPATCH(i0_stub, &i0_kernel_cuda);
REGISTER_DISPATCH(special_i0e_stub, &i0e_kernel_cuda);
REGISTER_DISPATCH(sigmoid_stub, &sigmoid_kernel_cuda);
REGISTER_DISPATCH(sinc_stub, &sinc_kernel_cuda);
REGISTER_DISPATCH(logit_stub, &logit_kernel_cuda);
REGISTER_DISPATCH(erf_stub, &erf_kernel_cuda);
REGISTER_DISPATCH(erfc_stub, &erfc_kernel_cuda);
REGISTER_DISPATCH(erfinv_stub, &erfinv_kernel_cuda);
REGISTER_DISPATCH(kaiser_window_stub, &kaiser_window_kernel_cuda);
REGISTER_DISPATCH(special_entr_stub, &entr_kernel_cuda);

} // namespace native
} // namespace at
