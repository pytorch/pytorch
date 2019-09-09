#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>

namespace at { namespace native {

namespace {

template <typename T>
static inline __host__ __device__ T powi(T a, T b) {
  T result = 1;
  while (b) {
    if (b & 1) {
       result *= a;
    }
    b /= 2;
    a *= a;
  }
  return result;
}

template <typename T>
static inline __host__ __device__ T sqrt(T x) {
  return std::sqrt(x);
}

void pow_tensor_tensor_kernel(TensorIterator& iter) {
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "pow_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
        return std::pow(base, exp);
      });
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
        return powi(base, exp);
      });
    });
  }
}

void pow_tensor_scalar_kernel(TensorIterator& iter, Scalar exp_scalar) {
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "pow_cuda", [&]() {
      const auto exp = exp_scalar.to<scalar_t>();
      if (exp == 0.5) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return ::sqrt(base);
        });
      } else if (exp == 2) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return base * base;
        });
      } else if (exp == 3) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return base * base * base;
        });
      } else if (exp == -0.5) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return 1.0 / ::sqrt(base);
        });
      } else if (exp == -1) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return 1.0 / base;
        });
      } else if (exp == -2) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return 1.0 / (base * base);
        });
      } else {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return std::pow(base, exp);
        });
      }
    });
  } else {
    const auto exp = exp_scalar.to<float>();
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow_cuda", [&]() {
      if (exp == 0.5) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return ::sqrt(base);
        });
      } else if (exp == 2) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return base * base;
        });
      } else if (exp == 3) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return base * base * base;
        });
      } else if (exp == -0.5) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return 1.0 / ::sqrt(base);
        });
      } else if (exp == -1) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return 1.0 / base;
        });
      } else if (exp == -2) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return 1.0 / (base * base);
        });
      } else {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return std::pow(base, exp);
        });
      }
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel);
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel);

}} // namespace at::native
