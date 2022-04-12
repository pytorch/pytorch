#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/BinaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at {
namespace native {

const char sigmoid_backward_name[] = "sigmoid_backward";
void sigmoid_backward_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if(isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto sigmoid_backward_string = jiterator_stringify(
        template <typename T>
        T sigmoid_backward(T a, T b) {
          return a * std::conj((T{1.} - b) * b);
        }
    ); // sigmoid_backward_string
    AT_DISPATCH_COMPLEX_TYPES(dtype, "sigmoid_backward_cuda", [&]() {
        jitted_gpu_kernel<
          /*name=*/ sigmoid_backward_name,
          /*return_dtype=*/ scalar_t,
          /*common_dtype=*/ scalar_t,
          /*arity=*/ 2>(iter, sigmoid_backward_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES(dtype, "sigmoid_backward_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * std::conj((scalar_t{1.} - b) * b);
      });
    });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, dtype, "sigmoid_backward_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * (scalar_t(1.) - b) * b;
      });
    });
  }
}

void logit_backward_kernel_cuda(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "logit_cuda",
      [&]() {
        using T_ACC = acc_type<scalar_t, true>;
        const T_ACC eps = eps_scalar.to<T_ACC>();
        if (eps < T_ACC(0)) {
          gpu_kernel(
              iter, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
                const T_ACC dy_acc = static_cast<T_ACC>(dy);
                const T_ACC x_acc = static_cast<T_ACC>(x);
                return (x_acc < T_ACC(0) || x_acc > T_ACC(1))
                    ? std::numeric_limits<T_ACC>::quiet_NaN()
                    : dy_acc / (x_acc * (T_ACC(1) - x_acc));
              });
        } else {
          const T_ACC lo = eps;
          const T_ACC hi = T_ACC(1) - eps;
          gpu_kernel(
              iter, [lo, hi] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
                const T_ACC dy_acc = static_cast<T_ACC>(dy);
                const T_ACC x_acc = static_cast<T_ACC>(x);
                return (x_acc < lo || x_acc > hi)
                    ? T_ACC(0)
                    : dy_acc / (x_acc * (T_ACC(1) - x_acc));
              });
        }
      });
}

void tanh_backward_kernel_cuda(TensorIteratorBase& iter) {
  if(isComplexType(iter.dtype())) {
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "tanh_backward_complex_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * std::conj(scalar_t{1.} - b * b);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "tanh_backward_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * (scalar_t{1.} - b * b);
      });
    });
  }
}

REGISTER_DISPATCH(sigmoid_backward_stub, &sigmoid_backward_kernel_cuda);
REGISTER_DISPATCH(logit_backward_stub, &logit_backward_kernel_cuda);
REGISTER_DISPATCH(tanh_backward_stub, &tanh_backward_kernel_cuda);

} // namespace native
} // namespace at
