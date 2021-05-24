#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/NumericUtils.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void smooth_l1_kernel_cuda(TensorIterator& iter, double beta) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "smooth_l1_cuda", [&iter, beta]() {
    scalar_t beta_val(beta);
    gpu_kernel(iter, [beta_val] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      auto z = ::abs(a - b);
      return z < beta_val ? scalar_t(0.5) * z * z / beta_val : z - scalar_t(0.5) * beta_val;
    });
  });
}

void huber_kernel_cuda(TensorIterator& iter, double delta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "huber_cuda", [&iter, delta] {
    scalar_t delta_val(delta);
    gpu_kernel(iter, [delta_val] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      auto z = ::abs(a - b);
      return z < delta_val ? scalar_t(0.5) * z * z : delta_val * (z - scalar_t(0.5) * delta_val);
    });
  });
}

void mse_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "mse_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      auto diff = a - b;
      return diff * diff;
    });
  });
}

void xlogy_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "xlogy_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      if (at::_isnan(y)){
        return NAN;
      }
      if (x == 0){
        return 0;
      }
      return x * std::log(y);
    });
  });
}

void xlog1py_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "xlog1py_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      if (at::_isnan(y)){
        return NAN;
      }
      if (x == 0){
        return 0;
      }
      return x * std::log1p(y);
    });
  });
}

void betainc_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "betainc_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t x, scalar_t a, scalar_t b) -> scalar_t {
      if (at::_isnan(x)){
        return NAN;
      }
      if (at::_isnan(a)){
        return NAN;
      }
      if (at::_isnan(b)){
        return NAN;
      }
      if (x < 0.0 || x > 1.0 || a < 0.0 || b < 0.0){
        return NAN;
      }

      /*The continued fraction converges nicely for x < (a+1)/(a+b+2)*/
      /*Use the fact that beta is symmetrical.*/
      bool return_inverse = false;
      if (x > (a + 1.0) / (a + b + 2.0)) {
        std::swap(a, b);
        x = 1.0 - x;
        return_inverse = true;
      }

      /*Find the first part before the continued fraction.*/
      const scalar_t lbeta_ab = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
      const scalar_t front = std::exp(std::log(x) * a + std::log(1.0 - x) * b - lbeta_ab) / a;

      /*Use Lentz's algorithm to evaluate the continued fraction.*/
      scalar_t f = 1.0, c = 1.0, d = 0.0;

      scalar_t TINY = 1.0e-30, STOP = 1.0e-8, numerator, cd;
      int i, m;
      for (i = 0; i <= 200; ++i) {
          m = i / 2;
          
          if (i == 0) {
              numerator = 1.0; /*First numerator is 1.0.*/
          } else if (i % 2 == 0) {
              numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m)); /*Even term.*/
          } else {
              numerator = - ((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1)); /*Odd term.*/
          }

          /*Do an iteration of Lentz's algorithm.*/
          d = 1.0 + numerator * d;
          if (std::abs(d) < TINY) {
            d = TINY;
          }
          d = 1.0 / d;

          c = 1.0 + numerator / c;
          if (std::abs(c) < TINY) {
            c = TINY;
          }

          cd = c * d;
          f *= cd;

          /*Check for stop.*/
          if (std::abs(1.0 - cd) < STOP) {
            if (return_inverse) {
              return 1 - front * (f - 1.0);
            } else {
              return front * (f - 1.0);
            }
          }
      }

      return NAN; /*Needed more loops, did not converge.*/
    });
  });
}

REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_kernel_cuda);
REGISTER_DISPATCH(huber_stub, &huber_kernel_cuda);
REGISTER_DISPATCH(mse_stub, &mse_kernel_cuda);
REGISTER_DISPATCH(xlogy_stub, &xlogy_kernel_cuda);
REGISTER_DISPATCH(xlog1py_stub, &xlog1py_kernel_cuda);
REGISTER_DISPATCH(betainc_stub, &betainc_kernel_cuda);

// DO NOT ADD ANY NEW KERNELS HERE
// CUDA compilation times grow quickly.  It's perfectly acceptable to have a file per kernel.

}} // namespace at::native
