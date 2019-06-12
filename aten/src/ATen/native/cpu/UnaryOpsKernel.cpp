#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <cmath>
#include <type_traits>
#include "ATen/Config.h"
#include "ATen/Dispatch.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CheckGenerator.h"
#include "ATen/Generator.h"
#include "ATen/cpu/vml.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/native/DispatchStub.h"
#include "ATen/native/Distributions.h"
#ifdef __AVX2__
#include "ATen/native/cpu/avx_mathfun.h"
#endif

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

#include "TH/THGenerator.hpp"
#include "TH/THRandom.h"

namespace at { namespace native {
namespace {

using namespace vec256;

template <typename scalar_t>
static int64_t _sigmoid(scalar_t* x, scalar_t* y, int64_t size);

// This should be a temporary solution until we understand why SLEEF is slower
// for sigmoid

template <>
int64_t _sigmoid(float* x, float* y, int64_t size) {
  using Vec = Vec256<float>;
  int64_t i = 0;
  for (; i < size - (size % (2 * Vec::size)); i += 2 * Vec::size) {
    Vec ret = Vec::loadu(y + i);
    Vec ret2 = Vec::loadu(y + i + Vec::size);
    ret = ret.neg();
    ret2 = ret2.neg();
#if defined(__AVX2__) && !defined(_MSC_VER)
    ret = exp256_ps(ret);
    ret2 = exp256_ps(ret2);
#else
    ret = ret.exp();
    ret2 = ret2.exp();
#endif
    ret = Vec((float)(1)) + ret;
    ret2 = Vec((float)(1)) + ret2;
    ret = ret.reciprocal();
    ret2 = ret2.reciprocal();
    ret.store(x + i);
    ret2.store(x + i + Vec::size);
  }
  return i;
}

template <>
int64_t _sigmoid(double* x, double* y, int64_t size) {
  using Vec = Vec256<double>;
  int64_t i = 0;
  for (; i < size - (size % (2 * Vec::size)); i += 2 * Vec::size) {
    Vec ret = Vec::loadu(y + i);
    Vec ret2 = Vec::loadu(y + i + Vec::size);
    ret = ret.neg();
    ret2 = ret2.neg();
    ret = ret.exp();
    ret2 = ret2.exp();
    ret = Vec((double)(1)) + ret;
    ret2 = Vec((double)(1)) + ret2;
    ret = ret.reciprocal();
    ret2 = ret2.reciprocal();
    ret.store(x + i);
    ret2.store(x + i + Vec::size);
  }
  return i;
}

static void sigmoid_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "sigmoid", [&] {
    using Vec = Vec256<scalar_t>;
    CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(
        result,
        self,
        [](int64_t size,
           scalar_t* x,
           scalar_t* y,
           int64_t stridex,
           int64_t stridey) {
          int64_t i = 0;
          if (stridex == 1 && stridey == 1) {
            i = _sigmoid(x, y, size);
          }
          for (; i < size; i += Vec::size) {
            scalar_t buffer[Vec::size];
            int64_t width = Vec::size;
            width = std::min(width, size - i);
            for (int64_t j = 0; j < width; j++) {
              buffer[j] = y[stridey * (i + j)];
            }
            Vec ret = Vec::loadu(buffer);
            ret = Vec((scalar_t)(0)) - ret;
            ret = ret.exp();
            ret = Vec((scalar_t)(1)) + ret;
            ret = ret.reciprocal();
            ret.store(buffer);
            for (int64_t j = 0; j < width; j++)
              x[stridex * (i + j)] = buffer[j];
          }
        });
  });
}


#if !AT_MKL_ENABLED()
void bernoulli_mkl_kernel(Tensor &output, const double p, Generator* gen) {
  // Use AT_ASSERTM because this should never be reached, and AT_ASSERTM tells
  // users to report this as a bug.
  AT_ASSERTM(false, "ATen not compiled with MKL");
}
#else
void bernoulli_mkl_kernel(Tensor &self, const double p, Generator* gen) {
  THGenerator* generator = get_generator(gen);
  int64_t seed;
  {
    std::lock_guard<std::mutex> lock(generator->mutex);
    seed = THRandom_random(generator);
  }
  int64_t n = self.numel();
  bool contig = self.is_contiguous();

  AT_DISPATCH_ALL_TYPES(self.type(), "bernoulli_scalar_cpu_", [&] {
    at::Tensor tmp_int_tensor;
    if (std::is_same<scalar_t, int>::value && contig) {
      tmp_int_tensor = self;
    } else {
      tmp_int_tensor = at::empty(self.sizes(), self.options().dtype(at::kInt));
    }

    scalar_t *self_ptr = self.data<scalar_t>();
    int *sample_int_ptr = tmp_int_tensor.data<int>();

    auto sample = [&](int64_t begin, int64_t end) {
      int64_t len = end - begin;
      if (len > 0) {
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MCG31, seed);
        vslSkipAheadStream(stream, begin);
        viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, len,
          sample_int_ptr + begin, p);
        vslDeleteStream(&stream);

        // vectorized copy if using buffer and contiguous, i.e., being non-int
        // type and contiguous
        if (!std::is_same<scalar_t, int>::value && contig) {
          scalar_t *self_seg = self_ptr + begin;
          int* tmp_seg = sample_int_ptr + begin;
          at::vec256::convert<int, scalar_t>(tmp_seg, self_seg, len);
        }
      }
    };

    parallel_for(0, n, /* grain_size= */ 800, sample);

    // copy_ if using buffer and non contiguous
    if (!contig) {
      self.copy_(tmp_int_tensor);
    }
  });
}
#endif

#define IMPLEMENT_FLOAT_KERNEL(dispatchtypes, op)                          \
  static void op##_kernel(Tensor& result, const Tensor& self) {            \
    checkBackend(#op, {result}, Backend::CPU);                             \
    AT_DISPATCH_##dispatchtypes##_TYPES(self.type(), #op, [&] {            \
      if (self.is_contiguous() && result.is_contiguous()) {                \
        vml::v##op(                                                        \
            result.data<scalar_t>(), self.data<scalar_t>(), self.numel()); \
                                                                           \
      } else {                                                             \
        static constexpr int64_t WIDTH = 131072 / sizeof(scalar_t);        \
        CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(             \
            result,                                                        \
            self,                                                          \
            [](int64_t size,                                               \
               scalar_t* x,                                                \
               scalar_t* y,                                                \
               int64_t stridex,                                            \
               int64_t stridey) {                                          \
              if (stridex == 1 && stridey == 1) {                          \
                vml::v##op(x, y, size);                                    \
              } else {                                                     \
                for (int64_t i = 0; i < size; i += WIDTH) {                \
                  scalar_t buffer[WIDTH];                                  \
                  int64_t width = WIDTH;                                   \
                  width = std::min(width, size - i);                       \
                  for (int64_t j = 0; j < width; j++)                      \
                    buffer[j] = y[stridey * (i + j)];                      \
                  vml::v##op(buffer, buffer, width);                       \
                  for (int64_t j = 0; j < width; j++)                      \
                    x[stridex * (i + j)] = buffer[j];                      \
                }                                                          \
              }                                                            \
            });                                                            \
      }                                                                    \
    });                                                                    \
  }                                                                        \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)

} // anonymous namespace

REGISTER_DISPATCH(sigmoidImpl, &sigmoid_kernel)
REGISTER_DISPATCH(bernoulli_mkl_stub, &bernoulli_mkl_kernel);

// IMPLEMENT_FLOAT_KERNEL(ALL, abs)
IMPLEMENT_FLOAT_KERNEL(FLOATING, acos)
IMPLEMENT_FLOAT_KERNEL(FLOATING, asin)
IMPLEMENT_FLOAT_KERNEL(FLOATING, atan)
IMPLEMENT_FLOAT_KERNEL(FLOATING, ceil)
IMPLEMENT_FLOAT_KERNEL(FLOATING, cos)
// IMPLEMENT_FLOAT_KERNEL(FLOATING, cosh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, erf)
IMPLEMENT_FLOAT_KERNEL(FLOATING, erfc)
IMPLEMENT_FLOAT_KERNEL(FLOATING, exp)
IMPLEMENT_FLOAT_KERNEL(FLOATING, expm1)
IMPLEMENT_FLOAT_KERNEL(FLOATING, floor)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log10)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log1p)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log2)
IMPLEMENT_FLOAT_KERNEL(FLOATING, round)
IMPLEMENT_FLOAT_KERNEL(FLOATING, rsqrt)
IMPLEMENT_FLOAT_KERNEL(FLOATING, sin)
// IMPLEMENT_FLOAT_KERNEL(FLOATING, sinh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, sqrt)
IMPLEMENT_FLOAT_KERNEL(FLOATING, tan)
IMPLEMENT_FLOAT_KERNEL(FLOATING, tanh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, trunc)

}} // namespace at::native
