#include <cmath>
#include <type_traits>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/CPUGenerator.h>
#include <ATen/CheckGenerator.h>
#include <ATen/Generator.h>

#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>

#include <ATen/native/UnaryOps.h>
#include <ATen/native/cpu/Loops.h>

#include <ATen/native/Distributions.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

#include <TH/THGenerator.hpp>
#include <TH/THRandom.h>

namespace at { namespace native {
namespace {

using namespace vec256;

static void sigmoid_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "sigmoid_cpu", [&]() {
    unary_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return (1 / (1 + std::exp((-a)))); },
        [=](Vec256<scalar_t> a) {
          a = Vec256<scalar_t>((scalar_t)(0)) - a;
          a = a.exp();
          a = Vec256<scalar_t>((scalar_t)(1)) + a;
          a = a.reciprocal();
          return a;
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

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "bernoulli_scalar_cpu_", [&] {
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

static void rsqrt_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "rsqrt", [&] {
    unary_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t {
          return ((scalar_t)1) / std::sqrt(a);
        },
        [=](Vec256<scalar_t> a) { return a.rsqrt(); });
  });
}

#define IMPLEMENT_FLOAT_KERNEL(dispatchtypes, op)                          \
  static void op##_kernel(TensorIterator& iter) {                          \
    AT_DISPATCH_##dispatchtypes##_TYPES(iter.dtype(), #op, [&] {           \
      unary_kernel_vec(                                                   \
          iter,                                                            \
          [=](scalar_t a) -> scalar_t { return std::op(a); },  \
          [=](Vec256<scalar_t> a) { return a.op(); }); \
    });                                                                    \
  }                                                                        \
  REGISTER_DISPATCH(op##_stub, &op##_kernel)
} // anonymous namespace

REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel)
REGISTER_DISPATCH(sigmoid_stub, &sigmoid_kernel)
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
IMPLEMENT_FLOAT_KERNEL(FLOATING, sin)
// IMPLEMENT_FLOAT_KERNEL(FLOATING, sinh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, sqrt)
IMPLEMENT_FLOAT_KERNEL(FLOATING, tan)
IMPLEMENT_FLOAT_KERNEL(FLOATING, tanh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, trunc)

}} // namespace at::native
