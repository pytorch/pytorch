#include <cmath>
#include <type_traits>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/CPUGenerator.h>
#include <ATen/Utils.h>
#include <ATen/Generator.h>
#include <ATen/Parallel.h>

#include <ATen/cpu/vml.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vec256/functional.h>

#include <ATen/native/Distributions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>

#include <ATen/native/cpu/Loops.h>


#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

namespace at { namespace native {
namespace {

using namespace vec256;

static void sigmoid_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "sigmoid_cpu", [&]() {
    cpu_kernel_vec(
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

static void abs_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "abs_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return std::abs(a); },
        [=](Vec256<scalar_t> a) { return a.abs(); });
  });
}

static void bitwise_not_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    // Boolean type does not work with ~ (bitwise NOT) in C++. bitwise_not wraps this operation for both Boolean and
    // integral types.
    cpu_kernel(
          iter,
          [](bool a) {
            return !a;
          });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_cpu", [&]() {
      cpu_kernel(
          iter,
          [](scalar_t a) -> scalar_t {
            return ~a;
      });
    });
  }
}

static void frac_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "frac_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return a - std::trunc(a); },
        [=](Vec256<scalar_t> a) { return a.frac(); });
  });
}

static void reciprocal_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "reciprocal_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return decltype(a)(1.0) / a; },
        [=](Vec256<scalar_t> a) { return a.reciprocal(); });
  });
}

static void neg_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "neg_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return -a; },
        [=](Vec256<scalar_t> a) { return a.neg(); });
  });
}

static void sinh_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "sinh_cpu", [&]() {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return std::sinh(a); });
  });
}

static void cosh_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "cosh_cpu", [&]() {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return std::cosh(a); });
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
  CPUGenerator* generator = get_generator_or_default<CPUGenerator>(gen, detail::getDefaultCPUGenerator());
  int64_t seed;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    seed = generator->random();
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
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "rsqrt_cpu", [&] {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t {
          return ((scalar_t)1) / std::sqrt(a);
        },
        [=](Vec256<scalar_t> a) { return a.rsqrt(); });
  });
}

// TODO: Disable cont. branch to test more risky code

#define IMPLEMENT_FLOAT_KERNEL(dispatchtypes, op)                             \
  static void op##_kernel(TensorIterator& iter) {                             \
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                              \
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), op##_vml_cpu, [&]() {            \
      iter.serial_for_each(                                                   \
          [&](char** data_, const int64_t* strides, int64_t n) { \
            scalar_t* out_data = reinterpret_cast<scalar_t*>(data_[0]);       \
            scalar_t* in_data = reinterpret_cast<scalar_t*>(data_[1]);        \
            int64_t out_stride = strides[0] / sizeof(scalar_t);               \
            int64_t in_stride = strides[1] / sizeof(scalar_t);                \
            if (out_stride == 1 && in_stride == 1) {                          \
              vml::v##op(out_data, in_data, n);                               \
            } else {                                                          \
              static constexpr int64_t WIDTH = 131072 / sizeof(scalar_t);     \
              for (int64_t i = 0; i < n; i += WIDTH) {                        \
                scalar_t buffer[WIDTH];                                       \
                int64_t width = WIDTH;                                        \
                width = std::min(width, n - i);                               \
                for (int64_t j = 0; j < width; j++)                           \
                  buffer[j] = in_data[in_stride * (i + j)];                   \
                vml::v##op(buffer, buffer, width);                            \
                for (int64_t j = 0; j < width; j++)                           \
                  out_data[out_stride * (i + j)] = buffer[j];                 \
              }                                                               \
            }                                                                 \
          },                                                                  \
          {0, iter.numel()});                                                 \
    });                                                                       \
  }                                                                           \
  REGISTER_DISPATCH(op##_stub, &op##_kernel)

} // anonymous namespace

REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel)
REGISTER_DISPATCH(sigmoid_stub, &sigmoid_kernel)
REGISTER_DISPATCH(bernoulli_mkl_stub, &bernoulli_mkl_kernel);
REGISTER_DISPATCH(abs_stub, &abs_kernel);
REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel);
REGISTER_DISPATCH(frac_stub, &frac_kernel);
REGISTER_DISPATCH(reciprocal_stub, &reciprocal_kernel);
REGISTER_DISPATCH(neg_stub, &neg_kernel);
REGISTER_DISPATCH(sinh_stub, &sinh_kernel);
REGISTER_DISPATCH(cosh_stub, &cosh_kernel);

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
