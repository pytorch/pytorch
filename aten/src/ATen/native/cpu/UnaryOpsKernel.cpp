#include <ATen/native/UnaryOps.h>

#include <cmath>
#include <limits>
#include <type_traits>

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vml.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <c10/util/MathConstants.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#include <cpuinfo.h>
#endif

namespace at {
namespace native {

namespace {

using namespace vec256;

static void sigmoid_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, iter.dtype(), "sigmoid_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return (static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp((-a)))); },
        [=](Vec256<scalar_t> a) {
          a = Vec256<scalar_t>(static_cast<scalar_t>(0)) - a;
          a = a.exp();
          a = Vec256<scalar_t>(static_cast<scalar_t>(1)) + a;
          a = a.reciprocal();
          return a;
        });
  });
}

#if AT_MKL_ENABLED()

template <typename T>
void VmlLog(int64_t N, const T* X, T* Y) {
  constexpr int64_t K = Vec256<T>::size();
  at::parallel_for(0, N, K, [=](int64_t begin, int64_t end) {
    vec256::map(
        [](Vec256<T> x_vec) { return x_vec.log(); },
        Y + begin,
        X + begin,
        end - begin);
  });
}

template <>
void VmlLog<float>(int64_t N, const float* X, float* Y) {
  vsLn(N, X, Y);
}

template <>
void VmlLog<double>(int64_t N, const double* X, double* Y) {
  vdLn(N, X, Y);
}

template <typename T>
void LogitMKLKernel(T eps, TensorIterator* it) {
  if (!it->can_use_32bit_indexing()) {
    for (auto& sub_it : it->with_32bit_indexing()) {
      LogitMKLKernel<T>(eps, &sub_it);
    }
    return;
  }

  constexpr int64_t K = Vec256<T>::size();
  const int64_t N = it->numel();
  const T* X_data = static_cast<T*>(it->data_ptr(1));
  T* Y_data = static_cast<T*>(it->data_ptr(0));
  if (eps < T(0)) {
    at::parallel_for(0, N, K, [=](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        Y_data[i] = X_data[i] == T(1) ? std::numeric_limits<T>::infinity()
                                      : X_data[i] / (T(1) - X_data[i]);
      }
      VmlLog<T>(end - begin, Y_data + begin, Y_data + begin);
    });
  } else {
    const T lo = eps;
    const T hi = T(1) - eps;
    at::parallel_for(0, N, K, [=](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const T x = X_data[i] < lo ? lo : (X_data[i] > hi ? hi : X_data[i]);
        Y_data[i] =
            x == T(1) ? std::numeric_limits<T>::infinity() : (x / (T(1) - x));
      }
      VmlLog<T>(end - begin, Y_data + begin, Y_data + begin);
    });
  }
}

#else

template <typename T>
void LogitMKLKernel(T eps, TensorIterator* it) {
  TORCH_CHECK(false, "ATen not compiled with MKL");
}

#endif // AT_MKL_ENABLED

void logit_kernel(TensorIterator& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      kBFloat16, iter.common_dtype(), "logit_cpu", [&]() {
        const scalar_t eps = eps_scalar.to<scalar_t>();
        if (at::hasMKL() && iter.is_contiguous()) {
          LogitMKLKernel<scalar_t>(eps, &iter);
        } else if (eps < scalar_t(0)) {
          const Vec256<scalar_t> kOneVec(scalar_t(1));
          cpu_kernel_vec(
              iter,
              [](scalar_t x) {
                return x == scalar_t(1)
                    ? std::numeric_limits<scalar_t>::infinity()
                    : std::log(x / (scalar_t(1) - x));
              },
              [kOneVec](Vec256<scalar_t> x_vec) {
                return (x_vec / (kOneVec - x_vec)).log();
              });
        } else {
          const scalar_t lo = eps;
          const scalar_t hi = scalar_t(1) - eps;
          const Vec256<scalar_t> kOneVec(scalar_t(1));
          const Vec256<scalar_t> lo_vec(lo);
          const Vec256<scalar_t> hi_vec(hi);
          cpu_kernel_vec(
              iter,
              [lo, hi](scalar_t x) {
                x = x < lo ? lo : (x > hi ? hi : x);
                return x == scalar_t(1)
                    ? std::numeric_limits<scalar_t>::infinity()
                    : std::log(x / (scalar_t(1) - x));
              },
              [kOneVec, lo_vec, hi_vec](Vec256<scalar_t> x_vec) {
                x_vec = vec256::clamp(x_vec, lo_vec, hi_vec);
                return (x_vec / (kOneVec - x_vec)).log();
              });
        }
      });
}

static void abs_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "abs_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return abs_impl(a); },
        [=](Vec256<scalar_t> a) { return a.abs(); });
  });
}

static void angle_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "angle_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return angle_impl(a); },
        [=](Vec256<scalar_t> a) { return a.angle(); });
  });
}

static void real_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "real_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return real_impl(a); },
        [=](Vec256<scalar_t> a) { return a.real(); });
  });
}

static void imag_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "imag_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return imag_impl(a); },
        [=](Vec256<scalar_t> a) { return a.imag(); });
  });
}

static void conj_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool, kBFloat16, kHalf, iter.common_dtype(), "conj_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return conj_impl(a); },
            [=](Vec256<scalar_t> a) { return a.conj(); });
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
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [](scalar_t a) -> scalar_t {
            return ~a;
          },
          [](Vec256<scalar_t> a) -> Vec256<scalar_t> {
            return ~a;
          });
    });
  }
}

static void frac_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "frac_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return a - std::trunc(a); },
        [=](Vec256<scalar_t> a) { return a.frac(); });
  });
}

static void logical_not_kernel(TensorIterator& iter) {
  // NOTE: this implementation differs from the CUDA implementation which only does single dispatch
  // (to avoid expensive compilation) because CPU kernels don't handle dynamic_casting
  // (see needs_dynamic_casting).
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(1), "logical_not_cpu", [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(0), "logical_not_cpu", [&]() {
      cpu_kernel(iter, [](self_t a) -> scalar_t { return static_cast<scalar_t>(!a); });
    });
  });
}

static void reciprocal_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "reciprocal_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) __ubsan_ignore_float_divide_by_zero__ -> scalar_t { return static_cast<scalar_t>(1.0) / a; },
        [=](Vec256<scalar_t> a) { return a.reciprocal(); });
  });
}

static void neg_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "neg_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return -a; },
        [=](Vec256<scalar_t> a) { return a.neg(); });
  });
}

static void sign_kernel(TensorIterator& iter){
  if(iter.dtype() == ScalarType::Bool){
      cpu_kernel(iter, [=](bool x) -> bool { return x; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, ScalarType::Half, iter.dtype(), "sign_cpu", [&]() {
        auto zero_vec = Vec256<scalar_t>(static_cast<scalar_t>(0));
        auto one_vec = Vec256<scalar_t>(static_cast<scalar_t>(1));

        cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return (0 < a) - (a < 0); },
          [=](Vec256<scalar_t> self_vec){

              // Comparison operators returns bitmask.
              auto left = Vec256<scalar_t>::blendv(zero_vec, one_vec, zero_vec < self_vec);
              auto right = Vec256<scalar_t>::blendv(zero_vec, one_vec, self_vec < zero_vec);

              return left - right;
          });
    });
  }
}

static void signbit_kernel(TensorIterator& iter){
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, ScalarType::Half, iter.input_dtype(), "signbit_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a) -> bool { return a < 0; });
  });
}

static void sgn_kernel(TensorIterator& iter){
  AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "sgn_cpu", [&]() {
    cpu_kernel_vec(
      iter,
      [=](scalar_t a) -> scalar_t { return sgn_impl(a); },
      [=](Vec256<scalar_t> a) { return a.sgn(); });
  });
}

static void sinc_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, iter.common_dtype(), "sinc_cpu", [&]() {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t {
          if (a == scalar_t(0)) {
            return scalar_t(1);
          } else {
            scalar_t product = c10::pi<scalar_t> * a;
            return std::sin(product) / product;
          }
        });
  });
}

static void sinh_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "sinh_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return std::sinh(a); },
        [=](Vec256<scalar_t> self_vec){return self_vec.sinh();});
  });
}

static void cosh_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "cosh_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return std::cosh(a); },
        [=](Vec256<scalar_t> self_vec){return self_vec.cosh();});
  });
}

static void acosh_kernel(TensorIterator& iter) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "acosh_cpu", [&]() {
      cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return std::acosh(a); });
    });
}

static void asinh_kernel(TensorIterator& iter) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "asinh_cpu", [&]() {
      cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return std::asinh(a); });
    });
}

static void atanh_kernel(TensorIterator& iter) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "atanh_cpu", [&]() {
      cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return std::atanh(a); });
    });
}

static void digamma_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "digamma", [&]() {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return calc_digamma(a); });
  });
}

static void trigamma_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "trigamma", [&]() {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return trigamma(a); });
  });
}

static void exp2_kernel(TensorIterator& iter) {
  // Supports only floating types as std::exp2 doesn't have
  // complex overloads.
  AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.dtype(), "exp2", [&]() {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return std::exp2(a); });
  });
}

static void polygamma_kernel(TensorIterator& iter, int64_t n) {
  if (n == 0) {
    digamma_kernel(iter);
  } else if (n == 1) {
    trigamma_kernel(iter);
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "polygamma", [&]() {
      cpu_kernel(
          iter, [=](scalar_t a) -> scalar_t { return calc_polygamma(n, a); });
    });
  }
}

static void nan_to_num_kernel(
    TensorIterator& iter,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.dtype(), "nan_to_num", [&]() {
    scalar_t nan_replacement = static_cast<scalar_t>(nan.value_or(0.));
    scalar_t pos_inf_replacement = pos_inf.has_value()
        ? static_cast<scalar_t>(pos_inf.value())
        : std::numeric_limits<scalar_t>::max();
    scalar_t neg_inf_replacement = neg_inf.has_value()
        ? static_cast<scalar_t>(neg_inf.value())
        : std::numeric_limits<scalar_t>::lowest();

    cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
      return (
          at::_isnan(a)
              ? nan_replacement
              : (a == std::numeric_limits<scalar_t>::infinity()
                     ? pos_inf_replacement
                     : (a == -std::numeric_limits<scalar_t>::infinity()
                            ? neg_inf_replacement
                            : a)));
    });
  });
}

static void clamp_kernel(TensorIterator& iter, const Scalar& min_scalar, const Scalar& max_scalar) {
  AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "clamp_cpu", [&]() {
    auto min = min_scalar.to<scalar_t>();
    auto max = max_scalar.to<scalar_t>();
    auto min_vec = Vec256<scalar_t>(min);
    auto max_vec = Vec256<scalar_t>(max);
    cpu_kernel_vec(iter,
     [=](scalar_t a) -> scalar_t { return std::min(std::max(a, min), max); },
     [=](Vec256<scalar_t> a) { return vec256::clamp(a, min_vec, max_vec); });
  });
}

static void clamp_max_kernel(TensorIterator& iter, const Scalar& max_scalar) {
  AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "clamp_max_cpu", [&]() {
    auto max = max_scalar.to<scalar_t>();
    auto max_vec = Vec256<scalar_t>(max);
    cpu_kernel_vec(iter,
     [=](scalar_t a) -> scalar_t { return std::min(a, max); },
     [=](Vec256<scalar_t> a) { return vec256::clamp_max(a, max_vec); });
  });
}

static void clamp_min_kernel(TensorIterator& iter, const Scalar& min_scalar) {
  AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "clamp_min_cpu", [&]() {
    auto min = min_scalar.to<scalar_t>();
    auto min_vec = Vec256<scalar_t>(min);
    cpu_kernel_vec(iter,
     [=](scalar_t a) -> scalar_t { return std::max(a, min); },
     [=](Vec256<scalar_t> a) { return vec256::clamp_min(a, min_vec); });
  });
}

static void kaiser_window_kernel(TensorIterator& iter, int64_t window_length, double beta){
  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), "kaiser_window_cpu", [&](){
    const scalar_t alpha = static_cast<scalar_t>((window_length - 1) / 2.0);
    cpu_kernel(iter, [=](scalar_t a){
        return calc_i0(static_cast<scalar_t>(beta) * std::sqrt(1 - std::pow((a - alpha) / alpha, static_cast<scalar_t>(2.0)))) / calc_i0(static_cast<scalar_t>(beta));
    });
  });
}

static void cauchy_kernel(TensorIterator& iter, double median, double sigma, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::cauchy_kernel(iter, median, sigma, generator);
}

void bernoulli_tensor_kernel(Tensor& self, const Tensor& p_, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::bernoulli_kernel(self, p_, generator);
}

void bernoulli_scalar_kernel_default(Tensor& self, double p, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::bernoulli_kernel(self, p, generator);
}

#if !AT_MKL_ENABLED()
void bernoulli_scalar_kernel(Tensor& self, double p, c10::optional<Generator> gen) {
  bernoulli_scalar_kernel_default(self, p, gen);
}
#else
void bernoulli_scalar_kernel(Tensor &self, double p, c10::optional<Generator> gen) {
  if (cpuinfo_initialize() && cpuinfo_vendor_intel == cpuinfo_get_processor(0)->core->vendor) {
    CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
    int64_t seed;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(generator->mutex_);
      seed = generator->random();
    }
    int64_t n = self.numel();
    bool contig = self.is_contiguous();

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool, self.scalar_type(), "bernoulli_scalar_cpu_", [&] {
      at::Tensor tmp_int_tensor;
      if (std::is_same<scalar_t, int>::value && contig) {
        tmp_int_tensor = self;
      } else {
        tmp_int_tensor = at::empty(self.sizes(), self.options().dtype(at::kInt));
      }

      scalar_t *self_ptr = self.data_ptr<scalar_t>();
      int *sample_int_ptr = tmp_int_tensor.data_ptr<int>();

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
  } else {
    // The situation of AMD, move to using the default version
    bernoulli_scalar_kernel_default(self, p, gen);
  }
}
#endif

static void exponential_kernel(TensorIterator& iter, double lambda, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::exponential_kernel(iter, lambda, generator);
}

static void geometric_kernel(TensorIterator& iter, double p, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::geometric_kernel(iter, p, generator);
}

static void log_normal_kernel(TensorIterator& iter, double mean, double std, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::log_normal_kernel(iter, mean, std, generator);
}

void uniform_kernel(TensorIterator& iter, double from, double to, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::uniform_kernel(iter, from, to, generator);
}

void normal_kernel(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::normal_kernel(self, mean, std, generator);
}

static void random_from_to_kernel(TensorIterator& iter, uint64_t range, int64_t base, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::random_from_to_kernel(iter, range, base, generator);
}

static void random_kernel(TensorIterator& iter, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::random_kernel(iter, generator);
}

// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
static void random_full_64_bits_range_kernel(TensorIterator& iter, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::random_full_64_bits_range_kernel(iter, generator);
}

static void rsqrt_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "rsqrt_cpu", [&] {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          return (static_cast<scalar_t>(1)) / std::sqrt(a);
        },
        [=](Vec256<scalar_t> a) { return a.rsqrt(); });
  });
}

static void frexp_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(kHalf,
    // The iter.dtype() here is the dtype of mantissa output.
    // It's a floating point type and must be the same as the input's dtype.
    iter.dtype(),
    "frexp_cpu", [&]() {
      cpu_kernel_multiple_outputs(
        iter,
        [](scalar_t a) -> std::tuple<scalar_t, int32_t> {
          int32_t exponent;
          scalar_t mantissa = std::frexp(a, &exponent);
          return std::tuple<scalar_t, int32_t>(mantissa, exponent);
        }
      );
  });
}

// TODO: Disable cont. branch to test more risky code

#define IMPLEMENT_ITERATOR_LAMBDA(op)                                         \
          [&](char** data_, const int64_t* strides, int64_t n) {              \
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
          }

#define IMPLEMENT_FLOAT_KERNEL(op)                                                  \
  static void op##_kernel(TensorIterator& iter) {                                   \
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                    \
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), #op "_vml_cpu", [&]() { \
      iter.serial_for_each(                                                         \
          IMPLEMENT_ITERATOR_LAMBDA(op),                                            \
          {0, iter.numel()});                                                       \
    });                                                                             \
  }                                                                                 \
  REGISTER_DISPATCH(op##_stub, &op##_kernel)

#define IMPLEMENT_COMPLEX_KERNEL(op)                                                             \
  static void op##_kernel(TensorIterator& iter) {                                                \
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                                 \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, iter.dtype(), #op "_vml_cpu", [&]() { \
      iter.serial_for_each(                                                                      \
          IMPLEMENT_ITERATOR_LAMBDA(op),                                                         \
          {0, iter.numel()});                                                                    \
    });                                                                                          \
  }                                                                                              \
  REGISTER_DISPATCH(op##_stub, &op##_kernel)

  #define IMPLEMENT_COMPLEX_STRUCTURED_KERNEL(op)                                                \
  static void op##_kernel(TensorIteratorBase& iter) {                                            \
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                                 \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, iter.dtype(), #op "_vml_cpu", [&]() { \
      iter.serial_for_each(                                                                      \
          IMPLEMENT_ITERATOR_LAMBDA(op),                                                         \
          {0, iter.numel()});                                                                    \
    });                                                                                          \
    iter.cast_outputs();                                                                         \
  }                                                                                              \
  REGISTER_DISPATCH(op##_stub, &op##_kernel)

} // anonymous namespace

REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel);
REGISTER_DISPATCH(sigmoid_stub, &sigmoid_kernel);
REGISTER_DISPATCH(logit_stub, &logit_kernel);
REGISTER_DISPATCH(bernoulli_tensor_stub, &bernoulli_tensor_kernel);
REGISTER_DISPATCH(bernoulli_scalar_stub, &bernoulli_scalar_kernel);
REGISTER_DISPATCH(cauchy_stub, &cauchy_kernel);
REGISTER_DISPATCH(exponential_stub, &exponential_kernel);
REGISTER_DISPATCH(geometric_stub, &geometric_kernel);
REGISTER_DISPATCH(log_normal_stub, &log_normal_kernel);
REGISTER_DISPATCH(normal_stub, &normal_kernel);
REGISTER_DISPATCH(uniform_stub, &uniform_kernel);
REGISTER_DISPATCH(random_from_to_stub, &random_from_to_kernel);
REGISTER_DISPATCH(random_full_64_bits_range_stub, &random_full_64_bits_range_kernel);
REGISTER_DISPATCH(random_stub, &random_kernel);
REGISTER_DISPATCH(abs_stub, &abs_kernel);
REGISTER_DISPATCH(angle_stub, &angle_kernel);
REGISTER_DISPATCH(real_stub, &real_kernel);
REGISTER_DISPATCH(imag_stub, &imag_kernel);
REGISTER_DISPATCH(conj_stub, &conj_kernel);
REGISTER_DISPATCH(exp2_stub, &exp2_kernel);
REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel);
REGISTER_DISPATCH(logical_not_stub, &logical_not_kernel);
REGISTER_DISPATCH(frac_stub, &frac_kernel);
REGISTER_DISPATCH(reciprocal_stub, &reciprocal_kernel);
REGISTER_DISPATCH(nan_to_num_stub, &nan_to_num_kernel);
REGISTER_DISPATCH(neg_stub, &neg_kernel);
REGISTER_DISPATCH(sign_stub, &sign_kernel);
REGISTER_DISPATCH(signbit_stub, &signbit_kernel);
REGISTER_DISPATCH(sgn_stub, &sgn_kernel);
REGISTER_DISPATCH(sinc_stub, &sinc_kernel);
REGISTER_DISPATCH(sinh_stub, &sinh_kernel);
REGISTER_DISPATCH(cosh_stub, &cosh_kernel);
REGISTER_DISPATCH(acosh_stub, &acosh_kernel);
REGISTER_DISPATCH(asinh_stub, &asinh_kernel);
REGISTER_DISPATCH(atanh_stub, &atanh_kernel);
REGISTER_DISPATCH(digamma_stub, &digamma_kernel);
REGISTER_DISPATCH(trigamma_stub, &trigamma_kernel);
REGISTER_DISPATCH(polygamma_stub, &polygamma_kernel);
REGISTER_DISPATCH(clamp_stub, &clamp_kernel);
REGISTER_DISPATCH(clamp_max_stub, &clamp_max_kernel);
REGISTER_DISPATCH(clamp_min_stub, &clamp_min_kernel);
REGISTER_DISPATCH(kaiser_window_stub, &kaiser_window_kernel)
REGISTER_DISPATCH(frexp_stub, &frexp_kernel)


IMPLEMENT_COMPLEX_KERNEL(acos)
IMPLEMENT_COMPLEX_KERNEL(asin)
IMPLEMENT_COMPLEX_KERNEL(atan)
IMPLEMENT_FLOAT_KERNEL(ceil)
IMPLEMENT_COMPLEX_KERNEL(cos)
IMPLEMENT_FLOAT_KERNEL(erf)
IMPLEMENT_FLOAT_KERNEL(erfc)
IMPLEMENT_FLOAT_KERNEL(erfinv)
IMPLEMENT_COMPLEX_KERNEL(exp)
IMPLEMENT_FLOAT_KERNEL(expm1)
IMPLEMENT_FLOAT_KERNEL(floor)
IMPLEMENT_COMPLEX_KERNEL(log)
IMPLEMENT_COMPLEX_KERNEL(log10)
IMPLEMENT_FLOAT_KERNEL(log1p)
IMPLEMENT_COMPLEX_KERNEL(log2)
IMPLEMENT_FLOAT_KERNEL(i0)
IMPLEMENT_FLOAT_KERNEL(round)
IMPLEMENT_COMPLEX_STRUCTURED_KERNEL(sin)
IMPLEMENT_COMPLEX_KERNEL(sqrt)
IMPLEMENT_COMPLEX_KERNEL(tan)
IMPLEMENT_COMPLEX_KERNEL(tanh)
IMPLEMENT_FLOAT_KERNEL(trunc)
IMPLEMENT_FLOAT_KERNEL(lgamma)

} // namespace native
} // namespace at
