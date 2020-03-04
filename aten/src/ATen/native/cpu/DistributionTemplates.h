#pragma once

#include <ATen/Dispatch.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <limits>
#include <mutex>

namespace at {
namespace native {
namespace templates {
namespace cpu {

template<typename RNG>
void random_from_to_kernel(TensorIterator& iter, uint64_t range, int64_t base, RNG* generator) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_from_to_kernel_cpu", [&] {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    if ((
      std::is_same<scalar_t, int64_t>::value ||
      std::is_same<scalar_t, double>::value ||
      std::is_same<scalar_t, float>::value ||
      std::is_same<scalar_t, at::BFloat16>::value) && range >= 1ULL << 32)
    {
      cpu_serial_kernel(iter, [range, base, generator]() -> scalar_t {
        return static_cast<scalar_t>(static_cast<int64_t>((generator->random64() % range) + base));
      });
    } else {
      cpu_serial_kernel(iter, [range, base, generator]() -> scalar_t {
        return static_cast<scalar_t>(static_cast<int64_t>((generator->random() % range) + base));
      });
    }
  });
}

// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
template<typename RNG>
void random_full_64_bits_range_kernel(TensorIterator& iter, RNG* generator) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel_cpu", [&] {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    if (std::is_same<scalar_t, int64_t>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, at::BFloat16>::value) {
      cpu_serial_kernel(iter, [generator]() -> scalar_t {
        return static_cast<scalar_t>(static_cast<int64_t>(generator->random64()));
      });
    } else {
      TORCH_CHECK(false, "random_full_64_bits_range_kernel_cpu handles only int64, double, float and bfloat16");
    }
  });
}

template<typename RNG>
struct RandomFromToKernel {
  void operator()(TensorIterator& iter, uint64_t range, int64_t base, RNG* gen) {
    random_from_to_kernel(iter, range, base, gen);
  }
  void operator()(TensorIterator& iter, RNG* gen) {
    random_full_64_bits_range_kernel(iter, gen);
  }
};

template<typename RNG>
void random_kernel(TensorIterator& iter, RNG* generator) {
  std::lock_guard<std::mutex> lock(generator->mutex_);
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_kernel_fp_cpu", [&] {
      if (std::is_same<scalar_t, double>::value) {
        cpu_serial_kernel(iter, [generator]() -> scalar_t {
          return static_cast<scalar_t>(generator->random64() % static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1));
        });
      } else {
        cpu_serial_kernel(iter, [generator]() -> scalar_t {
          return static_cast<scalar_t>(generator->random() % static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1));
        });
      }
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, iter.dtype(), "random_kernel_int_cpu", [&] {
      if (std::is_same<scalar_t, int64_t>::value) {
        cpu_serial_kernel(iter, [generator]() -> scalar_t {
          return static_cast<scalar_t>(generator->random64() % (static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1));
        });
      } else if (std::is_same<scalar_t, bool>::value) {
        cpu_serial_kernel(iter, [generator]() -> scalar_t {
          return static_cast<scalar_t>(generator->random() & 1);
        });
      } else {
        cpu_serial_kernel(iter, [generator]() -> scalar_t {
          return static_cast<scalar_t>(generator->random() % (static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1));
        });
      }
    });
  } else {
    TORCH_CHECK(false, "random_kernel_cpu handles only integral, floating-point and boolean types");
  }
}

template<typename RNG>
struct RandomKernel {
  void operator()(TensorIterator& iter, RNG* gen) {
    random_kernel(iter, gen);
  }
};

// =======================================================================================================================================

#ifdef __AVX2__
#include <ATen/native/cpu/avx_mathfun.h>

static void normal_fill_16_AVX2(float *data,
                         const __m256* two_pi,
                         const __m256* one,
                         const __m256* minus_two,
                         const __m256* mean,
                         const __m256* std_v) {
  const __m256 u1 = _mm256_sub_ps(*one, _mm256_loadu_ps(data));
  const __m256 u2 = _mm256_loadu_ps(data + 8);
  // sincos256_ps and log256_ps are from avx_mathfun.h
  const __m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(*minus_two, log256_ps(u1)));
  const __m256 theta = _mm256_mul_ps(*two_pi, u2);
  __m256 sintheta, costheta;
  sincos256_ps(theta, &sintheta, &costheta);
  const __m256 n1 = _mm256_mul_ps(radius, costheta);
  const __m256 n2 = _mm256_mul_ps(radius, sintheta);
  _mm256_storeu_ps(data, _mm256_fmadd_ps(n1, *std_v, *mean));
  _mm256_storeu_ps(data + 8, _mm256_fmadd_ps(n2, *std_v, *mean));
}

void normal_fill_AVX2(Tensor& self, const float mean, const float std, Generator* gen) {
  float *data = self.data_ptr<float>();
  auto size = self.numel();
  CPUGenerator* generator = get_generator_or_default<CPUGenerator>(gen, detail::getDefaultCPUGenerator());
  std::lock_guard<std::mutex> lock(generator->mutex_);
  for (int64_t i = 0; i < size; ++i) {
    at::uniform_real_distribution<float> uniform(0, 1);
    data[i] = uniform(generator);
  }
   const __m256 two_pi = _mm256_set1_ps(2.0f * M_PI);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 minus_two = _mm256_set1_ps(-2.0f);
  const __m256 mean_v = _mm256_set1_ps(mean);
  const __m256 std_v = _mm256_set1_ps(std);

  for (int64_t i = 0; i < size - 15; i += 16) {
    normal_fill_16_AVX2(data + i, &two_pi, &one, &minus_two, &mean_v, &std_v);
  }

  if (size % 16 != 0) {
    // Recompute the last 16 values.
    data = data + size - 16;
    for (int64_t i = 0; i < 16; ++i) {
      at::uniform_real_distribution<float> uniform(0, 1);
      data[i] = uniform(generator);
    }
    normal_fill_16_AVX2(data, &two_pi, &one, &minus_two, &mean_v, &std_v);
  }
}
#endif

template <typename scalar_t>
static void normal_fill_16(scalar_t *data, const scalar_t mean, const scalar_t std) {
  for (int j = 0; j < 8; ++j) {
    const scalar_t u1 = 1 - data[j]; // [0, 1) -> (0, 1] for log.
    const scalar_t u2 = data[j + 8];
    const scalar_t radius = std::sqrt(-2 * std::log(u1));
    const scalar_t theta = 2.0f * M_PI * u2;
    data[j] = radius * std::cos(theta) * std + mean;
    data[j + 8] = radius * std::sin(theta) * std + mean;
  }
}

template <typename scalar_t, typename RNG>
void normal_fill(Tensor& self, const scalar_t mean, const scalar_t std, RNG* generator) {
  scalar_t *data = self.data_ptr<scalar_t>();
  auto size = self.numel();
  std::lock_guard<std::mutex> lock(generator->mutex_);
  for (int64_t i = 0; i < size; ++i) {
    at::uniform_real_distribution<scalar_t> uniform(0, 1);
    data[i] = uniform(generator);
  }

  for (int64_t i = 0; i < size - 15; i += 16) {
    normal_fill_16<scalar_t>(data + i, mean, std);
  }
  if (size % 16 != 0) {
    // Recompute the last 16 values.
    data = data + size - 16;
    for (int64_t i = 0; i < 16; ++i) {
      at::uniform_real_distribution<scalar_t> uniform(0, 1);
      data[i] = uniform(generator);
    }
    normal_fill_16<scalar_t>(data, mean, std);
  }
}

static std::vector<int64_t> computeStrideForComplex(IntArrayRef oldstride) {
  auto res = oldstride.vec();
  for(size_t i = 0; i < res.size(); i++) {
    res[i] = res[i] * 2;
  }
  res.emplace_back(1);
  return res;
}

// expects as input a complex tensor and returns back a float tensor
// containing the complex values in the last two dimensions
static Tensor view_complex_as_float(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.is_complex());
  auto new_sizes = self.sizes().vec();
  // last dimension will always have two elements containing the real and imag vals
  new_sizes.emplace_back(2);
  auto new_strides = computeStrideForComplex(self.strides());
  if(self.scalar_type() == at::kComplexFloat) {
    float* data = reinterpret_cast<float*>(self.data_ptr<std::complex<float>>());
    return at::from_blob(data, new_sizes, new_strides, dtype(at::kFloat));
  } else {
    double* data = reinterpret_cast<double*>(self.data_ptr<std::complex<double>>());
    return at::from_blob(data, new_sizes, new_strides, dtype(at::kDouble));
  }
}

template<typename RNG>
void normal_kernel(Tensor& self, double mean, double std, RNG* generator) {
  if(self.is_complex()) {
    // note: float_tensor lives only as long as the self tensor lives
    auto float_tensor = view_complex_as_float(self);
    // variance for normal distribution of the real and imaginary values
    // is half of the input variance
    return normal_kernel(float_tensor, mean, std/(std::sqrt(2)), generator);
  }
  auto size = self.numel();
  if (self.scalar_type() == ScalarType::Float && size >= 16 && self.is_contiguous()) {
#ifdef __AVX2__
    normal_fill_AVX2(self, static_cast<float>(mean), static_cast<float>(std), generator);
#else
    normal_fill(self, static_cast<float>(mean), static_cast<float>(std), generator);
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "normal_kernel_cpu", [&] {
      if (size >= 16 && self.is_contiguous()) {
        normal_fill<scalar_t>(self, static_cast<scalar_t>(mean), static_cast<scalar_t>(std), generator);
      } else {
        auto iter = TensorIterator::nullary_op(self);
        std::lock_guard<std::mutex> lock(generator->mutex_);
        cpu_serial_kernel(iter, [mean, std, generator]() -> scalar_t {
          at::normal_distribution<double> normal(mean, std);
          return (scalar_t)normal(generator);
        });
      }
    });
  }
}

template<typename RNG>
struct NormalKernel {
  void operator()(Tensor& self, double mean, double std, RNG* gen) {
    normal_kernel(self, mean, std, gen);
  }
};

// =======================================================================================================================================

template<typename RNG>
void cauchy_kernel(TensorIterator& iter, double median, double sigma, RNG* generator) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "cauchy_cpu", [&]() {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    cpu_serial_kernel(iter, [median, sigma, generator]() -> scalar_t {
      at::cauchy_distribution<double> cauchy(median, sigma);
      return (scalar_t)cauchy(generator);
    });
  });
}

}}}}
