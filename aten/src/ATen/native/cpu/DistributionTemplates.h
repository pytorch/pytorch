#pragma once

#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandBase.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <mutex>

#ifdef CPU_CAPABILITY_AVX2
#include <ATen/native/cpu/avx_mathfun.h>
#include <c10/util/irange.h>
#endif




namespace at::native::templates::cpu {
namespace {

// ==================================================== Random ========================================================

template<typename RNG>
void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, RNG generator) {
  AT_DISPATCH_V2(iter.dtype(), "random_from_to_kernel_cpu", AT_WRAP([&] {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    cpu_serial_kernel(iter, [range, base, generator]() -> scalar_t {
      uniform_int_from_to_distribution<scalar_t> random(range, base);
      return random(generator);
    });
  }), kBool, kHalf, kBFloat16, AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
template<typename RNG>
void random_full_64_bits_range_kernel(TensorIteratorBase& iter, RNG generator) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel_cpu", [&] {
    if constexpr (std::is_same_v<scalar_t, int64_t> ||
        std::is_same_v<scalar_t, double> ||
        std::is_same_v<scalar_t, float> ||
        std::is_same_v<scalar_t, at::BFloat16>) {
      std::lock_guard<std::mutex> lock(generator->mutex_);
      cpu_serial_kernel(iter, [generator]() -> scalar_t {
        uniform_int_full_range_distribution<scalar_t> random;
        return random(generator);
      });
    } else {
      TORCH_CHECK(false, "random_full_64_bits_range_kernel_cpu handles only int64, double, float and bfloat16");
    }
  });
}

template<typename RNG>
struct RandomFromToKernel {
  void operator()(TensorIteratorBase& iter, uint64_t range, int64_t base, std::optional<Generator> gen) {
    random_from_to_kernel(iter, range, base, check_generator<RNG>(gen));
  }
  void operator()(TensorIteratorBase& iter, std::optional<Generator> gen) {
    random_full_64_bits_range_kernel(iter, check_generator<RNG>(gen));
  }
};

template<typename RNG>
void random_kernel(TensorIteratorBase& iter, RNG generator) {
  std::lock_guard<std::mutex> lock(generator->mutex_);
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, iter.dtype(), "random_kernel_cpu", [&] {
    cpu_serial_kernel(iter, [generator]() -> scalar_t {
      uniform_int_distribution<scalar_t> random;
      return random(generator);
    });
  });
}

template<typename RNG>
struct RandomKernel {
  void operator()(TensorIteratorBase& iter, std::optional<Generator> gen) {
    random_kernel(iter, check_generator<RNG>(gen));
  }
};

// ==================================================== Normal ========================================================

template<typename RNG>
void normal_kernel(const TensorBase &self, double mean, double std, RNG generator) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "normal_kernel_cpu", [&] {
    auto iter = TensorIterator::borrowing_nullary_op(self);
    std::lock_guard<std::mutex> lock(generator->mutex_);
    cpu_serial_kernel(iter, [mean, std, generator]() -> scalar_t {
      at::normal_distribution<double> normal(mean, std);
      return static_cast<scalar_t>(normal(generator));
    });
  });
}

template<typename RNG>
struct NormalKernel {
  void operator()(Tensor& self, double mean, double std, std::optional<Generator> gen) {
    normal_kernel(self, mean, std, check_generator<RNG>(gen));
  }
};

// ==================================================== Uniform =======================================================

template<typename RNG>
void uniform_kernel(TensorIteratorBase& iter, double from_, double to_, RNG generator) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "uniform_kernel_cpu", [&]() {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    auto from = static_cast<scalar_t>(from_);
    auto to = static_cast<scalar_t>(to_);
    at::uniform_real_distribution<scalar_t> uniform(from, to);
    cpu_serial_kernel(iter, [&uniform, generator]() -> scalar_t {
      return static_cast<scalar_t>(uniform(generator));
    });
  });
}

template<typename RNG>
struct UniformKernel {
  void operator()(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
    uniform_kernel(iter, from, to, check_generator<RNG>(gen));
  }
};

// ==================================================== Cauchy ========================================================

template<typename RNG>
void cauchy_kernel(TensorIteratorBase& iter, double median, double sigma, RNG generator) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "cauchy_cpu", [&]() {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    at::cauchy_distribution<double> cauchy(median, sigma);
    cpu_serial_kernel(iter, [&cauchy, generator]() -> scalar_t {
      return static_cast<scalar_t>(cauchy(generator));
    });
  });
}

template<typename RNG>
struct CauchyKernel {
  void operator()(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
    cauchy_kernel(iter, median, sigma, check_generator<RNG>(gen));
  }
};

// ================================================== LogNormal =======================================================

template<typename RNG>
void log_normal_kernel(TensorIteratorBase& iter, double mean, double std, RNG generator) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "log_normal_cpu", [&]() {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    at::lognormal_distribution<double> logNormal(mean, std);
    cpu_serial_kernel(iter, [&logNormal, generator]() -> scalar_t {
      return static_cast<scalar_t>(logNormal(generator));
    });
  });
}

template<typename RNG>
struct LogNormalKernel {
  void operator()(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
    log_normal_kernel(iter, mean, std, check_generator<RNG>(gen));
  }
};

// =================================================== Geometric ======================================================

template<typename RNG>
void geometric_kernel(TensorIteratorBase& iter, double p, RNG generator) {
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "geometric_cpu", [&]() {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    at::geometric_distribution<double> geometric(p);
    cpu_serial_kernel(iter, [&geometric, generator]() -> scalar_t {
      return static_cast<scalar_t>(geometric(generator));
    });
  });
}

template<typename RNG>
struct GeometricKernel {
  void operator()(TensorIteratorBase& iter, double p, std::optional<Generator> gen) {
    geometric_kernel(iter, p, check_generator<RNG>(gen));
  }
};

// ================================================== Exponential =====================================================

template<typename RNG>
void exponential_kernel(TensorIteratorBase& iter, double lambda, RNG generator) {
  TORCH_CHECK(isFloatingType(iter.dtype()), "Exponential distribution is a continuous probability distribution. dtype must be a floating point but you specified ", iter.dtype());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "exponential_cpu", [&]() {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    at::exponential_distribution<double> exponential(lambda);
    cpu_serial_kernel(iter, [&exponential, generator]() -> scalar_t {
      return static_cast<scalar_t>(exponential(generator));
    });
  });
}

template<typename RNG>
struct ExponentialKernel {
  void operator()(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
    exponential_kernel(iter, lambda, check_generator<RNG>(gen));
  }
};

// ================================================== Bernoulli =======================================================

template<typename RNG>
void bernoulli_kernel(const TensorBase &self, const TensorBase &p_, RNG generator) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
  self.scalar_type(), "bernoulli_tensor_cpu_self_", [&] {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    using self_t = scalar_t;
    auto p_cpu = p_.to(kCPU);
    auto p = expand_inplace(self, p_cpu);
    auto iter = TensorIteratorConfig()
        .add_output(self)
        .add_const_input(*p)
        .check_all_same_dtype(false)
        .build();
    if (p->scalar_type() == kDouble) {
      cpu_serial_kernel(iter, [&](const double p_val) -> self_t {
        at::bernoulli_distribution<double> bernoulli(p_val);
        return static_cast<self_t>(bernoulli(generator));
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half,
      p->scalar_type(), "bernoulli_tensor_cpu_p_", [&] {
        using p_t = scalar_t;
        cpu_serial_kernel(iter, [&](const p_t p_val) -> self_t {
          at::bernoulli_distribution<float> bernoulli(p_val);
          return static_cast<self_t>(bernoulli(generator));
        });
      });
    }
  });
}

template<typename RNG>
void bernoulli_kernel(const TensorBase &self, double p, RNG generator) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
  self.scalar_type(), "bernoulli_scalar_cpu_", [&] {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    auto iter = TensorIterator::borrowing_nullary_op(self);
    cpu_serial_kernel(iter, [p, generator]() -> scalar_t {
      at::bernoulli_distribution<double> bernoulli(p);
      return static_cast<scalar_t>(bernoulli(generator));
    });
  });
}

template<typename RNG>
struct BernoulliKernel {
  void operator()(const TensorBase &self, double p, std::optional<Generator> gen) {
    bernoulli_kernel(self, p, check_generator<RNG>(gen));
  }
  void operator()(const TensorBase &self, const TensorBase &p_, std::optional<Generator> gen) {
    bernoulli_kernel(self, p_, check_generator<RNG>(gen));
  }
};

}}
