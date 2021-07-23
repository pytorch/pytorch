#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>
#include <c10/util/math_compat.h>
#include <c10/util/Optional.h>

#include <ATen/Utils.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/cpu/Loops.h>

#include <type_traits>
#include <functional>
// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <assert.h>
// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <float.h>

namespace {
/*
 * This section is a counterpart to Distributions.cu
 *
 */

// The function `sample_poisson`
// is adapted from Numpy's distributions.c implementation.
// It is MIT licensed, so here is the copyright:

/* Copyright 2005 Robert Kern (robert.kern@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


int64_t sample_poisson(double lambda, at::CPUGeneratorImpl* generator) {
  TORCH_CHECK(lambda >= 0, "invalid Poisson rate, expected rate to be non-negative");
  at::uniform_real_distribution<double> standard_uniform(0.0, 1.0);
  if (lambda >= 10) {
    // transformed rejection method, (Hoermann, 1993)
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t k;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    double U, V, a, b, invalpha, vr, us;

    double slam = std::sqrt(lambda);
    double loglam = std::log(lambda);
    b = 0.931 + 2.53 * slam;
    a = -0.059 + 0.02483 * b;
    invalpha = 1.1239 + 1.1328 / (b - 3.4);
    vr = 0.9277 - 3.6224 / (b - 2);

    // NOLINTNEXTLINE(modernize-use-bool-literals)
    while (1) {
      U = standard_uniform(generator) - 0.5;
      V = standard_uniform(generator);
      us = 0.5 - std::fabs(U);
      k = (int64_t)std::floor((2 * a / us + b) * U + lambda + 0.43);
      if ((us >= 0.07) && (V <= vr)) {
        return k;
      }
      if ((k < 0) || ((us < 0.013) && (V > us))) {
        continue;
      }
      if ((std::log(V) + std::log(invalpha) - std::log(a / (us * us) + b)) <=
          (-lambda + k * loglam - std::lgamma((double)k + 1))) {
        return k;
      }
    }
  } else if (lambda == 0) {
    return 0;
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t X;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    double prod, U, enlam;

    enlam = std::exp(-lambda);
    X = 0;
    prod = 1.0;
    // NOLINTNEXTLINE(modernize-use-bool-literals)
    while (1) {
      U = standard_uniform(generator);
      prod *= U;
      if (prod > enlam) {
        X += 1;
      } else {
        return X;
      }
    }
  }
}

} // namespace

namespace at {
namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(bernoulli_tensor_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(bernoulli_scalar_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(cauchy_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(exponential_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(multinomial_with_replacement_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(geometric_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log_normal_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(uniform_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(normal_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(random_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(random_from_to_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(random_full_64_bits_range_stub);

// ==================================================== Bernoulli =====================================================

template<typename RNG>
struct BernoulliStub {
  void operator()(Tensor& self, const Tensor& p_, c10::optional<Generator> gen) {
    bernoulli_tensor_stub(self.device().type(), self, p_, gen);
  }

  void operator()(Tensor& self, double p, c10::optional<Generator> gen) {
    bernoulli_scalar_stub(self.device().type(), self, p, gen);
  }
};

Tensor bernoulli(const Tensor& self, c10::optional<Generator> gen) {
  Tensor result = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  result.bernoulli_(self, gen);
  return result;
}

Tensor bernoulli(const Tensor& self, double p, c10::optional<Generator> gen) {
  Tensor result = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  result.bernoulli_(p, gen);
  return result;
}

Tensor& bernoulli_out(const Tensor& self, c10::optional<Generator> gen, Tensor& result) {
  return at::native::templates::bernoulli_out_impl<BernoulliStub, Generator>(result, self, gen);
}

Tensor& bernoulli_(Tensor& self, const Tensor& p_, c10::optional<Generator> gen) {
  return at::native::templates::bernoulli_impl_<BernoulliStub, Generator>(self, p_, gen);
}

Tensor& bernoulli_(Tensor& self, double p, c10::optional<Generator> gen) {
  return at::native::templates::bernoulli_impl_<BernoulliStub, Generator>(self, p, gen);
}

// ================================================== LogNormal =======================================================

template<typename RNG>
struct LogNormalStub {
  void operator()(TensorIteratorBase& iter, double mean, double std, c10::optional<Generator> gen) {
    log_normal_stub(iter.device_type(), iter, mean, std, gen);
  }
};

Tensor& log_normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  return at::native::templates::log_normal_impl_<LogNormalStub, Generator>(self, mean, std, gen);
}

// ==================================================== Cauchy ========================================================

template<typename RNG>
struct CauchyStub {
  void operator()(TensorIteratorBase& iter, double median, double sigma, c10::optional<Generator> gen) {
    cauchy_stub(iter.device_type(), iter, median, sigma, gen);
  }
};

Tensor& cauchy_(Tensor& self, double median, double sigma, c10::optional<Generator> gen) {
  return at::native::templates::cauchy_impl_<CauchyStub, Generator>(self, median, sigma, gen);
}

// ================================================== Exponential =====================================================

template<typename RNG>
struct ExponentialStub {
  void operator()(TensorIteratorBase& iter, double lambda, c10::optional<Generator> gen) {
    exponential_stub(iter.device_type(), iter, lambda, gen);
  }
};

Tensor& exponential_(Tensor& self, double lambda, c10::optional<Generator> gen) {
  return at::native::templates::exponential_impl_<ExponentialStub, Generator>(self, lambda, gen);
}

// =================================================== Geometric ======================================================

template<typename RNG>
struct GeometricStub {
  void operator()(TensorIteratorBase& iter, double p, c10::optional<Generator> gen) {
    geometric_stub(iter.device_type(), iter, p, gen);
  }
};

Tensor& geometric_(Tensor& self, double p, c10::optional<Generator> gen) {
  return at::native::templates::geometric_impl_<GeometricStub, Generator>(self, p, gen);
}

// ==================================================== Uniform =======================================================

template<typename RNG>
struct UniformStub {
  void operator()(TensorIteratorBase& iter, double from, double to, c10::optional<Generator> gen) {
    uniform_stub(iter.device_type(), iter, from, to, gen);
  }
};

template<typename RNG>
struct UniformMeta {
  // No-op!
  void operator()(TensorIteratorBase& iter, double from, double to, c10::optional<Generator> gen) {
  }
};

Tensor& uniform_(Tensor& self, double from, double to, c10::optional<Generator> gen) {
  return at::native::templates::uniform_impl_<UniformStub, Generator>(self, from, to, gen);
}

Tensor& uniform_meta_(Tensor& self, double from, double to, c10::optional<Generator> gen) {
  return at::native::templates::uniform_impl_<UniformMeta, Generator>(self, from, to, gen);
}

// ==================================================== Normal ========================================================

template<typename RNG>
struct NormalStub {
  void operator()(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
    normal_stub(self.device().type(), self, mean, std, gen);
  }
};

Tensor& normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  return at::native::templates::normal_impl_<NormalStub, Generator>(self, mean, std, gen);
}

Tensor& normal_meta_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);  // TODO: dedupe
  return self;
}

Tensor& normal_out(const Tensor& mean, double std, c10::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, gen);
}

Tensor& normal_out(double mean, const Tensor& std, c10::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, gen);
}

Tensor& normal_out(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, gen);
}

Tensor normal(const Tensor& mean, double std, c10::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalStub, Generator>(mean, std, gen);
}

Tensor normal(double mean, const Tensor& std, c10::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalStub, Generator>(mean, std, gen);
}

Tensor normal(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalStub, Generator>(mean, std, gen);
}

// ==================================================== Random ========================================================

template<typename RNG>
struct RandomStub {
  void operator()(TensorIteratorBase& iter, c10::optional<Generator> gen) {
    random_stub(iter.device_type(), iter, gen);
  }
};

Tensor& random_(Tensor& self, c10::optional<Generator> gen) {
  return at::native::templates::random_impl<RandomStub, Generator>(self, gen);
}

template<typename RNG>
struct RandomFromToStub {
  void operator()(TensorIteratorBase& iter, uint64_t range, int64_t from, c10::optional<Generator> gen) {
    random_from_to_stub(iter.device_type(), iter, range, from, gen);
  }
  void operator()(TensorIteratorBase& iter, c10::optional<Generator> gen) {
    random_full_64_bits_range_stub(iter.device_type(), iter, gen);
  }
};

template<typename RNG>
struct RandomFromToMeta {
  // No-op!
  void operator()(TensorIteratorBase& iter, uint64_t range, int64_t from, c10::optional<Generator> gen) {
  }
  void operator()(TensorIteratorBase& iter, c10::optional<Generator> gen) {
  }
};

Tensor& random_(Tensor& self, int64_t from, optional<int64_t> to, c10::optional<Generator> gen) {
  return at::native::templates::random_from_to_impl<RandomFromToStub, Generator>(self, from, to, gen);
}

Tensor& random_(Tensor& self, int64_t to, c10::optional<Generator> gen) {
  return random_(self, 0, to, gen);
}

Tensor& random_meta_(Tensor& self, c10::optional<Generator> gen) {
  // No error checking yay
  return self;
}

Tensor& random_meta_(Tensor& self, int64_t from, optional<int64_t> to, c10::optional<Generator> gen) {
  return at::native::templates::random_from_to_impl<RandomFromToMeta, Generator>(self, from, to, gen);
}

Tensor& random_meta_(Tensor& self, int64_t to, c10::optional<Generator> gen) {
  return random_meta_(self, 0, to, gen);
}

// ====================================================================================================================

Tensor _standard_gamma_grad_cpu(const Tensor& self, const Tensor& output) {
  Tensor ret = at::empty(self.sizes(), self.options());
  auto iter = TensorIteratorConfig()
    .add_output(ret)
    .add_input(self)
    .add_input(output)
    .build();
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "_standard_gamma_grad_cpu", [&] {
    cpu_serial_kernel(iter, [](scalar_t self_val, scalar_t output_val) -> scalar_t{
      return standard_gamma_grad_one<scalar_t, double>(self_val, output_val);
    });
  });
  return ret;
}

Tensor _dirichlet_grad_cpu(const Tensor& x, const Tensor& alpha, const Tensor& total) {
  Tensor ret = at::empty(x.sizes(), x.options());
  auto iter = TensorIteratorConfig()
    .add_output(ret)
    .add_input(x)
    .add_input(alpha)
    .add_input(total)
    .build();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "_dirichlet_grad_cpu", [&] {
    cpu_serial_kernel(iter, [](scalar_t x_val, scalar_t alpha_val, scalar_t total_val) -> scalar_t{
      return dirichlet_grad_one<scalar_t, double>(x_val, alpha_val, total_val);
    });
  });
  return ret;
}

/*
 * This section is a counterpart to Distributions.cu
 */

Tensor _s_binomial_cpu(const Tensor& count, const Tensor& prob, c10::optional<Generator> gen) {
  Tensor ret = at::zeros(count.sizes(), count.options());
  auto iter = TensorIteratorConfig()
    .add_output(ret)
    .add_input(count)
    .add_input(prob)
    .build();
  AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "binomial_cpu", [&] {
    CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    cpu_serial_kernel(iter, [generator](scalar_t count_val, scalar_t prob_val) -> scalar_t{
      auto uniform_lambda = [generator] () {
        at::uniform_real_distribution<double> standard_uniform(0.0, 1.0);
        return standard_uniform(generator);
      };
      BaseSampler<double, decltype(uniform_lambda)> standard_uniform(uniform_lambda);

      auto sample = sample_binomial<scalar_t, double, decltype(uniform_lambda)>(count_val, prob_val, standard_uniform);
      return static_cast<scalar_t>(sample);
    });
  });
  return ret;
}

Tensor _s_poisson_cpu(const Tensor& lambda, c10::optional<Generator> gen) {
  Tensor ret = at::zeros(lambda.sizes(), lambda.options());
  auto iter = TensorIteratorConfig()
    .add_output(ret)
    .add_input(lambda)
    .build();
  AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "poisson_cpu", [&] {
    CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    cpu_serial_kernel(iter, [generator](scalar_t lambda_val) -> scalar_t{
      return static_cast<scalar_t>(sample_poisson(static_cast<double>(lambda_val), generator));
    });
  });
  return ret;
}

Tensor _s_gamma_cpu(const Tensor& alpha, c10::optional<Generator> gen) {
  Tensor ret = at::zeros(alpha.sizes(), alpha.options());
  auto iter = TensorIteratorConfig()
    .add_output(ret)
    .add_input(alpha)
    .build();
  AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "gamma_cpu", [&] {
    CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    cpu_serial_kernel(iter, [generator](scalar_t alpha_val) -> scalar_t{
      auto uniform_lambda = [generator] () {
        at::uniform_real_distribution<double> standard_uniform(0.0, 1.0);
        return standard_uniform(generator);
      };
      BaseSampler<double, decltype(uniform_lambda)> standard_uniform(uniform_lambda);

      auto normal_lambda = [generator] () {
        at::normal_distribution<double> normal(0.0, 1.0);
        return normal(generator);
      };
      BaseSampler<double, decltype(normal_lambda)> standard_normal(normal_lambda);
      auto sample = sample_gamma<scalar_t, double, decltype(uniform_lambda), decltype(normal_lambda)>(alpha_val, standard_uniform, standard_normal);
      return std::max(std::numeric_limits<scalar_t>::min(), (scalar_t) sample);
    });
  });

  return ret;
}

Tensor _s_dirichlet_cpu(const Tensor& alpha, c10::optional<Generator> gen) {
  Tensor ret = at::zeros(alpha.sizes(), alpha.options());
  AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "dirichlet", [&] {
    Tensor gamma = at::zeros(alpha.sizes(), alpha.options().dtype(ScalarType::Double));
    CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    /* Generate gamma sample by casting alpha to double to prevent underflow. */
    auto iter1 = TensorIteratorConfig()
      .add_output(gamma)
      .add_input(alpha)
      .check_all_same_dtype(false)
      .build();
    cpu_serial_kernel(iter1, [generator](scalar_t alpha_val) -> double{
      auto uniform_lambda = [generator] () {
        at::uniform_real_distribution<double> standard_uniform(0.0, 1.0);
        return standard_uniform(generator);
      };
      BaseSampler<double, decltype(uniform_lambda)> standard_uniform(uniform_lambda);

      auto normal_lambda = [generator] () {
        at::normal_distribution<double> normal(0.0, 1.0);
        return normal(generator);
      };
      BaseSampler<double, decltype(normal_lambda)> standard_normal(normal_lambda);
      auto sample = sample_gamma<double, double, decltype(uniform_lambda), decltype(normal_lambda)>
        (alpha_val, standard_uniform, standard_normal);
      return std::max(std::numeric_limits<double>::min(), sample);
    });
    /* Normalize and cast back to scalar_t. */
    Tensor gamma_sum = gamma.sum(-1, true).expand(alpha.sizes());
    auto iter2 = TensorIteratorConfig()
      .add_output(ret)
      .add_input(gamma)
      .add_input(gamma_sum)
      .check_all_same_dtype(false)
      .build();
    cpu_serial_kernel(iter2, [](double gamma_val, double gamma_sum_val) -> scalar_t{
      auto ret_val = gamma_val / gamma_sum_val;
      auto min_val = std::numeric_limits<scalar_t>::min();
      auto max_val = std::nexttoward(static_cast<scalar_t>(1.0f), 0.0f);
      return std::min(max_val, std::max(min_val, static_cast<scalar_t>(ret_val)));
    });
  });
  return ret;
}

/* The largest consecutive integer representable in float32 (2^24) */
constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (FLT_MANT_DIG);

Tensor& multinomial_out(const Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    c10::optional<Generator> gen,
    Tensor& result) {
  TORCH_CHECK(
      result.device() == self.device(),
      "multinomial arguments must have the same device");
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "multinomial only supports floating-point dtypes for input, got: ",
      self.scalar_type());
  TORCH_CHECK(result.scalar_type() == ScalarType::Long,
      "multinomial expects Long tensor out, got: ", result.scalar_type());
  TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
  int64_t n_categories = self.size(-1);
  TORCH_CHECK(with_replacement || (n_sample <= n_categories),
      "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  TORCH_CHECK(
      n_categories <= FLOAT32_MAX_CONSECUTIVE_INT,
      "number of categories cannot exceed 2^24");

  if (self.dim() == 1) {
    result.resize_({n_sample});
  } else {
    const int64_t n_dist = self.size(0);
    result.resize_({n_dist, n_sample});
  }
  if (result.numel() == 0) {
    return result;
  }

  // Fast-path for no replacement.
  // Reference:
  // https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
  // Half is not supported on CPU.
  TORCH_CHECK(
      !(self.device().is_cpu() && self.scalar_type() == ScalarType::Half),
      "multinomial is not implemented for half on CPU");
  if (!with_replacement) {
    // Sanity checks on `self`.
    auto is_valid = ((self.max() < INFINITY) & (self.min() >= 0)).item();
    TORCH_CHECK(
        is_valid.to<bool>(),
        "probability tensor contains either `inf`, `nan` or element < 0");
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool zero_prob_condition;
    if (self.dim() == 1){
      zero_prob_condition = (self.sum() == 0).item().to<bool>();
    } else {
      zero_prob_condition = (self.sum(1) == 0).sum().item().to<bool>();
    }
    TORCH_CHECK(
        !zero_prob_condition,
        "invalid multinomial distribution (sum of probabilities <= 0)");

    // The algorithm is from gumbel softmax.
    // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
    // Here we can apply exp to the formula which will not affect result of
    // argmax or topk. Then we have
    // s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
    // We can also simplify the formula above by
    // s = argmax( p / q ) where q ~ Exp(1)
    Tensor q = at::empty_like(self).exponential_(1, gen);
    // In theory the probability to generate 0 from exponential distribution is
    // 0. However, on CUDA side there is a protection to avoid 0s, but on CPU
    // side, there is a very low probability to generate 0 from
    // exponential<double>. The probability is about 2^(-DBL_MANT_DIG). We just
    // ignore it here, but there may be some risk to get invalid output on CPU.
    at::div_out(q, self, q);
    if (n_sample == 1) {
      at::argmax_out(result, q, /*dim=*/-1, /*keepdim=*/true);
    } else {
      Tensor vals = at::empty(result.sizes(), self.options());
      at::topk_out(vals, result, q, n_sample);
    }
    return result;
  }

  multinomial_with_replacement_stub(
      result.device().type(), result, self, n_sample, gen);
  return result;
}

Tensor multinomial(
    const Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    c10::optional<Generator> gen) {
  Tensor result = at::empty({0}, self.options().dtype(kLong));
  native::multinomial_out(self, n_sample, with_replacement, gen, result);
  return result;
}

}} // namespace at::native
