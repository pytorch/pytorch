#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/cpu/DistributionTemplates.h>

#include <ATen/native/UnaryOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <cmath>
#include <limits>
#include <type_traits>

#if AT_MKL_ENABLED()
#include <mkl.h>
#include <ATen/mklrng/MKLGeneratorImpl.h>
#include <cpuinfo.h>
#endif

namespace at::native {
namespace {

void cauchy_kernel(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::cauchy_kernel(iter, median, sigma, generator);
}

void bernoulli_tensor_kernel(const TensorBase &self, const TensorBase &p_, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::bernoulli_kernel(self, p_, generator);
}

#if !AT_MKL_ENABLED()
void bernoulli_scalar_kernel_default(const TensorBase &self, double p, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::bernoulli_kernel(self, p, generator);
}

void bernoulli_scalar_kernel(const TensorBase &self, double p, std::optional<Generator> gen) {
  bernoulli_scalar_kernel_default(self, p, gen);
}
#else
void bernoulli_scalar_kernel(const TensorBase &self, double p, std::optional<Generator> gen) {
  int64_t n = self.numel();
  bool contig = self.is_contiguous();

  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
  self.scalar_type(), "bernoulli_scalar_cpu_", [&] {
    at::Tensor tmp_int_tensor;
    if (std::is_same_v<scalar_t, int> && contig) {
      tmp_int_tensor = self;
    } else {
      tmp_int_tensor = at::empty(self.sizes(), self.options().dtype(at::kInt));
    }

    scalar_t *self_ptr = self.data_ptr<scalar_t>();
    int *sample_int_ptr = tmp_int_tensor.data_ptr<int>();

    auto mklGenerator = check_generator<MKLGeneratorImpl>(detail::getDefaultMKLGenerator());
    VSLStreamStatePtr main_stream;

    // Get a local copy of the global stream and immediately advance the global
    // state before the generation step to avoid multiple threads using the same state.
    {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(mklGenerator->mutex_);
    mklGenerator->get_stream_copy(main_stream);
    mklGenerator->skip_ahead(n);
    }

    auto sample = [&](int64_t begin, int64_t end) {
      int64_t len = end - begin;
      if (len > 0) {
        VSLStreamStatePtr sample_stream;
        vslCopyStream(&sample_stream, main_stream);
        vslSkipAheadStream(sample_stream, begin);

        viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, sample_stream, len,
          sample_int_ptr + begin, p);
        vslDeleteStream(&sample_stream);

        // vectorized copy if using buffer and contiguous, i.e., being non-int
        // type and contiguous
        if (!std::is_same_v<scalar_t, int> && contig) {
          scalar_t *self_seg = self_ptr + begin;
          int* tmp_seg = sample_int_ptr + begin;
          at::vec::convert<int, scalar_t>(tmp_seg, self_seg, len);
        }
      }
    };

    parallel_for(0, n, /* grain_size= */ 800, sample);
    vslDeleteStream(&main_stream);

    // copy_ if using buffer and non contiguous
    if (!contig) {
      OptionalTensorRef(self)->copy_(tmp_int_tensor);
    }
  });
}
#endif

void exponential_kernel_default(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::exponential_kernel(iter, lambda, generator);
}

#if (!AT_MKL_ENABLED() || defined(FBCODE_CAFFE2))
void exponential_kernel(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
  exponential_kernel_default(iter, lambda, gen);
}
#else
void exponential_kernel(TensorIteratorBase &iter, double lambda, std::optional<Generator> gen) {
  TORCH_CHECK(isFloatingType(iter.dtype()), "Exponential distribution is a continuous probability distribution. dtype must be a floating point but you specified ", iter.dtype());
  Tensor self = iter.tensor(0);
  if (lambda > 0 && !std::isinf(lambda) && !std::isnan(lambda)) {
    int64_t n = self.numel();
    bool contig = self.is_contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "exponential_cpu", [&] {
      at::Tensor tmp_tensor;
      constexpr bool is_df = std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>;
      if (is_df && contig) {
        tmp_tensor = self;
      } else if (std::is_same_v<scalar_t, double>) {
        tmp_tensor = at::empty(self.sizes(), self.options().dtype(at::kDouble));
      } else {
        tmp_tensor = at::empty(self.sizes(), self.options().dtype(at::kFloat));
      }

      scalar_t *self_ptr = self.data_ptr<scalar_t>();
      using tmp_scalar_t = typename std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
      tmp_scalar_t *sample_ptr = tmp_tensor.data_ptr<tmp_scalar_t>();

      // Intel MKL vRngExponential variate originally does not exclude 0.
      // However, to align with pytorch exponential variate definition which excludes 0,
      // we shift the MKL vRngExponential distribution location by adding a very small constant, eps.
      // If X ~ Exp(lambda), then E(X) = 1/lambda, and V(X) = 1/lambda**2.
      // If Y = X + eps, where eps ~= 0, then E(Y) = (1/lambda) + eps, and V(Y) = 1/lambda**2.
      // If eps is very small, the two distributions are indistinguishable, and are almost identical.
      // The detail of location-shifted MKL vRngExponential is as follows.
      // PDF:         f(x) = lambda * exp( -lambda * (x - eps) )
      // CDF:         F(x) = 1 - exp( -lambda * (x - eps) )
      // Mean:        E[X+eps] = (1/lambda) + eps
      // Variance:    V[X+eps] = 1/lambda**2
      auto eps = std::numeric_limits<tmp_scalar_t>::min();

      auto mklGenerator = check_generator<MKLGeneratorImpl>(detail::getDefaultMKLGenerator());
      VSLStreamStatePtr main_stream;

      // Get a local copy of the global stream and immediately advance the global
      // state before the generation step to avoid multiple threads using the same state.
      {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(mklGenerator->mutex_);
        mklGenerator->get_stream_copy(main_stream);
        mklGenerator->skip_ahead(n);
      }

      auto sample = [&](int64_t begin, int64_t end) {
        int64_t len = end - begin;
        if (len > 0) {
          VSLStreamStatePtr sample_stream;
          vslCopyStream(&sample_stream, main_stream);
          vslSkipAheadStream(sample_stream, begin);

          if constexpr (std::is_same_v<scalar_t, double>) {
            vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF, sample_stream, len,
              (double *)(sample_ptr + begin), eps, 1./lambda);
            vslDeleteStream(&sample_stream);
          } else {
            vsRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF, sample_stream, len,
              (float *) (sample_ptr + begin), eps, 1./lambda);
            vslDeleteStream(&sample_stream);
          }

          // vectorized copy if using buffer and contiguous
          if (!is_df && contig) {
            scalar_t *self_seg = self_ptr + begin;
            tmp_scalar_t *tmp_seg = sample_ptr + begin;
            at::vec::convert<tmp_scalar_t, scalar_t>(tmp_seg, self_seg, len);
          }
        }
      };

      parallel_for(0, n, /* grain_size= */ 800, sample);
      vslDeleteStream(&main_stream);

      // copy_ if using buffer and non contiguous
      if (!contig) {
        self.copy_(tmp_tensor);
      }
    });
  } else {
    // The situation of inf and nan, move to using the default version
    exponential_kernel_default(iter, lambda, gen);
  }
}
#endif

void geometric_kernel(TensorIteratorBase& iter, double p, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::geometric_kernel(iter, p, generator);
}

void log_normal_kernel(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::log_normal_kernel(iter, mean, std, generator);
}

void uniform_kernel(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::uniform_kernel(iter, from, to, generator);
}

void normal_kernel(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::normal_kernel(self, mean, std, generator);
}

void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::random_from_to_kernel(iter, range, base, generator);
}

void random_kernel(TensorIteratorBase& iter, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::random_kernel(iter, generator);
}

// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
void random_full_64_bits_range_kernel(TensorIteratorBase& iter, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::random_full_64_bits_range_kernel(iter, generator);
}

} // namespace (anonymous)

REGISTER_DISPATCH(bernoulli_tensor_stub, &bernoulli_tensor_kernel)
REGISTER_DISPATCH(bernoulli_scalar_stub, &bernoulli_scalar_kernel)
REGISTER_DISPATCH(cauchy_stub, &cauchy_kernel)
REGISTER_DISPATCH(exponential_stub, &exponential_kernel)
REGISTER_DISPATCH(geometric_stub, &geometric_kernel)
REGISTER_DISPATCH(log_normal_stub, &log_normal_kernel)
REGISTER_DISPATCH(normal_stub, &normal_kernel)
REGISTER_DISPATCH(uniform_stub, &uniform_kernel)
REGISTER_DISPATCH(random_from_to_stub, &random_from_to_kernel)
REGISTER_DISPATCH(random_full_64_bits_range_stub, &random_full_64_bits_range_kernel)
REGISTER_DISPATCH(random_stub, &random_kernel)

} // namespace at::native
