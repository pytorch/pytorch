#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Generator.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/cpu/DistributionTemplates.h>

#include <ATen/native/UnaryOps.h>

#include <cmath>
#include <limits>
#include <type_traits>

#if AT_MKL_ENABLED()
#include <mkl.h>
#include <cpuinfo.h>
#endif

namespace at { namespace native {
namespace {

static void cauchy_kernel(TensorIteratorBase& iter, double median, double sigma, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::cauchy_kernel(iter, median, sigma, generator);
}

void bernoulli_tensor_kernel(const TensorBase &self, const TensorBase &p_, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::bernoulli_kernel(self, p_, generator);
}

void bernoulli_scalar_kernel_default(const TensorBase &self, double p, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::bernoulli_kernel(self, p, generator);
}

#if !AT_MKL_ENABLED()
void bernoulli_scalar_kernel(const TensorBase &self, double p, c10::optional<Generator> gen) {
  bernoulli_scalar_kernel_default(self, p, gen);
}
#else
void bernoulli_scalar_kernel(const TensorBase &self, double p, c10::optional<Generator> gen) {
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

    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "bernoulli_scalar_cpu_", [&] {
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
            at::vec::convert<int, scalar_t>(tmp_seg, self_seg, len);
          }
        }
      };

      parallel_for(0, n, /* grain_size= */ 800, sample);

      // copy_ if using buffer and non contiguous
      if (!contig) {
        OptionalTensorRef(self)->copy_(tmp_int_tensor);
      }
    });
  } else {
    // The situation of AMD, move to using the default version
    bernoulli_scalar_kernel_default(self, p, gen);
  }
}
#endif

static void exponential_kernel(TensorIteratorBase& iter, double lambda, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::exponential_kernel(iter, lambda, generator);
}

static void geometric_kernel(TensorIteratorBase& iter, double p, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::geometric_kernel(iter, p, generator);
}

static void log_normal_kernel(TensorIteratorBase& iter, double mean, double std, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::log_normal_kernel(iter, mean, std, generator);
}

void uniform_kernel(TensorIteratorBase& iter, double from, double to, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::uniform_kernel(iter, from, to, generator);
}

void normal_kernel(const TensorBase &self, double mean, double std, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::normal_kernel(self, mean, std, generator);
}

static void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::random_from_to_kernel(iter, range, base, generator);
}

static void random_kernel(TensorIteratorBase& iter, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::random_kernel(iter, generator);
}

// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
static void random_full_64_bits_range_kernel(TensorIteratorBase& iter, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::random_full_64_bits_range_kernel(iter, generator);
}

} // namespace (anonymous)

REGISTER_DISPATCH(bernoulli_tensor_stub, &bernoulli_tensor_kernel);
REGISTER_DISPATCH(bernoulli_scalar_stub, &bernoulli_scalar_kernel);
REGISTER_DISPATCH(cauchy_stub, &cauchy_kernel);
REGISTER_DISPATCH(exponential_stub, &exponential_kernel);
REGISTER_DISPATCH(geometric_stub, &geometric_kernel);
REGISTER_DISPATCH(log_normal_stub, &log_normal_kernel);
#ifdef CPU_CAPABILITY_AVX512
// normal_stub isn't being dispatched to AVX512 because it exposes
// flakiness in test_sgd of test/test_optim.py
REGISTER_NO_AVX512_DISPATCH(normal_stub);
#else
REGISTER_DISPATCH(normal_stub, &normal_kernel);
#endif
REGISTER_DISPATCH(uniform_stub, &uniform_kernel);
REGISTER_DISPATCH(random_from_to_stub, &random_from_to_kernel);
REGISTER_DISPATCH(random_full_64_bits_range_stub, &random_full_64_bits_range_kernel);
REGISTER_DISPATCH(random_stub, &random_kernel);

} // namespace native
} // namespace at
