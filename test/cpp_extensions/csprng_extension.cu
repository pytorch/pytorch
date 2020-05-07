#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/core/DistributionsHelper.h>
#include <memory>
#include "block_cipher.cuh"
#include "aes.cuh"

using namespace at;
using namespace at::native::templates;
using namespace torch::custom_prng;

// CUDA CSPRNG is actually CPU generator which is used only to generate a random key on CPU for AES running in a block mode on CUDA 
struct CUDA_CSPRNG_GeneratorImpl : public CPUGeneratorImpl {
  CUDA_CSPRNG_GeneratorImpl(uint64_t seed_in = default_rng_seed_val) : CPUGeneratorImpl(seed_in) {
    this->key_set_ = DispatchKeySet(DispatchKey::CustomRNGKeyId);
  }
};

// ====================================================================================================================

// Applies AES in CTR mode with the `key` for passed TensorIterator iter.
// `scalar_t`       is a scalar type equivalent of target tensor dtype
// `uint_t`         is an unsigned integral type of sub-blocks that random state is divided to
//                  (e.g, 16 bytes random state block can be divided into 16 uint8_t sub-blocks 
//                  or 8 uint16_t sub-block or 4 uint32_t sub-block or 2 uint64_t sub-blocks)
// `N`              is a number of sub-block which is used by `transform_func` 
//                  to generate a random value of specific distribution (e.g. `normal` uses 2)
// `key`            is a CUDA pointer to random key memory block
// `transform_func` is a callable that converts N `uint_t` random state sub-blocks passed in RNGValues into target dtype `scalar_t`
template<typename scalar_t, typename uint_t, size_t N = 1, typename transform_t>
void aes_helper(TensorIterator& iter, const uint8_t* key, transform_t transform_func) {
  block_cipher_ctr_mode<scalar_t, uint_t, N>(iter, aes::block_t_size,
    [key] __device__ (unsigned int idx) -> aes::block_t {
      aes::block_t block;
      memset(&block, 0, aes::block_t_size);
      *(reinterpret_cast<unsigned int*>(&block)) = idx;
      aes::encrypt(reinterpret_cast<uint8_t*>(&block), key);
      return block;
    },
    transform_func
  );
}

// ====================================================================================================================

// A mapping between scalar type and corresponding unsigned integer type of random state sub-block.
// uint64_t for double and long, uint32_t for the rest
template <typename T>
struct UIntType {};

template <> struct UIntType<double> { using type = uint64_t; };
template <> struct UIntType<float> { using type = uint32_t; };
template <> struct UIntType<int64_t> { using type = uint64_t; };
template <> struct UIntType<int32_t> { using type = uint32_t; };
template <> struct UIntType<int16_t> { using type = uint32_t; };
template <> struct UIntType<int8_t> { using type = uint32_t; };
template <> struct UIntType<uint8_t> { using type = uint32_t; };
template <> struct UIntType<bool> { using type = uint32_t; };

// ==================================================== Random ========================================================

template<typename RNG>
struct RandomKernel {
  void operator()(TensorIterator& iter, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator, aes::block_t_size, iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, iter.dtype(), "my_random_kernel_cuda", [&] {
      aes_helper<scalar_t, UIntType<scalar_t>::type>(iter, key,
        [] __device__ (RNGValues<1>* generator) -> scalar_t {
          uniform_int_distribution<scalar_t> random;
          return random(generator);
        }
      );
    });
  }
};

template<typename scalar_t, typename uint_t>
void random_from_to_kernel_helper(TensorIterator& iter, uint64_t range, int64_t base, const uint8_t* key) {
  aes_helper<scalar_t, uint_t>(iter, key,
    [range, base] __device__ (RNGValues<1>* generator) -> scalar_t {
      uniform_int_from_to_distribution<scalar_t> random(range, base);
      return random(generator);
    }
  );
}

template<typename scalar_t, typename uint_t>
void random_full_range_kernel_helper(TensorIterator& iter, const uint8_t* key) {
  aes_helper<scalar_t, uint_t>(iter, key,
    [] __device__ (RNGValues<1>* generator) -> scalar_t {
      uniform_int_full_range_distribution<scalar_t> random;
      return random(generator);
    }
  );
}

template<typename RNG>
struct RandomFromToKernel {
  void operator()(TensorIterator& iter, uint64_t range, int64_t base, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator, aes::block_t_size, iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_from_to_kernel_cuda", [&] {
      if ((
        std::is_same<scalar_t, int64_t>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, at::BFloat16>::value) && range >= 1ULL << 32)
      {
        random_from_to_kernel_helper<scalar_t, uint64_t>(iter, range, base, key);
      } else {
        random_from_to_kernel_helper<scalar_t, uint32_t>(iter, range, base, key);
      }
    });
  }
  void operator()(TensorIterator& iter, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator, aes::block_t_size, iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel_cuda", [&] {
      if (std::is_same<scalar_t, int64_t>::value ||
          std::is_same<scalar_t, double>::value ||
          std::is_same<scalar_t, float>::value ||
          std::is_same<scalar_t, at::BFloat16>::value)
      {
        random_full_range_kernel_helper<scalar_t, uint64_t>(iter, key);
      } else {
        TORCH_CHECK(false, "random_full_64_bits_range_kernel_cuda handles only int64, double, float and bfloat16");
      }
    });
  }
};

Tensor& random_(Tensor& self, c10::optional<Generator> generator) {
  return random_impl<RandomKernel, CUDA_CSPRNG_GeneratorImpl>(self, generator);
}

Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, c10::optional<Generator> generator) {
  return random_from_to_impl<RandomFromToKernel, CUDA_CSPRNG_GeneratorImpl>(self, from, to, generator);
}

Tensor& random_to(Tensor& self, int64_t to, c10::optional<Generator> generator) {
  return random_from_to(self, 0, to, generator);
}

// ==================================================== Uniform =======================================================

template<typename RNG>
struct UniformKernel {
  void operator()(TensorIterator& iter, double from, double to, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator, aes::block_t_size, iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "uniform_kernel_cuda", [&] {
      aes_helper<scalar_t, UIntType<scalar_t>::type>(iter, key,
        [from, to] __device__ (RNGValues<1>* generator) -> scalar_t {
          uniform_real_distribution<scalar_t> uniform(from, to);
          return uniform(generator);
        }
      );
    });
  }
};

Tensor& uniform_(Tensor& self, double from, double to, c10::optional<Generator> generator) {
  return uniform_impl_<UniformKernel, CUDA_CSPRNG_GeneratorImpl>(self, from, to, generator);
}

// ==================================================== Normal ========================================================

template<typename RNG>
struct NormalKernel {
  void operator()(Tensor& self, double mean, double std, c10::optional<Generator> generator) {
    auto iter = TensorIterator::nullary_op(self);
    const auto key_t = key_tensor(generator, aes::block_t_size, iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "normal_kernel_cuda", [&] {
      aes_helper<scalar_t, UIntType<scalar_t>::type, 2>(iter, key,
        [mean, std] __device__ (RNGValues<2>* gen) -> scalar_t {
          normal_distribution<scalar_t> normal(mean, std);
          return normal(gen);
        }
      );
    });
  }
};

Tensor& normal_(Tensor& self, double mean, double std, c10::optional<Generator> generator) {
  return normal_impl_<NormalKernel, CUDA_CSPRNG_GeneratorImpl>(self, mean, std, generator);
}

Tensor& normal_Tensor_float_out(Tensor& output, const Tensor& mean, double std, c10::optional<Generator> gen) {
  return normal_out_impl<NormalKernel, CUDA_CSPRNG_GeneratorImpl>(output, mean, std, gen);
}

Tensor& normal_float_Tensor_out(Tensor& output, double mean, const Tensor& std, c10::optional<Generator> gen) {
  return normal_out_impl<NormalKernel, CUDA_CSPRNG_GeneratorImpl>(output, mean, std, gen);
}

Tensor& normal_Tensor_Tensor_out(Tensor& output, const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  return normal_out_impl<NormalKernel, CUDA_CSPRNG_GeneratorImpl>(output, mean, std, gen);
}

Tensor normal_Tensor_float(const Tensor& mean, double std, c10::optional<Generator> gen) {
  return normal_impl<NormalKernel, CUDA_CSPRNG_GeneratorImpl>(mean, std, gen);
}

Tensor normal_float_Tensor(double mean, const Tensor& std, c10::optional<Generator> gen) {
  return normal_impl<NormalKernel, CUDA_CSPRNG_GeneratorImpl>(mean, std, gen);
}

Tensor normal_Tensor_Tensor(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  return normal_impl<NormalKernel, CUDA_CSPRNG_GeneratorImpl>(mean, std, gen);
}

// ==================================================== Cauchy ========================================================

template<typename RNG>
struct CauchyKernel {
  void operator()(TensorIterator& iter, double median, double sigma, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator, aes::block_t_size, iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "cauchy_kernel_cuda", [&] {
      aes_helper<scalar_t, UIntType<scalar_t>::type, 1>(iter, key,
        [median, sigma] __device__ (RNGValues<1>* gen) -> scalar_t {
          cauchy_distribution<scalar_t> cauchy(median, sigma);
          return cauchy(gen);
        }
      );
    });
  }
};

Tensor& cauchy_(Tensor& self, double median, double sigma, c10::optional<Generator> generator) {
  return cauchy_impl_<CauchyKernel, CUDA_CSPRNG_GeneratorImpl>(self, median, sigma, generator);
}

// ================================================== LogNormal =======================================================

template<typename RNG>
struct LogNormalKernel {
  void operator()(TensorIterator& iter, double mean, double std, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator, aes::block_t_size, iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "log_normal_cuda", [&] {
      aes_helper<scalar_t, UIntType<scalar_t>::type, 2>(iter, key,
        [mean, std] __device__ (RNGValues<2>* gen) -> scalar_t {
          lognormal_distribution<scalar_t> logNormal(mean, std);
          return logNormal(gen);
        }
      );
    });
  }
};

Tensor& log_normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  return log_normal_impl_<LogNormalKernel, CUDA_CSPRNG_GeneratorImpl>(self, mean, std, gen);
}

// ================================================== Geometric =======================================================

template<typename RNG>
struct GeometricKernel {
  void operator()(TensorIterator& iter, double p, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator, aes::block_t_size, iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "geometric_kernel_cuda", [&] {
      aes_helper<scalar_t, UIntType<scalar_t>::type, 1>(iter, key,
        [p] __device__ (RNGValues<1>* gen) -> scalar_t {
          geometric_distribution<scalar_t> geometric(p);
          return geometric(gen);
        }
      );
    });
  }
};

Tensor& geometric_(Tensor& self, double p, c10::optional<Generator> gen) {
  return geometric_impl_<GeometricKernel, CUDA_CSPRNG_GeneratorImpl>(self, p, gen);
}

// ================================================== Exponential =====================================================

template<typename RNG>
struct ExponentialKernel {
  void operator()(TensorIterator& iter, double lambda, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator, aes::block_t_size, iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "exponential_kernel_cuda", [&] {
      aes_helper<scalar_t, UIntType<scalar_t>::type, 1>(iter, key,
        [lambda] __device__ (RNGValues<1>* gen) -> scalar_t {
          exponential_distribution<scalar_t> exponential(lambda);
          return exponential(gen);
        }
      );
    });
  }
};

Tensor& exponential_(Tensor& self, double lambda, c10::optional<Generator> gen) {
  return exponential_impl_<ExponentialKernel, CUDA_CSPRNG_GeneratorImpl>(self, lambda, gen);
}

// ====================================================================================================================

Generator create_CUDA_CSPRNG_Generator() {
  return make_generator<CUDA_CSPRNG_GeneratorImpl>();
}
  
void registerOps() {
  // TODO: replace with new ops registration API
  static auto registry = torch::RegisterOperators()
    // Random
    .op(torch::RegisterOperators::options()
      .schema("aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(random_from_to), &random_from_to>(DispatchKey::CustomRNGKeyId))
    .op(torch::RegisterOperators::options()
      .schema("aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(random_to), &random_to>(DispatchKey::CustomRNGKeyId))
    .op(torch::RegisterOperators::options()
      .schema("aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(random_), &random_>(DispatchKey::CustomRNGKeyId))
    // Uniform
    .op(torch::RegisterOperators::options()
      .schema("aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(uniform_), &uniform_>(DispatchKey::CustomRNGKeyId))
    // Normal
    .op(torch::RegisterOperators::options()
      .schema("aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(normal_), &normal_>(DispatchKey::CustomRNGKeyId))
    .op(torch::RegisterOperators::options()
      .schema("aten::normal.Tensor_float_out(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(normal_Tensor_float_out), &normal_Tensor_float_out>(DispatchKey::CustomRNGKeyId))
    .op(torch::RegisterOperators::options()
      .schema("aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(normal_float_Tensor_out), &normal_float_Tensor_out>(DispatchKey::CustomRNGKeyId))
    .op(torch::RegisterOperators::options()
      .schema("aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(normal_Tensor_Tensor_out), &normal_Tensor_Tensor_out>(DispatchKey::CustomRNGKeyId))
    .op(torch::RegisterOperators::options()
      .schema("aten::normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor")
      .impl_unboxedOnlyKernel<decltype(normal_Tensor_float), &normal_Tensor_float>(DispatchKey::CustomRNGKeyId))
    .op(torch::RegisterOperators::options()
      .schema("aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor")
      .impl_unboxedOnlyKernel<decltype(normal_float_Tensor), &normal_float_Tensor>(DispatchKey::CustomRNGKeyId))
    .op(torch::RegisterOperators::options()
      .schema("aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor")
      .impl_unboxedOnlyKernel<decltype(normal_Tensor_Tensor), &normal_Tensor_Tensor>(DispatchKey::CustomRNGKeyId))
    // Cauchy
    .op(torch::RegisterOperators::options()
      .schema("aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(cauchy_), &cauchy_>(DispatchKey::CustomRNGKeyId))
    // LogNormal
    .op(torch::RegisterOperators::options()
      .schema("aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(log_normal_), &log_normal_>(DispatchKey::CustomRNGKeyId))
    // Geometric
    .op(torch::RegisterOperators::options()
      .schema("aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(geometric_), &geometric_>(DispatchKey::CustomRNGKeyId))
    // Exponential
    .op(torch::RegisterOperators::options()
      .schema("aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(exponential_), &exponential_>(DispatchKey::CustomRNGKeyId))
  ;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("registerOps", &registerOps);
  m.def("create_CUDA_CSPRNG_Generator", &create_CUDA_CSPRNG_Generator);
}
