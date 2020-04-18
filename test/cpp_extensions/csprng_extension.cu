#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/core/DistributionsHelper.h>
#include <memory>
#include "aes.cuh"
#include "block_cipher.cuh"

using namespace at;
using namespace at::native::templates;

struct CUDA_CSPRNG_GeneratorImpl : public CPUGeneratorImpl {
  CUDA_CSPRNG_GeneratorImpl(uint64_t seed_in = default_rng_seed_val) : CPUGeneratorImpl(seed_in) {
    this->key_set_ = DispatchKeySet(DispatchKey::CustomRNGKeyId);
  }
};

// ===========================================================================================================================

template<typename scalar_t, typename uint_t, size_t N = 1, typename transform_t>
void aes_helper(TensorIterator& iter, const uint8_t* key, transform_t transform_func) {
  block_cipher_ctr_mode<scalar_t, uint_t, N>(iter,
    [key] __device__ (unsigned int idx) -> block_t {
      block_t block;
      memset(&block, 0, block_t_size);
      *(reinterpret_cast<unsigned int*>(&block)) = idx;
      encrypt(reinterpret_cast<uint8_t*>(&block), key);
      return block;
    },
    transform_func
  );
}

// ===========================================================================================================================

template<typename scalar_t, typename uint_t>
void random_kernel_helper_fp(TensorIterator& iter, const uint8_t* key) {
  aes_helper<scalar_t, uint_t>(iter, key,
    [] __device__ (DummyRNG<1>* generator) -> scalar_t {
      uniform_int_distribution<scalar_t> random;
      return random(generator);
    }
  );
}

template<typename scalar_t, typename uint_t>
void random_kernel_helper_int(TensorIterator& iter, const uint8_t* key) {
  aes_helper<scalar_t, uint_t>(iter, key,
    [] __device__ (DummyRNG<1>* generator) -> scalar_t {
      uniform_int_distribution<scalar_t> random;
      return random(generator);
    }
  );
}

void random_kernel_helper_bool(TensorIterator& iter, const uint8_t* key) {
  aes_helper<bool, uint32_t>(iter, key,
    [] __device__ (DummyRNG<1>* generator) -> bool {
      uniform_int_distribution<bool> random;
      return random(generator);
    }
  );
}

template<typename RNG>
struct RandomKernel {
  void operator()(TensorIterator& iter, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator);
    const auto key = key_t.data_ptr<uint8_t>();
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "random_kernel_fp_cuda", [&] {
        if (std::is_same<scalar_t, double>::value) {
          random_kernel_helper_fp<scalar_t, uint64_t>(iter, key);
        } else {
          random_kernel_helper_fp<scalar_t, uint32_t>(iter, key);
        }
      });
    } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
      AT_DISPATCH_INTEGRAL_TYPES_AND(ScalarType::Bool, iter.dtype(), "random_kernel_int_cuda", [&] {
        if (std::is_same<scalar_t, int64_t>::value) {
          random_kernel_helper_int<scalar_t, uint64_t>(iter, key);
        } else if (std::is_same<scalar_t, bool>::value) {
          random_kernel_helper_bool(iter, key);
        } else {
          random_kernel_helper_int<scalar_t, uint32_t>(iter, key);
        }
      });
    }
  }
};

template<typename scalar_t>
void random_from_to_kernel_helper(TensorIterator& iter, uint64_t range, int64_t base, const uint8_t* key) {
  aes_helper<scalar_t, int32_t>(iter, key,
    [range, base] __device__ (DummyRNG<1>* generator) -> scalar_t {
      uniform_int_from_to_distribution<scalar_t> random(range, base);
      return random(generator);
    }
  );
}

template<typename scalar_t>
void random_from_to_64_kernel_helper(TensorIterator& iter, uint64_t range, int64_t base, const uint8_t* key) {
  aes_helper<scalar_t, int64_t>(iter, key,
    [range, base] __device__ (DummyRNG<1>* generator) -> scalar_t {
      uniform_int_from_to_distribution<scalar_t> random(range, base);
      return random(generator);
    }
  );
}

template<typename scalar_t, typename uint_t>
void random_full_range_kernel_helper(TensorIterator& iter, const uint8_t* key) {
  aes_helper<scalar_t, uint_t>(iter, key,
    [] __device__ (DummyRNG<1>* generator) -> scalar_t {
      uniform_int_full_range_distribution<scalar_t> random;
      return random(generator);
    }
  );
}

template<typename RNG>
struct RandomFromToKernel {
  void operator()(TensorIterator& iter, uint64_t range, int64_t base, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator);
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_from_to_kernel_cuda", [&] {
      if ((
        std::is_same<scalar_t, int64_t>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, at::BFloat16>::value) && range >= 1ULL << 32)
      {
        random_from_to_64_kernel_helper<scalar_t>(iter, range, base, key);
      } else {
        random_from_to_kernel_helper<scalar_t>(iter, range, base, key);
      }
    });
  }
  void operator()(TensorIterator& iter, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator);
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

// ===========================================================================================================================

template<typename scalar_t, typename uint_t>
void uniform_kernel_helper_fp(TensorIterator& iter, scalar_t from, scalar_t to, const uint8_t* key) {
  aes_helper<scalar_t, uint_t>(iter, key,
    [from, to] __device__ (DummyRNG<1>* generator) -> scalar_t {
      uniform_real_distribution<scalar_t> uniform(from, to);
      return uniform(generator);
    }
  );
}

template<typename RNG>
struct UniformKernel {
  void operator()(TensorIterator& iter, double from, double to, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator);
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "uniform_kernel_cuda", [&] {
      if (std::is_same<scalar_t, double>::value) {
        uniform_kernel_helper_fp<scalar_t, uint64_t>(iter, from, to, key);
      } else {
        uniform_kernel_helper_fp<scalar_t, uint32_t>(iter, from, to, key);
      }
    });
  }
};

Tensor& uniform_(Tensor& self, double from, double to, c10::optional<Generator> generator) {
  return uniform_impl_<UniformKernel, CUDA_CSPRNG_GeneratorImpl>(self, from, to, generator);
}

// ===========================================================================================================================

template<typename scalar_t, typename uint_t>
void normal_kernel_helper_fp(TensorIterator& iter, scalar_t mean, scalar_t std, const uint8_t* key) {
  aes_helper<scalar_t, uint_t, 2>(iter, key,
    [mean, std] __device__ (DummyRNG<2>* generator) -> scalar_t {
      normal_distribution<scalar_t> normal(mean, std);
      return normal(generator);
    }
  );
}

template<typename RNG>
struct NormalKernel {
  void operator()(Tensor& self, double mean, double std, c10::optional<Generator> generator) {
    const auto key_t = key_tensor(generator);
    const auto key = key_t.data_ptr<uint8_t>();
    auto iter = TensorIterator::nullary_op(self);
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "normal_kernel_cuda", [&] {
      if (std::is_same<scalar_t, double>::value) {
        normal_kernel_helper_fp<scalar_t, uint64_t>(iter, mean, std, key);
      } else {
        normal_kernel_helper_fp<scalar_t, uint32_t>(iter, mean, std, key);
      }
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

// ===========================================================================================================================

Generator create_CUDA_CSPRNG_Generator() {
  return make_generator<CUDA_CSPRNG_GeneratorImpl>();
}
  
void registerOps() {
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
  ;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("registerOps", &registerOps);
  m.def("create_CUDA_CSPRNG_Generator", &create_CUDA_CSPRNG_Generator);
}
