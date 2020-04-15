#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/core/MT19937RNGEngine.h>
#include <memory>
#include "aes.cuh"

using namespace at;

struct CUDA_CSPRNG_GeneratorImpl : public at::CPUGeneratorImpl {
  CUDA_CSPRNG_GeneratorImpl(uint64_t seed_in = default_rng_seed_val) : CPUGeneratorImpl(seed_in) {
    this->key_set_ = DispatchKeySet(DispatchKey::CustomRNGKeyId);
  }
};

typedef ulonglong2 block_t;
constexpr size_t block_t_size = sizeof(block_t);

Tensor key_tensor(c10::optional<Generator> generator) {
  return torch::empty({16}, torch::kUInt8).random_(0, 256, generator).to(kCUDA);
}

template<size_t size>
struct DummyRNG {
  __device__ DummyRNG(uint64_t* vals) {
    for (auto i = 0; i < size; i++) {
      vals_[i] = vals[i];
    }
  }
  uint32_t __device__ random() { return static_cast<uint32_t>(vals_[index++]); }
  uint64_t __device__ random64() { return vals_[index++]; }
  c10::optional<float> __device__ next_float_normal_sample() { return c10::nullopt; }
  c10::optional<double> __device__ next_double_normal_sample() { return c10::nullopt; }
  void __device__ set_next_float_normal_sample(c10::optional<float> randn) {}
  void __device__ set_next_double_normal_sample(c10::optional<double> randn) {}
private:
  uint64_t vals_[size];
  int index = 0;
};

template<typename scalar_t, typename uint_t, size_t N = 1, typename cipher_t, typename transform_t>
__global__ void block_cipher_contiguous_kernel(scalar_t* data, int numel, cipher_t cipher, transform_t transform_func) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr auto unroll_factor = block_t_size / sizeof(uint_t) / N;
  if (unroll_factor * idx < numel) {
    auto block = cipher(idx);
    #pragma unroll
    for (auto i = 0; i < unroll_factor; ++i) {
      const auto li = unroll_factor * idx + i;
      if (li < numel) {
        uint64_t vals[N];
        #pragma unroll
        for (auto j = 0; j < N; j++) {
          vals[j] = (reinterpret_cast<uint_t*>(&block))[N * i + j];
        }
        DummyRNG<N> rng(vals);
        data[li] = transform_func(&rng);
      }
    }
  }
}

template<typename scalar_t, typename uint_t, size_t N = 1, typename cipher_t, typename transform_t>
__global__ void block_cipher_kernel(scalar_t* data, int numel, cipher_t cipher, transform_t transform_func, OffsetCalculator<1> offset_calc) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr auto unroll_factor = block_t_size / sizeof(uint_t) / N;
  if (unroll_factor * idx < numel) {
    auto block = cipher(idx);
    #pragma unroll
    for (auto i = 0; i < unroll_factor; ++i) {
      const auto li = unroll_factor * idx + i;
      if (li < numel) {
        uint64_t vals[N];
        #pragma unroll
        for (auto j = 0; j < N; j++) {
          vals[j] = (reinterpret_cast<uint_t*>(&block))[N * i + j];
        }
        DummyRNG<N> rng(vals);
        data[offset_calc.get(li)[0] / sizeof(scalar_t)] = transform_func(&rng);
      }
    }
  }
}

template<typename scalar_t, typename uint_t, size_t N = 1, typename cipher_t, typename transform_t>
void block_cipher_ctr_mode(at::TensorIterator& iter, cipher_t cipher, transform_t transform_func) {
  const auto numel = iter.numel();
  if (numel == 0) {
    return;
  }
  constexpr auto unroll_factor = block_t_size / sizeof(uint_t) / N;
  const auto block = 256;
  const auto grid = (numel + (block * unroll_factor) - 1) / (block * unroll_factor);
  scalar_t* data = (scalar_t*)iter.data_ptr(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (iter.output(0).is_contiguous()) {
    block_cipher_contiguous_kernel<scalar_t, uint_t, N, cipher_t, transform_t><<<grid, block, 0, stream>>>(data, numel, cipher, transform_func);
  } else {
    auto offset_calc = make_offset_calculator<1>(iter);
    block_cipher_kernel<scalar_t, uint_t, N, cipher_t, transform_t><<<grid, block, 0, stream>>>(data, numel, cipher, transform_func, offset_calc);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================================================================

template<typename scalar_t, typename uint_t, size_t N = 1, typename transform_t>
void block_cipher_helper(TensorIterator& iter, uint8_t* key, transform_t transform_func) {
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
void random_kernel_helper_fp(TensorIterator& iter, uint8_t* key) {
  block_cipher_helper<scalar_t, uint_t>(iter, key,
    [] __device__ (DummyRNG<1>* generator) -> scalar_t {
      if (std::is_same<scalar_t, double>::value) {
        return static_cast<scalar_t>(generator->random64() % static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1));
      } else {
        return static_cast<scalar_t>(generator->random() % static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1));
      }
    }
  );
}

template<typename scalar_t, typename uint_t>
void random_kernel_helper_int(TensorIterator& iter, uint8_t* key) {
  block_cipher_helper<scalar_t, uint_t>(iter, key,
    [] __device__ (DummyRNG<1>* generator) -> scalar_t {
      if (std::is_same<scalar_t, long>::value) {
        return static_cast<scalar_t>(generator->random64() % (static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1));
      } else {
        return static_cast<scalar_t>(generator->random() % (static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1));
      }
    }
  );
}

void random_kernel_helper_bool(TensorIterator& iter, uint8_t* key) {
  block_cipher_helper<bool, uint32_t>(iter, key,
    [] __device__ (DummyRNG<1>* generator) -> bool {
      return static_cast<bool>(generator->random() & 1);
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
      AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, iter.dtype(), "random_kernel_int_cuda", [&] {
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

Tensor& random_(Tensor& self, c10::optional<Generator> generator) {
  return native::templates::random_impl<RandomKernel, CUDA_CSPRNG_GeneratorImpl>(self, generator);
}

// ===========================================================================================================================

// Using DistAccumType in accumulate types for distributions.
// Note: Ideally we'd be using ATen/AccumulateType.h but looks
// like the there is some inconsistency in how accumulate types
// are mapped currently, e.g. for the cpu side, float is mapped
// to double.
template <typename T>
struct DistAccumType {  };

#if defined(__CUDACC__) || defined(__HIPCC__)
template <> struct DistAccumType<half> { using type = float; };
#endif
template <> struct DistAccumType<Half> { using type = float; };
template <> struct DistAccumType<float> { using type = float; };
template <> struct DistAccumType<double> { using type = double; };

template <typename T>
using dist_acctype = typename DistAccumType<T>::type;

// Constants for uniform distribution
// doubles have 52 bits of mantissa (fractional part)
constexpr uint64_t DOUBLE_MASK = (1ULL << std::numeric_limits<double>::digits) - 1;
constexpr double DOUBLE_DIVISOR = 1.0 / (1ULL << std::numeric_limits<double>::digits);

// floats have 23 bits of mantissa (fractional part)
constexpr uint32_t FLOAT_MASK = (1 << std::numeric_limits<float>::digits) - 1;
constexpr float FLOAT_DIVISOR = 1.0f / (1 << std::numeric_limits<float>::digits);

template <typename T>
struct uniform_real_distribution {

  inline __device__ uniform_real_distribution(T a_in, T b_in) {
    // TORCH_CHECK(a_in <= b_in);
    // TORCH_CHECK(b_in-a_in <= std::numeric_limits<T>::max());
    a = a_in;
    b = b_in;
  }

  template <typename RNG>
  inline __device__ dist_acctype<T> operator()(RNG* generator){
    dist_acctype<T> x;
    if(std::is_same<T, double>::value) {
      x = (generator->random64() & DOUBLE_MASK) * DOUBLE_DIVISOR;
    } else {
      x = (generator->random() & FLOAT_MASK) * FLOAT_DIVISOR;
    }
    return (x * (b - a) + a);
  }

  private:
    T a;
    T b;
};

template<typename scalar_t, typename uint_t>
void uniform_kernel_helper_fp(TensorIterator& iter, uint8_t* key, scalar_t from, scalar_t to) {
  block_cipher_helper<scalar_t, uint_t>(iter, key,
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
        uniform_kernel_helper_fp<scalar_t, uint64_t>(iter, key, from, to);
      } else {
        uniform_kernel_helper_fp<scalar_t, uint32_t>(iter, key, from, to);
      }
    });
  }
};

Tensor& uniform_(Tensor& self, double from, double to, c10::optional<Generator> generator) {
  return at::native::templates::uniform_impl_<UniformKernel, CUDA_CSPRNG_GeneratorImpl>(self, from, to, generator);
}

// ===========================================================================================================================

template<typename RNG>
struct NormalKernel {
  void operator()(Tensor& self, double mean, double std, c10::optional<Generator> generator) {
    // TODO
  }
};

Tensor& normal_(Tensor& self, double mean, double std, Generator gen) {
  return at::native::templates::normal_impl_<NormalKernel, CUDA_CSPRNG_GeneratorImpl>(self, mean, std, gen);
}

// ===========================================================================================================================

Generator create_CUDA_CSPRNG_Generator() {
  return make_generator<CUDA_CSPRNG_GeneratorImpl>();
}
  
void registerOps() {
  static auto registry = torch::import()
    .impl_UNBOXED("aten::random_", DispatchKey::CustomRNGKeyId, random_)
    .impl_UNBOXED("aten::uniform_", DispatchKey::CustomRNGKeyId, uniform_)
    .impl_UNBOXED("aten::normal_", DispatchKey::CustomRNGKeyId, normal_);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("registerOps", &registerOps);
  m.def("create_CUDA_CSPRNG_Generator", &create_CUDA_CSPRNG_Generator);
}
