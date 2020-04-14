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

// ===========================================================================================================================

template<typename RNG>
struct RandomKernel {
  void operator()(TensorIterator& iter, c10::optional<Generator> generator) {
    // TODO
  }
};

Tensor& random_(Tensor& self, c10::optional<Generator> generator) {
  return native::templates::random_impl<RandomKernel, CUDA_CSPRNG_GeneratorImpl>(self, generator);
}

// ===========================================================================================================================

template<typename RNG>
struct UniformKernel {
  void operator()(TensorIterator& iter, double from, double to, c10::optional<Generator> generator) {
    // TODO
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
