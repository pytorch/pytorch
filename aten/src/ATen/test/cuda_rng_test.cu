#include <gtest/gtest.h>
#include <ATen/test/rng_test.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/util/Optional.h>
#include <torch/all.h>
#include <stdexcept>

using namespace at;

namespace {

struct TestCPUGenerator : public Generator {
  TestCPUGenerator(uint64_t value) : Generator{Device(DeviceType::CUDA), DispatchKeySet(DispatchKey::CustomRNGKeyId)} { }
  ~TestCPUGenerator() = default;
  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  TestCPUGenerator* clone_impl() const override { throw std::runtime_error("not implemented"); }
};

template<typename RNG>
void random_from_to_kernel(TensorIterator& iter, uint64_t range, int64_t base, RNG* gen) {
  std::cout << "random_from_to_kernel" << std::endl;
}

template<typename RNG>
void random_full_64_bits_range_kernel(TensorIterator& iter, RNG* gen) {
  std::cout << "random_full_64_bits_range_kernel" << std::endl;
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
void random_kernel(TensorIterator& iter, RNG* gen) {
  std::cout << "random_kernel" << std::endl;
// #ifdef _WIN32
//   // TODO: https://github.com/pytorch/pytorch/issues/33793
//   if (iter.dtype() == ScalarType::BFloat16) {
//     TORCH_CHECK(false, "random_() is not supported for bfloat16 CUDA tensors on Windows. Please see https://github.com/pytorch/pytorch/issues/33793");
//   }
// #endif
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_kernel_fp_cuda", [&] {
      if (std::is_same<scalar_t, double>::value) {
        // auto random_func = [] __device__ (uint64_t rand) {
        //   return static_cast<scalar_t>(rand % static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1));
        // };
        // distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter,
        //   gen,
        //   [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
        //     ulonglong2 ret;
        //     uint4 rand_val = curand4(state);
        //     ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
        //     ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
        //     return ret;
        //   },
        //   random_func);
      } else {
        // auto random_func = [] __device__ (uint32_t rand) {
        //   return static_cast<scalar_t>(rand % static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1));
        // };
        // distribution_nullary_kernel<scalar_t, uint32_t, curand4_engine_calls>(iter,
        //   gen,
        //   [] __device__ (curandStatePhilox4_32_10_t* state) {
        //     return curand4(state);
        //   },
        //   random_func);
      }
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, iter.dtype(), "random_kernel_int_cuda", [&] {
      if (std::is_same<scalar_t, int64_t>::value) {
        // auto random_func = [] __device__ (uint64_t rand) {
        //   return static_cast<scalar_t>(rand % (static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1));
        // };
        // distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter,
        //   gen,
        //   [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
        //     ulonglong2 ret;
        //     uint4 rand_val = curand4(state);
        //     ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
        //     ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
        //     return ret;
        //   },
        //   random_func);
      } else if (std::is_same<scalar_t, bool>::value) {
        // auto random_func = [] __device__ (uint32_t rand) {
        //   return static_cast<scalar_t>(rand & 1);
        // };
        // distribution_nullary_kernel<scalar_t, uint32_t, curand4_engine_calls>(iter,
        //   gen,
        //   [] __device__ (curandStatePhilox4_32_10_t* state) {
        //     return curand4(state);
        //   },
        //   random_func);
      } else {
        // auto random_func = [] __device__ (uint32_t rand) {
        //   return static_cast<scalar_t>(rand % (static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1));
        // };
        // distribution_nullary_kernel<scalar_t, uint32_t, curand4_engine_calls>(iter,
        //   gen,
        //   [] __device__ (curandStatePhilox4_32_10_t* state) {
        //     return curand4(state);
        //   },
        //   random_func);
      }
    });
  } else {
    TORCH_CHECK(false, "random_kernel_cuda handles only integral, floating-point and boolean types");
  }
}

template<typename RNG>
struct RandomKernel {
  void operator()(TensorIterator& iter, RNG* gen) {
    random_kernel(iter, gen);
  }
};

Tensor& random_(Tensor& self, Generator* generator) {
  return at::native::templates::random_impl<RandomKernel, TestCPUGenerator>(self, generator);
}

Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, Generator* generator) {
  return at::native::templates::random_from_to_impl<RandomFromToKernel, TestCPUGenerator>(self, from, to, generator);
}

Tensor& random_to(Tensor& self, int64_t to, Generator* generator) {
  return random_from_to(self, 0, to, generator);
}

class RNGTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static auto registry = torch::RegisterOperators()
      .op(torch::RegisterOperators::options()
        .schema("aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(random_from_to), &random_from_to>(DispatchKey::CustomRNGKeyId))
      .op(torch::RegisterOperators::options()
        .schema("aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(random_to), &random_to>(DispatchKey::CustomRNGKeyId))
      .op(torch::RegisterOperators::options()
        .schema("aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(random_), &random_>(DispatchKey::CustomRNGKeyId));
  }
};

TEST_F(RNGTest, Random) {
  const at::Device device("cuda");
  auto gen = new TestCPUGenerator(42);
  auto actual = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  actual.random_(gen);
  actual.random_(100, gen);
  actual.random_(100, 200, gen);
  actual.random_(std::numeric_limits<int64_t>::min(), c10::nullopt, gen);
}

} 
