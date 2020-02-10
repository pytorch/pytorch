#include <gtest/gtest.h>
#include <ATen/test/rng_test.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/cuda/DistributionTemplates.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/util/Optional.h>
#include <torch/all.h>
#include <stdexcept>

using namespace at;

namespace {

struct TestCUDAGenerator : public Generator {
  TestCUDAGenerator(uint64_t value) : Generator{Device(DeviceType::CUDA), DispatchKeySet(DispatchKey::CustomRNGKeyId)}, value_(value) { }
  ~TestCUDAGenerator() = default;
  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  void set_philox_offset_per_thread(uint64_t offset) {}
  uint64_t philox_offset_per_thread() {}
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment) {
    uint64_t offset = this->philox_offset_per_thread_;
    this->philox_offset_per_thread_ += increment;
    return std::make_pair(value_, offset);
  }

  TestCUDAGenerator* clone_impl() const override { throw std::runtime_error("not implemented"); }

  uint64_t value_;
  uint64_t philox_offset_per_thread_ = 0;
};

Tensor& random_(Tensor& self, Generator* generator) {
  return test::random_<native::templates::cuda::RandomKernel, TestCUDAGenerator>(self, generator);
}
  
Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, Generator* generator) {
  return test::random_from_to<native::templates::cuda::RandomFromToKernel, TestCUDAGenerator>(self, from, to, generator);
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

TEST_F(RNGTest, RandomFromTo_CUDA) {
  using test::test_random_from_to;
  const at::Device device("cuda");
  test_random_from_to<TestCUDAGenerator, torch::kBool, bool>(device);
  test_random_from_to<TestCUDAGenerator, torch::kUInt8, uint8_t>(device);
  test_random_from_to<TestCUDAGenerator, torch::kInt8, int8_t>(device);
  test_random_from_to<TestCUDAGenerator, torch::kInt16, int16_t>(device);
  test_random_from_to<TestCUDAGenerator, torch::kInt32, int32_t>(device);
  test_random_from_to<TestCUDAGenerator, torch::kInt64, int64_t>(device);
  test_random_from_to<TestCUDAGenerator, torch::kFloat32, float>(device);
  test_random_from_to<TestCUDAGenerator, torch::kFloat64, double>(device);
}
  
TEST_F(RNGTest, Random_CUDA) {
  using test::test_random;
  const at::Device device("cuda");
  test_random<TestCUDAGenerator, torch::kBool, bool>(device);
  test_random<TestCUDAGenerator, torch::kUInt8, uint8_t>(device);
  test_random<TestCUDAGenerator, torch::kInt8, int8_t>(device);
  test_random<TestCUDAGenerator, torch::kInt16, int16_t>(device);
  test_random<TestCUDAGenerator, torch::kInt32, int32_t>(device);
  test_random<TestCUDAGenerator, torch::kInt64, int64_t>(device);
  test_random<TestCUDAGenerator, torch::kFloat32, float>(device);
  test_random<TestCUDAGenerator, torch::kFloat64, double>(device);
}

}
