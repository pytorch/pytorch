#include <gtest/gtest.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/all.h>
#include <stdexcept>

using namespace at;

namespace {

constexpr uint32_t FLOAT_MASK = (1 << 24) - 1;
constexpr float FLOAT_DIVISOR = 1.0f / (1 << 24);

constexpr uint64_t DOUBLE_MASK = (1ULL << 53) - 1;
constexpr double DOUBLE_DIVISOR = 1.0 / (1ULL << 53);

struct CustomCPUGenerator : public Generator {
  CustomCPUGenerator() : Generator{Device(DeviceType::CPU), DispatchKeySet(DispatchKey::CustomRNGKeyId)} { }
  ~CustomCPUGenerator() = default;
  uint32_t random() { return 42; }
  uint64_t random64() { return 42; }
  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  CustomCPUGenerator* clone_impl() const override { throw std::runtime_error("not implemented"); }
};

Tensor& custom_rng_cauchy_(Tensor& self, double median, double sigma, Generator * generator) {
  auto custom_generator = (CustomCPUGenerator*)generator;
  auto iter = TensorIterator::nullary_op(self);
  native::templates::cauchy_kernel(iter, median, sigma, custom_generator);
  return self;
}

}

TEST(RNGTest, RegisterCustomRNG) {
  static auto registry = torch::RegisterOperators()
    .op(torch::RegisterOperators::options()
      .schema("aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(custom_rng_cauchy_), &custom_rng_cauchy_>(DispatchKey::CustomRNGKeyId));

  const auto median = 123.45;
  const auto sigma = 67.89;
  auto custom_generator = new CustomCPUGenerator();

  auto actual = torch::empty({3, 3});
  actual.cauchy_(median, sigma, custom_generator);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cauchy_kernel(iter, median, sigma, custom_generator);

  ASSERT_TRUE(torch::allclose(actual, expected));
}
