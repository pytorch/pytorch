#include <gtest/gtest.h>
#include <torch/extension.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <ATen/core/op_registration/op_registration.h>

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
  void set_current_seed(uint64_t seed) override { throw "not implemented"; }
  uint64_t current_seed() const override { throw "not implemented"; }
  uint64_t seed() override { throw "not implemented"; }
  CustomCPUGenerator* clone_impl() const override { throw "not implemented"; }
};

Tensor& custom_rng_cauchy_(Tensor& self, double median, double sigma, Generator * generator) {
  auto custom_generator = (CustomCPUGenerator*)generator;
  auto iter = TensorIterator::unary_op(self, self);
  native::templates::cauchy_kernel(iter, median, sigma, custom_generator);
  return self;
}

template <typename T, typename RNG>
auto uniform(T a, T b, RNG* generator){
  T x;
  if(std::is_same<T, double>::value) {
    x = (generator->random64() & DOUBLE_MASK) * DOUBLE_DIVISOR;
  } else {
    x = (generator->random() & FLOAT_MASK) * FLOAT_DIVISOR;
  }
  return (x * (b - a) + a);
}

template <typename T, typename RNG>
auto cauchy(T median, T sigma, RNG* generator){
  return median + sigma * ::tan(static_cast<T>(M_PI) * (uniform(0.0, 1.0, generator)-static_cast<T>(0.5)));
}

} 

TEST(RNGTest, RegisterCustomRNG) {
  static auto registry = torch::RegisterOperators()
    .op(torch::RegisterOperators::options()
      .schema("aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<decltype(custom_rng_cauchy_), &custom_rng_cauchy_>(DispatchKey::CustomRNGKeyId));

  auto gen = new CustomCPUGenerator();
  auto t = torch::empty({3, 3});
  const auto median = 123.45;
  const auto sigma = 67.89;
  t.cauchy_(median, sigma, gen);
  const auto expected = cauchy(median, sigma, gen);
  ASSERT_TRUE(torch::allclose(t, torch::full_like(t, expected)));
}
