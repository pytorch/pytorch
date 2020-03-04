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
  TestCPUGenerator(uint64_t value) : Generator{Device(DeviceType::CPU), DispatchKeySet(DispatchKey::CustomRNGKeyId)}, value_(value) { }
  ~TestCPUGenerator() = default;
  uint32_t random() { return value_; }
  uint64_t random64() { return value_; }
  c10::optional<float> next_float_normal_sample() { return next_float_normal_sample_; }
  c10::optional<double> next_double_normal_sample() { return next_double_normal_sample_; }
  void set_next_float_normal_sample(c10::optional<float> randn) { next_float_normal_sample_ = randn; }
  void set_next_double_normal_sample(c10::optional<double> randn) { next_double_normal_sample_ = randn; }
  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  TestCPUGenerator* clone_impl() const override { throw std::runtime_error("not implemented"); }

  uint64_t value_;
  c10::optional<float> next_float_normal_sample_;
  c10::optional<double> next_double_normal_sample_;
};

// =======================================================================================================================================

Tensor& random_(Tensor& self, Generator* generator) {
  return at::native::templates::random_impl<native::templates::cpu::RandomKernel, TestCPUGenerator>(self, generator);
}

Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, Generator* generator) {
  return at::native::templates::random_from_to_impl<native::templates::cpu::RandomFromToKernel, TestCPUGenerator>(self, from, to, generator);
}

Tensor& random_to(Tensor& self, int64_t to, Generator* generator) {
  return random_from_to(self, 0, to, generator);
}

// =======================================================================================================================================

Tensor& normal_(Tensor& self, double mean, double std, Generator* gen) {
  return at::native::templates::normal_impl_<native::templates::cpu::NormalKernel, TestCPUGenerator>(self, mean, std, gen);
}

Tensor& normal_Tensor_float_out(Tensor& output, const Tensor& mean, double std, Generator* gen) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

Tensor& normal_float_Tensor_out(Tensor& output, double mean, const Tensor& std, Generator* gen) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

Tensor& normal_Tensor_Tensor_out(Tensor& output, const Tensor& mean, const Tensor& std, Generator* gen) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

Tensor normal_Tensor_float(const Tensor& mean, double std, Generator* gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

Tensor normal_float_Tensor(double mean, const Tensor& std, Generator* gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

Tensor normal_Tensor_Tensor(const Tensor& mean, const Tensor& std, Generator* gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

// =======================================================================================================================================

Tensor& custom_rng_cauchy_(Tensor& self, double median, double sigma, Generator * generator) {
  auto gen = (TestCPUGenerator*)generator;
  auto iter = TensorIterator::nullary_op(self);
  native::templates::cpu::cauchy_kernel(iter, median, sigma, gen);
  return self;
}

// =======================================================================================================================================

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
        .impl_unboxedOnlyKernel<decltype(random_), &random_>(DispatchKey::CustomRNGKeyId))
// =======================================================================================================================================
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
// =======================================================================================================================================
      .op(torch::RegisterOperators::options()
        .schema("aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(custom_rng_cauchy_), &custom_rng_cauchy_>(DispatchKey::CustomRNGKeyId));
  }
};

// =======================================================================================================================================

TEST_F(RNGTest, RandomFromTo) {
  const at::Device device("cpu");
  test_random_from_to<TestCPUGenerator, torch::kBool, bool>(device);
  test_random_from_to<TestCPUGenerator, torch::kUInt8, uint8_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt8, int8_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt16, int16_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt32, int32_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt64, int64_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kFloat32, float>(device);
  test_random_from_to<TestCPUGenerator, torch::kFloat64, double>(device);
}

TEST_F(RNGTest, Random) {
  const at::Device device("cpu");
  test_random<TestCPUGenerator, torch::kBool, bool>(device);
  test_random<TestCPUGenerator, torch::kUInt8, uint8_t>(device);
  test_random<TestCPUGenerator, torch::kInt8, int8_t>(device);
  test_random<TestCPUGenerator, torch::kInt16, int16_t>(device);
  test_random<TestCPUGenerator, torch::kInt32, int32_t>(device);
  test_random<TestCPUGenerator, torch::kInt64, int64_t>(device);
  test_random<TestCPUGenerator, torch::kFloat32, float>(device);
  test_random<TestCPUGenerator, torch::kFloat64, double>(device);
}

// This test proves that Tensor.random_() distribution is able to generate unsigned 64 bit max value(64 ones)
// https://github.com/pytorch/pytorch/issues/33299
TEST_F(RNGTest, Random64bits) {
  auto gen = new TestCPUGenerator(std::numeric_limits<uint64_t>::max());
  auto actual = torch::empty({1}, torch::kInt64);
  actual.random_(std::numeric_limits<int64_t>::min(), c10::nullopt, gen);
  ASSERT_EQ(static_cast<uint64_t>(actual[0].item<int64_t>()), std::numeric_limits<uint64_t>::max());
}

// =======================================================================================================================================

TEST_F(RNGTest, Normal) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = new TestCPUGenerator(42.0);

  auto actual = torch::empty({3, 3});
  actual.normal_(mean, std, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, gen);

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_float_Tensor_out) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = new TestCPUGenerator(42.0);

  auto actual = torch::empty({3, 3});
  at::normal_out(actual, mean, torch::full({3, 3}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, gen);

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_float_out) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = new TestCPUGenerator(42.0);

  auto actual = torch::empty({3, 3});
  at::normal_out(actual, torch::full({3, 3}, mean), std, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, gen);

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_Tensor_out) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = new TestCPUGenerator(42.0);

  auto actual = torch::empty({3, 3});
  at::normal_out(actual, torch::full({3, 3}, mean), torch::full({3, 3}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, gen);

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_float_Tensor) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = new TestCPUGenerator(42.0);

  auto actual = at::normal(mean, torch::full({3, 3}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, gen);

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_float) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = new TestCPUGenerator(42.0);

  auto actual = at::normal(torch::full({3, 3}, mean), std, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, gen);

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_Tensor) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = new TestCPUGenerator(42.0);

  auto actual = at::normal(torch::full({3, 3}, mean), torch::full({3, 3}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, gen);

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// =======================================================================================================================================

TEST_F(RNGTest, Cauchy) {
  const auto median = 123.45;
  const auto sigma = 67.89;
  auto gen = new TestCPUGenerator(42.0);

  auto actual = torch::empty({3, 3});
  actual.cauchy_(median, sigma, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::cauchy_kernel(iter, median, sigma, gen);

  ASSERT_TRUE(torch::allclose(actual, expected));
}

}
