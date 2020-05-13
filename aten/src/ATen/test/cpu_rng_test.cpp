#include <gtest/gtest.h>
#include <ATen/test/rng_test.h>
#include <ATen/Generator.h>
#include <c10/core/GeneratorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <torch/library.h>
#include <c10/util/Optional.h>
#include <torch/all.h>
#include <stdexcept>

using namespace at;

namespace {

constexpr auto kCustomRNG = DispatchKey::CustomRNGKeyId;

struct TestCPUGenerator : public c10::GeneratorImpl {
  TestCPUGenerator(uint64_t value) : GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(kCustomRNG)}, value_(value) { }
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

  static DeviceType device_type() { return DeviceType::CPU; }

  uint64_t value_;
  c10::optional<float> next_float_normal_sample_;
  c10::optional<double> next_double_normal_sample_;
};

// ==================================================== Random ========================================================

Tensor& random_(Tensor& self, c10::optional<Generator> generator) {
  return at::native::templates::random_impl<native::templates::cpu::RandomKernel, TestCPUGenerator>(self, generator);
}

Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, c10::optional<Generator> generator) {
  return at::native::templates::random_from_to_impl<native::templates::cpu::RandomFromToKernel, TestCPUGenerator>(self, from, to, generator);
}

Tensor& random_to(Tensor& self, int64_t to, c10::optional<Generator> generator) {
  return random_from_to(self, 0, to, generator);
}

// ==================================================== Normal ========================================================

Tensor& normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  return at::native::templates::normal_impl_<native::templates::cpu::NormalKernel, TestCPUGenerator>(self, mean, std, gen);
}

Tensor& normal_Tensor_float_out(Tensor& output, const Tensor& mean, double std, c10::optional<Generator> gen) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

Tensor& normal_float_Tensor_out(Tensor& output, double mean, const Tensor& std, c10::optional<Generator> gen) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

Tensor& normal_Tensor_Tensor_out(Tensor& output, const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

Tensor normal_Tensor_float(const Tensor& mean, double std, c10::optional<Generator> gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

Tensor normal_float_Tensor(double mean, const Tensor& std, c10::optional<Generator> gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

Tensor normal_Tensor_Tensor(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

// ==================================================== Uniform =======================================================

Tensor& uniform_(Tensor& self, double from, double to, c10::optional<Generator> generator) {
  return at::native::templates::uniform_impl_<native::templates::cpu::UniformKernel, TestCPUGenerator>(self, from, to, generator);
}

// ==================================================== Cauchy ========================================================

Tensor& cauchy_(Tensor& self, double median, double sigma, c10::optional<Generator> generator) {
  return at::native::templates::cauchy_impl_<native::templates::cpu::CauchyKernel, TestCPUGenerator>(self, median, sigma, generator);
}

// ================================================== LogNormal =======================================================

Tensor& log_normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  return at::native::templates::log_normal_impl_<native::templates::cpu::LogNormalKernel, TestCPUGenerator>(self, mean, std, gen);
}

// ================================================== Geometric =======================================================

Tensor& geometric_(Tensor& self, double p, c10::optional<Generator> gen) {
  return at::native::templates::geometric_impl_<native::templates::cpu::GeometricKernel, TestCPUGenerator>(self, p, gen);
}

// ================================================== Exponential =====================================================

Tensor& exponential_(Tensor& self, double lambda, c10::optional<Generator> gen) {
  return at::native::templates::exponential_impl_<native::templates::cpu::ExponentialKernel, TestCPUGenerator>(self, lambda, gen);
}

TORCH_LIBRARY_IMPL(aten, CustomRNGKeyId, m) {
  // Random
  m.impl_UNBOXED("random_.from",             random_from_to);
  m.impl_UNBOXED("random_.to",               random_to);
  m.impl_UNBOXED("random_",                  random_);
  // Normal
  m.impl_UNBOXED("normal_",                  normal_);
  m.impl_UNBOXED("normal.Tensor_float_out",  normal_Tensor_float_out);
  m.impl_UNBOXED("normal.float_Tensor_out",  normal_float_Tensor_out);
  m.impl_UNBOXED("normal.Tensor_Tensor_out", normal_Tensor_Tensor_out);
  m.impl_UNBOXED("normal.Tensor_float",      normal_Tensor_float);
  m.impl_UNBOXED("normal.float_Tensor",      normal_float_Tensor);
  m.impl_UNBOXED("normal.Tensor_Tensor",     normal_Tensor_Tensor);
  m.impl_UNBOXED("uniform_",                 uniform_);
  // Cauchy
  m.impl_UNBOXED("cauchy_",                  cauchy_);
  // LogNormal
  m.impl_UNBOXED("log_normal_",              log_normal_);
  // Geometric
  m.impl_UNBOXED("geometric_",               geometric_);
  // Exponential
  m.impl_UNBOXED("exponential_",             exponential_);
}

class RNGTest : public ::testing::Test {
};

static constexpr auto MAGIC_NUMBER = 424242424242424242ULL;

// ==================================================== Random ========================================================

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
  auto gen = at::make_generator<TestCPUGenerator>(std::numeric_limits<uint64_t>::max());
  auto actual = torch::empty({1}, torch::kInt64);
  actual.random_(std::numeric_limits<int64_t>::min(), c10::nullopt, gen);
  ASSERT_EQ(static_cast<uint64_t>(actual[0].item<int64_t>()), std::numeric_limits<uint64_t>::max());
}

// ==================================================== Normal ========================================================

TEST_F(RNGTest, Normal) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({10});
  actual.normal_(mean, std, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_float_Tensor_out) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({10});
  at::normal_out(actual, mean, torch::full({10}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_float_out) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({10});
  at::normal_out(actual, torch::full({10}, mean), std, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_Tensor_out) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({10});
  at::normal_out(actual, torch::full({10}, mean), torch::full({10}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_float_Tensor) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = at::normal(mean, torch::full({10}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_float) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = at::normal(torch::full({10}, mean), std, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_Tensor) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = at::normal(torch::full({10}, mean), torch::full({10}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ==================================================== Uniform =======================================================

TEST_F(RNGTest, Uniform) {
  const auto from = -24.24;
  const auto to = 42.42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3});
  actual.uniform_(from, to, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::uniform_kernel(iter, from, to, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ==================================================== Cauchy ========================================================

TEST_F(RNGTest, Cauchy) {
  const auto median = 123.45;
  const auto sigma = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3});
  actual.cauchy_(median, sigma, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::cauchy_kernel(iter, median, sigma, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ================================================== LogNormal =======================================================

TEST_F(RNGTest, LogNormal) {
  const auto mean = 12.345;
  const auto std = 6.789;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({10});
  actual.log_normal_(mean, std, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::log_normal_kernel(iter, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ================================================== Geometric =======================================================

TEST_F(RNGTest, Geometric) {
  const auto p = 0.42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3});
  actual.geometric_(p, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::geometric_kernel(iter, p, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ================================================== Exponential =====================================================

TEST_F(RNGTest, Exponential) {
  const auto lambda = 42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3});
  actual.exponential_(lambda, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::exponential_kernel(iter, lambda, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

}
