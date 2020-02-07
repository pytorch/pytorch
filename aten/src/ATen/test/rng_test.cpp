#include <gtest/gtest.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
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
  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  TestCPUGenerator* clone_impl() const override { throw std::runtime_error("not implemented"); }

  uint64_t value_;
};

Tensor& random_(Tensor& self, Generator* generator) {
  auto gen = (TestCPUGenerator*)generator;
  auto iter = TensorIterator::nullary_op(self);
  native::templates::random_kernel(iter, gen);
  return self;
}

Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, Generator* generator) {
  auto gen = (TestCPUGenerator*)generator;
  uint64_t range;
  auto iter = TensorIterator::nullary_op(self);
  if (to) {
    // [from, to)
    TORCH_CHECK(from < *to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", *to);
    range = *to - from;
    native::templates::random_from_to_kernel(iter, range, from, gen);
  } else if (from != std::numeric_limits<int64_t>::lowest()) {
    // [from, std::numeric_limits<int64_t>::max()]
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool, self.scalar_type(), "random_from_to_range_calc", [&] {
      if (std::is_same<scalar_t, bool>::value) {
        range = 2;
      } else {
        const auto t_max_val = std::numeric_limits<scalar_t>::max();
        const auto int64_max_val = std::numeric_limits<int64_t>::max();
        const int64_t max_val = std::is_floating_point<scalar_t>::value ? int64_max_val : static_cast<int64_t>(t_max_val);
        range = max_val - from + 1;
      }
    });
    native::templates::random_from_to_kernel(iter, range, from, gen);
  } else {
    // [std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max()]
    // range = 2^64
    native::templates::random_full_64_range_kernel(iter, gen);
  }
  return self;
}

Tensor& random_to(Tensor& self, int64_t to, Generator* generator) {
  return random_from_to(self, 0, to, generator);
}

Tensor& custom_rng_cauchy_(Tensor& self, double median, double sigma, Generator * generator) {
  auto gen = (TestCPUGenerator*)generator;
  auto iter = TensorIterator::nullary_op(self);
  native::templates::cauchy_kernel(iter, median, sigma, gen);
  return self;
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
        .impl_unboxedOnlyKernel<decltype(random_), &random_>(DispatchKey::CustomRNGKeyId))
      .op(torch::RegisterOperators::options()
        .schema("aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(custom_rng_cauchy_), &custom_rng_cauchy_>(DispatchKey::CustomRNGKeyId));
  }
};

template<c10::ScalarType S, typename T>
void test_random_from_to() {
  const auto t_min_val = std::numeric_limits<T>::lowest();
  const auto int64_min_val = std::numeric_limits<int64_t>::lowest();
  const int64_t min_val = std::is_floating_point<T>::value ? int64_min_val : static_cast<int64_t>(t_min_val);

  const auto t_max_val = std::numeric_limits<T>::max();
  const auto int64_max_val = std::numeric_limits<int64_t>::max();
  const int64_t max_val = std::is_floating_point<T>::value ? int64_max_val : static_cast<int64_t>(t_max_val);

  const auto uint64_max_val = std::numeric_limits<uint64_t>::max();

  std::vector<int64_t> froms;
  std::vector<c10::optional<int64_t>> tos;
  if (std::is_same<T, bool>::value) {
    froms = {
      0L
    };
    tos = {
      1L,
      static_cast<c10::optional<int64_t>>(c10::nullopt)
    };
  } else if (std::is_signed<T>::value) {
    froms = {
      min_val,
      -42L,
      0L,
      42L
    };
    tos = {
      c10::optional<int64_t>(-42L),
      c10::optional<int64_t>(0L),
      c10::optional<int64_t>(42L),
      c10::optional<int64_t>(max_val),
      static_cast<c10::optional<int64_t>>(c10::nullopt)
    };
  } else {
    froms = {
      0L,
      42L
    };
    tos = {
      c10::optional<int64_t>(42L),
      c10::optional<int64_t>(max_val),
      static_cast<c10::optional<int64_t>>(c10::nullopt)
    };
  }

  const std::vector<uint64_t> vals = {
    0L,
    42L,
    static_cast<uint64_t>(max_val),
    static_cast<uint64_t>(max_val) + 1,
    uint64_max_val
  };

  bool full_64_bit_range_case_covered = false;
  bool from_to_case_covered = false;
  bool from_case_covered = false;
  for (const int64_t from : froms) {
    for (const c10::optional<int64_t> to : tos) {
      if (!to.has_value() || from < *to) {
        for (const uint64_t val : vals) {
          auto gen = new TestCPUGenerator(val);

          auto actual = torch::empty({3, 3}, S);
          actual.random_(from, to, gen);

          T exp;
          uint64_t range;
          if (!to.has_value() && from == int64_min_val) {
            exp = val;
            full_64_bit_range_case_covered = true;
          } else {
            if (to.has_value()) {
              range = *to - from;
              from_to_case_covered = true;
            } else {
              range = max_val - from + 1;
              from_case_covered = true;
            }
            if (range < (1ULL << 32)) {
              exp = static_cast<T>(static_cast<int64_t>((static_cast<uint32_t>(val) % range + from)));
            } else {
              exp = static_cast<T>(static_cast<int64_t>((val % range + from)));
            }
          }
          ASSERT_TRUE(from <= exp);
          if (to) {
            ASSERT_TRUE(static_cast<int64_t>(exp) < *to);
          }
          const auto expected = torch::full_like(actual, exp);
          if (std::is_same<T, bool>::value) {
            ASSERT_TRUE(torch::allclose(actual.toType(torch::kInt), expected.toType(torch::kInt)));
          } else {
            ASSERT_TRUE(torch::allclose(actual, expected));
          }
        }
      }
    }
  }
  if (std::is_same<T, int64_t>::value) {
    ASSERT_TRUE(full_64_bit_range_case_covered);
  }
  ASSERT_TRUE(from_to_case_covered);
  ASSERT_TRUE(from_case_covered);
}

TEST_F(RNGTest, RandomFromTo) {
  test_random_from_to<torch::kBool, bool>();
  test_random_from_to<torch::kUInt8, uint8_t>();
  test_random_from_to<torch::kInt8, int8_t>();
  test_random_from_to<torch::kInt16, int16_t>();
  test_random_from_to<torch::kInt32, int32_t>();
  test_random_from_to<torch::kInt64, int64_t>();
  test_random_from_to<torch::kFloat32, float>();
  test_random_from_to<torch::kFloat64, double>();
}

template<c10::ScalarType S, typename T>
void test_random() {
  const auto min_val = std::numeric_limits<T>::lowest();
  const auto max_val = std::numeric_limits<T>::max();
  const auto uint64_max_val = std::numeric_limits<uint64_t>::max();

  const std::vector<uint64_t> vals = {
    0L,
    42L,
    static_cast<uint64_t>(max_val),
    static_cast<uint64_t>(max_val) + 1,
    uint64_max_val
  };

  for (const uint64_t val : vals) {
    auto gen = new TestCPUGenerator(val);

    auto actual = torch::empty({3, 3}, S);
    actual.random_(gen);

    uint64_t range;
    if (std::is_floating_point<T>::value) {
      range = static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1);
    } else if (std::is_same<T, bool>::value) {
      range = 2;
    } else {
      range = static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1;
    }
    T exp;
    if (std::is_same<T, double>::value || std::is_same<T, int64_t>::value) {
      exp = val % range;
    } else {
      exp = static_cast<uint32_t>(val) % range;
    }

    ASSERT_TRUE(0 <= static_cast<int64_t>(exp));
    ASSERT_TRUE(static_cast<int64_t>(exp) < range);

    const auto expected = torch::full_like(actual, exp);
    if (std::is_same<T, bool>::value) {
      ASSERT_TRUE(torch::allclose(actual.toType(torch::kInt), expected.toType(torch::kInt)));
    } else {
      ASSERT_TRUE(torch::allclose(actual, expected));
    }
  }
}

TEST_F(RNGTest, Random) {
  test_random<torch::kBool, bool>();
  test_random<torch::kUInt8, uint8_t>();
  test_random<torch::kInt8, int8_t>();
  test_random<torch::kInt16, int16_t>();
  test_random<torch::kInt32, int32_t>();
  test_random<torch::kInt64, int64_t>();
  test_random<torch::kFloat32, float>();
  test_random<torch::kFloat64, double>();
}

TEST_F(RNGTest, Cauchy) {
  const auto median = 123.45;
  const auto sigma = 67.89;
  auto gen = new TestCPUGenerator(42.0);

  auto actual = torch::empty({3, 3});
  actual.cauchy_(median, sigma, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cauchy_kernel(iter, median, sigma, gen);

  ASSERT_TRUE(torch::allclose(actual, expected));
}

}
