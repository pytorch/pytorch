#include <gtest/gtest.h>

#include <ATen/native/Pow.h>
#include <c10/util/irange.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <iostream>
#include <vector>
#include <type_traits>

using namespace at;

namespace {

constexpr auto int_min = std::numeric_limits<int>::min();
constexpr auto int_max = std::numeric_limits<int>::max();
constexpr auto long_min = std::numeric_limits<int64_t>::min();
constexpr auto long_max = std::numeric_limits<int64_t>::max();
constexpr auto float_lowest = std::numeric_limits<float>::lowest();
constexpr auto float_min = std::numeric_limits<float>::min();
constexpr auto float_max = std::numeric_limits<float>::max();
constexpr auto double_lowest = std::numeric_limits<double>::lowest();
constexpr auto double_min = std::numeric_limits<double>::min();
constexpr auto double_max = std::numeric_limits<double>::max();

const std::vector<int> ints {
  int_min,
  int_min + 1,
  int_min + 2,
  static_cast<int>(-sqrt(static_cast<double>(int_max))),
  -3, -2, -1, 0, 1, 2, 3,
  static_cast<int>(sqrt(static_cast<double>(int_max))),
  int_max - 2,
  int_max - 1,
  int_max
};
const std::vector<int> non_neg_ints {
  0, 1, 2, 3,
  static_cast<int>(sqrt(static_cast<double>(int_max))),
  int_max - 2,
  int_max - 1,
  int_max
};
const std::vector<int64_t> longs {
  long_min,
  long_min + 1,
  long_min + 2,
  static_cast<int64_t>(-sqrt(static_cast<double>(long_max))),
  -3, -2, -1, 0, 1, 2, 3,
  static_cast<int64_t>(sqrt(static_cast<double>(long_max))),
  long_max - 2,
  long_max - 1,
  long_max
};
const std::vector<int64_t> non_neg_longs {
  0, 1, 2, 3,
  static_cast<int64_t>(sqrt(static_cast<double>(long_max))),
  long_max - 2,
  long_max - 1,
  long_max
};
const std::vector<float> floats {
  float_lowest,
  -3.0f, -2.0f, -1.0f, -1.0f/2.0f, -1.0f/3.0f,
  -float_min,
  0.0,
  float_min,
  1.0f/3.0f, 1.0f/2.0f, 1.0f, 2.0f, 3.0f,
  float_max,
};
const std::vector<double> doubles {
  double_lowest,
  -3.0, -2.0, -1.0, -1.0/2.0, -1.0/3.0,
  -double_min,
  0.0,
  double_min,
  1.0/3.0, 1.0/2.0, 1.0, 2.0, 3.0,
  double_max,
};

template <class T,
  typename std::enable_if_t<std::is_floating_point_v<T>, T>* = nullptr>
void assert_eq(T val, T act, T exp) {
  if (std::isnan(act) || std::isnan(exp)) {
    return;
  }
  ASSERT_FLOAT_EQ(act, exp);
}

template <class T,
  typename std::enable_if_t<std::is_integral_v<T>, T>* = nullptr>
void assert_eq(T val, T act, T exp) {
  if (val != 0 && act == 0) {
    return;
  }
  if (val != 0 && exp == 0) {
    return;
  }
  const auto min = std::numeric_limits<T>::min();
  if (exp == min && val != min) {
    return;
  }
  ASSERT_EQ(act, exp);
}

template <class T,
  typename std::enable_if_t<std::is_floating_point_v<T>, T>* = nullptr>
T typed_pow(T base, T exp) {
  return std::pow(base, exp);
}
template <class T,
  typename std::enable_if_t<std::is_integral_v<T>, T>* = nullptr>
T typed_pow(T base, T exp) {
  return native::powi(base, exp);
}

template<typename Vals, typename Pows>
void tensor_pow_scalar(const Vals vals, const Pows pows, const torch::ScalarType valsDtype, const torch::ScalarType dtype) {
  const auto tensor = torch::tensor(vals, valsDtype);

  for (const auto pow : pows) {
    // NOLINTNEXTLINE(clang-diagnostic-implicit-const-int-float-conversion)
    if ( dtype == kInt && pow > static_cast<float>(std::numeric_limits<int>::max())) {
      // value cannot be converted to type int without overflow
      // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
      EXPECT_THROW(tensor.pow(pow), std::runtime_error);
      continue;
    }
    auto actual_pow = tensor.pow(pow);

    auto actual_pow_ = torch::empty_like(actual_pow);
    actual_pow_.copy_(tensor);
    actual_pow_.pow_(pow);

    auto actual_pow_out = torch::empty_like(actual_pow);
    torch::pow_out(actual_pow_out, tensor, pow);

    auto actual_torch_pow = torch::pow(tensor, pow);

    int i = 0;
    for (const auto val : vals) {
      const auto exp = torch::pow(torch::tensor({val}, dtype), torch::tensor(pow, dtype)).template item<double>();

      const auto act_pow = actual_pow[i].to(at::kDouble).template item<double>();
      assert_eq<long double>(val, act_pow, exp);

      const auto act_pow_ = actual_pow_[i].to(at::kDouble).template item<double>();
      assert_eq<long double>(val, act_pow_, exp);

      const auto act_pow_out = actual_pow_out[i].to(at::kDouble).template item<double>();
      assert_eq<long double>(val, act_pow_out, exp);

      const auto act_torch_pow = actual_torch_pow[i].to(at::kDouble).template item<double>();
      assert_eq<long double>(val, act_torch_pow, exp);

      i++;
    }
  }
}

template<typename Vals, typename Pows>
void scalar_pow_tensor(const Vals vals, c10::ScalarType vals_dtype, const Pows pows, c10::ScalarType pows_dtype) {
  using T = typename Pows::value_type;

  const auto pow_tensor = torch::tensor(pows, pows_dtype);

  for (const auto val : vals) {
    const auto actual_pow = torch::pow(val, pow_tensor);
    auto actual_pow_out1 = torch::empty_like(actual_pow);
    const auto actual_pow_out2 =
      torch::pow_out(actual_pow_out1, val, pow_tensor);

    int i = 0;
    for (const auto pow : pows) {
      const auto exp = typed_pow(static_cast<T>(val), T(pow));

      const auto act_pow = actual_pow[i].template item<T>();
      assert_eq<T>(val, act_pow, exp);

      const auto act_pow_out1 = actual_pow_out1[i].template item<T>();
      assert_eq<T>(val, act_pow_out1, exp);

      const auto act_pow_out2 = actual_pow_out2[i].template item<T>();
      assert_eq<T>(val, act_pow_out2, exp);

      i++;
    }
  }
}

template<typename Vals, typename Pows>
void tensor_pow_tensor(const Vals vals, c10::ScalarType vals_dtype, Pows pows, c10::ScalarType pows_dtype) {
  using T = typename Vals::value_type;

  typedef std::numeric_limits< double > dbl;
  std::cout.precision(dbl::max_digits10);

  const auto vals_tensor = torch::tensor(vals, vals_dtype);
  for ([[maybe_unused]] const auto shirt : c10::irange(pows.size())) {
    const auto pows_tensor = torch::tensor(pows, pows_dtype);

    const auto actual_pow = vals_tensor.pow(pows_tensor);

    auto actual_pow_ = vals_tensor.clone();
    actual_pow_.pow_(pows_tensor);

    auto actual_pow_out = torch::empty_like(vals_tensor);
    torch::pow_out(actual_pow_out, vals_tensor, pows_tensor);

    auto actual_torch_pow = torch::pow(vals_tensor, pows_tensor);

    int i = 0;
    for (const auto val : vals) {
      const auto pow = pows[i];
      const auto exp = typed_pow(T(val), T(pow));

      const auto act_pow = actual_pow[i].template item<T>();
      assert_eq(val, act_pow, exp);

      const auto act_pow_ = actual_pow_[i].template item<T>();
      assert_eq(val, act_pow_, exp);

      const auto act_pow_out = actual_pow_out[i].template item<T>();
      assert_eq(val, act_pow_out, exp);

      const auto act_torch_pow = actual_torch_pow[i].template item<T>();
      assert_eq(val, act_torch_pow, exp);

      i++;
    }

    std::rotate(pows.begin(), pows.begin() + 1, pows.end());
  }
}

template<typename T>
void test_pow_one(const std::vector<T> vals) {
  for (const auto val : vals) {
    ASSERT_EQ(native::powi(val, T(1)), val);
  }
}

template<typename T>
void test_squared(const std::vector<T> vals) {
  for (const auto val : vals) {
    ASSERT_EQ(native::powi(val, T(2)), val * val);
  }
}

template<typename T>
void test_cubed(const std::vector<T> vals) {
  for (const auto val : vals) {
    ASSERT_EQ(native::powi(val, T(3)), val * val * val);
  }
}
template<typename T>
void test_inverse(const std::vector<T> vals) {
  for (const auto val : vals) {
    // 1 has special checks below
    if ( val != 1 && val != -1) {
      ASSERT_EQ(native::powi(val, T(-4)), 0);
      ASSERT_EQ(native::powi(val, T(-1)), val==1);
    }
  }
  T neg1 = -1;
  ASSERT_EQ(native::powi(neg1, T(0)), 1);
  ASSERT_EQ(native::powi(neg1, T(-1)), -1);
  ASSERT_EQ(native::powi(neg1, T(-2)), 1);
  ASSERT_EQ(native::powi(neg1, T(-3)), -1);
  ASSERT_EQ(native::powi(neg1, T(-4)), 1);

  T one = 1;
  ASSERT_EQ(native::powi(one, T(0)), 1);
  ASSERT_EQ(native::powi(one, T(-1)), 1);
  ASSERT_EQ(native::powi(one, T(-2)), 1);
  ASSERT_EQ(native::powi(one, T(-3)), 1);
  ASSERT_EQ(native::powi(one, T(-4)), 1);

}

}

TEST(PowTest, IntTensorPowAllScalars) {
  tensor_pow_scalar(ints, non_neg_ints, kInt, kInt);
  tensor_pow_scalar(ints, non_neg_longs, kInt, kInt);
  tensor_pow_scalar(ints, floats, kInt, kFloat);
  tensor_pow_scalar(ints, doubles, kInt, kDouble);
}

TEST(PowTest, LongTensorPowAllScalars) {
  tensor_pow_scalar(longs, non_neg_ints, kLong, kLong);
  tensor_pow_scalar(longs, non_neg_longs, kLong, kLong);
  tensor_pow_scalar(longs, floats, kLong, kFloat);
  tensor_pow_scalar(longs, doubles, kLong, kDouble);
}

TEST(PowTest, FloatTensorPowAllScalars) {
  tensor_pow_scalar(floats, ints, kFloat, kDouble);
  tensor_pow_scalar(floats, longs, kFloat, kDouble);
  tensor_pow_scalar(floats, floats, kFloat, kFloat);
  tensor_pow_scalar(floats, doubles, kFloat, kDouble);
}

TEST(PowTest, DoubleTensorPowAllScalars) {
  tensor_pow_scalar(doubles, ints, kDouble, kDouble);
  tensor_pow_scalar(doubles, longs, kDouble, kDouble);
  tensor_pow_scalar(doubles, floats, kDouble, kDouble);
  tensor_pow_scalar(doubles, doubles, kDouble, kDouble);
}

TEST(PowTest, IntScalarPowAllTensors) {
  scalar_pow_tensor(ints, c10::kInt, ints, c10::kInt);
  scalar_pow_tensor(ints, c10::kInt, longs, c10::kLong);
  scalar_pow_tensor(ints, c10::kInt, floats, c10::kFloat);
  scalar_pow_tensor(ints, c10::kInt, doubles, c10::kDouble);
}

TEST(PowTest, LongScalarPowAllTensors) {
  scalar_pow_tensor(longs, c10::kLong, longs, c10::kLong);
  scalar_pow_tensor(longs, c10::kLong, floats, c10::kFloat);
  scalar_pow_tensor(longs, c10::kLong, doubles, c10::kDouble);
}

TEST(PowTest, FloatScalarPowAllTensors) {
  scalar_pow_tensor(floats, c10::kFloat, floats, c10::kFloat);
  scalar_pow_tensor(floats, c10::kFloat, doubles, c10::kDouble);
}

TEST(PowTest, DoubleScalarPowAllTensors) {
  scalar_pow_tensor(doubles, c10::kDouble, doubles, c10::kDouble);
}

TEST(PowTest, IntTensorPowIntTensor) {
  tensor_pow_tensor(ints, c10::kInt, ints, c10::kInt);
}

TEST(PowTest, LongTensorPowLongTensor) {
  tensor_pow_tensor(longs, c10::kLong, longs, c10::kLong);
}

TEST(PowTest, FloatTensorPowFloatTensor) {
  tensor_pow_tensor(floats, c10::kFloat, floats, c10::kFloat);
}

TEST(PowTest, DoubleTensorPowDoubleTensor) {
  tensor_pow_tensor(doubles, c10::kDouble, doubles, c10::kDouble);
}

TEST(PowTest, TestIntegralPow) {
  test_pow_one(longs);
  test_pow_one(ints);

  test_squared(longs);
  test_squared(ints);

  test_cubed(longs);
  test_cubed(ints);

  test_inverse(longs);
  test_inverse(ints);
}
