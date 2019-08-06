#include <gtest/gtest.h>

#include <ATen/native/Pow.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <iostream>
#include <vector>
#include <type_traits>

using namespace at;

namespace {

const auto int_min = std::numeric_limits<int>::min();
const auto int_max = std::numeric_limits<int>::max();
const auto long_min = std::numeric_limits<int64_t>::min();
const auto long_max = std::numeric_limits<int64_t>::max();
const auto float_lowest = std::numeric_limits<float>::lowest();
const auto float_min = std::numeric_limits<float>::min();
const auto float_max = std::numeric_limits<float>::max();
const auto double_lowest = std::numeric_limits<double>::lowest();
const auto double_min = std::numeric_limits<double>::min();
const auto double_max = std::numeric_limits<double>::max();

const std::vector<int> ints {
  int_min,
  int_min + 1,
  int_min + 2,
  static_cast<int>(-sqrt(int_max)),
  -3, -2, -1, 0, 1, 2, 3,
  static_cast<int>(sqrt(int_max)),
  int_max - 2,
  int_max - 1,
  int_max
};
const std::vector<int> non_neg_ints {
  0, 1, 2, 3,
  static_cast<int>(sqrt(int_max)),
  int_max - 2,
  int_max - 1,
  int_max
};
const std::vector<int64_t> longs {
  long_min,
  long_min + 1,
  long_min + 2,
  static_cast<int64_t>(-sqrt(long_max)),
  -3, -2, -1, 0, 1, 2, 3,
  static_cast<int64_t>(sqrt(long_max)),
  long_max - 2,
  long_max - 1,
  long_max
};
const std::vector<int64_t> non_neg_longs {
  0, 1, 2, 3,
  static_cast<int64_t>(sqrt(long_max)),
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
  typename std::enable_if<std::is_floating_point<T>::value,T>::type* = nullptr>
void assert_eq(T val, T act, T exp) {
  if (std::isnan(act) || std::isnan(exp)) {
    return;
  }
  ASSERT_FLOAT_EQ(act, exp);
}

template <class T,
  typename std::enable_if<std::is_integral<T>::value, T>::type* = nullptr>
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

template<typename Vals, typename Pows>
void tensor_pow_scalar(const Vals vals, const Pows pows) {
  using T = typename Vals::value_type;

  const auto tensor = torch::tensor(vals);

  for (const auto pow : pows) {
    auto actual_pow = tensor.pow(pow);

    auto actual_pow_ = tensor.clone();
    actual_pow_.pow_(pow);

    auto actual_pow_out = torch::empty_like(tensor);
    torch::pow_out(actual_pow_out, tensor, pow);

    auto actual_torch_pow = torch::pow(tensor, pow);

    int i = 0;
    for (const auto val : vals) {
      const auto exp = static_cast<T>(
        std::pow(static_cast<long double>(val), static_cast<double>(pow)));

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
  }
}

template<typename Vals, typename Pows>
void scalar_pow_tensor(const Vals vals, const Pows pows) {
  using T = typename Pows::value_type;

  const auto pow_tensor = torch::tensor(pows);

  for (const auto val : vals) {
    const auto actual_pow = torch::pow(val, pow_tensor);
    auto actual_pow_out1 = torch::empty_like(pow_tensor);
    const auto actual_pow_out2 =
      torch::pow_out(actual_pow_out1, val, pow_tensor);

    int i = 0;
    for (const auto pow : pows) {
      const auto exp = static_cast<T>(
        std::pow(static_cast<T>(val), pow));

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
void tensor_pow_tensor(const Vals vals, Pows pows) {
  using T = typename Vals::value_type;

  typedef std::numeric_limits< double > dbl;
  std::cout.precision(dbl::max_digits10);

  const auto vals_tensor = torch::tensor(vals);
  for (size_t shift = 0; shift < pows.size(); shift++) {
    const auto pows_tensor = torch::tensor(pows);

    const auto actual_pow = vals_tensor.pow(pows_tensor);

    auto actual_pow_ = vals_tensor.clone();
    actual_pow_.pow_(pows_tensor);

    auto actual_pow_out = torch::empty_like(vals_tensor);
    torch::pow_out(actual_pow_out, vals_tensor, pows_tensor);

    auto actual_torch_pow = torch::pow(vals_tensor, pows_tensor);

    int i = 0;
    for (const auto val : vals) {
      const auto pow = pows[i];
      const auto exp = static_cast<T>(std::pow(val, pow));

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

}

TEST(PowTest, IntTensorPowAllScalars) {
  tensor_pow_scalar(ints, non_neg_ints);
  tensor_pow_scalar(ints, non_neg_longs);
  tensor_pow_scalar(ints, floats);
  tensor_pow_scalar(ints, doubles);
}

TEST(PowTest, LongTensorPowAllScalars) {
  tensor_pow_scalar(longs, non_neg_ints);
  tensor_pow_scalar(longs, non_neg_longs);
  tensor_pow_scalar(longs, floats);
  tensor_pow_scalar(longs, doubles);
}

TEST(PowTest, FloatTensorPowAllScalars) {
  tensor_pow_scalar(floats, ints);
  tensor_pow_scalar(floats, longs);
  tensor_pow_scalar(floats, floats);
  tensor_pow_scalar(floats, doubles);
}

TEST(PowTest, DoubleTensorPowAllScalars) {
  tensor_pow_scalar(doubles, ints);
  tensor_pow_scalar(doubles, longs);
  tensor_pow_scalar(doubles, floats);
  tensor_pow_scalar(doubles, doubles);
}

TEST(PowTest, IntScalarPowAllTensors) {
  scalar_pow_tensor(ints, ints);
  scalar_pow_tensor(ints, longs);
  scalar_pow_tensor(ints, floats);
  scalar_pow_tensor(ints, doubles);
}

TEST(PowTest, LongScalarPowAllTensors) {
  scalar_pow_tensor(longs, longs);
  scalar_pow_tensor(longs, floats);
  scalar_pow_tensor(longs, doubles);
}

TEST(PowTest, FloatScalarPowAllTensors) {
  scalar_pow_tensor(floats, floats);
  scalar_pow_tensor(floats, doubles);
}

TEST(PowTest, DoubleScalarPowAllTensors) {
  scalar_pow_tensor(doubles, doubles);
}

TEST(PowTest, IntTensorPowIntTensor) {
  tensor_pow_tensor(ints, ints);
}

TEST(PowTest, LongTensorPowLongTensor) {
  tensor_pow_tensor(longs, longs);
}

TEST(PowTest, FloatTensorPowFloatTensor) {
  tensor_pow_tensor(floats, floats);
}

TEST(PowTest, DoubleTensorPowDoubleTensor) {
  tensor_pow_tensor(doubles, doubles);
}
