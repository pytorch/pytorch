#include <gtest/gtest.h>

#include <ATen/native/Pow.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <iostream>
#include <vector>

using namespace at;

const std::vector<int> ints {
  std::numeric_limits<int>::min(),
  std::numeric_limits<int>::min() + 1,
  std::numeric_limits<int>::min() + 2,
  -sqrt(std::numeric_limits<int>::max()),
  -3, -2, -1, 0, 1, 2, 3,
  sqrt(std::numeric_limits<int>::max()),
  std::numeric_limits<int>::max() - 2,
  std::numeric_limits<int>::max() - 1,
  std::numeric_limits<int>::max()
};
const std::vector<int> non_neg_ints {
  0, 1, 2, 3,
  sqrt(std::numeric_limits<int>::max()),
  std::numeric_limits<int>::max() - 2,
  std::numeric_limits<int>::max() - 1,
  std::numeric_limits<int>::max()
};
const std::vector<long> longs {
  std::numeric_limits<long>::min(),
  std::numeric_limits<long>::min() + 1,
  std::numeric_limits<long>::min() + 2,
  -sqrt(std::numeric_limits<long>::max()),
  -3, -2, -1, 0, 1, 2, 3,
  sqrt(std::numeric_limits<long>::max()),
  std::numeric_limits<long>::max() - 2,
  std::numeric_limits<long>::max() - 1,
  std::numeric_limits<long>::max()
};
const std::vector<long> non_neg_longs {
  0, 1, 2, 3,
  sqrt(std::numeric_limits<long>::max()),
  std::numeric_limits<long>::max() - 2,
  std::numeric_limits<long>::max() - 1,
  std::numeric_limits<long>::max()
};
const std::vector<float> floats {
  std::numeric_limits<float>::lowest(),
  -3.0, -2.0, -1.0, -1.0/2.0, -1.0/3.0,
  -std::numeric_limits<float>::min(),
  0.0,
  std::numeric_limits<float>::min(),
  1.0/3.0, 1.0/2.0, 1.0, 2.0, 3.0,
  std::numeric_limits<float>::max(),
};
const std::vector<double> doubles {
  std::numeric_limits<double>::lowest(),
  -3.0, -2.0, -1.0, -1.0/2.0, -1.0/3.0,
  -std::numeric_limits<double>::min(),
  0.0,
  std::numeric_limits<double>::min(),
  1.0/3.0, 1.0/2.0, 1.0, 2.0, 3.0,
  std::numeric_limits<double>::max(),
};

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
      const auto exp = static_cast<T>(std::pow(static_cast<long double>(val), static_cast<double>(pow)));

      const auto act_pow = actual_pow[i].template item<T>();
      if ((!std::isnan(act_pow) || !std::isnan(exp)) && act_pow != exp) {
        std::cout << val << " pow " << pow << " = exp " << exp << " act " << act_pow << std::endl;
      }
      // ASSERT_EQ(act_pow, exp);

      const auto act_pow_ = actual_pow_[i].template item<T>();
      if ((!std::isnan(act_pow_) || !std::isnan(exp)) && act_pow_ != exp) {
        std::cout << val << " pow_ " << pow << " = exp " << exp << " act " << act_pow_ << std::endl;
      }
      // ASSERT_EQ(act_pow_, exp);

      const auto act_pow_out = actual_pow_out[i].template item<T>();
      if ((!std::isnan(act_pow_out) || !std::isnan(exp)) && act_pow_out != exp) {
        std::cout << val << " pow_out " << pow << " = exp " << exp << " act " << act_pow_out << std::endl;
      }
      // ASSERT_EQ(act_pow_out, exp);

      const auto act_torch_pow = actual_torch_pow[i].template item<T>();
      if ((!std::isnan(act_torch_pow) || !std::isnan(exp)) && act_torch_pow != exp) {
        std::cout << val << " pow_out " << pow << " = exp " << exp << " act " << act_torch_pow << std::endl;
      }
      // ASSERT_EQ(act_torch_pow, exp);

      i++;
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

template<typename T, typename Vals, typename Pows>
void scalar_pow_tensor(const Vals vals, const Pows pows) {
  const auto pow_tensor = torch::tensor(pows);

  for (const auto val : vals) {
    auto actual_pow = torch::pow(val, pow_tensor);

    int i = 0;
    for (const auto pow : pows) {
      const auto exp = static_cast<T>(std::pow(static_cast<long double>(val), static_cast<long double>(pow)));

      const auto act_pow = actual_pow[i].template item<T>();

      if ((!std::isnan(act_pow) || !std::isnan(exp)) && act_pow != exp) {
        std::cout << val << " pow " << pow << " = exp " << exp << " act " << act_pow << std::endl;
      }
      // ASSERT_EQ(act_pow, exp);

      i++;
    }
  }
}

TEST(PowTest, IntScalarPowAllTensors) {
  scalar_pow_tensor<long>(ints, ints);
  scalar_pow_tensor<long>(ints, longs);
  scalar_pow_tensor<long>(ints, floats);
  scalar_pow_tensor<long>(ints, doubles);
}

TEST(PowTest, LongScalarPowAllTensors) {
  scalar_pow_tensor<long>(longs, ints);
  scalar_pow_tensor<long>(longs, longs);
  scalar_pow_tensor<long>(longs, floats);
  scalar_pow_tensor<long>(longs, doubles);
}

TEST(PowTest, FloatScalarPowAllTensors) {
  scalar_pow_tensor<double>(floats, ints);
  scalar_pow_tensor<double>(floats, longs);
  scalar_pow_tensor<double>(floats, floats);
  scalar_pow_tensor<double>(floats, doubles);
}

TEST(PowTest, DoubleScalarPowAllTensors) {
  scalar_pow_tensor<double>(doubles, ints);
  scalar_pow_tensor<double>(doubles, longs);
  scalar_pow_tensor<double>(doubles, floats);
  scalar_pow_tensor<double>(doubles, doubles);
}

template<typename Vals, typename Pows>
void tensor_pow_tensor(const Vals vals, Pows pows) {
  using T = typename Vals::value_type;

  const auto vals_tensor = torch::tensor(vals);
  for (size_t shift = 0; shift < pows.size(); shift++) {
    const auto pows_tensor = torch::tensor(pows);

    auto actual_pow = vals_tensor.pow(pows_tensor);

    int i = 0;
    for (const auto val : vals) {
      const auto pow = pows[i];
      const auto exp = static_cast<T>(std::pow(val, pow));

      const auto act_pow = actual_pow[i].template item<T>();

      if ((!std::isnan(act_pow) || !std::isnan(exp)) && act_pow != exp) {
        std::cout << val << " pow " << pow << " = exp " << exp << " act " << act_pow << std::endl;
      }
      // ASSERT_EQ(act_pow, exp);

      i++;
    }

    std::rotate(pows.begin(), pows.begin() + 1, pows.end());
  }
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
