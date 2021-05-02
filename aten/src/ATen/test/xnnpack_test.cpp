#include <gtest/gtest.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Pooling.h>

#if defined(C10_MOBILE) && defined(USE_XNNPACK)

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  return diff.abs().max().item<float>() < (0.01 + 2e-2 * maxValue);
}
bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.f;
}

void test_hardswish(const at::Tensor& input, const at::Tensor& expected) {
  ASSERT_TRUE(at::native::xnnpack::use_hardswish(input));
  auto result = at::native::xnnpack::hardswish(input);
  auto check = almostEqual(expected, result);
  ASSERT_TRUE(check);
}

void test_hardswish_(at::Tensor input, const at::Tensor& expected) {
  ASSERT_TRUE(at::native::xnnpack::use_hardswish(input));
  at::native::xnnpack::hardswish_(input);
  auto check = almostEqual(expected, input);
  ASSERT_TRUE(check);
}

void test_global_average_pool(at::Tensor input, const at::Tensor& expected) {
  ASSERT_TRUE(at::native::xnnpack::use_global_average_pool(input));
  auto result = at::native::xnnpack::global_average_pool(input);
  auto check = almostEqual(expected, result);
  ASSERT_TRUE(check);
}

// Since XNNPACK path is only taken #if defined(C10_MOBILE) && defined(USE_XNNPACK)
// We can't compare regular CPU path with XNNPACK path in the same test binary
// Instead we precompute regular results and compare with XNNPACK path here
TEST(TestXNNPackOps, TestHardSwish) {
  // input, expected_result pair
  std::vector<std::pair<at::Tensor, at::Tensor>> input_result_pairs = {
    {
      torch::tensor({1, 2, 3, 4, 5}, {torch::kFloat32}),
      torch::tensor({0.6667, 1.6667, 3.0000, 4.0000, 5.0000}, {torch::kFloat32})
    },
    {
      torch::tensor({0.3330}, {torch::kFloat32}),
      torch::tensor({0.1850}, {torch::kFloat32})
    },
    {
      torch::tensor({
        {0.4523, 0.8131, 0.9829},
        {0.0782, 0.7395, 0.0787}
      }),
      torch::tensor({
        {0.2602, 0.5167, 0.6525},
        {0.0401, 0.4609, 0.0404}
      })
    },
  };

  for (const auto& input_result : input_result_pairs) {
    test_hardswish(input_result.first, input_result.second);
    test_hardswish_(input_result.first, input_result.second);
  }
}

TEST(TestXNNPackOps, TestGlobal) {
  // input, expected_result pair
  std::vector<std::pair<at::Tensor, at::Tensor>> input_result_pairs = {
    {
      torch::tensor({{
        {{0.0852, 0.7312, 0.9943, 0.7105},
        {0.0956, 0.9072, 0.3124, 0.9362},
        {0.5878, 0.8883, 0.5086, 0.9494}},
        {{0.1056, 0.4968, 0.7740, 0.7593},
        {0.8519, 0.3543, 0.8078, 0.5517},
        {0.1413, 0.4608, 0.1706, 0.0314}}
      }}, {torch::kFloat32}),
      torch::tensor({{
        {{0.6422}},
        {{0.4588}}
      }}, {torch::kFloat32})
    },
    {
      torch::tensor({{
          {{0.0280, 0.9073},
          {0.2103, 0.5298}},
          {{0.5335, 0.9901},
          {0.2902, 0.2955}}
        },
        {
          {{0.2363, 0.7024},
          {0.7903, 0.8260}},
          {{0.3802, 0.5959},
          {0.5749, 0.8855}}
        }}, {torch::kFloat32}),
        torch::tensor(
          {{{{0.4188}},
          {{0.5273}}},
          {{{0.6388}},
          {{0.6091}}}},
          {torch::kFloat32}
        )
    }
  };

  for (const auto& input_result : input_result_pairs) {
    test_global_average_pool(input_result.first, input_result.second);
  }
}
#endif
