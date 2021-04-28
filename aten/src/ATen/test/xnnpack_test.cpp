#include <gtest/gtest.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/xnnpack/Common.h>

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
  }
}

#endif
