#ifdef USE_VULKAN_API

#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/irange.h>

// TODO: These functions should move to a common place.

namespace {

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor>& inputs) {
  float maxValue = 0.0f;

  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }

#ifdef USE_VULKAN_FP16_INFERENCE
  constexpr float tolerance = 1e-2;
#else
  constexpr float tolerance = 1e-5;
#endif

  return diff.abs().max().item<float>() <= (tolerance * maxValue);
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

void showRtol(const at::Tensor& a, const at::Tensor& b) {
  const auto diff = (a - b).abs();

  float maxValue = a.abs().max().item<float>();
  maxValue = fmax(b.abs().max().item<float>(), maxValue);

#ifdef USE_VULKAN_FP16_INFERENCE
  constexpr float tolerance = 1e-2;
#else
  constexpr float tolerance = 1e-5;
#endif

  const float maxDiff = maxValue * tolerance;
  std::cout << "Max Diff allowed: " << maxDiff << std::endl;
  if (diff.sizes().size() == 2) {
    for (const auto y : c10::irange(diff.sizes()[0])) {
      std::cout << y << ":";
      for (const auto x : c10::irange(diff.sizes()[1])) {
        float diff_xy = diff[y][x].item<float>();
        if (diff_xy > maxDiff) {
          std::cout << std::setw(5) << x;
        }
        else {
          std::cout << std::setw(5) << " ";
        }
      }
      std::cout << std::endl;
    }
  }
}

} // namespace

namespace {

class VulkanAPITest : public ::testing::Test {
public:
#if defined (__ANDROID__)  // to avoid `Undefined symbols for architecture arm64` error
    static void SetUpTestSuite() {
    }

    static void TearDownTestSuite() {
    }
#endif
};

TEST_F(VulkanAPITest, adaptive_avg_pool2d) {
  if (!at::is_vulkan_available()) {
    return;
  }
  c10::InferenceMode mode;

  const auto in_cpu = at::rand({5, 7, 47, 31}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::adaptive_avg_pool2d(in_cpu, {3, 3});
  const auto out_vulkan = at::adaptive_avg_pool2d(in_cpu.vulkan(), {3, 3});

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

} // namespace

#endif /* USE_VULKAN_API */
