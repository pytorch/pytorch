#include <gtest/gtest.h>

#include <torch/headeronly/core/DeviceType.h>

TEST(TestDeviceType, TestDeviceType) {
  using torch::headeronly::DeviceType;
  constexpr DeviceType expected_device_types[] = {
      torch::headeronly::kCPU,
      torch::headeronly::kCUDA,
      DeviceType::MKLDNN,
      DeviceType::OPENGL,
      DeviceType::OPENCL,
      DeviceType::IDEEP,
      torch::headeronly::kHIP,
      torch::headeronly::kFPGA,
      torch::headeronly::kMAIA,
      torch::headeronly::kXLA,
      torch::headeronly::kVulkan,
      torch::headeronly::kMetal,
      torch::headeronly::kXPU,
      torch::headeronly::kMPS,
      torch::headeronly::kMeta,
      torch::headeronly::kHPU,
      torch::headeronly::kVE,
      torch::headeronly::kLazy,
      torch::headeronly::kIPU,
      torch::headeronly::kMTIA,
      torch::headeronly::kPrivateUse1,
  };
  for (int8_t i = 0; i <
       static_cast<int8_t>(torch::headeronly::COMPILE_TIME_MAX_DEVICE_TYPES);
       i++) {
    EXPECT_EQ(static_cast<DeviceType>(i), expected_device_types[i]);
  }
}
