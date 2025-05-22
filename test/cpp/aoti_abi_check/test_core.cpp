#include <gtest/gtest.h>

#include <torch/standalone/core/DeviceType.h>

namespace torch {
namespace standalone {
TEST(TestCore, TestDeviceType) {
  // clang-format off
  constexpr DeviceType expected_device_types[] = {
    kCPU,
    kCUDA,
    kMKLDNN,
    kOPENGL,
    kOPENCL,
    kIDEEP,
    kHIP,
    kFPGA,
    kMAIA,
    kXLA,
    kVulkan,
    kMetal,
    kXPU,
    kMPS,
    kMeta,
    kHPU,
    kVE,
    kLazy,
    kIPU,
    kMTIA,
    kPrivateUse1,
  };
  // clang-format on
  for (int8_t i = 0;
       i < static_cast<int8_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
       ++i) {
    EXPECT_EQ(static_cast<DeviceType>(i), expected_device_types[i]);
  }
}
} // namespace standalone
} // namespace torch
