#include <gtest/gtest.h>

#include <torch/headeronly/core/DeviceCapability.h>

TEST(TestDeviceCapability, TestDeviceCapability) {
  using torch::headeronly::DeviceCapability;

  EXPECT_GT(torch::headeronly::NUMBER_OF_DEVICE_CAPABILITIES, 0u);

  // Default-constructed capability enables every supported scalar type.
  DeviceCapability cap;
  int count = 0;
  cap.forEachSupportedScalarType(
      [&](torch::headeronly::ScalarType) { ++count; });
  EXPECT_GT(count, 0);

  // Unscoped enum index alias is reachable from torch::headeronly.
  torch::headeronly::ScalarTypeIndex idx = torch::headeronly::kIndex_Byte;
  EXPECT_EQ(static_cast<int>(idx), 0);

  // Backward-compatible c10 alias resolves to the same struct.
  c10::DeviceCapability c10cap;
  EXPECT_EQ(
      c10cap.capability_data.capability_bits,
      cap.capability_data.capability_bits);
}
