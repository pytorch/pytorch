#include <gtest/gtest.h>

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

// -- Device -------------------------------------------------------

struct ExpectedDeviceTestResult {
  std::string device_string;
  c10::DeviceType device_type;
  c10::DeviceIndex device_index;
};

TEST(DeviceTest, BasicConstruction) {
  std::vector<ExpectedDeviceTestResult> valid_devices = {
      {"cpu", c10::DeviceType::CPU, -1},
      {"cuda", c10::DeviceType::CUDA, -1},
      {"cpu:0", c10::DeviceType::CPU, 0},
      {"cuda:0", c10::DeviceType::CUDA, 0},
      {"cuda:1", c10::DeviceType::CUDA, 1},
  };
  std::vector<std::string> invalid_device_strings = {
      "cpu:x",
      "cpu:foo",
      "cuda:cuda",
      "cuda:",
      "cpu:0:0",
      "cpu:0:",
      "cpu:-1",
      "::",
      ":",
      "cpu:00",
      "cpu:01"};

  for (auto& ds : valid_devices) {
    c10::Device d(ds.device_string);
    ASSERT_EQ(d.type(), ds.device_type)
        << "Device String: " << ds.device_string;
    ASSERT_EQ(d.index(), ds.device_index)
        << "Device String: " << ds.device_string;
  }

  auto make_device = [](const std::string& ds) { return c10::Device(ds); };

  for (auto& ds : invalid_device_strings) {
    EXPECT_THROW(make_device(ds), c10::Error) << "Device String: " << ds;
  }
}
