#include <gtest/gtest.h>

#include <c10/core/Device.h>

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
  };

  for (auto& ds : valid_devices) {
    c10::Device d(ds.device_string);
    ASSERT_EQ(d.type(), ds.device_type)
        << "Device String: " << ds.device_string;
    ASSERT_EQ(d.index(), ds.device_index)
        << "Device String: " << ds.device_string;
  }

  for (auto& ds : invalid_device_strings) {
    bool got_exception = false;
    try {
      c10::Device d(ds);
    } catch (c10::Error& ex) {
      got_exception = true;
    }
    ASSERT_TRUE(got_exception) << "Device String: " << ds;
  }
}
