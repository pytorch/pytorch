#include <gtest/gtest.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
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
      "cpu:1",
      "cpu:2",
      "cpu:127",
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

TEST(DeviceTest, CpuDeviceIndexValidation) {
  EXPECT_NO_THROW(c10::Device(c10::DeviceType::CPU));
  EXPECT_NO_THROW(c10::Device(c10::DeviceType::CPU, -1));
  EXPECT_NO_THROW(c10::Device(c10::DeviceType::CPU, 0));

  for (const c10::DeviceIndex index : {1, 2, 127}) {
    EXPECT_THROW(c10::Device(c10::DeviceType::CPU, index), c10::Error);
    EXPECT_NO_THROW(c10::Device(c10::DeviceType::CUDA, index));
  }
}

TEST(DeviceTest, UnknownTypeErrorOmitsXla) {
  ASSERT_EQ(c10::Device("xla:0").type(), c10::DeviceType::XLA);
  try {
    c10::Device("tpu");
    FAIL() << "Expected c10::Error";
  } catch (const c10::Error& e) {
    std::string msg = e.what_without_backtrace();
    EXPECT_NE(msg.find("Expected one of cpu, cuda"), std::string::npos) << msg;
    EXPECT_EQ(msg.find("xla"), std::string::npos) << msg;
  }
}

TEST(DeviceTypeTest, PrivateUseOneDeviceType) {
  c10::register_privateuse1_backend("my_privateuse1_backend");
  ASSERT_TRUE(c10::is_privateuse1_backend_registered());
  ASSERT_EQ(c10::get_privateuse1_backend(true), "my_privateuse1_backend");
  ASSERT_EQ(c10::get_privateuse1_backend(false), "MY_PRIVATEUSE1_BACKEND");
}

TEST(DeviceTypeTest, PrivateUseOneRegister) {
  ASSERT_THROW(c10::register_privateuse1_backend("cpu"), c10::Error);
  ASSERT_THROW(c10::register_privateuse1_backend("cuda"), c10::Error);
  ASSERT_THROW(c10::register_privateuse1_backend("hip"), c10::Error);
  ASSERT_THROW(c10::register_privateuse1_backend("mps"), c10::Error);
  ASSERT_THROW(c10::register_privateuse1_backend("xpu"), c10::Error);
  ASSERT_THROW(c10::register_privateuse1_backend("mtia"), c10::Error);
}
