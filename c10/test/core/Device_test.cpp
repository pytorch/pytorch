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
  ASSERT_THROW(c10::register_privateuse1_backend("metal"), c10::Error);
  ASSERT_THROW(c10::register_privateuse1_backend("mps"), c10::Error);
  ASSERT_THROW(c10::register_privateuse1_backend("xpu"), c10::Error);
  ASSERT_THROW(c10::register_privateuse1_backend("mtia"), c10::Error);
}

// The canonical printed representation of a device constructed from "metal"
// remains "mps" for backward compatibility during the transition.
TEST(DeviceTest, MetalAliasStrReturnsMps) {
  c10::Device d("metal");
  ASSERT_EQ(d.str(), "mps");

  // Verify C++ ostream operator<< support
  std::stringstream ss;
  ss << d;
  ASSERT_EQ(ss.str(), "mps");
}

TEST(DeviceTest, MetalAliasParsesToMPS) {
  c10::Device device_lowercase("metal");
  EXPECT_EQ(device_lowercase.type(), c10::DeviceType::MPS);
  EXPECT_EQ(device_lowercase.index(), -1);

  c10::Device device_with_index("metal:0");
  EXPECT_EQ(device_with_index.type(), c10::DeviceType::MPS);
  EXPECT_EQ(device_with_index.index(), 0);
}

TEST(DeviceTest, ErrorMessageIncludesMetal) {
  try {
    c10::Device bad("notadevice");
    FAIL() << "Expected c10::Error";
  } catch (const c10::Error& e) {
    ASSERT_NE(std::string(e.what()).find("metal"), std::string::npos)
        << "Error message should list 'metal' as an accepted device string";
  }
}

TEST(DeviceTest, MetalPrefixedGarbageIsRejected) {
  auto make = [](const std::string& s) { return c10::Device(s); };
  EXPECT_THROW(make("metalx"), c10::Error);
  EXPECT_THROW(make("metal:"), c10::Error);
  EXPECT_THROW(make("metal:01"), c10::Error);
  EXPECT_THROW(make("metal:-1"), c10::Error);
}
