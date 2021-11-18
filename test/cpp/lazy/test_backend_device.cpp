#include <gtest/gtest.h>

#include <sstream>

#include <torch/csrc/lazy/backend/backend_device.h>

namespace torch {
namespace lazy {

TEST(BackendDeviceTest, BackendDeviceType) {
  auto type = BackendDeviceType();

  EXPECT_EQ(type.type, 0);
  EXPECT_STREQ(type.toString().c_str(), "Unknown");
}

TEST(BackendDeviceTest, Basic1) {
  auto device = BackendDevice();

  EXPECT_EQ(device.type(), 0);
  EXPECT_EQ(device.ordinal(), 0);
  EXPECT_STREQ(device.toString().c_str(), "Unknown0");
}

TEST(BackendDeviceTest, Basic2) {
  auto type = std::make_shared<BackendDeviceType>();
  type->type = 1;
  auto device = BackendDevice(std::move(type), 1);

  EXPECT_EQ(device.type(), 1);
  EXPECT_EQ(device.ordinal(), 1);
  EXPECT_STREQ(device.toString().c_str(), "Unknown1");
}

TEST(BackendDeviceTest, Basic3) {
  struct TestType : public BackendDeviceType {
    std::string toString() const override { return "Test"; }
  };

  auto device = BackendDevice(std::make_shared<TestType>(), 1);

  EXPECT_EQ(device.type(), 0);
  EXPECT_EQ(device.ordinal(), 1);
  EXPECT_STREQ(device.toString().c_str(), "Test1");
}

TEST(BackendDeviceTest, Compare) {
  auto type = std::make_shared<BackendDeviceType>();
  type->type = 1;

  auto device1 = BackendDevice(std::make_shared<BackendDeviceType>(), 1);
  auto device2 = BackendDevice(std::move(type), 0);
  auto device3 = BackendDevice(std::make_shared<BackendDeviceType>(), 2);
  auto device4 = BackendDevice(std::make_shared<BackendDeviceType>(), 1);

  EXPECT_NE(device1, device2);
  EXPECT_NE(device1, device3);
  EXPECT_EQ(device1, device4);
  EXPECT_LT(device1, device2);
  EXPECT_LT(device1, device3);
}

TEST(BackendDeviceTest, Ostream) {
  auto device = BackendDevice();
  std::stringstream ss;
  ss << device;

  EXPECT_EQ(device.toString(), ss.str());
}

}  // namespace lazy
}  // namespace torch
