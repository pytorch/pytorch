#include <gtest/gtest.h>
#include <include/openreg.h>

namespace {

class DeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    orSetDevice(0);
  }
};

TEST_F(DeviceTest, GetDeviceCountValid) {
  int count = -1;
  EXPECT_EQ(orGetDeviceCount(&count), orSuccess);
  EXPECT_EQ(count, 2);
}

TEST_F(DeviceTest, GetDeviceValid) {
  int device = -1;
  EXPECT_EQ(orGetDevice(&device), orSuccess);
  EXPECT_EQ(device, 0);
}

TEST_F(DeviceTest, SetDeviceValid) {
  EXPECT_EQ(orSetDevice(1), orSuccess);

  int device = -1;
  EXPECT_EQ(orGetDevice(&device), orSuccess);
  EXPECT_EQ(device, 1);

  EXPECT_EQ(orSetDevice(0), orSuccess);
  EXPECT_EQ(orGetDevice(&device), orSuccess);
  EXPECT_EQ(device, 0);
}

TEST_F(DeviceTest, SetDeviceInvalidNegative) {
  EXPECT_EQ(orSetDevice(-1), orErrorUnknown);
}

} // namespace
