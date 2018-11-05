#include <gtest/gtest.h>

#include "c10/detail/InlineDeviceGuard.h"
#include "c10/detail/FakeGuardImpl.h"

using namespace c10;
using namespace c10::detail;

constexpr auto TestDeviceType = DeviceType::CUDA;
using TestGuardImpl = FakeGuardImpl<TestDeviceType>;
using TestGuard = InlineDeviceGuard<TestGuardImpl>;

Device cpu_dev() {
  return Device(DeviceType::CPU);
}

Device dev(DeviceIndex index) {
  return Device(TestDeviceType, index);
}

TEST(InlineDeviceGuard, Constructor) {
  for (DeviceIndex i : {-1, 0, 1}) {
    DeviceIndex init_i = 0;
    TestGuardImpl::setDeviceIndex(init_i);
    auto test_body = [&](TestGuard& g) -> void {
      ASSERT_EQ(g.original_device(), dev(init_i));
      ASSERT_EQ(g.current_device(), dev(i == -1 ? init_i : i));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i == -1 ? init_i : i);
      // Test un-bracketed write to device index
      TestGuardImpl::setDeviceIndex(4);
    };
    {
      // Index constructor
      TestGuard g(i);
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
    {
      // Device constructor
      TestGuard g(dev(i));
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
    {
      // Optional constructor
      TestGuard g(make_optional(dev(i)));
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
  }
}

TEST(InlineDeviceGuard, NullaryConstructor) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  auto test_body = [&](TestGuard& g) -> void {
    ASSERT_EQ(g.original_device(), dev(init_i));
    ASSERT_EQ(g.current_device(), dev(init_i));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
    // Test un-bracketed write to device index
    TestGuardImpl::setDeviceIndex(4);
  };
  {
    TestGuard g;
    test_body(g);
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
  {
    TestGuard g(nullopt);
    test_body(g);
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
}

TEST(InlineDeviceGuard, ConstructorError) {
  EXPECT_ANY_THROW(InlineDeviceGuard<FakeGuardImpl<DeviceType::CUDA>>
                   g(Device(DeviceType::HIP, 1)));
}

TEST(InlineDeviceGuard, MoveConstructor) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = init_i + 1;
  TestGuard g(i);
  TestGuard g2(std::move(g));
  ASSERT_TRUE(!g.initialized()); // use-after-move for testing
  ASSERT_EQ(g2.original_device(), dev(init_i));
  ASSERT_EQ(g2.current_device(), dev(i));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}

TEST(InlineDeviceGuard, MoveConstructorFromTemporary) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = init_i + 1;
  TestGuard g2(i);
  ASSERT_EQ(g2.original_device(), dev(init_i));
  ASSERT_EQ(g2.current_device(), dev(i));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}

/*
TEST(InlineDeviceGuard, MoveAssignment) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = init_i + 1;
  TestGuard g(i);
  DeviceIndex i2 = init_i + 2;
  TestGuard g2(i2);
  g = std::move(g2);
  ASSERT_TRUE(!g2.initialized()); // use-after-move for testing
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

TEST(InlineDeviceGuard, MoveAssignmentFromTemporary) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = init_i + 1;
  TestGuard g(i);
  DeviceIndex i2 = init_i + 2;
  g = TestGuard(i2);
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

// Suppress -Wself-move
void move_it(TestGuard& g, TestGuard&& g2) {
  g = std::move(g2);
}

TEST(InlineDeviceGuard, MoveAssignmentSelf) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = init_i + 1;
  TestGuard g(i);
  move_it(g, std::move(g));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}
*/

TEST(InlineDeviceGuard, SetDevice) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = init_i + 1;
  TestGuard g(i);
  DeviceIndex i2 = init_i + 2;
  g.set_device(dev(i2));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
  g.set_device(dev(i2));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

TEST(InlineDeviceGuard, SetIndex) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = init_i + 1;
  TestGuard g(i);
  DeviceIndex i2 = init_i + 2;
  g.set_index(i2);
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
  g.set_index(i2);
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}
