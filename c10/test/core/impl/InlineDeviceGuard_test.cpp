#include <gtest/gtest.h>
#include <initializer_list>

#include <c10/core/impl/FakeGuardImpl.h>
#include <c10/core/impl/InlineDeviceGuard.h>

using namespace c10;
using namespace c10::impl;

constexpr auto TestDeviceType = DeviceType::CUDA;
using TestGuardImpl = FakeGuardImpl<TestDeviceType>;

static Device dev(DeviceIndex index) {
  return Device(TestDeviceType, index);
}

// -- InlineDeviceGuard -------------------------------------------------------

using TestGuard = InlineDeviceGuard<TestGuardImpl>;

TEST(InlineDeviceGuard, Constructor) {
  for (DeviceIndex i : std::initializer_list<DeviceIndex>{-1, 0, 1}) {
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
    /*
    {
      // Optional constructor
      TestGuard g(std::make_optional(dev(i)));
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
    */
  }
}

TEST(InlineDeviceGuard, ConstructorError) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(InlineDeviceGuard<FakeGuardImpl<DeviceType::CUDA>> g(
      Device(DeviceType::HIP, 1)));
}

TEST(InlineDeviceGuard, SetDevice) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = 1;
  TestGuard g(i);
  DeviceIndex i2 = 2;
  g.set_device(dev(i2));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
  g.set_device(dev(i2));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

TEST(InlineDeviceGuard, ResetDevice) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = 1;
  TestGuard g(i);
  DeviceIndex i2 = 2;
  g.reset_device(dev(i2));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
  g.reset_device(dev(i2));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

TEST(InlineDeviceGuard, SetIndex) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = 1;
  TestGuard g(i);
  DeviceIndex i2 = 2;
  g.set_index(i2);
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
  g.set_index(i2);
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

// -- InlineOptionalDeviceGuard
// --------------------------------------------------

using MaybeTestGuard = InlineOptionalDeviceGuard<TestGuardImpl>;

TEST(InlineOptionalDeviceGuard, Constructor) {
  for (DeviceIndex i : std::initializer_list<DeviceIndex>{-1, 0, 1}) {
    DeviceIndex init_i = 0;
    TestGuardImpl::setDeviceIndex(init_i);
    auto test_body = [&](MaybeTestGuard& g) -> void {
      ASSERT_EQ(g.original_device(), dev(init_i));
      ASSERT_EQ(g.current_device(), dev(i == -1 ? init_i : i));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i == -1 ? init_i : i);
      // Test un-bracketed write to device index
      TestGuardImpl::setDeviceIndex(4);
    };
    {
      // Index constructor
      MaybeTestGuard g(i);
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
    {
      // Device constructor
      MaybeTestGuard g(dev(i));
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
    {
      // Optional constructor
      MaybeTestGuard g(std::make_optional(dev(i)));
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
  }
}

TEST(InlineOptionalDeviceGuard, NullaryConstructor) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  auto test_body = [&](MaybeTestGuard& g) -> void {
    ASSERT_EQ(g.original_device(), std::nullopt);
    ASSERT_EQ(g.current_device(), std::nullopt);
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
  };
  {
    MaybeTestGuard g;
    test_body(g);
  }
  {
    // If you want nullopt directly to work, define a nullopt_t
    // overload.  But I don't really see why you'd want this lol.
    std::optional<Device> dev_opt = std::nullopt;
    MaybeTestGuard g(dev_opt);
    test_body(g);
  }
}

TEST(InlineOptionalDeviceGuard, SetDevice) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  MaybeTestGuard g;
  DeviceIndex i = 1;
  g.set_device(dev(i));
  ASSERT_EQ(g.original_device(), std::make_optional(dev(init_i)));
  ASSERT_EQ(g.current_device(), std::make_optional(dev(i)));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
  g.set_device(dev(i));
  ASSERT_EQ(g.original_device(), std::make_optional(dev(init_i)));
  ASSERT_EQ(g.current_device(), std::make_optional(dev(i)));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}

TEST(InlineOptionalDeviceGuard, SetIndex) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = 1;
  MaybeTestGuard g;
  g.set_index(i);
  ASSERT_EQ(g.original_device(), std::make_optional(dev(init_i)));
  ASSERT_EQ(g.current_device(), std::make_optional(dev(i)));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
  g.set_index(i);
  ASSERT_EQ(g.original_device(), std::make_optional(dev(init_i)));
  ASSERT_EQ(g.current_device(), std::make_optional(dev(i)));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}
