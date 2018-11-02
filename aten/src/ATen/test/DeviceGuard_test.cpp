#include "gtest/gtest.h"

#include "ATen/DeviceGuard.h"
#include "c10/detail/FakeGuardImpl.h"

using namespace at;
using c10::detail::FakeGuardImpl;

constexpr auto TestDeviceType = DeviceType::CUDA;
using TestGuardImpl = FakeGuardImpl<TestDeviceType>;

Device cpu_dev() {
  return Device(DeviceType::CPU);
}

Device dev(DeviceIndex index) {
  return Device(TestDeviceType, index);
}

bool is_inert(const DeviceGuard& g) {
  return g.original_device() == cpu_dev() && g.current_device() == cpu_dev();
}

TEST(DeviceGuard, Constructor) {
  for (DeviceIndex i : {-1, 0, 1}) {
    TestGuardImpl impl;
    DeviceIndex init_i = 0;
    TestGuardImpl::setDeviceIndex(init_i);
    {
      DeviceGuard g(dev(i), &impl);
      ASSERT_EQ(g.original_device(), dev(init_i));
      ASSERT_EQ(g.current_device(), dev(i == -1 ? init_i : i));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i == -1 ? init_i : i);
      // Test un-bracketed write to device index
      TestGuardImpl::setDeviceIndex(4);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  }
}

TEST(DeviceGuard, ConstructorCPUDevice) {
  DeviceGuard g(cpu_dev());
}

TEST(DeviceGuard, MoveConstructor) {
  TestGuardImpl impl;
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = init_i + 1;
  DeviceGuard g(dev(i), &impl);
  DeviceGuard g2(std::move(g));
  ASSERT_TRUE(is_inert(g)); // use-after-move for testing
  ASSERT_EQ(g2.original_device(), dev(init_i));
  ASSERT_EQ(g2.current_device(), dev(i));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}

TEST(DeviceGuard, MoveConstructorFromTemporary) {
  TestGuardImpl impl;
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = init_i + 1;
  DeviceGuard g2(DeviceGuard(dev(i), &impl));
  ASSERT_EQ(g2.original_device(), dev(init_i));
  ASSERT_EQ(g2.current_device(), dev(i));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}

TEST(DeviceGuard, MoveAssignmentSameDeviceType) {
  TestGuardImpl impl;
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = init_i + 1;
  DeviceGuard g(dev(i), &impl);
  DeviceIndex i2 = init_i + 2;
  DeviceGuard g2(dev(i2), &impl);
  g = std::move(g2);
  ASSERT_TRUE(is_inert(g2)); // use-after-move for testing
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

TEST(DeviceGuard, MoveAssignmentSameDeviceTypeFromTemporary) {
  TestGuardImpl impl;
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = init_i + 1;
  DeviceGuard g(dev(i), &impl);
  DeviceIndex i2 = init_i + 2;
  g = DeviceGuard(dev(i2), &impl);
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

TEST(DeviceGuard, MoveAssignmentDifferentDeviceType) {
  FakeGuardImpl<DeviceType::CUDA> cuda_impl;
  FakeGuardImpl<DeviceType::HIP> hip_impl;
  DeviceIndex cuda_init_i = 0;
  FakeGuardImpl<DeviceType::CUDA>::setDeviceIndex(cuda_init_i);
  DeviceIndex hip_init_i = 0;
  FakeGuardImpl<DeviceType::HIP>::setDeviceIndex(hip_init_i);
  DeviceIndex cuda_i = cuda_init_i + 1;
  DeviceGuard g(Device(DeviceType::CUDA, cuda_i), &cuda_impl);
  DeviceIndex hip_i = hip_init_i + 2;
  DeviceGuard g2(Device(DeviceType::HIP, hip_i), &hip_impl);
  g = std::move(g2);
  ASSERT_TRUE(is_inert(g2)); // use-after-move for testing
  ASSERT_EQ(g.original_device(), Device(DeviceType::HIP, hip_init_i));
  ASSERT_EQ(g.current_device(), Device(DeviceType::HIP, hip_i));
  ASSERT_EQ(FakeGuardImpl<DeviceType::CUDA>::getDeviceIndex(), cuda_init_i);
  ASSERT_EQ(FakeGuardImpl<DeviceType::HIP>::getDeviceIndex(), hip_i);
}

TEST(DeviceGuard, MoveAssignmentDifferentDeviceTypeFromTemporary) {
  FakeGuardImpl<DeviceType::CUDA> cuda_impl;
  FakeGuardImpl<DeviceType::HIP> hip_impl;
  DeviceIndex cuda_init_i = 0;
  FakeGuardImpl<DeviceType::CUDA>::setDeviceIndex(cuda_init_i);
  DeviceIndex hip_init_i = 0;
  FakeGuardImpl<DeviceType::HIP>::setDeviceIndex(hip_init_i);
  DeviceIndex cuda_i = cuda_init_i + 1;
  DeviceGuard g(Device(DeviceType::CUDA, cuda_i), &cuda_impl);
  DeviceIndex hip_i = hip_init_i + 2;
  g = DeviceGuard(Device(DeviceType::HIP, hip_i), &hip_impl);
  ASSERT_EQ(g.original_device(), Device(DeviceType::HIP, hip_init_i));
  ASSERT_EQ(g.current_device(), Device(DeviceType::HIP, hip_i));
  ASSERT_EQ(FakeGuardImpl<DeviceType::CUDA>::getDeviceIndex(), cuda_init_i);
  ASSERT_EQ(FakeGuardImpl<DeviceType::HIP>::getDeviceIndex(), hip_i);
}

// Suppress -Wself-move
void move_it(DeviceGuard& g, DeviceGuard&& g2) {
  g = std::move(g2);
}

TEST(DeviceGuard, MoveAssignmentSelf) {
  TestGuardImpl impl;
  DeviceIndex init_i = TestGuardImpl::getDeviceIndex();
  DeviceIndex i = init_i + 1;
  DeviceGuard g(dev(i), &impl);
  move_it(g, std::move(g));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}
