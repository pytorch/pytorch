#include <gtest/gtest.h>

#include <sstream>

#include <c10/core/Device.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/torch.h>

namespace torch {
namespace lazy {

TEST(BackendDeviceTest, BackendDeviceType) {
  auto type = BackendDeviceType();

  EXPECT_EQ(type.type, 0);
  EXPECT_STREQ(type.toString().c_str(), "Unknown");
}

TEST(BackendDeviceTest, Basic1) {
  auto device = BackendDevice();

  EXPECT_EQ(device.ordinal(), 0);
  if (std::getenv("LTC_TS_CUDA") != nullptr) {
    EXPECT_EQ(device.type(), 1);
    EXPECT_STREQ(device.toString().c_str(), "CUDA0");
  } else {
    EXPECT_EQ(device.type(), 0);
    EXPECT_STREQ(device.toString().c_str(), "CPU0");
  }
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
    std::string toString() const override {
      return "Test";
    }
  };

  auto device = BackendDevice(std::make_shared<TestType>(), 1);

  EXPECT_EQ(device.type(), 0);
  EXPECT_EQ(device.ordinal(), 1);
  EXPECT_STREQ(device.toString().c_str(), "Test1");
}

TEST(BackendDeviceTest, Basic4) {
  // Seems weird to have setters in BackendImplInterface given getBackend()
  // returns a const pointer.
  auto default_type = getBackend()->GetDefaultDeviceType();
  auto default_ordinal = getBackend()->GetDefaultDeviceOrdinal();
  const_cast<BackendImplInterface*>(getBackend())
      ->SetDefaultDeviceType(static_cast<int8_t>(c10::kCUDA));
  const_cast<BackendImplInterface*>(getBackend())->SetDefaultDeviceOrdinal(1);

  auto device = BackendDevice();

  EXPECT_EQ(device.type(), 1);
  EXPECT_EQ(device.ordinal(), 1);
  EXPECT_STREQ(device.toString().c_str(), "CUDA1");

  const_cast<BackendImplInterface*>(getBackend())
      ->SetDefaultDeviceType(default_type->type);
  const_cast<BackendImplInterface*>(getBackend())
      ->SetDefaultDeviceOrdinal(default_ordinal);
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

TEST(BackendDeviceTest, FromAten) {
  auto device = c10::Device(c10::kCPU);
  EXPECT_THROW(atenDeviceToBackendDevice(device), c10::Error);

  device = c10::Device(c10::kLazy);
#ifndef FBCODE_CAFFE2
  auto backend_device = atenDeviceToBackendDevice(device);
#else
  // Lazy Tensor is disabled in FBCODE until addressing non-virtual methods
  // (e.g. sizes) in TensorImpl
  EXPECT_THROW(atenDeviceToBackendDevice(device), c10::Error);
#endif // FBCODE_CAFFE2
}

TEST(BackendDeviceTest, ToAten) {
  auto device = backendDeviceToAtenDevice(BackendDevice());
  EXPECT_EQ(device.type(), c10::kLazy);
  EXPECT_TRUE(device.has_index());
  EXPECT_EQ(device.index(), 0);
}

// TODO(alanwaketan): Update the following test once we have TorchScript backend
// upstreamed.
TEST(BackendDeviceTest, GetBackendDevice1) {
  auto tensor = torch::rand({0, 1, 3, 0});
  EXPECT_FALSE(GetBackendDevice(tensor));
}

TEST(BackendDeviceTest, GetBackendDevice2) {
  auto tensor1 = torch::rand({0, 1, 3, 0});
  auto tensor2 = torch::rand({0, 1, 3, 0});
  // TODO(alanwaketan): Cover the test case for GetBackendDevice().
  EXPECT_FALSE(GetBackendDevice(tensor1, tensor2));
}

} // namespace lazy
} // namespace torch
