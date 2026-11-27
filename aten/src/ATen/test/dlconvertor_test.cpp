#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/test/test_assert.h>

#include <cstdlib>

using namespace at;

TEST(TestDlconvertor, TestDlconvertor) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensor* dlMTensor = toDLPack(a);

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertor, TestDlconvertorNoStrides) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensor* dlMTensor = toDLPack(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorUnversioned, TestDlconvertor) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorUnversioned, TestDlconvertorNoStrides) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

#if GTEST_HAS_DEATH_TEST
namespace {
struct TestPrivateUse1Hooks : PrivateUse1HooksInterface {
  int32_t type = PrivateUse1HooksInterface::getDLDeviceType();

  int32_t getDLDeviceType() const override {
    return type;
  }
};
} // namespace

TEST(TestDlconvertorDeathTest, TestPrivateUse1DeviceHook) {
  // Hook registration is permanent, so isolate it from other tests.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_EXIT(
      {
        const Device device(kPrivateUse1, 1);
        auto dl_device = torchDeviceToDLDevice(device);
        ASSERT(dl_device.device_type == kDLExtDev);
        ASSERT(dl_device.device_id == 1);
        ASSERT_THROWSM(
            dlDeviceToTorchDevice(kDLCUDAManaged, 1),
            "Unsupported device_type: 13");

        TestPrivateUse1Hooks hooks;
        RegisterPrivateUse1HooksInterface(&hooks);
        // Only metadata is inspected; no kernels access this host data.
        float value = 0;
        const auto options = TensorOptions().device(device).dtype(kFloat);
        const auto tensor = for_blob(&value, {1})
                                .options(options)
                                .target_device(device)
                                .make_tensor();
        const auto ext_device = torchDeviceToDLDevice(device);
        for (auto type : {kDLExtDev, kDLCUDAManaged, kDLTrn}) {
          hooks.type = type;
          dl_device = torchDeviceToDLDevice(Device(kPrivateUse1));
          ASSERT(dl_device.device_type == type);
          ASSERT(dl_device.device_id == 255);
          dl_device = torchDeviceToDLDevice(device);
          ASSERT(dl_device.device_type == type);
          ASSERT(dl_device.device_id == 1);
          ASSERT(dlDeviceToTorchDevice(type, 1) == device);
          ASSERT(dlDeviceToTorchDevice(kDLExtDev, 1) == device);
          ASSERT(maybeCopyTensor(tensor, dl_device, false).is_same(tensor));
          ASSERT(maybeCopyTensor(tensor, std::nullopt, false).is_same(tensor));
          if (type != kDLExtDev) {
            ASSERT_THROWSM(
                maybeCopyTensor(tensor, ext_device, false),
                "Requested DLPack device type 12 does not match PrivateUse1 export device type");
            ASSERT_THROWSM(
                maybeCopyTensor(tensor, ext_device, true),
                "Requested DLPack device type 12 does not match PrivateUse1 export device type");
          }
        }
        ASSERT_THROWSM(
            dlDeviceToTorchDevice(kDLCUDAManaged, 1),
            "Unsupported device_type: 13");

        hooks.type = kDLCPU;
        ASSERT_THROWSM(
            torchDeviceToDLDevice(device),
            "getDLDeviceType() returned 1; expected kDLExtDev");
        hooks.type = kDLCUDA;
        ASSERT_THROWSM(
            torchDeviceToDLDevice(device),
            "getDLDeviceType() returned 2; expected kDLExtDev");

        std::_Exit(EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS),
      "");
}
#endif // GTEST_HAS_DEATH_TEST
