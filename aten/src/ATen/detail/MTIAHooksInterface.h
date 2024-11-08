#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <c10/core/Stream.h>
#include <c10/util/Registry.h>

#include <c10/core/Allocator.h>

#include <c10/util/python_stub.h>
#include <ATen/detail/AcceleratorHooksInterface.h>

#include <string>
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")
namespace at {
class Context;
}

namespace at {
constexpr const char* MTIA_HELP =
    "The MTIA backend requires MTIA extension for PyTorch;"
    "this error has occurred because you are trying "
    "to use some MTIA's functionality without MTIA extension included.";

struct TORCH_API MTIAHooksInterface : AcceleratorHooksInterface {
// this fails the implementation if MTIAHooks functions are called, but
// MTIA backend is not present.
#define FAIL_MTIAHOOKS_FUNC(func) \
  TORCH_CHECK(false, "Cannot execute ", func, "() without MTIA backend.");

  ~MTIAHooksInterface() override = default;

  void init() const override {
    // Avoid logging here, since MTIA needs init devices first then it will know
    // how many devices are available. Make it as no-op if mtia extension is not
    // dynamically loaded.
    return;
  }

  virtual bool hasMTIA() const {
    return false;
  }

  DeviceIndex deviceCount() const override {
    return 0;
  }

  virtual void deviceSynchronize(c10::DeviceIndex device_index) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual std::string showConfig() const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    return false;
  }

  void setCurrentDevice(DeviceIndex device) const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  DeviceIndex getCurrentDevice() const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  DeviceIndex exchangeDevice(DeviceIndex device) const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  DeviceIndex maybeExchangeDevice(DeviceIndex device) const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  virtual c10::Stream getCurrentStream(DeviceIndex device) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return c10::Stream::unpack3(-1, 0, c10::DeviceType::MTIA);
  }

  virtual c10::Stream getDefaultStream(DeviceIndex device) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return c10::Stream::unpack3(-1, 0, c10::DeviceType::MTIA);
  }

  virtual void setCurrentStream(const c10::Stream& stream) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  bool isPinnedPtr(const void* data) const override {
    return false;
  }

  Allocator* getPinnedMemoryAllocator() const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

  virtual PyObject* memoryStats(DeviceIndex device) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

  virtual PyObject* getDeviceCapability(DeviceIndex device) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }
};

struct TORCH_API MTIAHooksArgs {};

C10_DECLARE_REGISTRY(MTIAHooksRegistry, MTIAHooksInterface, MTIAHooksArgs);
#define REGISTER_MTIA_HOOKS(clsname) \
  C10_REGISTER_CLASS(MTIAHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MTIAHooksInterface& getMTIAHooks();
TORCH_API bool isMTIAHooksBuilt();
} // namespace detail
} // namespace at
C10_DIAGNOSTIC_POP()
