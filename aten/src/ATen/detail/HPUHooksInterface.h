#pragma once

#include <ATen/detail/AcceleratorHooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/util/Registry.h>

namespace at {

struct TORCH_API HPUHooksInterface : AcceleratorHooksInterface {
  ~HPUHooksInterface() override = default;

  virtual void initHPU() const {
    TORCH_CHECK(false, "Cannot initialize HPU without HPU backend");
  }

  virtual bool hasHPU() const {
    return false;
  }
  virtual const Generator& getDefaultHPUGenerator(
      C10_UNUSED DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Cannot get default HPU generator without HPU backend");
  }
  virtual Device getDeviceFromPtr(void* /*data*/) const {
    TORCH_CHECK(
        false, "Cannot get device of pointer on HPU without HPU backend");
  }

  virtual bool isPinnedPtr(const void*) const override {
    return false;
  }

  virtual Allocator* getPinnedMemoryAllocator() const override {
    TORCH_CHECK(
        false,
        "You should register `HPUHooksInterface` for HPU before call `getPinnedMemoryAllocator`.");
  }
};

struct TORCH_API HPUHooksArgs {};

TORCH_DECLARE_REGISTRY(HPUHooksRegistry, HPUHooksInterface, HPUHooksArgs);
#define REGISTER_HPU_HOOKS(clsname) \
  C10_REGISTER_CLASS(HPUHooksRegistry, clsname, clsname)

namespace detail {

TORCH_API const at::HPUHooksInterface& getHPUHooks();

} // namespace detail
} // namespace at
