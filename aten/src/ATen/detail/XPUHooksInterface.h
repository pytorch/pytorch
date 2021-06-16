#pragma once

#include <ATen/core/Generator.h>
#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>

#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace at {
class Context;
}

namespace at {

constexpr const char* XPU_HELP =
  "PyTorch splits its backend into two shared libraries: a CPU library "
  "and a XPU library; this error has occurred because you are trying "
  "to use some XPU functionality, but the XPU library has not been "
  "loaded by the dynamic linker for some reason.  The XPU library MUST "
  "be loaded, EVEN IF you don't directly use any symbols from the XPU library!";

struct TORCH_API XPUHooksInterface {
  virtual ~XPUHooksInterface() {}

  virtual void initXPU() const {
    TORCH_CHECK(false, "Cannot initialize XPU without XPU library.", XPU_HELP);
  }

  virtual bool hasXPU() const {
    return false;
  }

  virtual bool hasOneMKL() const {
    return false;
  }

  virtual bool hasOneDNN() const {
    return false;
  }

  virtual std::string showConfig() const {
    TORCH_CHECK(false, "Cannot query detailed XPU version without XPU library. ", XPU_HELP);
  }

  virtual int64_t getCurrentDevice() const {
    return -1;
  }

  virtual int getDeviceCount() const {
    return 0;
  }

  virtual Device getDeviceFromPtr(void* data) const {
    TORCH_CHECK(false, "Cannot get device of pointer on XPU without XPU library.", XPU_HELP);
  }

  virtual bool isPinnedPtr(void* data) const {
    return false;
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    TORCH_CHECK(false, "Pinned Memory requires XPU library support", XPU_HELP);
  }

  virtual const Generator& getDefaultXPUGenerator(DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Cannot get default XPU generator without XPU library.", XPU_HELP);
  }
};

struct TORCH_API XPUHooksArgs {};

C10_DECLARE_REGISTRY(XPUHooksRegistry, XPUHooksInterface, XPUHooksArgs);
#define REGISTER_XPU_HOOKS(clsname) \
  C10_REGISTER_CLASS(XPUHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const XPUHooksInterface& getXPUHooks();
} // namespace detail
} // namespace at
