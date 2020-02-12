#pragma once

// The legacy mechanism for initializing device types; this is used by
// LegacyTypeDispatch.

#include <c10/core/DeviceType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Registry.h>
#include <ATen/core/ScalarType.h>

namespace at {

struct CAFFE2_API LegacyDeviceTypeInitInterface {
  virtual ~LegacyDeviceTypeInitInterface() {}
  virtual void initCPU() const {
    AT_ERROR("cannot use CPU without ATen library");
  }
  virtual void initCUDA() const {
    AT_ERROR("cannot use CUDA without ATen CUDA library");
  }
  virtual void initHIP() const {
    AT_ERROR("cannot use HIP without ATen HIP library");
  }
};

struct CAFFE2_API LegacyDeviceTypeInitArgs {};

C10_DECLARE_REGISTRY(
    LegacyDeviceTypeInitRegistry,
    LegacyDeviceTypeInitInterface,
    LegacyDeviceTypeInitArgs);
#define REGISTER_LEGACY_TYPE_INIT(clsname) \
  C10_REGISTER_CLASS(LegacyDeviceTypeInitRegistry, clsname, clsname)

CAFFE2_API const LegacyDeviceTypeInitInterface& getLegacyDeviceTypeInit();

} // namespace at
