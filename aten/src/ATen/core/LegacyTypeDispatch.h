#pragma once

// The legacy mechanism for dispatching operators in ATen is a Type
// object, which is essentially a giant virtual dispatch table
// for every operation we support dynamically dispatching over.
//
// This has been deprecated in favor of ATenDispatch, and in the future,
// c10 dispatcher.
// TODO: Clean up what remains here

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <ATen/core/LegacyDeviceTypeInit.h>
#include <c10/core/TensorImpl.h>

namespace at {

class CAFFE2_API LegacyTypeDispatch {
 public:
  void initForBackend(Backend b) {
    auto p = backendToDeviceType(b);
    static std::once_flag cpu_once;
    static std::once_flag cuda_once;
    if (p == DeviceType::CPU) {
      std::call_once(cpu_once, [] {
        getLegacyDeviceTypeInit().initCPU();
      });
    } else if (p == DeviceType::CUDA) {
      std::call_once(cuda_once, [] {
        getLegacyDeviceTypeInit().initCUDA();
      });
    } else if (p == DeviceType::HIP) {
      std::call_once(cuda_once, [] {
        getLegacyDeviceTypeInit().initHIP();
      });
    }
  }
};

CAFFE2_API LegacyTypeDispatch& globalLegacyTypeDispatch();

// A RAII, thread local (!) guard that has the following effect:
//
// Upon construction: sets NonVariableTypeMode_enabled for the current thread to
// control whether we are in non-Variable-type mode.
//
// Upon destruction: sets NonVariableTypeMode_enabled back to the original value.
//
// See NOTE [ Treating Variables as non-Variables in type dispatch ] for details.
struct CAFFE2_API AutoNonVariableTypeMode {
  AutoNonVariableTypeMode(bool enabled) : prev_mode(NonVariableTypeMode::is_enabled()) {
    NonVariableTypeMode::set_enabled(enabled);
  }
  ~AutoNonVariableTypeMode() {
    NonVariableTypeMode::set_enabled(prev_mode);
  }
  bool prev_mode;
};

} // namespace at
