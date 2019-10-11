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
#include <c10/core/impl/LocalTensorTypeSet.h>
#include <c10/core/TensorImpl.h>
#include <ATen/core/ATenDispatch.h>
#include <ATen/core/TensorBody.h>

namespace at {

class CAFFE2_API LegacyTypeDispatch {
 public:
  void initForTensorTypeSet(TensorTypeSet ts) {
    // TODO: Avoid use of legacyExtractTypeId here.  The key
    // problem is that you may get a TensorTypeSet with
    // VariableTensorId set; should you initialize the "underlying"
    // type in that case?  Hard to say.
    auto b = tensorTypeIdToBackend(legacyExtractTypeId(ts));
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

// A RAII, thread local (!) guard that will disable dispatch to variable
// handler.
//
// See NOTE [ Treating Variables as non-Variables in type dispatch ] for details.
struct CAFFE2_API AutoNonVariableTypeMode {
  // NB: The enabled parameter must ALWAYS be black, as Henry Ford used to say.
  // TODO: Eliminate this parameter entirely
  AutoNonVariableTypeMode(bool enabled = true) :
    guard_(TensorTypeId::VariableTensorId) {

    TORCH_INTERNAL_ASSERT(enabled);
  }
  c10::impl::ExcludeTensorTypeIdGuard guard_;
};

} // namespace at
