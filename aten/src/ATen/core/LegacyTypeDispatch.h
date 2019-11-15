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
// NOTE [ Treating Variables as non-Variables in type dispatch ]
//
// What exactly does AutoNonVariableType do?  The short answer is, it causes
// dispatches on ATen functions to go to the non-variable implementation,
// bypassing autograd handling (and also profiling and tracing).
//
// To understand why this guard exists, it's helpful to understand the history
// behind how Variable was implemented.  Previously, Variables were implemented
// as a wrapper on Tensors; so the act of processing a Variable involved
// unwrapping the underlying Tensor, and then calling the underlying base
// operation on /that/ operation
//
// However, after the Variable/Tensor merge, there is no concept of unwrapping
// a tensor anymore.  If you just call the operation on the same variable
// again inside your VariableType handler, you'll dispatch back to
// VariableType, which is not what we want.
//
// The solution to the above problem is to add `at::NonVariableTypeMode`, which
// when enabled will cause `legacyTensorType()` and `getType()` to always return
// non-Variable type, even if the tensor being called on is a variable.
//
// TODO: Since `torch::NoGradGuard` serves almost the same purpose in libtorch,
// we should merge these two thread-local guards.  However, NoGradGuard does
// something subtly different: it turns off gradient recording, but DOES NOT
// skip VariableType implementation (as we still might need to profile or
// trace).  To unify the two, we would first have to move profiling and tracing
// out of VariableType.

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
