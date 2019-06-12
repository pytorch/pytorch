#pragma once

// LegacyTHDispatcher is the legacy mechanism for dispatching directly
// to TH/THNN/THC/THCUNN functions in ATen, which is essentially a giant virtual
// dispatch table for every TH function we support dynamically dispatching over.
//
// NB: We do not actually dispatch to *operators* here, the usual pattern is for
// ATen operators to call this mechanism for their implementation, but the
// operator itself is declared separately (e.g. as a native function "wrapper").
//
// Q: Why don't we just use LegacyTypeDispatch here?
// A: Mainly separation of concerns:
//   1) Type is for implementation of operators, which requires codegen of
//      Variables, JIT, etc.  That is handled by the native function "wrappers";
//      just calling into TH does not require that.
//   2) Type does not require scalar-specific dispatch, whereas calling into TH
//      does.  Thus, this separation allows us to evolve operator dispatch
//      separately (i.e. to use the C10 dispatcher) from details of how to
//      call TH functionality.
//
// The implmentation here is very similar to the LegacyTypeDispatch design, with
// the following simplications:
// 1) This is not required for a mobile build, so does not have to live in /core.
// 2) Because these only contain function implementations, we do not have to
//    handle the Variable/Tensor split; that is handled at the native function
//    "wrapper" level.
// 3) Because an operator must have been previously dispatched via the Type
//    mechanism, we do need to handle device initialization.  This means it is
//    WRONG to call directly into these functions without first going through
//    Type dispatch (i.e. the usual operator -> Type -> LegacyTHDispatch pattern).
// 4) Because an operator must have been previously dispatched via the Type
//    mechanism, we do not need to handle undefined Tensors.
//
// NB: We don't use Registry for this, because we don't want to
// pay for a hash table lookup every time we do an operation.
//
// NB: we can delete this when we don't call into any TH implementations.

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <ATen/core/LegacyDeviceTypeInit.h>
#include <ATen/LegacyTHDispatcher.h>

namespace at {

struct Type;

struct CAFFE2_API LegacyTHDispatcherDeleter {
  using LegacyTHDispatcherDeleterFun = void(LegacyTHDispatcher*);
  LegacyTHDispatcherDeleterFun *fn_ = nullptr;
  LegacyTHDispatcherDeleter() {}
  /* implicit */ LegacyTHDispatcherDeleter(LegacyTHDispatcherDeleterFun *fn) : fn_(fn) {}
  void operator()(LegacyTHDispatcher * ptr) {
    if (fn_) {
      (*fn_)(ptr);
    }
  }
};

class CAFFE2_API LegacyTHDispatch {
 public:
  using LegacyTHDispatcherUniquePtr = std::unique_ptr<LegacyTHDispatcher, LegacyTHDispatcherDeleter>;
  // WARNING: This function has the precondition that you have
  // initialized the type you want to call.  This initialization
  // step is generally done by Context, or assumed because you
  // have a Tensor and thus the Type of that Tensor must already
  // be initialized.

  void registerDispatcher(Backend b, ScalarType s, LegacyTHDispatcherUniquePtr&& t) {
    dispatcher_registry[static_cast<int>(b)][static_cast<int>(s)] = std::move(t);
  }

  LegacyTHDispatcher & getLegacyTHDispatcher(Backend p, ScalarType s) {
    auto* dispatcher = getLegacyTHDispatcherOpt(p, s);
    if (!dispatcher) AT_ERROR(toString(p), toString(s), "THDispatcher is not enabled.");
    return *dispatcher;
  }
private:
  LegacyTHDispatcher* getLegacyTHDispatcherRaw(Backend p, ScalarType s) {
    return dispatcher_registry[static_cast<int>(p)][static_cast<int>(s)].get();
  }

  LegacyTHDispatcher* getLegacyTHDispatcherOpt(Backend p, ScalarType s) {
    if (p != Backend::Undefined) {
      initForDeviceType(backendToDeviceType(p));
      // NB: there is no Complex for TH, so no initialization to be done.
    }
    auto dispatcher = getLegacyTHDispatcherRaw(p, s);

    if(!dispatcher) {
      if (p == Backend::Undefined || s == ScalarType::Undefined) {
        AT_ERROR("Requested Undefined THDispatcher which is invalid.  Backend:",
                 toString(p), "ScalarType: ", toString(s));
      }
    }

    return dispatcher;
  }

  void initForDeviceType(DeviceType p) {
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

  // NB: dispatcher_registry has nullptr for all CUDA backends until
  // CUDA initialization has occurred
  LegacyTHDispatcherUniquePtr dispatcher_registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
};

CAFFE2_API LegacyTHDispatch& globalLegacyTHDispatch();

} // namespace at
