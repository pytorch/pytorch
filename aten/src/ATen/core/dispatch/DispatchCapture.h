#pragma once

#include <ATen/core/ivalue.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>

#include <atomic>

namespace c10 {

class TORCH_API OperatorHandle;

// A DispatchCaptureSink observes the kernel invocations the dispatcher makes
// on the thread it is installed on. For now that covers Dispatcher::call and
// Dispatcher::redispatch; the boxed entry points (callBoxed, redispatchBoxed)
// do not report yet. Unlike a mode dispatch key, which only
// sees the entry of a call, a sink sees each step of the resolution chain: a
// call reports the DispatchKeySet computed by key extraction, and every
// redispatch made by the kernels underneath (Autograd -> ADInplaceOrView ->
// backend) reports the DispatchKeySet it was handed. The last redispatch of a
// chain is therefore the post-dispatch kernel, which is what a compiler needs
// to record in order to replay the operator without dispatching again (see
// torch.cfg_compile).
//
// Enter/exit calls are properly nested, so a sink can rebuild the call tree.
// Sinks must not throw and must not call back into the dispatcher.
class TORCH_API DispatchCaptureSink {
 public:
  virtual ~DispatchCaptureSink();

  virtual void onKernelEnter(
      const OperatorHandle& op,
      DispatchKeySet dispatchKeySet,
      bool isRedispatch,
      ArrayRef<const IValue> inputs) = 0;

  // outputs is empty when the kernel threw.
  virtual void onKernelExit(
      const OperatorHandle& op,
      ArrayRef<const IValue> outputs,
      bool threw) = 0;
};

// Installs sink on the current thread (nullptr uninstalls) and returns the
// previously installed one.
TORCH_API DispatchCaptureSink* setDispatchCaptureSink(
    DispatchCaptureSink* sink);

namespace impl {

// Number of threads with a sink installed. The dispatcher hot path only pays
// for a load of this counter; the thread-local lookup happens behind it.
TORCH_API extern std::atomic<int> dispatch_capture_sink_count;

inline bool dispatch_capture_enabled() {
  return dispatch_capture_sink_count.load(std::memory_order_relaxed) != 0;
}

TORCH_API DispatchCaptureSink* tls_dispatch_capture_sink();

// Reports the exit of a kernel to the sink, also when the kernel throws.
class DispatchCaptureExitGuard {
 public:
  DispatchCaptureExitGuard(DispatchCaptureSink& sink, const OperatorHandle& op)
      : sink_(sink), op_(op) {}
  DispatchCaptureExitGuard(const DispatchCaptureExitGuard&) = delete;
  DispatchCaptureExitGuard& operator=(const DispatchCaptureExitGuard&) = delete;
  ~DispatchCaptureExitGuard() {
    if (!finished_) {
      sink_.onKernelExit(op_, {}, /*threw=*/true);
    }
  }
  void finish(ArrayRef<const IValue> outputs) {
    finished_ = true;
    sink_.onKernelExit(op_, outputs, /*threw=*/false);
  }

 private:
  DispatchCaptureSink& sink_;
  const OperatorHandle& op_;
  bool finished_ = false;
};

} // namespace impl
} // namespace c10
