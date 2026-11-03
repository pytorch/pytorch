#pragma once

#include <cstddef>

namespace torch::inductor {

// Lifecycle events an AOTInductor model container reports to an observer.
enum class AOTIContainerEvent {
  kLoadConstants, // constants blob loaded into the container
  kUpdateConstantBuffer, // in-place constants/weights update
  // Same operation as kUpdateConstantBuffer, but the source tensors live on the
  // host so it also pays a host-to-device copy of every constant. Separate
  // because the two have very different cost profiles.
  kUpdateConstantBufferFromCpu,
  kSwapConstantBuffer, // active <-> inactive constants-buffer swap
  kRunConstantFolding, // constant-folding pass
  // run() / boxed_run(); warmup is inference too. Single-threaded mode is a
  // construction-time flag that swaps run_func_ inside these, not a separate
  // entry point.
  kInference,
  kFreeInactiveBuffer, // release of the inactive constants buffer
};

// Context passed with each event. Fields that don't apply to a given event keep
// their defaults. Deliberately trivial so it stays cheap to construct on the
// hot path.
struct AOTIObserverContext {
  size_t num_constants = 0;
  bool use_inactive = false; // update/fold target buffer
};

// Observer for AOTInductor container lifecycle events. Attach a subclass to a
// container runner via set_observer() to receive begin/end callbacks bracketing
// each event; measure durations yourself between on_begin and on_end. Attaching
// is optional -- a null observer is zero overhead.
//
// on_end always runs, including when the bracketed operation threw; `succeeded`
// says which happened, so a failed call is not silently recorded as a
// normal-latency sample. Implementations must be cheap and must not throw:
// callbacks can run on the serving hot path, and on_end runs during stack
// unwinding, where an escaping exception would terminate the process.
class AOTIModelContainerObserver {
 public:
  virtual ~AOTIModelContainerObserver() = default;

  virtual void on_begin(
      AOTIContainerEvent event,
      const AOTIObserverContext& ctx) = 0;
  virtual void on_end(
      AOTIContainerEvent event,
      const AOTIObserverContext& ctx,
      bool succeeded) = 0;
};

} // namespace torch::inductor
