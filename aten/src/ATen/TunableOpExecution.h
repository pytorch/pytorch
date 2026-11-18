#pragma once

#include <c10/macros/Export.h>

#include <cstdint>
#include <memory>
#include <utility>

namespace at {

namespace cuda::tunable {
class ServingTunableOpResults;
}

enum class TunableOpPermission { Disabled, Enabled };

// Whether the current call may use TunableOp results, on which device, and
// which read-only results table it uses.
// A serving engine installs its binding for the duration of each call; the
// binding follows the call into PyTorch worker threads through
// ThreadLocalState.
struct TORCH_API TunableOpExecutionBinding {
  TunableOpPermission permission{TunableOpPermission::Disabled};
  int device{-1};
  uint64_t generation{0};
  std::shared_ptr<const cuda::tunable::ServingTunableOpResults> results;
};

using TunableOpBinding = std::shared_ptr<const TunableOpExecutionBinding>;

// Once called, TunableOp reads the per-call binding instead of the
// process-wide enable switch, and never tunes or writes results.
TORCH_API void InitializeManagedTunableOp();
TORCH_API bool IsManagedTunableOp();
TORCH_API const TunableOpBinding& GetTunableOpBinding();
TORCH_API void SetTunableOpBinding(TunableOpBinding binding);
TORCH_API TunableOpBinding DisabledTunableOpBinding();

class TORCH_API TunableOpExecutionGuard {
 public:
  explicit TunableOpExecutionGuard(TunableOpBinding binding)
      : previous_(GetTunableOpBinding()) {
    SetTunableOpBinding(std::move(binding));
  }
  ~TunableOpExecutionGuard() {
    SetTunableOpBinding(std::move(previous_));
  }
  TunableOpExecutionGuard(const TunableOpExecutionGuard&) = delete;
  TunableOpExecutionGuard& operator=(const TunableOpExecutionGuard&) = delete;
  TunableOpExecutionGuard(TunableOpExecutionGuard&&) = delete;
  TunableOpExecutionGuard& operator=(TunableOpExecutionGuard&&) = delete;

 private:
  TunableOpBinding previous_;
};

} // namespace at
