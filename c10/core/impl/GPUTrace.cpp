#include <c10/core/impl/GPUTrace.h>

namespace c10::impl {

std::atomic<const PyInterpreter*> GPUTrace::gpuTraceState{nullptr};

std::atomic<bool> GPUTrace::haveState{false};

void GPUTrace::set_trace(const PyInterpreter* trace) {
  static bool once_flag [[maybe_unused]] = [&]() {
    gpuTraceState.store(trace, std::memory_order_release);
    return true;
  }();
  haveState.store(true, std::memory_order_release);
}

void GPUTrace::unset_trace() {
  haveState.store(false, std::memory_order_release);
}

} // namespace c10::impl
