#include <mutex>

#include <c10/core/impl/GPUTrace.h>
#include <c10/util/CallOnce.h>

namespace c10 {
namespace impl {

std::atomic<const PyInterpreter*> GPUTrace::gpuTraceState{nullptr};

bool GPUTrace::haveState{false};

void GPUTrace::set_trace(const PyInterpreter* trace) {
  static c10::once_flag flag;
  c10::call_once(flag, [&]() {
    gpuTraceState.store(trace, std::memory_order_release);
    haveState = true;
  });
}

} // namespace impl
} // namespace c10
