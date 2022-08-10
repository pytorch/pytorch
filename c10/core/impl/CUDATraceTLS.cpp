#include <c10/core/impl/CUDATraceTLS.h>

#include <mutex>

namespace c10 {
namespace impl {

static std::atomic<const PyInterpreter*> cudaTraceState;

void CUDATraceTLS::set_trace(const PyInterpreter* trace) {
  static std::once_flag flag;
  std::call_once(
    flag,
    [&](){ cudaTraceState.store(trace); }
  );
  cudaTraceState.store(trace);
}

const PyInterpreter* CUDATraceTLS::get_trace() {
  return cudaTraceState.load();
}

} // namespace impl
} // namespace c10
