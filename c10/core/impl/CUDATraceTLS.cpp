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

template<typename... Ts>
static void noop_trace_cuda_fn(const PyInterpreter*, Ts...) {}

void CUDATraceFunctionWrapper::disarm() {
  event_creation_fn_ = &noop_trace_cuda_fn;
  event_deletion_fn_ = &noop_trace_cuda_fn;
  event_record_fn_ = &noop_trace_cuda_fn;
  event_wait_fn_ = &noop_trace_cuda_fn;
  memory_allocation_fn_ = &noop_trace_cuda_fn;
  memory_deallocation_fn_ = &noop_trace_cuda_fn;
  stream_allocation_fn_ = &noop_trace_cuda_fn;
}

} // namespace impl
} // namespace c10
