#include <c10/core/impl/CUDATraceTLS.h>

namespace c10 {
namespace impl {

static const PyInterpreter* cudaTraceState;

void CUDATraceTLS::set_trace(const PyInterpreter* trace) {
  cudaTraceState = trace;
}

const PyInterpreter* CUDATraceTLS::get_trace() {
  return cudaTraceState;
}

} // namespace impl
} // namespace c10
