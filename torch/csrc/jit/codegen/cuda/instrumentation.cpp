
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace fuser {
namespace inst {

Trace::Trace() {
  const char* trace_filename = getenv("PYTORCH_CUDA_FUSER_TRACE");
  if (trace_filename != nullptr) {
    log_file_ = fopen(trace_filename, "w");
    TORCH_CHECK(log_file_ != nullptr, "Can't open trace file");
    fprintf(log_file_, "{\n\"traceEvents\": [\n");
    start_timestamp_ = Clock::now();
    logEvent('I', "TRACE_START");
  }
}

Trace::~Trace() {
  if (log_file_ != nullptr) {
    logEvent('I', "TRACE_END", ' ');
    fprintf(log_file_, "],\n\"displayTimeUnit\": \"ms\"\n}\n");
    fclose(log_file_);
  }
}

void Trace::logEvent(char ph, const char* name, char sep) {
  const std::chrono::duration<double> d = Clock::now() - start_timestamp_;
  const double elapsed = d.count() * 1e6;

  // TODO: add support for tracing multi-process & multi-threaded execution
  const unsigned int pid = 0;
  const unsigned int tid = 0;

  fprintf(
      log_file_,
      "{ \"name\": \"%s\", \"ph\": \"%c\", \"pid\": %u, \"tid\": %u, \"ts\": %.0f }%c\n",
      name,
      ph,
      pid,
      tid,
      elapsed,
      sep);
}

} // namespace inst
} // namespace fuser
} // namespace jit
} // namespace torch
