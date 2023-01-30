#include <instrumentation.h>

#include <c10/macros/Export.h>

#ifdef _WIN32
#include <c10/util/win32-headers.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace inst {

Trace::Trace() {
  const char* trace_filename = getenv("PYTORCH_NVFUSER_TRACE");
  if (trace_filename != nullptr) {
    log_file_ = fopen(trace_filename, "w");
    TORCH_CHECK(log_file_ != nullptr, "Can't open trace file");

    // Disable the file stream buffering, since it may result
    // in torn writes in multi-threaded tracing
    setbuf(log_file_, nullptr);

    // Print the trace prologue
    // (including a dummy TRACE_START event)
    fprintf(log_file_, "{\n\"traceEvents\": [\n");
    start_timestamp_ = Clock::now();
    logEvent('I', "TRACE_START");
  }

  if (isOptionDisabled(DisableOption::Nvtx)) {
    record_nvtx_range_ = false;
  }
}

Trace::~Trace() {
  if (log_file_ != nullptr) {
    // Print trace epilogue
    logEvent('I', "TRACE_END", ' ');
    fprintf(log_file_, "],\n\"displayTimeUnit\": \"ms\"\n}\n");
    fclose(log_file_);
  }
}

void Trace::logEvent(char ph, const char* name, char sep) {
  const std::chrono::duration<double> d = Clock::now() - start_timestamp_;
  const double elapsed = d.count() * 1e6;

#ifdef _WIN32
  const unsigned int pid = GetCurrentProcessId();
  const unsigned int tid = GetCurrentThreadId();
#else
  const unsigned int pid = getpid();
  const unsigned int tid = std::hash<pthread_t>{}(pthread_self());
#endif // _WIN32

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
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
