#include <torch/csrc/profiler/orchestration/python_tracer.h>

namespace torch {
namespace profiler {
namespace impl {
namespace python_tracer {
namespace {
GetFn get_fn;

struct NoOpPythonTracer : public PythonTracerBase {
  static NoOpPythonTracer& singleton() {
    static NoOpPythonTracer singleton_;
    return singleton_;
  }
  void start(RecordQueue*) override {}
  void stop() override {}
  void clear() override {}
  std::vector<std::shared_ptr<Result>> getEvents(
      std::function<time_t(approx_time_t)>,
      std::vector<CompressedEvent>&,
      time_t) override {
    return {};
  }
  ~NoOpPythonTracer() = default;
};
} // namespace

void registerTracer(GetFn get_tracer) {
  get_fn = get_tracer;
}

PythonTracerBase& PythonTracerBase::get() {
  if (get_fn == nullptr) {
    return NoOpPythonTracer::singleton();
  }
  return get_fn();
}
} // namespace python_tracer
} // namespace impl
} // namespace profiler
} // namespace torch
