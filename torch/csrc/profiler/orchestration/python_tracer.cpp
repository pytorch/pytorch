#include <torch/csrc/profiler/orchestration/python_tracer.h>

namespace torch {
namespace profiler {
namespace impl {
namespace python_tracer {
namespace {
MakeFn make_fn;

struct NoOpPythonTracer : public PythonTracerBase {
  NoOpPythonTracer() = default;
  ~NoOpPythonTracer() = default;

  void stop() override {}
  std::vector<std::shared_ptr<Result>> getEvents(
      std::function<time_t(approx_time_t)>,
      std::vector<CompressedEvent>&,
      time_t) override {
    return {};
  }
};
} // namespace

void registerTracer(MakeFn make_tracer) {
  make_fn = make_tracer;
}

std::unique_ptr<PythonTracerBase> PythonTracerBase::make(RecordQueue* queue) {
  if (make_fn == nullptr) {
    return std::make_unique<NoOpPythonTracer>();
  }
  return make_fn(queue);
}
} // namespace python_tracer
} // namespace impl
} // namespace profiler
} // namespace torch
