#include <torch/csrc/profiler/orchestration/python_tracer.h>

namespace torch::profiler::impl::python_tracer {
namespace {
MakeFn make_fn;

struct NoOpPythonTracer : public PythonTracerBase {
  NoOpPythonTracer() = default;
  ~NoOpPythonTracer() override = default;

  void stop() override {}
  void restart() override {}
  std::vector<std::shared_ptr<Result>> getEvents(
      std::function<c10::time_t(c10::approx_time_t)>,
      std::vector<CompressedEvent>&,
      c10::time_t) override {
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
} // namespace torch::profiler::impl::python_tracer
