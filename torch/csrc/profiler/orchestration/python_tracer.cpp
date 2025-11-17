#include <torch/csrc/profiler/orchestration/python_tracer.h>

namespace torch::profiler::impl::python_tracer {
namespace {
MakeFn make_fn;
MakeMemoryFn memory_make_fn;

struct NoOpPythonTracer : public PythonTracerBase {
  NoOpPythonTracer() = default;
  ~NoOpPythonTracer() override = default;

  void stop() override {}
  void restart() override {}
  void register_gc_callback() override {}
  std::vector<std::shared_ptr<Result>> getEvents(
      std::function<c10::time_t(c10::approx_time_t)>,
      std::vector<CompressedEvent>&,
      c10::time_t) override {
    return {};
  }
};

struct NoOpMemoryPythonTracer : public PythonMemoryTracerBase {
  NoOpMemoryPythonTracer() = default;
  ~NoOpMemoryPythonTracer() override = default;
  void start() override {}
  void stop() override {}
  void export_memory_history(const std::string&) override {}
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

void registerMemoryTracer(MakeMemoryFn make_memory_tracer) {
  memory_make_fn = make_memory_tracer;
}

std::unique_ptr<PythonMemoryTracerBase> PythonMemoryTracerBase::make() {
  if (memory_make_fn == nullptr) {
    return std::make_unique<NoOpMemoryPythonTracer>();
  }
  return memory_make_fn();
}
} // namespace torch::profiler::impl::python_tracer
