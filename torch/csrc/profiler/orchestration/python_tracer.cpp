#include <torch/csrc/profiler/orchestration/python_tracer.h>

#include <c10/util/overloaded.h>
#include <torch/csrc/profiler/collection.h>

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
      std::function<c10::time_t(c10::approx_time_t)> /*time_converter*/,
      std::vector<CompressedEvent>& /*enters*/,
      c10::time_t /*end_time_ns*/) override {
    return {};
  }
};

struct NoOpMemoryPythonTracer : public PythonMemoryTracerBase {
  NoOpMemoryPythonTracer() = default;
  ~NoOpMemoryPythonTracer() override = default;
  void start() override {}
  void stop() override {}
  void export_memory_history(const std::string& /*path*/) override {}
};

} // namespace

void clampOverrunningPythonEvents(
    const std::vector<std::shared_ptr<Result>>& sorted_events) {
  std::vector<std::vector<Result*>> stacks;
  for (const auto& event : sorted_events) {
    const auto tag = event->tag();
    TORCH_INTERNAL_ASSERT(
        tag == EventType::PyCall || tag == EventType::PyCCall,
        "Expected a Python event");

    size_t python_tid = 0;
    event->visit_if_base<PyExtraFieldsBase>(
        [&](const auto& fields) { python_tid = fields.python_tid_; });
    if (python_tid >= stacks.size()) {
      stacks.resize(python_tid + 1);
    }
    auto& stack = stacks[python_tid];

    while (!stack.empty() &&
           stack.back()->endTimeNS() <= event->start_time_ns_) {
      stack.pop_back();
    }

    if (!stack.empty() &&
        event->endTimeNS() > stack.back()->endTimeNS()) {
      const auto parent_end_ns = stack.back()->endTimeNS();
      event->visit(c10::overloaded(
          [parent_end_ns](ExtraFields<EventType::PyCall>& fields) {
            fields.end_time_ns_ = parent_end_ns;
          },
          [parent_end_ns](ExtraFields<EventType::PyCCall>& fields) {
            fields.end_time_ns_ = parent_end_ns;
          },
          [](auto&) {}));
    }

    if (event->endTimeNS() > event->start_time_ns_) {
      stack.push_back(event.get());
    }
  }
}

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
