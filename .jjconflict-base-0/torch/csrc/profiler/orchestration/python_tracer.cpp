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
  // If a Python event is pushed to the CALL queue, but a corresponding RETURN
  // event is never pushed, this can cause imbalance between the stacks during
  // replay. When this happens we attempt to remedy the situation by assigning
  // the imbalanced event an end time equal to the end time of the profiling
  // session. This can cause weird visualizations in the profile.
  //
  // To fix this, we perform an initial pass through the events to identify
  // mismatches and clamp the end timestamp to the parent event to avoid this
  // overrun.

  // One stack per thread, dynamically resized below. We only need to track the
  // end timestamp for the events we've seen before.
  std::vector<std::vector<c10::time_t>> stacks;
  for (const auto& event : sorted_events) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        event->tag() == EventType::PyCall || event->tag() == EventType::PyCCall,
        "Expected a Python event");

    const auto start_time_ns = event->start_time_ns_;
    size_t python_tid = 0;
    c10::time_t end_time_ns = 0;
    event->visit_if_base<PyExtraFieldsBase>([&](const auto& fields) {
      python_tid = fields.python_tid_;
      end_time_ns = fields.end_time_ns_;
    });

    if (python_tid >= stacks.size()) {
      stacks.resize(python_tid + 1);
    }
    auto& stack = stacks[python_tid];

    // For the current python_tid's stack, remove events that
    // ended <= the start time of the current event. Afterward,
    // stack.back() is the nearest active candidate parent.
    c10::time_t parent_end_ns = 0;
    while (!stack.empty()) {
      parent_end_ns = stack.back();
      if (parent_end_ns > start_time_ns) {
        break;
      }
      stack.pop_back();
    }

    // If the current event has end time greater than parent end time,
    // perform clamping by assigning it the parent's end time.
    if (!stack.empty() && end_time_ns > parent_end_ns) {
      end_time_ns = parent_end_ns;
      event->visit(c10::overloaded(
          [parent_end_ns](ExtraFields<EventType::PyCall>& fields) {
            fields.end_time_ns_ = parent_end_ns;
          },
          [parent_end_ns](ExtraFields<EventType::PyCCall>& fields) {
            fields.end_time_ns_ = parent_end_ns;
          },
          [](auto&) {}));
    }

    stack.push_back(end_time_ns);
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
