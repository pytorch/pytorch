#pragma once

#include <ATen/record_function.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch {
namespace jit {

inline int64_t timeSinceEpoch() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

typedef struct TORCH_API MemoryEvent {
  enum class EventType { ALLOCATE = 0, FREE };
  friend std::ostream& operator<<(std::ostream& out, const EventType& rhs) {
    out << (rhs == EventType::ALLOCATE ? "ALLOCATE" : "FREE");
    return out;
  }

  int64_t ts{};
  c10::optional<std::string> stack_trace{};
  intptr_t addr{};
  int64_t size{};
  EventType type{EventType::ALLOCATE};
  c10::optional<FrameNodeId> frame_node_id{};

  MemoryEvent() = default;
  MemoryEvent(
      int64_t ts,
      c10::optional<std::string> stack_trace,
      intptr_t addr,
      int64_t s,
      EventType e,
      c10::optional<FrameNodeId> frame_nodeid = c10::nullopt)
      : ts(ts),
        stack_trace(std::move(stack_trace)),
        addr(addr),
        size(s),
        type(e),
        frame_node_id(frame_nodeid) {}

  friend std::ostream& operator<<(std::ostream& out, const MemoryEvent& rhs) {
    out << "MEMORY_EVENT: type: " << rhs.type << ", ts: " << rhs.ts
        << ", size: " << rhs.size << ", addr: " << rhs.addr << "\n";
    return out;
  }

  std::ostream& dump(std::ostream& out, bool include_st = false) {
    out << *this;
    if (this->frame_node_id) {
      out << ", pc: " << this->frame_node_id.value().pc << "\n"
          << ", node_schema: "
          << (this->frame_node_id->node->maybeSchema()
                  ? canonicalSchemaString(this->frame_node_id->node->schema())
                  : "no schema")
          << "\n"
          << ", node_header: " << getHeader(this->frame_node_id->node) << "\n";
    }
    if (include_st && this->stack_trace) {
      out << ", stack trace: " << this->stack_trace.value() << "\n";
    }
    return out;
  }

} MemoryEvent;

typedef struct FunctionFrameEvent {
  std::vector<intptr_t> input_ival_addrs;
  std::vector<intptr_t> output_ival_addrs;
  std::vector<std::string> input_val_names;
  std::vector<std::string> output_val_names;
  std::string fn_name;
  int64_t start_time{};
  int64_t end_time{};

  friend std::ostream& operator<<(
      std::ostream& out,
      const FunctionFrameEvent& rhs) {
    out << "FUNCTION_EVENT: fn_name: " << rhs.fn_name
        << ", start_time: " << rhs.start_time << ", end_time: " << rhs.end_time
        << "\n"
        << "input_val_names: "
        << (rhs.input_val_names.size() ? c10::Join(", ", rhs.input_val_names)
                                       : "no val names")
        << "\n"
        << "output_val_names: "
        << (rhs.output_val_names.size() ? c10::Join(", ", rhs.output_val_names)
                                        : "no val names")
        << "\n"
        << "input_ival_addrs: "
        << (rhs.input_ival_addrs.size() ? c10::Join(", ", rhs.input_ival_addrs)
                                        : "no ival addrs")
        << "\n"
        << "output_ival_addrs: "
        << (rhs.output_ival_addrs.size()
                ? c10::Join(", ", rhs.output_ival_addrs)
                : "no ival addrs")
        << "\n";
    return out;
  }

} FunctionFrameEvent;

typedef struct TORCH_API MemoryObserverEvent {
  MemoryObserverEvent() = default;
  enum { MEMORY_EVENT, FUNCTION_EVENT } type{MEMORY_EVENT};

  MemoryEvent mem_event{};
  FunctionFrameEvent function_event{};

  friend std::ostream& operator<<(
      std::ostream& out,
      const MemoryObserverEvent& evt) {
    if (evt.type == MEMORY_EVENT) {
      out << evt.mem_event << "\n";
    } else {
      out << evt.function_event << "\n";
    }

    return out;
  }
  ~MemoryObserverEvent() = default;
} MemoryObserverEvent;

struct TORCH_API MemoryObserverThreadLocalState
    : public c10::MemoryReportingInfoBase {
  explicit MemoryObserverThreadLocalState();
  ~MemoryObserverThreadLocalState() override = default;

  void callbackHandle(at::CallbackHandle handle) {
    this->handle = handle;
  }

  at::CallbackHandle callbackHandle() const {
    return handle;
  }

  bool hasCallbackHandle() {
    return handle > 0;
  }

  void reportMemoryUsage(
      void* ptr,
      int64_t alloc_size,
      int64_t total_ALLOCATEd,
      int64_t total_reserved,
      c10::Device device) override;

  bool memoryProfilingEnabled() const override {
    return true;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::mutex state_mutex;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  at::CallbackHandle handle = 0;

  std::vector<c10::optional<std::vector<std::string>>> stack{};
  std::vector<MemoryObserverEvent> events;
};

TORCH_API void enableMemoryObserver();

TORCH_API std::vector<MemoryObserverEvent> disableMemoryObserver();

} // namespace jit
} // namespace torch