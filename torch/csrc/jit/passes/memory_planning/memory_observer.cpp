//
// Created by Maksim Levental on 9/11/21.
//

#include <iostream>
#include <utility>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/memory_planning/memory_observer.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch {
namespace jit {

MemoryObserverThreadLocalState* getMemoryObserverTLSState() {
  const auto& state =
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE);
  return static_cast<MemoryObserverThreadLocalState*>(state);
}

struct MemoryObserverContext : public at::ObserverContext {
  MemoryObserverContext(
      std::vector<intptr_t> input_ival_addrs,
      std::vector<std::string> input_val_names,
      std::vector<std::string> output_val_names,
      std::string fn_name,
      int64_t start_time)
      : input_ival_addrs(std::move(input_ival_addrs)),
        input_val_names(std::move(input_val_names)),
        output_val_names(std::move(output_val_names)),
        fn_name(std::move(fn_name)),
        start_time(start_time) {}
  std::vector<intptr_t> input_ival_addrs;
  std::vector<std::string> input_val_names;
  std::vector<std::string> output_val_names;
  std::string fn_name;
  int64_t start_time;
};

void pushObserverCallbacks() {
  auto state_ptr = getMemoryObserverTLSState();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected memory observer state set");
  auto start_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    auto state_ptr = getMemoryObserverTLSState();
    if (!state_ptr) {
      return nullptr;
    }
    std::lock_guard<std::mutex> guard(state_ptr->state_mutex);

    std::vector<intptr_t> input_ival_addrs;
    for (size_t i = fn.inputs().size() - fn.num_inputs();
         i < fn.inputs().size();
         i++) {
      auto ival = fn.inputs()[i];
      input_ival_addrs.emplace_back(
          ival.isTensor()
              ? reinterpret_cast<intptr_t>(ival.toTensor().data_ptr())
              : 0);
    }

    std::vector<std::string> input_val_names;
    std::vector<std::string> output_val_names;
    if (jit::currentFrameId()) {
      auto frame_id = jit::currentFrameId().value();
      std::transform(
          frame_id.node->inputs().begin(),
          frame_id.node->inputs().end(),
          std::back_inserter(input_val_names),
          [](auto v) { return v->debugName(); });
      std::transform(
          frame_id.node->outputs().begin(),
          frame_id.node->outputs().end(),
          std::back_inserter(output_val_names),
          [](auto v) { return v->debugName(); });
    }

    auto ctx_ptr = std::make_unique<MemoryObserverContext>(
        input_ival_addrs,
        input_val_names,
        output_val_names,
        fn.name().str(),
        timeSinceEpoch());

    return ctx_ptr;
  };
  auto end_cb = [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
    auto state_ptr = getMemoryObserverTLSState();
    if (!state_ptr) {
      return;
    }
    std::lock_guard<std::mutex> guard(state_ptr->state_mutex);
    auto mem_ctx_ptr = static_cast<MemoryObserverContext*>(ctx_ptr);
    TORCH_INTERNAL_ASSERT(mem_ctx_ptr != nullptr);

    MemoryObserverEvent evt;
    evt.type = MemoryObserverEvent::FUNCTION_EVENT;
    evt.function_event.input_ival_addrs = mem_ctx_ptr->input_ival_addrs;
    evt.function_event.input_val_names = mem_ctx_ptr->input_val_names;
    evt.function_event.output_val_names = mem_ctx_ptr->output_val_names;
    evt.function_event.fn_name = mem_ctx_ptr->fn_name;
    evt.function_event.start_time = mem_ctx_ptr->start_time;
    evt.function_event.end_time = timeSinceEpoch();

    std::vector<intptr_t> output_ival_addrs;
    for (size_t i = fn.outputs().size() - fn.num_outputs();
         i < fn.outputs().size();
         i++) {
      auto oval = fn.outputs()[i];
      output_ival_addrs.emplace_back(
          oval.isTensor()
              ? reinterpret_cast<intptr_t>(oval.toTensor().data_ptr())
              : 0);
    }
    evt.function_event.output_ival_addrs = output_ival_addrs;

    state_ptr->events.emplace_back(std::move(evt));
  };
  auto record_function_cb = at::RecordFunctionCallback(start_cb, end_cb)
                                .needsInputs(true)
                                .needsOutputs(true)
                                .needsIds(true)
                                .scopes({});
  state_ptr->callbackHandle(at::addThreadLocalCallback(record_function_cb));
}

void enableMemoryObserver() {
  auto state = std::make_shared<MemoryObserverThreadLocalState>();
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);
  pushObserverCallbacks();
}

std::vector<MemoryObserverEvent> disableMemoryObserver() {
  auto state =
      c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);

  auto state_ptr = static_cast<MemoryObserverThreadLocalState*>(state.get());
  if (state_ptr->hasCallbackHandle()) {
    at::removeCallback(state_ptr->callbackHandle());
  }
  return std::move(state_ptr->events);
}

MemoryObserverThreadLocalState::MemoryObserverThreadLocalState() = default;
void MemoryObserverThreadLocalState::reportMemoryUsage(
    void* ptr,
    int64_t alloc_size,
    int64_t total_allocated,
    int64_t total_reserved,
    c10::Device device) {
  //  auto bt = c10::get_backtrace(1, 200, true);
  std::lock_guard<std::mutex> guard(state_mutex);
  MemoryObserverEvent evt;
  evt.type = MemoryObserverEvent::MEMORY_EVENT;
  evt.mem_event.addr = reinterpret_cast<intptr_t>(ptr);
  evt.mem_event.size = alloc_size;
  evt.mem_event.ts = timeSinceEpoch();
  evt.mem_event.type =
      (alloc_size >= 0 ? MemoryEvent::EventType::ALLOCATE
                       : MemoryEvent::EventType::FREE);
  evt.mem_event.frame_node_id = jit::currentFrameId();
  events.emplace_back(std::move(evt));
}
} // namespace jit
} // namespace torch
