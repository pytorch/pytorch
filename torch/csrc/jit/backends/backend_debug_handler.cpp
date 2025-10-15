#include <torch/csrc/jit/backends/backend_debug_handler.h>

#include <stack>

namespace torch::jit {

std::atomic<DebugHandleType> BackendDebugInfoRecorder::unique_debug_handle_{0};

int64_t BackendDebugInfoRecorder::getNextDebugHandle(const Node* node) {
  InlinedCallStackPtr cs_ptr;
  if (node->callstack().has_value()) {
    cs_ptr = node->callstack().value();
  } else {
    cs_ptr = c10::intrusive_ptr<InlinedCallStack>();
  }
  DebugHandleType debug_handle = unique_debug_handle_;
  const SourceRange& range = node->sourceRange();
  handles_to_inlined_callstack_ptrs_[debug_handle] =
      std::make_tuple(range, node->kind().toQualString(), cs_ptr);
  // This increment is with seq memory order.
  // Not trying to perf optimizing this for now.
  unique_debug_handle_++;
  return debug_handle;
}

BackendDebugInfoMapType BackendDebugInfoRecorder::stopRecording() {
  // Note that this is return by copy and since
  // InlinedCallStackPtrs are intrusive ptr it will result in
  // bump of refcount. Not performant, but this is not intended
  // to be used in perf critical path.
  // Alternate might be do move but that will be destructive
  return handles_to_inlined_callstack_ptrs_;
}

} // namespace torch::jit
