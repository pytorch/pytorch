#pragma once

#include <c10/core/impl/PyInterpreter.h>

namespace c10::impl {

struct C10_API GPUTrace {
  // On the x86 architecture the atomic operations are lock-less.
  static std::atomic<const PyInterpreter*> gpuTraceState;

  static std::atomic<bool> haveState;

  // This function will only register the first interpreter that tries to invoke
  // it. For all of the next ones it will be a no-op.
  static void set_trace(const PyInterpreter* /*trace*/);
  static void unset_trace();

  static const PyInterpreter* get_trace() {
    if (!haveState.load(std::memory_order_acquire))
      return nullptr;
    return gpuTraceState.load(std::memory_order_acquire);
  }
};

} // namespace c10::impl
