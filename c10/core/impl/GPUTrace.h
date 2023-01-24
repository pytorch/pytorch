#pragma once

#include <c10/core/impl/PyInterpreter.h>

namespace c10 {
namespace impl {

struct C10_API GPUTrace {
  // On the x86 architecture the atomic operations are lock-less.
  static std::atomic<const PyInterpreter*> gpuTraceState;

  // When PyTorch migrates to C++20, this should be changed to an atomic flag.
  // Currently, the access to this variable is not synchronized, on the basis
  // that it will only be flipped once and by the first interpreter that
  // accesses it.
  static bool haveState;

  // This function will only register the first interpreter that tries to invoke
  // it. For all of the next ones it will be a no-op.
  static void set_trace(const PyInterpreter*);

  static const PyInterpreter* get_trace() {
    if (!haveState)
      return nullptr;
    return gpuTraceState.load(std::memory_order_acquire);
  }
};

} // namespace impl
} // namespace c10
