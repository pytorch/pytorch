#include <ATen/core/boxing/KernelFunction.h>

namespace c10 {

// This a "fake" kernel which doesn't actually do anything.  Instead, it is a
// distinguished kernel which is special cased by the dispatch table to
// be handled specially.  Its semantics is that it redispatches to the
// *next* dispatch key that would have been processed, skipping the current
// one.
void fallthrough_kernel(OperatorKernel*, const OperatorHandle&, Stack*) {
  TORCH_INTERNAL_ASSERT(0, "fallthrough_kernel was executed but it should have been short-circuited by the dispatcher");
}

} // namespace c10
