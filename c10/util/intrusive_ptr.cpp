#include <c10/util/intrusive_ptr.h>

// Checked here rather than in the header so device compilers (CUDA, HIP, and
// other CUDA-like backends that may not define a recognizable device-pass
// macro) parsing intrusive_ptr.h aren't forced to satisfy a host-only
// invariant. combined_refcount_ is only ever touched from host code.
// See https://github.com/pytorch/pytorch/pull/163394 and
// https://github.com/pytorch/pytorch/issues/171775.
static_assert(std::atomic<uint64_t>::is_always_lock_free);
