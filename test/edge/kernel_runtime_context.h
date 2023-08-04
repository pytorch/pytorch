#pragma once

namespace torch {
namespace executor {

/**
 * Bucket type abstraction that contains many elements of runtime state that
 * a kernel author may want available, but would otherwise be unable to access.
 *
 * Forwarded along to all operators when running in lean mode. NOTE: Will not be
 * forwarded to operators if running in ATen mode as those operators do not
 * expect to receive a KernelRuntimeContext and would not use it.
 *
 * This includes things like setting an error state, a scratch allocator for
 * operators that need more then constant space, and a TensorResizer for dynamic
 * shape tensors allowing programs to be more flexible with Tensor shape.
 */
class KernelRuntimeContext {};

} // namespace executor
} // namespace torch
