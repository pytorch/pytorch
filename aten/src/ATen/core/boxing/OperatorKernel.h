#pragma once
#include <c10/util/intrusive_ptr.h>

namespace c10 {

/**
 * Inherit from OperatorKernel to implement a c10 kernel.
 *
 * Example:
 * > namespace {
 * >   class my_kernel_cpu final : public c10::OperatorKernel {
 * >   public:
 * >     Tensor operator()(Tensor a, Tensor b) {...}
 * >   };
 * > }
 *
 * The kernel class is allowed to have members but these are equivalent
 * to global variables. The kernel implementation is responsible for
 * preventing race conditions on them.
 *
 * See below for how to register this kernel with PyTorch.
 */
struct TORCH_API OperatorKernel : public c10::intrusive_ptr_target {
  virtual ~OperatorKernel() = default;
};

}  // namespace c10
