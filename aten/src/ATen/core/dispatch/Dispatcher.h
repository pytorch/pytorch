#pragma once

#include <ATen/core/dispatch/DispatchTable.h>

namespace c10 {

/**
 * This class represents an operator kernel, i.e. an operator *after* it was
 * dispatched to a certain device. You can use it to call the kernel.
 *
 * You can keep this OpKernel instance around to avoid future dispatch
 * when you know it'd dispatch to the same kernel anyhow.
 *
 * Also, keeping around the OpKernel instance will keep around a local cache
 * that is used by some kernels to get better performance when they're called
 * multiple times (mostly Caffe2 kernels do that).
 */
class OpKernel final {
public:
  explicit OpKernel(KernelFunction* kernel, KernelStateCreatorFunction* state_creator)
  : kernel_(kernel), state_creator_(state_creator) {}

  OpKernel(OpKernel&&) = default;
  OpKernel& operator=(OpKernel&&) = default;
  OpKernel(const OpKernel&) = delete;
  OpKernel& operator=(const OpKernel&) = delete;

  /**
   * Call the operator kernel with the given arguments.
   */
  IValue call(ArrayRef<IValue> args) {
    if (state_.get() == nullptr) {
      AT_ASSERT(state_creator_ != nullptr);
      state_ = (*state_creator_)();
    }
    return (*kernel_)(args, state_.get());
  }

private:
  // The kernel function is a global C function, not a std::function.
  // That is, ownership is not an issue.
  KernelFunction* kernel_;

  KernelStateCreatorFunction* state_creator_;
  std::unique_ptr<c10::KernelState> state_;
};

/**
 * Top-level dispatch interface for dispatching via the dynamic dispatcher.
 */
template<class OpSchemaDef>
class Dispatcher final {
private:
  using Schema = OpSchema<OpSchemaDef>;
public:
  // Implementation note: this class abstracts over the fact that we have per-operator
  // dispatch tables.  This could be easily adjusted to have a single global hash
  // table.

  /**
   * Register an operator to the dispatch table for some operator schema.
   */
  static void registerKernel(typename Schema::dispatch::dispatch_key_type dispatch_key, KernelFunction* kernel_func, KernelStateCreatorFunction* state_creator_func) {
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    return dispatch_table_for_this_op.registerKernel(std::move(dispatch_key), DispatchTableEntry{kernel_func, state_creator_func});
  }

  /**
   * Remove an operator from the dispatch table for some operator schema.
   */
  static void deregisterKernel(const typename Schema::dispatch::dispatch_key_type& dispatch_key) {
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    return dispatch_table_for_this_op.deregisterKernel(dispatch_key);
  }

  /**
   * Perform a dynamic dispatch and get the kernel for an operator
   */
  static OpKernel lookup(ArrayRef<IValue> args) {
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    const DispatchTableEntry& kernel = dispatch_table_for_this_op.lookup(args);
    return OpKernel(kernel.kernel_func, kernel.state_creator_func);
  }

};

} // namespace c10
