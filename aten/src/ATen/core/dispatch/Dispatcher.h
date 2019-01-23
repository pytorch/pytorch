#pragma once

#include <ATen/core/dispatch/DispatchTable.h>

namespace c10 {

class OpKernel final {
public:
  explicit OpKernel(KernelFunction* kernel, KernelStateCreatorFunction* state_creator)
  : kernel_(kernel), state_creator_(state_creator) {}

  OpKernel(OpKernel&&) = default;
  OpKernel& operator=(OpKernel&&) = default;
  OpKernel(const OpKernel&) = delete;
  OpKernel& operator=(const OpKernel&) = delete;

  void call(Stack* stack) {
    if (state_.get() == nullptr) {
      state_ = (*state_creator_)();
    }
    return (*kernel_)(stack, state_.get());
  }

private:
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
  static OpKernel lookup(const Stack* stack) {
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    const DispatchTableEntry& kernel = dispatch_table_for_this_op.lookup(stack);
    return OpKernel(kernel.kernel_func, kernel.state_creator_func);
  }

};

} // namespace c10
