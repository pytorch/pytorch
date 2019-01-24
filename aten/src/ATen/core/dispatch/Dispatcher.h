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
  explicit constexpr OpKernel(KernelFunction* kernel): kernel_(kernel) {}

  OpKernel(OpKernel&&) = default;
  OpKernel& operator=(OpKernel&&) = default;
  OpKernel(const OpKernel&) = delete;
  OpKernel& operator=(const OpKernel&) = delete;

  IValue call(ArrayRef<IValue> args) const {
    return (*kernel_)(args);
  }

private:
  // TODO Store kernel state
  // The kernel function is a global C function, not a std::function.
  // That is, ownership is not an issue.
  KernelFunction* kernel_;
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
  static void registerKernel(KernelFunction kernel_func, typename Schema::dispatch::dispatch_key_type dispatch_key) {
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    return dispatch_table_for_this_op.registerKernel(std::move(kernel_func), std::move(dispatch_key));
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
    return OpKernel(dispatch_table_for_this_op.lookup(args));
  }

};

} // namespace c10
