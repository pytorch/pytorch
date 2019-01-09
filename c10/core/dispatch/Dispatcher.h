#pragma once

#include <c10/core/dispatch/DispatchTable.h>

namespace c10 {

/**
 * Top-level dispatch interface for dispatching via the dynamic dispatcher.
 */
template<class OpSchemaDef>
class Dispatcher final {
public:
  // Implementation note: this class abstracts over the fact that we have per-operator
  // dispatch tables.  This could be easily adjusted to have a single global hash
  // table.

  /**
   * Register an operator to the dispatch table for some operator schema.
   *
   * @tparam OpSchemaDef Operator schema to register this operator to (mandatory)
   * @tparam Args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::registerOp (inferred)
   * @param args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::registerOp
   * @return void
   */
  template<class... Args>
  static void registerKernel(Args&&... args) {
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    return dispatch_table_for_this_op.registerKernel(std::forward<Args>(args)...);
  }

  /**
   * Remove an operator from the dispatch table for some operator schema.
   *
   * @tparam OpSchemaDef Operator schema to deregister from (mandatory)
   * @tparam Args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::deregisterOp (inferred)
   * @param args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::deregisterOp
   * @return void
   */
  template<class... Args>
  static void deregisterKernel(Args&&... args) {
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    return dispatch_table_for_this_op.deregisterKernel(std::forward<Args>(args)...);
  }

  /**
   * Perform a dynamic dispatch to some operator
   *
   * @tparam OpSchemaDef Operator schema to dispatch with (mandatory)
   * @tparam Args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::call (inferred)
   * @param args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::call
   * @return Return type of this operator
   */
  template<class... Args>
  static typename OpSchema<OpSchemaDef>::signature::return_type call(Args&&... args) {
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    return dispatch_table_for_this_op.call(std::forward<Args>(args)...);
  }
};

} // namespace c10
