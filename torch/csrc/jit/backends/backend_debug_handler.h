#pragma once
#include <torch/csrc/jit/ir/scope.h>

#include <atomic>

namespace torch {
namespace jit {

/*
 *  There are two parts to BackendDebugHandleManager:
 *  1. static std::atomic debug_handle
 *  2. Map of debug-handle-to-inlined-callstack-ptr
 *
 *  About 1:
 *  Why do they have to be unique. The reason is that by ensuring
 *  uniqueness of debug handles, we remove the burden of another layer of
 * mapping where we need to say this set of debug handles were generated for
 * this lowered module or this bytecode function. This simplifies the API for
 * serialization since we only need to ask backends to returns the
 * debug-handles-to-inlined-callstack-ptrs we provided. Furthermore it also
 * simplifies runtime API when the exception is thrown. The thrown exception
 * only needs to know debug_handle and not which module or method threw it.
 *  There are 2 issues to keep in mind for static std::atomic debug_handle:
 *  1. Performance implications of using atomic variable. However this is only
 *     used for compilation so we assume to absorb some of that penalty.
 *     Plus if there is no contention then we should have less to worry about.
 *  2. If repeated compilation is part of a long running process then we may
 * overflow int64_t. We may detect and fail on this. Not sure if this is good.
 * Will seek opinions about this.
 *
 *  Now about 2:
 *  There are two usecases for this:
 *  1. During bytecode generation the inlined callstack ptrs, corresponding
 *     to the nodes of the inlined graph being serialized, is stored in this
 *     object and a unique debug handle is returned. This unique debug handle
 *     is stored in mobile_debug info. It will be used for raising exceptions
 *     as well as profiling.
 *  2. During backend lowering, each backend's preprocess/compile method can
 *     compile method's graph and serialize those methods. Once the method is
 *     lowered to backend, graph is essentially lost. Without access to graph
 *     it is hard to generate model level debug info via source range.
 *     For this purpose, we will provide utility,
 * generate_debug_handles_and_table that, given an inlined graph of a method,
 * returns map of node*-to-debug-handles and map of
 * debug-handles-to-inlined-callstack-ptrs. generate_debug_handles_and_table
 * will utilize BackendDebugHandleManager, to generate map of
 * debug-handles-to-inlined-callstack-ptrs.
 *
 *  Once all the needed debug handles are generated, one can obtain the map of
 *  debug-handles-to-inlined-callstack-ptrs which can be serialized via
 *  InlinedCallStackPickler. Backends can serialize the debug handles and
 *  raise exception using debug handle that on device runtime can use to
 * symoblicate, using InlinedCallStackPtr. If preprocess/compile and execute
 * happen in same session then the map, debug-handles-to-inlined-callstack-ptrs,
 * is not serialized. However, exception will be raised using inlined callstack
 * ptr directly. Relevant API will be in following diffs.
 */
class BackendDebugHandleManager {
 public:
  BackendDebugHandleManager() = default;
  int64_t getNextDebugHandleForInlinedCallStackPtr(
      const SourceRange& range,
      const InlinedCallStackPtr& cs_ptr);
  std::unordered_map<int64_t, DelegateDebugInfoType> getCallStackPtrMap();

 private:
  static std::atomic<int64_t> unique_debug_handle_;
  std::unordered_map<int64_t, DelegateDebugInfoType>
      handles_to_inlined_callstack_ptrs_;
};

} // namespace jit
} // namespace torch
