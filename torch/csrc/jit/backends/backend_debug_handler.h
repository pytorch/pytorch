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
 *  mapping where we need to say this set of debug handles were generated for
 *  this lowered module or this bytecode function. This simplifies the API for
 *  serialization since debug handles can uniquely identify inlined-callstack-ptr.
 *  Thus simplifies the runtime API for throwing exception. Exception throwing
 *  only needs to know debug_handle and not which module or method threw it.
 *  There are 2 issues to keep in mind, though,for static std::atomic debug_handle:
 *  A. Performance implications of using atomic variable. However this is only
 *     used for compilation so we assume to absorb some of that penalty.
 *     Plus if there is no contention then we should have less to worry about.
 *  B. If repeated compilation is part of a long running process then we may
 *     overflow int64_t. We may detect and fail on this. Not sure if this is good.
 *     Will seek opinions about this.
 *
 *  Now about 2:
 *  There are two usecases for debug-handle-to-inlined-callstack-ptr:
 *  A. During bytecode generation the inlined callstack ptrs, corresponding
 *     to the nodes of the inlined graph being serialized, are stored in this
 *     object and a unique debug handle is returned. This unique debug handle
 *     is stored in mobile_debug info for pytorch lite models. It will be used
 *     for raising exceptions as well as profiling.
 *  B. During backend lowering, each backend's preprocess/compile method can
 *     compile method's graph and serialize those methods. Once the method is
 *     lowered to backend, graph is essentially lost. Without access to graph
 *     it is hard to generate model level debug info via source range. Thus
 *     the debug handles provide a way to map nodes of the graph to the model level
 *     debug info. Following diffs will provide a API to generate debug handles
 *     given inlined graph. This API will, given an inlined graph of a method,
 *     returns a map of node*-to-debug-handles. Backends will serialize these debug
 *     handles and use them to raise exceptionm, much like lite interpreter.
 *     Underneath the API will utilize BackendDebugHandleManager, to generate map
 *     of debug-handles-to-inlined-callstack-ptrs.
 *
 *  During byte-code model serialization, debug-handles-to-inlined-callstack-ptrs is
 *  serialized.
 *  Now we know a. debug handles and b. how to map debug handles to model source code.
 *  Thus we can either do eager symbolication by converting debug handles to
 *  corresponding source code at runtime, or do lazy symbolicattion offline.
 *
 *  Note that it is not necessary to serialize debug-handles-to-inlined-callstack-ptrs
 *  lowered backend if the lowering process, that is preprocess/compile, and execution
 *  happens in the same session, then eager symbolication can be employed.
 *
 *  Now how does BackendDebugHandleManager capture all of the above?
 *  By providing two API.
 *  1. getNextDebugHandleForInlinedCallStackPtr which given a source range and
 *     inlined callstack ptr returns a unique debug handle, that will uniquely
 *     identify the tuple of source range and inlined callstack ptr.
 *     and
 *  2. getCallStackPtrMap which returns the map of debug-handle-to-inlined-callstack-ptr
 *     or more precisely map of debug-handle-to-tuple(source range, inlined-callstack-ptr)
 *
 *  1 provides debug handles to backends and 2 provides runtime a way to map debug handles
 *    to source level debug info.
 */
class TORCH_API BackendDebugHandleManager {
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
