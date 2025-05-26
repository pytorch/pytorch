#pragma once
#include <ATen/core/ivalue.h>

#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/scope.h>

#include <atomic>

namespace torch::jit {

/*
 *  BackendDebugHandleManager is responsible for issuing debug handles to
 *  backends. Debug handles are associated with nodes of a graph.
 *  BackendDebugHandleManager also maintains a map
 *  [debug-handle, DebugInfoTuple = {source range, inlined callstack ptr]} that
 *  will help generate a callstack for exception raised using debug handles.
 *  Effectively debug handles are something that is given to backend and later
 *  when an exception occurs in the backend, backend can tell, using debug
 *  handle, that an exception occurred here. Then the runtime can generate
 *  callstack correspoding to the exception.
 *  There are two parts to BackendDebugHandleManager:
 *  1. static std::atomic debug_handle
 *  2. Map of [debug-handle, DebugInfoTuple]
 *
 *  About 1:
 *  Why do they have to be unique. The reason is that by ensuring
 *  uniqueness of debug handles, we remove the burden of another layer of
 *  mapping where we need to say this set of debug handles were generated for
 *  this lowered module or this bytecode function. This simplifies the API for
 *  serialization since debug handles can uniquely identify DebugInfoTuple.
 *  Thus simplifies the runtime API for throwing exception. Exception throwing
 *  only needs to know debug_handle and not which module or method threw it.
 *  There are 2 issues to keep in mind, though,for static std::atomic
 *  debug_handle: A. Performance implications of using atomic variable. However
 *  this is only used for compilation so we assume to absorb some of that
 *  penalty. Plus if there is no contention then we should have less to worry
 *  about. B. If repeated compilation is part of a long running process then we
 *  may overflow int64_t. We may detect and fail on this. For now this is not
 *  done.
 *
 *  Now about 2:
 *  There are two usecases for [debug-handle, DebugInfoTuple]
 *  A. During bytecode generation the DebugInfoTuple corresponding to the nodes
 *  of the inlined graph being serialized, are stored in this object and a
 *  unique debug handle is returned. This unique debug handle is stored in
 *  mobile_debug info for pytorch lite models. It will be used for raising
 *  exceptions as well as profiling. B. During backend lowering, each backend's
 *  preprocess/compile method can compile method's graph and serialize those
 *  methods. Once the method is lowered to backend, graph is essentially lost.
 *  Without access to graph it is hard to generate model level debug info. Thus
 *  the debug handles provide a way to map nodes of the graph to the model level
 *  debug info.
 *
 *  During byte-code model serialization, [debug-handle, DebugInfoTuple] is
 *  serialized. Now we know a. debug handles and b. how to map debug handles to
 *  model source code. Thus we can either do eager symbolication by converting
 *  debug handles to corresponding source code at runtime, or do lazy
 *  symbolicattion offline.
 *
 *  Note that it is not necessary to serialize [debug-handle, DebugInfoTuple]
 *  corresponding to lowered backend if the lowering process, that is
 *  preprocess/compile, and execution happens in the same session, then eager
 *  symbolication can be employed.
 *
 *  Now how does BackendDebugHandleManager capture all of the above?
 *  By providing two API.
 *  1. getNextDebugHandle which given a Node* returns a unique debug handle,
 *     that will uniquely identify DebugInfoTuple.
 *     and
 *  2. getCallStackPtrMap which returns the map
 *     [debug-handle, DebugInfoTuple]
 *
 *  1 provides debug handles to backends and 2 provides runtime a way to map
 *  debug handles to source level debug info.
 *
 *  So why does debug handle map to DebugInfoTuple = {source range and inlined
 *  cs}? {debug_handle, source_range_tag, serialized_callstack} Take this
 *  example: class L(nn.Module): def __init__(self) -> None:
 *      ...
 *    def forward(self, x):
 *      return x * 5
 *  class M(nn.Module):
 *    def __init__(self) -> None:
 *      ...
 *    def forward(self, x):
 *      return x - 2
 *  class N(nn.Module):
 *    def __init__(self) -> None:
 *      self.m = M()
 *    def forward(self, x):
 *      return self.m(x) + 3
 *  m = torch.jit.script(N())
 *  Once you inline m's forward method, m.forward.graph will look something
 *  like this
 *  graph(%self...):
 *   %x = aten::mul(..)
 *   %x = aten::sub(x, ..)
 *   %y = aten::add(x, ..)
 *   ..
 *  Inlined callstack ptr for these two nodes will look like:
 *  aten::mul's inlined CS (callstack): [N.forward, source range] -> [M.forward,
 *  source range] aten::sub's inlined CS (callstack): [N.forward, source range]
 *  aten::add's inlined CS: null
 *  mul node's inlined CS contains only information about the callsites' source
 *  range The information about mul node's source range ('return x * 5') is not
 *  available in its inlined CS. It is rather part of node's source range
 *  instead of inlined CS. Thus to get full stack: [N.forward, source range] ->
 *  [M.forward, source range] -> [aten::mul's source range] We need to track
 *  mul's source range and inlined CS both.
 */

using BackendDebugInfoMapType =
    std::unordered_map<torch::jit::DebugHandleType, DebugInfoTuple>;

/*
 * This class is used to generate debug info map.
 * backend's preprocess will call generate_debug_handles (see
 * backend_detail.cpp), which uses debug_handle_manager to generate debug
 * handles. When lowering process finishes, calling stopRecording will
 * return debug info map from debug_handle_manager
 */
class TORCH_API BackendDebugInfoRecorder {
 public:
  BackendDebugInfoRecorder() = default;
  int64_t getNextDebugHandle(const Node* node);
  // Reason this is not done as RAII is that work done in stopRecording
  // can throw, and throwing with dtor will call terminate and thus voids any
  // exception catching at a higher level.
  BackendDebugInfoMapType stopRecording();
  NodeToDebugHandle generate_debug_handles(const std::shared_ptr<Graph>& graph);

 private:
  static std::atomic<DebugHandleType> unique_debug_handle_;
  BackendDebugInfoMapType handles_to_inlined_callstack_ptrs_;
};

} // namespace torch::jit
