#pragma once
#include <ATen/core/jit_type.h>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

using NodeToDebugHandle = std::unordered_map<Node*, DebugHandleType>;

/*
 * This is the API via which backend's preprocess function will obtain debug
 * handles corresponding to the nodes of the graph for the lowered methods of
 * the module. It is expected that the graphs of the methods are inlined. If
 * graph is not inlined, this method will throw exception. Implementation: Given
 * moudle with inlined methods:
 * 1. Query if a valid debug handle manager has been initialized
 * 2. If so use debug handle manager to generate debug handles, else all handles
 * are -1. -1 is not quite the great constant for invalid handle, so we will
 * probably fix this later. This will be used to generate debug handles and
 * debug info map:
 * 1. Inside to_backend, use BackendModuleDebugInfoRecorder to initialize thread
 * local debug handler context. for the lowered module ptr.
 * 2. Backend code for lowering module, preprocess, calls
 *    generate_debug_handles(graph)) which will return debug handles
 * corresponding to the Node* of the said graph.
 * 3. In to_backend, after lowering, call stopRecording on
 * BackendModuleDebugInfoRecorder: It will extract debug map. This map gets
 * stored in static instance of ModuleDebugInfoMap. Now there is a global map in
 * which module's callstack ptr maps are stored and can be queried during
 * serialization.
 */
NodeToDebugHandle TORCH_API
generate_debug_handles(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
