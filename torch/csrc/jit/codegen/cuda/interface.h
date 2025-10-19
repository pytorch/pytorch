#pragma once

#include <c10/macros/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

/*
 * This file contains APIs for cuda fuser;
 *
 * We use an empty static struct to hold the function pointers, which are
 * registered separately. This is to support cpu-only compilation.
 * Registration is done in torch/csrc/jit/codegen/cuda/register_interface.cpp
 */

namespace torch::jit::fuser::cuda {

TORCH_API std::atomic<bool>& getCudaFusionGuardMode();

TORCH_API bool getSingletonFusion();
TORCH_API bool setSingletonFusion(bool value);
TORCH_API bool getHorizontalFusion();
TORCH_API bool setHorizontalFusion(bool value);

// dummy struct to allow API registration
struct CudaFuserInterface {
  void (*fn_compile_n)(Node*) = nullptr;
  void (*fn_run_n_s)(const Node*, Stack&) = nullptr;
  void (*fn_fuse_graph)(std::shared_ptr<Graph>&) = nullptr;
  bool (*fn_can_fuse_n)(const Node*) = nullptr;
  void (*fn_insert_profile_inodes)(ProfilingRecord* pr) = nullptr;
  bool (*fn_profile_n)(const Node*) = nullptr;
  bool (*fn_skip_n)(const std::string&, bool flip) = nullptr;
};

// Get interface, this is used by registration and user facing API internally
TORCH_API CudaFuserInterface* getFuserInterface();

TORCH_API void compileFusionGroup(Node* fusion_node);
TORCH_API void runFusionGroup(const Node* fusion_node, Stack& stack);
TORCH_API void fuseGraph(std::shared_ptr<Graph>& /*graph*/);
TORCH_API bool canFuseNode(const Node* node);
TORCH_API void InsertProfileNodesForCUDAFuser(ProfilingRecord* pr);
TORCH_API bool profileNode(const Node* node);

TORCH_API bool skipNode(const std::string& symbol_str, bool flip = true);

TORCH_API bool isEnabled();
TORCH_API bool setEnabled(bool is_enabled);
TORCH_API bool canBeEnabled();

} // namespace torch::jit::fuser::cuda
