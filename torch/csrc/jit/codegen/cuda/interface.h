#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

/*
 * This file contains APIs for cuda fuser;
 *
 * We use an empty static struct to hold the function pointers, which are
 * registered separately. This is to support cpu-only compilation.
 * Registration is done in torch/csrc/jit/codegen/cuda/register_interface.cpp
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// dummy struct to allow API registration
struct CudaFuserInterface {
  void (*fn_compile_n_)(Node*) = nullptr;
  void (*fn_run_n_s_)(const Node*, Stack&) = nullptr;
  void (*fn_fuse_graph)(std::shared_ptr<Graph>&) = nullptr;
};

// Get interface, this is used by registration and user facing API internally
C10_EXPORT CudaFuserInterface* getFuserInterface();

C10_EXPORT void compileFusionGroup(Node* fusion_node);
C10_EXPORT void runFusionGroup(const Node* fusion_node, Stack& stack);
C10_EXPORT void fuseGraph(std::shared_ptr<Graph>&);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
