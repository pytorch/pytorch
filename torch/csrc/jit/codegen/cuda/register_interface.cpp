#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/codegen/cuda/manager.h>
#include <torch/csrc/jit/codegen/cuda/partition.h>

/*
 * Registers function pointers in interface.h
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
struct RegisterInterface {
  RegisterInterface() {
    auto ptr = getFuserInterface();
    ptr->fn_compile_n_ = &compileCudaFusionGroup;
    ptr->fn_run_n_s_ = &runCudaFusionGroup;
    ptr->fn_fuse_graph = &CudaFuseGraph;
  }
};

static RegisterInterface register_interface_;
} // namespace

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
