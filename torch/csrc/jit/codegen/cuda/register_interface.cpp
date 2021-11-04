#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/codegen/cuda/manager.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/partition.h>

#include <torch/csrc/jit/runtime/profiling_record.h>

/*
 * Registers function pointers in interface.h
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
class RegisterInterface {
 public:
  RegisterInterface() {
    auto ptr = getFuserInterface();
    ptr->fn_compile_n = &compileCudaFusionGroup;
    ptr->fn_run_n_s = &runCudaFusionGroup;
    ptr->fn_fuse_graph = &CudaFuseGraph;
    ptr->fn_can_fuse_n = &isFusibleCudaFusionGroup;
    ptr->fn_insert_profile_inodes = &InsertProfileNodes;
  }
};

static RegisterInterface register_interface_;
} // namespace

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
