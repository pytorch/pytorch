#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_manager.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace nvfuser;
using namespace torch::jit::fuser::cuda;

TEST_F(NVFuserTest, FusionManager_CUDA) {

  TORCH_CHECK(true, "Hello test world!");
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
