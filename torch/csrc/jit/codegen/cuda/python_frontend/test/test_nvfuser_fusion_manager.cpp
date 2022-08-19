#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_definition.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_manager.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace nvfuser;
using namespace torch::jit::fuser::cuda;

TEST_F(NVFuserTest, FusionManager_CUDA) {
  // Create a fusion manager with a maximum of 1 Fusion
  FusionManager* fm = FusionManager::get(1);

  ASSERT_FALSE(fm == nullptr);

  fm->reset();

  ASSERT_FALSE(fm == nullptr);

  try {
    fm->fusionPtr();
    FAIL() << "Expected a Fusion ptr check to fail!";
  } catch (...) {
    SUCCEED();
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
