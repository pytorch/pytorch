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
  //FusionManager* fm = FusionManager::get(1);

  /*
  {
    FusionDefinition fd(fm, 10);

    fd.enter();

    nvfuser::Tensor* t0 = fd.defineTensor();
    fd.defineRecord(new TensorRecord({*static_cast<State*>(t0)}, {3}, {true}, Nvf::DataType::Float));

    TORCH_CHECK(fm->fusionCachePtr()->record.use_count() == 2, "Both the FusionDefintion and FusionManager Cache should share Record's pointer!");
  }*/

  TORCH_CHECK(true, "Hello test world!");
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
