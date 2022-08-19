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

// RUN CMD: bin/test_jit --gtest_filter="NVFuserTest*FusionManager*"
TEST_F(NVFuserTest, FusionManager_CUDA) {
  // Create a fusion manager with a maximum of 1 Fusion
  FusionManager* fm = FusionManager::get(1);

  // You should never get a nullptr
  ASSERT_FALSE(fm == nullptr);

  // If you are not pointed to a terminal node, accessing a fusion pointer
  // should result in an assert.
  try {
    fm->fusionPtr();
    FAIL() << "Expected a Fusion ptr check to fail!";
  } catch (...) {
    SUCCEED();
  }

  {
    std::shared_ptr<RecordFunctor> bad_record(nullptr);
    std::shared_ptr<RecordFunctor> good_record(new TensorRecord(
        {State(StateType::Tensor, 0)}, 
        {3},
        {true},
        Nvf::DataType::Float));

    // Cache Lookup should not succeed becase no records are in the cache
    auto cache_entry_ptr = fm->lookupFusionCacheEntry(good_record);
    ASSERT_TRUE(cache_entry_ptr == c10::nullopt);
    
    // Traversal of the cache should fail because there is nothing to traverse
    try {
      fm->traverseFusionCache(good_record);
      FAIL() << "Expected the cache traversal to fail!";
    } catch(...) {
      SUCCEED();
    }
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
