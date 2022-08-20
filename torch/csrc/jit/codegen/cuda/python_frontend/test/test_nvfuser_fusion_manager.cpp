#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

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

  // Check that cache methods all assert when presented with a null record.
  {
    std::shared_ptr<RecordFunctor> null_record(nullptr);

    try {
      auto bad_cache_entry_ptr = fm->lookupFusionCacheEntry(null_record);
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch(...) {
      SUCCEED();
    }
    
    try {
      fm->traverseFusionCache(null_record);
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch(...) {
      SUCCEED();
    }
    
    try {
      fm->createFusionCacheEntry(null_record);
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch(...) {
      SUCCEED();
    }
    
    try {
      fm->createTerminalFusionCacheEntry(null_record);
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch(...) {
      SUCCEED();
    }
  }

  // Check that cache methods act appropriately when presenting a new
  // record to an empty cache. 
  {
    std::shared_ptr<RecordFunctor> test_record(new TensorRecord(
        {State(StateType::Tensor, 0)}, 
        {3},
        {true},
        Nvf::DataType::Float));

    // Check Methods prior to adding an entry to the cache

    // Cache Lookup should not succeed becase no records are in the cache
    auto empty_cache_entry_ptr = fm->lookupFusionCacheEntry(test_record);
    ASSERT_TRUE(empty_cache_entry_ptr == c10::nullopt);
    
    // Traversal of the cache should fail because there is nothing to traverse
    try {
      fm->traverseFusionCache(test_record);
      FAIL() << "Expected the cache traversal to fail!";
    } catch(...) {
      SUCCEED();
    }

    // Add a cache entry and check methods

    try {
      fm->createFusionCacheEntry(test_record);
      SUCCEED();
    } catch(...) {
      FAIL() << "An unexpected assert on Cache Entry creation!";
    }

    try {
      auto cache_entry_ptr = fm->lookupFusionCacheEntry(test_record);
      ASSERT_FALSE(cache_entry_ptr == c10::nullopt);
      SUCCEED();
    } catch(...) {
      FAIL() << "An unexpected assert on cache lookup!";
    }

    try {
      fm->traverseFusionCache(test_record);
      SUCCEED();
    } catch (...) {
      FAIL() << "An unexpected assert during Cache Traverse!";
    } 
    
    // Try to add terminal cache entry with a record that is not of End Type.

    try {
      fm->createTerminalFusionCacheEntry(test_record);
      FAIL() << "Terminal Cache Entries should only accept EndRecords!";
    } catch(...) {
      SUCCEED();
    }
    
    // Add a terminal cache entry and check methods

    std::shared_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      fm->createTerminalFusionCacheEntry(end_record);
      SUCCEED();
    } catch(...) {
      FAIL() << "An unexpected assert on Terminal Cache Entry creation!";
    }

    try {
      fm->traverseFusionCache(end_record);
      SUCCEED();
    } catch(...) {
      FAIL() << "An unexpected assert while traversing to a Terminal Entry!";
    }
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
