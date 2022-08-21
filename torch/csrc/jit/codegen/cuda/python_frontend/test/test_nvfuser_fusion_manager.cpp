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
    auto ptr = fm->fusionPtr();
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
    } catch (...) {
      SUCCEED();
    }

    try {
      fm->traverseFusionCache(null_record);
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fm->createFusionCacheEntry(null_record);
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fm->createTerminalFusionCacheEntry(null_record);
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Check that cache methods act appropriately when presenting a new
  // record to an empty cache.
  {
    std::shared_ptr<RecordFunctor> test_record(new TensorRecord(
        {State(StateType::Tensor, 0)}, {3}, {true}, Nvf::DataType::Float));

    // Check Methods prior to adding an entry to the cache

    // Cache Lookup should not succeed becase no records are in the cache
    try {
      auto empty_cache_entry_ptr = fm->lookupFusionCacheEntry(test_record);
      ASSERT_TRUE(empty_cache_entry_ptr == c10::nullopt);
      SUCCEED();
    } catch (...) {
      FAIL() << "Unexpected assert during cache lookup!";
    }

    // Traversal of the cache should fail because there is nothing to traverse
    try {
      fm->traverseFusionCache(test_record);
      FAIL() << "Expected the cache traversal to fail!";
    } catch (...) {
      SUCCEED();
    }

    // Add a cache entry and check methods

    try {
      fm->createFusionCacheEntry(test_record);
      SUCCEED();
    } catch (...) {
      FAIL() << "An unexpected assert on Cache Entry creation!";
    }

    try {
      auto cache_entry_ptr = fm->lookupFusionCacheEntry(test_record);
      ASSERT_FALSE(cache_entry_ptr == c10::nullopt);
      SUCCEED();
    } catch (...) {
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
    } catch (...) {
      SUCCEED();
    }

    // Add a terminal cache entry and check methods

    std::shared_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      fm->createTerminalFusionCacheEntry(end_record);
      SUCCEED();
    } catch (...) {
      FAIL() << "An unexpected assert on Terminal Cache Entry creation!";
    }

    try {
      fm->traverseFusionCache(end_record);
      SUCCEED();
    } catch (...) {
      FAIL() << "An unexpected assert while traversing to a Terminal Entry!";
    }

    try {
      auto no_cache_entry_ptr = fm->lookupFusionCacheEntry(test_record);
      FAIL() << "Expected an assert from a terminal entry!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fm->traverseFusionCache(test_record);
      FAIL() << "Expected an assert from a terminal entry!";
    } catch (...) {
      SUCCEED();
    }

    try {
      auto ptr = fm->fusionPtr();
      ASSERT_FALSE(ptr == nullptr);
      SUCCEED();
    } catch (...) {
      FAIL() << "An unexpected assert occurred while getting fusion ptr!";
    }
  }

  // Setup cache for a new cache lookup
  try {
    fm->resetFusionCachePtr();
    SUCCEED();
  } catch (...) {
    FAIL() << "Did not properly set cache to pointer to top of tree!";
  }

  // Check that cache methods act appropriately when presenting a new
  // record to a cache with 1 fusion.
  {
    std::shared_ptr<RecordFunctor> cached_record(new TensorRecord(
        {State(StateType::Tensor, 0)}, {3}, {true}, Nvf::DataType::Float));
    std::shared_ptr<RecordFunctor> new_record(
        new ScalarRecord({State(StateType::Scalar, 1)}, Nvf::DataType::Float));

    try {
      auto hit_cache_entry = fm->lookupFusionCacheEntry(cached_record);
      ASSERT_FALSE(hit_cache_entry == c10::nullopt);
      SUCCEED();
    } catch (...) {
      FAIL() << "Cache lookup unexpectedly asserted!";
    }

    try {
      fm->traverseFusionCache(cached_record);
      SUCCEED();
    } catch (...) {
      FAIL() << "Fusion cache traverse unexpectedly asserted!";
    }

    try {
      auto miss_cache_entry = fm->lookupFusionCacheEntry(new_record);
      ASSERT_TRUE(miss_cache_entry == c10::nullopt);
      SUCCEED();
    } catch (...) {
      FAIL() << "Cache lookup unexpectedly asserted!";
    }

    try {
      fm->createFusionCacheEntry(new_record);
      SUCCEED();
    } catch (...) {
      FAIL() << "An unexpected assert on Cache Entry creation!";
    }

    try {
      fm->traverseFusionCache(new_record);
      SUCCEED();
    } catch (...) {
      FAIL() << "Fusion cache traverse unexpectedly asserted!";
    }

    std::shared_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      fm->createTerminalFusionCacheEntry(end_record);
      FAIL() << "Expected the cache to assert because it is full!";
    } catch (...) {
      SUCCEED();
    }
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
