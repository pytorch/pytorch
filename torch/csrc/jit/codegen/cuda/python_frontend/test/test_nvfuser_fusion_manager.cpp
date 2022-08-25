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
    std::unique_ptr<RecordFunctor> null_record(nullptr);

    try {
      auto bad_cache_entry_ptr = fm->lookupFusionCacheEntry(null_record.get());
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fm->traverseFusionCache(null_record.get());
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fm->createFusionCacheEntry(null_record.get());
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fm->createTerminalFusionCacheEntry(null_record.get());
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Check that cache methods act appropriately when presenting a new
  // record to an empty cache.
  {
    std::unique_ptr<RecordFunctor> test_record(new TensorRecord(
        {State(0, StateType::Tensor)}, {3}, {true}, Nvf::DataType::Float));

    // Check Methods prior to adding an entry to the cache

    // Cache Lookup should not succeed becase no records are in the cache
    try {
      auto empty_cache_entry_ptr =
          fm->lookupFusionCacheEntry(test_record.get());
      ASSERT_TRUE(empty_cache_entry_ptr == c10::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during cache lookup!" << e.what();
    }

    // Traversal of the cache should fail because there is nothing to traverse
    try {
      fm->traverseFusionCache(test_record.get());
      FAIL() << "Expected the cache traversal to fail!";
    } catch (...) {
      SUCCEED();
    }

    // Add a cache entry and check methods

    try {
      fm->createFusionCacheEntry(test_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on Cache Entry creation!" << e.what();
    }

    try {
      auto cache_entry_ptr = fm->lookupFusionCacheEntry(test_record.get());
      ASSERT_FALSE(cache_entry_ptr == c10::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on cache lookup!" << e.what();
    }

    try {
      fm->traverseFusionCache(test_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert during Cache Traverse!" << e.what();
    }

    // Try to add terminal cache entry with a record that is not of End Type.

    try {
      fm->createTerminalFusionCacheEntry(test_record.get());
      FAIL() << "Terminal Cache Entries should only accept EndRecords!";
    } catch (...) {
      SUCCEED();
    }

    // Add a terminal cache entry and check methods

    std::unique_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      fm->createTerminalFusionCacheEntry(end_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on Terminal Cache Entry creation!"
             << e.what();
    }

    try {
      fm->traverseFusionCache(end_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert while traversing to a Terminal Entry!"
             << e.what();
    }

    try {
      auto no_cache_entry_ptr = fm->lookupFusionCacheEntry(test_record.get());
      FAIL() << "Expected an assert from a terminal entry!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fm->traverseFusionCache(test_record.get());
      FAIL() << "Expected an assert from a terminal entry!";
    } catch (...) {
      SUCCEED();
    }

    try {
      auto ptr = fm->fusionPtr();
      ASSERT_FALSE(ptr == nullptr);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert occurred while getting fusion ptr!"
             << e.what();
    }
  }

  // Setup cache for a new cache lookup
  try {
    fm->resetFusionCachePtr();
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Did not properly set cache to pointer to top of tree!"
           << e.what();
  }

  // Check that cache methods act appropriately when presenting a new
  // record to a cache with 1 fusion.
  {
    std::unique_ptr<RecordFunctor> cached_record(new TensorRecord(
        {State(0, StateType::Tensor)}, {3}, {true}, Nvf::DataType::Float));
    std::unique_ptr<RecordFunctor> new_record(
        new ScalarRecord({State(1, StateType::Scalar)}, Nvf::DataType::Float));

    try {
      auto hit_cache_entry = fm->lookupFusionCacheEntry(cached_record.get());
      ASSERT_FALSE(hit_cache_entry == c10::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Cache lookup unexpectedly asserted!" << e.what();
    }

    try {
      fm->traverseFusionCache(cached_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Fusion cache traverse unexpectedly asserted!" << e.what();
    }

    try {
      auto miss_cache_entry = fm->lookupFusionCacheEntry(new_record.get());
      ASSERT_TRUE(miss_cache_entry == c10::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Cache lookup unexpectedly asserted!" << e.what();
    }

    try {
      fm->createFusionCacheEntry(new_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on Cache Entry creation!" << e.what();
    }

    try {
      fm->traverseFusionCache(new_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Fusion cache traverse unexpectedly asserted!" << e.what();
    }

    std::unique_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      fm->createTerminalFusionCacheEntry(end_record.get());
      FAIL() << "Expected the cache to assert because it is full!";
    } catch (...) {
      SUCCEED();
    }
  }
  
  // Setup cache for a new cache lookup
  try {
    fm->resetFusionCachePtr();
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Did not properly set cache to pointer to top of tree!"
           << e.what();
  }
  
  // Verify proper cache lookup up of complete fusion already cached.
  // This tends to flush out pointer problems in the cache.
  {
    std::unique_ptr<RecordFunctor> test_record(new TensorRecord(
        {State(0, StateType::Tensor)}, {3}, {true}, Nvf::DataType::Float));
    std::unique_ptr<RecordFunctor> dummy_record(new TensorRecord(
        {State(0, StateType::Tensor)}, {3}, {true}, Nvf::DataType::Float));

    try {
      auto cache_entry_ptr = fm->lookupFusionCacheEntry(test_record.get());
      ASSERT_FALSE(cache_entry_ptr == c10::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on cache lookup!" << e.what();
    }

    try {
      fm->traverseFusionCache(test_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert during Cache Traverse!" << e.what();
    }
    
    std::unique_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      auto no_cache_entry_ptr = fm->lookupFusionCacheEntry(end_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on cache lookup!" << e.what();
    }

    try {
      fm->traverseFusionCache(end_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert while traversing to a Terminal Entry!"
             << e.what();
    }
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
