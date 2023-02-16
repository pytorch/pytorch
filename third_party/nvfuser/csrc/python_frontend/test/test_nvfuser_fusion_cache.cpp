#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <python_frontend/fusion_cache.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

namespace nvfuser {
using namespace nvfuser::python_frontend;

// RUN CMD: bin/test_jit --gtest_filter="NVFuserTest*PyFusionCache*"
TEST_F(NVFuserTest, PyFusionCache_CUDA) {
  // Reset cache before testing.
  try {
    FusionCache::reset();
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Did not properly reset cache!" << e.what();
  }

  // Create a fusion manager with a maximum of 1 Fusion
  FusionCache* fc = FusionCache::get(1);
  // You should never get a nullptr
  ASSERT_FALSE(fc == nullptr);
  ASSERT_TRUE(fc->numFusions() == 0);

  // Check that cache methods all assert when presented with a null record.
  {
    std::unique_ptr<RecordFunctor> null_record(nullptr);

    try {
      fc->queryChildren(null_record.get());
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fc->traverseTrie(null_record.get());
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fc->createChild(null_record.get());
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fc->createChild(null_record.get());
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Check that cache methods act appropriately when presenting a new
  // record to an empty cache.
  {
    std::unique_ptr<RecordFunctor> test_record(new TensorRecord(
        {State(0, StateType::Tensor)}, {3}, {true}, DataType::Float));

    // Check Methods prior to adding an entry to the cache

    // Cache Lookup should not succeed becase no records are in the cache
    try {
      auto empty_cache_entry_ptr = fc->queryChildren(test_record.get());
      ASSERT_TRUE(empty_cache_entry_ptr == c10::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during cache lookup!" << e.what();
    }

    // Traversal of the cache should fail because there is nothing to traverse
    try {
      fc->traverseTrie(test_record.get());
      FAIL() << "Expected the cache traversal to fail!";
    } catch (...) {
      SUCCEED();
    }

    // Add a cache entry and check methods

    try {
      fc->createChild(test_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on Cache Entry creation!" << e.what();
    }

    try {
      auto cache_entry_ptr = fc->queryChildren(test_record.get());
      ASSERT_FALSE(cache_entry_ptr == c10::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on cache lookup!" << e.what();
    }

    try {
      fc->traverseTrie(test_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert during Cache Traverse!" << e.what();
    }

    // Add a terminal cache entry and check methods

    std::unique_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      fc->createChild(end_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on Terminal Cache Entry creation!"
             << e.what();
    }

    try {
      fc->traverseTrie(end_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert while traversing to a Terminal Entry!"
             << e.what();
    }

    try {
      fc->queryChildren(test_record.get());
      FAIL() << "Expected an assert from a terminal entry!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fc->traverseTrie(test_record.get());
      FAIL() << "Expected an assert from a terminal entry!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Setup cache for a new cache lookup
  try {
    fc->resetTriePtr();
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Did not properly set cache to pointer to top of tree!"
           << e.what();
  }

  // Check that cache methods act appropriately when presenting a new
  // record to a cache with 1 fusion.
  {
    std::unique_ptr<RecordFunctor> cached_record(new TensorRecord(
        {State(0, StateType::Tensor)}, {3}, {true}, DataType::Float));
    std::unique_ptr<RecordFunctor> new_record(
        new ScalarRecord({State(1, StateType::Scalar)}, DataType::Float));

    try {
      auto hit_cache_entry = fc->queryChildren(cached_record.get());
      ASSERT_FALSE(hit_cache_entry == c10::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Cache lookup unexpectedly asserted!" << e.what();
    }

    try {
      fc->traverseTrie(cached_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Fusion cache traverse unexpectedly asserted!" << e.what();
    }

    try {
      auto miss_cache_entry = fc->queryChildren(new_record.get());
      ASSERT_TRUE(miss_cache_entry == c10::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Cache lookup unexpectedly asserted!" << e.what();
    }

    try {
      fc->createChild(new_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on Cache Entry creation!" << e.what();
    }

    try {
      fc->traverseTrie(new_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Fusion cache traverse unexpectedly asserted!" << e.what();
    }

    std::unique_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      fc->createChild(end_record.get());
      FAIL() << "Expected the cache to assert because it is full!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Setup cache for a new cache lookup
  try {
    fc->resetTriePtr();
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Did not properly set cache to pointer to top of tree!"
           << e.what();
  }

  // Verify proper cache lookup up of complete fusion already cached.
  // This tends to flush out pointer problems in the cache.
  {
    std::unique_ptr<RecordFunctor> test_record(new TensorRecord(
        {State(0, StateType::Tensor)}, {3}, {true}, DataType::Float));
    std::unique_ptr<RecordFunctor> dummy_record(new TensorRecord(
        {State(0, StateType::Tensor)}, {3}, {true}, DataType::Float));

    try {
      auto cache_entry_ptr = fc->queryChildren(test_record.get());
      ASSERT_FALSE(cache_entry_ptr == c10::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on cache lookup!" << e.what();
    }

    try {
      fc->traverseTrie(test_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert during Cache Traverse!" << e.what();
    }

    std::unique_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      fc->queryChildren(end_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on cache lookup!" << e.what();
    }

    try {
      fc->traverseTrie(end_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert while traversing to a Terminal Entry!"
             << e.what();
    }
  }
}

} // namespace nvfuser
