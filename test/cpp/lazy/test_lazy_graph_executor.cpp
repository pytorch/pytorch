#include <gtest/gtest.h>

#include <test/cpp/lazy/test_lazy_ops_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>

#include <vector>

namespace torch {
namespace lazy {
namespace {

class LazyGraphExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executor_ = LazyGraphExecutor::Get();
  }

  using CachedComputationType = LazyGraphExecutor::CachedComputation;

  std::shared_ptr<CachedComputationType> GetCachedComputation(hash_t hash) {
    return executor_->GetComputationCache()->Get(hash);
  }

  void EnsureComputationIsCached(
      std::vector<LazyTensorPtr>& tensors,
      hash_t hash) {
    // Force computation to be cached by syncing the tensors.
    executor_->SyncTensorsGraph(
        &tensors, /* devices */ {}, /* wait */ true, /* sync_ltc_data */ true);

    // Ensure that the computation cache entry exists.
    auto cached_computation = GetCachedComputation(hash);
    EXPECT_NE(cached_computation, nullptr)
        << "Computation should be cached after sync";
  }

  LazyGraphExecutor* executor_;
};

TEST_F(LazyGraphExecutorTest, TestClearComputationCache) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor tensor_a =
        torch::rand({2, 2}, at::TensorOptions(torch::kFloat));
    torch::Tensor tensor_b =
        torch::rand({2, 2}, at::TensorOptions(torch::kFloat));

    torch::Tensor xla_tensor_a = CopyToDevice(tensor_a, device);
    torch::Tensor xla_tensor_b = CopyToDevice(tensor_b, device);
    torch::Tensor result = xla_tensor_a + xla_tensor_b;

    std::vector<LazyTensorPtr> tensors{TryGetLtcTensor(result)};
    hash_t hash = executor_->GetGraphHash(tensors);
    EnsureComputationIsCached(tensors, hash);
    EXPECT_EQ(executor_->GetComputationCache()->Numel(), 1);

    // Clear the entire computation cache.
    executor_->ClearComputationCache();

    // Ensure that there are no cache entries.
    EXPECT_EQ(executor_->GetComputationCache()->Numel(), 0);
    auto cached_computation = GetCachedComputation(hash);
    EXPECT_EQ(cached_computation, nullptr)
        << "Cache entry should be null after clearing";
  });
}

TEST_F(LazyGraphExecutorTest, TestRemoveSpecificCacheEntry) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor tensor_a =
        torch::rand({2, 2}, at::TensorOptions(torch::kFloat));
    torch::Tensor tensor_b =
        torch::rand({2, 2}, at::TensorOptions(torch::kFloat));

    torch::Tensor xla_tensor_a = CopyToDevice(tensor_a, device);
    torch::Tensor xla_tensor_b = CopyToDevice(tensor_b, device);
    torch::Tensor result = xla_tensor_a + xla_tensor_b;

    std::vector<LazyTensorPtr> tensors{TryGetLtcTensor(result)};
    hash_t hash = executor_->GetGraphHash(tensors);
    EnsureComputationIsCached(tensors, hash);

    // Remove a specific cache entry.
    executor_->RemoveFromComputationCache(hash);

    // Ensure that the cache entry has been removed.
    auto cached_computation = GetCachedComputation(hash);
    EXPECT_EQ(cached_computation, nullptr)
        << "Cache entry should be null after removal";

    // Attempting to remove again should not do anything.
    executor_->RemoveFromComputationCache(hash);
  });
}

} // namespace
} // namespace lazy
} // namespace torch
