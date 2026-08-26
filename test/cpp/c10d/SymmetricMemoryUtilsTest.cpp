#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>

using c10d::symmetric_memory::StoreExchange;

TEST(SymmetricMemoryUtilsTest, StoreExchangeAllGather) {
  c10::intrusive_ptr<c10d::Store> store =
      c10::make_intrusive<c10d::HashStore>();
  StoreExchange rank0Exchange("SymmetricMemoryUtilsTest");
  StoreExchange rank1Exchange("SymmetricMemoryUtilsTest");

  std::vector<int> rank0Values;
  std::vector<int> rank1Values;
  std::thread rank0([&] {
    rank0Values = rank0Exchange.all_gather(
        store, /*rank=*/0, /*world_size=*/2, /*val=*/11);
  });
  std::thread rank1([&] {
    rank1Values = rank1Exchange.all_gather(
        store, /*rank=*/1, /*world_size=*/2, /*val=*/22);
  });

  rank0.join();
  rank1.join();

  const std::vector<int> expected{11, 22};
  EXPECT_EQ(rank0Values, expected);
  EXPECT_EQ(rank1Values, expected);
}

TEST(SymmetricMemoryUtilsTest, StoreExchangeAllGatherTimesOut) {
  c10::intrusive_ptr<c10d::Store> store =
      c10::make_intrusive<c10d::HashStore>();
  StoreExchange exchange("SymmetricMemoryUtilsTimeoutTest");

  auto start = std::chrono::steady_clock::now();
  EXPECT_THROW(
      exchange.all_gather(
          store,
          /*rank=*/0,
          /*world_size=*/2,
          /*val=*/11,
          std::chrono::milliseconds(10)),
      c10::DistStoreError);
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_LT(elapsed, std::chrono::seconds(5));
}
