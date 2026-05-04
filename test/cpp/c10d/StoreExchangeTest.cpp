#include <gtest/gtest.h>

#include <atomic>
#include <array>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>

using namespace c10d::symmetric_memory;

namespace {

struct BroadcastPayload {
  int64_t value;
  uint32_t tag;
};

void expect_payload_eq(
    const BroadcastPayload& actual,
    const BroadcastPayload& expected) {
  EXPECT_EQ(actual.value, expected.value);
  EXPECT_EQ(actual.tag, expected.tag);
}

std::vector<StoreExchange> make_store_exchanges(
    int world_size,
    const std::string& prefix) {
  std::vector<StoreExchange> exchanges;
  exchanges.reserve(world_size);
  for (int rank = 0; rank < world_size; ++rank) {
    exchanges.emplace_back(prefix);
  }
  return exchanges;
}

void test_broadcast_from_nonzero_source() {
  constexpr auto world_size = 4;
  constexpr auto src_rank = 2;
  const BroadcastPayload src_payload{123456789, 17};
  const BroadcastPayload peer_payload{-1, 99};

  auto store = c10::make_intrusive<c10d::HashStore>();
  auto exchanges =
      make_store_exchanges(world_size, "StoreExchangeTest.NonzeroSource");

  for (const auto rank : std::array<int, world_size>{src_rank, 0, 1, 3}) {
    auto payload = exchanges[rank].broadcast(
        store,
        rank,
        world_size,
        src_rank,
        rank == src_rank ? src_payload : peer_payload);
    expect_payload_eq(payload, src_payload);
  }
}

void test_broadcast_multiple_rounds() {
  constexpr auto world_size = 3;
  auto store = c10::make_intrusive<c10d::HashStore>();
  auto exchanges =
      make_store_exchanges(world_size, "StoreExchangeTest.MultipleRounds");

  for (const auto round : std::array<int, 2>{0, 1}) {
    const auto src_rank = round;
    const BroadcastPayload src_payload{
        round + 10, static_cast<uint32_t>(round)};
    for (const auto rank :
         std::array<int, world_size>{src_rank, 2, 1 - src_rank}) {
      auto payload = exchanges[rank].broadcast(
          store,
          rank,
          world_size,
          src_rank,
          rank == src_rank ? src_payload : BroadcastPayload{});
      expect_payload_eq(payload, src_payload);
    }
  }
}

void test_broadcast_waits_for_source_rank() {
  constexpr auto world_size = 4;
  constexpr auto src_rank = 0;
  const BroadcastPayload src_payload{987654321, 123};

  auto store = c10::make_intrusive<c10d::HashStore>();
  auto exchanges =
      make_store_exchanges(world_size, "StoreExchangeTest.WaitsForSourceRank");

  std::array<BroadcastPayload, world_size> outputs{};
  std::atomic<int> waiting_peers{0};
  std::vector<std::thread> threads;
  for (int rank = 1; rank < world_size; ++rank) {
    threads.emplace_back([&, rank] {
      waiting_peers.fetch_add(1);
      outputs[rank] = exchanges[rank].broadcast(
          store, rank, world_size, src_rank, BroadcastPayload{});
    });
  }

  while (waiting_peers.load() != world_size - 1) {
    std::this_thread::yield();
  }
  outputs[src_rank] = exchanges[src_rank].broadcast(
      store, src_rank, world_size, src_rank, src_payload);

  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto rank : c10::irange(world_size)) {
    expect_payload_eq(outputs[rank], src_payload);
  }
}

void test_broadcast_rejects_invalid_ranks() {
  auto store = c10::make_intrusive<c10d::HashStore>();
  StoreExchange exchange("StoreExchangeTest.InvalidRanks");

  EXPECT_THROW(exchange.broadcast(store, -1, 2, 0, 0), c10::Error);
  EXPECT_THROW(exchange.broadcast(store, 0, 2, 2, 0), c10::Error);
}

} // namespace

TEST(StoreExchangeTest, BroadcastFromNonzeroSource) {
  test_broadcast_from_nonzero_source();
}

TEST(StoreExchangeTest, BroadcastMultipleRounds) {
  test_broadcast_multiple_rounds();
}

TEST(StoreExchangeTest, BroadcastWaitsForSourceRank) {
  test_broadcast_waits_for_source_rank();
}

TEST(StoreExchangeTest, BroadcastRejectsInvalidRanks) {
  test_broadcast_rejects_invalid_ranks();
}
