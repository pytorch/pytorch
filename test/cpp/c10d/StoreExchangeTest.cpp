#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <future>
#include <memory>
#include <new>
#include <stdexcept>
#include <tuple>

namespace c10d::symmetric_memory {
namespace {

class ArrivalStore : public HashStore {
 public:
  std::promise<void> first_arrival;

  void barrier(
      const std::string& key,
      int64_t world_size,
      const std::chrono::milliseconds& timeout) override {
    if (!arrived_.exchange(true)) {
      first_arrival.set_value();
    }
    Store::barrier(key, world_size, timeout);
  }

 private:
  std::atomic<bool> arrived_{false};
};

TEST(StoreExchangeTest, HealthyRankObservesTheLastArrivingFailure) {
  auto store = c10::make_intrusive<ArrivalStore>();
  store->setTimeout(std::chrono::seconds(5));
  StoreExchange healthy("ordered-peer-setup");
  StoreExchange failing("ordered-peer-setup");
  auto arrival = store->first_arrival.get_future();
  auto outcome = std::async(std::launch::async, [&] {
    return healthy.barrier_with_failure(store, 2, false);
  });
  ASSERT_EQ(
      arrival.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_TRUE(failing.barrier_with_failure(store, 2, true));
  EXPECT_TRUE(outcome.get());
}

void expectAgreementAndNextExchange(
    const std::array<c10::intrusive_ptr<Store>, 2>& stores) {
  std::array<StoreExchange, 2> exchanges{
      StoreExchange("peer-setup"), StoreExchange("peer-setup")};
  const std::array<std::array<bool, 2>, 4> failures{
      {{false, false}, {true, false}, {false, true}, {false, false}}};
  for (const auto& failed : failures) {
    std::array<std::future<bool>, 2> outcomes;
    for (int rank = 0; rank < 2; ++rank) {
      outcomes[rank] = std::async(std::launch::async, [&, rank] {
        return exchanges[rank].barrier_with_failure(
            stores[rank], 2, failed[rank]);
      });
    }
    const bool expected = failed[0] || failed[1];
    EXPECT_EQ(outcomes[0].get(), expected);
    EXPECT_EQ(outcomes[1].get(), expected);
  }
  std::array<std::future<std::vector<int>>, 2> gathered;
  for (int rank = 0; rank < 2; ++rank) {
    gathered[rank] = std::async(std::launch::async, [&, rank] {
      return exchanges[rank].all_gather(stores[rank], rank, 2, rank + 10);
    });
  }
  for (auto& outcome : gathered) {
    EXPECT_EQ(outcome.get(), (std::vector<int>{10, 11}));
  }
}

TEST(StoreExchangeTest, AgreementDoesNotCarryFailureIntoTheNextExchange) {
  auto store = c10::make_intrusive<HashStore>();
  store->setTimeout(std::chrono::seconds(5));
  expectAgreementAndNextExchange({store, store});
}

class TCPStoreExchangeTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(TCPStoreExchangeTest, AgreementDoesNotCarryFailureIntoTheNextExchange) {
  const auto [use_libuv, use_prefix] = GetParam();
#ifndef TORCH_USE_LIBUV
  if (use_libuv) {
    GTEST_SKIP() << "PyTorch was built without libuv TCPStore support";
  }
#endif
  TCPStoreOptions server_options;
  server_options.port = 0;
  server_options.isServer = true;
  server_options.numWorkers = 2;
  server_options.waitWorkers = false;
  server_options.timeout = std::chrono::seconds(5);
  server_options.useLibUV = use_libuv;
  auto server = c10::make_intrusive<TCPStore>("127.0.0.1", server_options);
  auto client_options = server_options;
  client_options.port = server->getPort();
  client_options.isServer = false;
  auto client = c10::make_intrusive<TCPStore>("127.0.0.1", client_options);
  std::array<c10::intrusive_ptr<Store>, 2> stores{server, client};
  if (use_prefix) {
    for (auto& store : stores) {
      store = c10::make_intrusive<PrefixStore>("peer-group", store);
    }
  }
  expectAgreementAndNextExchange(stores);
}

INSTANTIATE_TEST_SUITE_P(
    Backends,
    TCPStoreExchangeTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool()));

class FailingAgreementStore : public HashStore {
 public:
  explicit FailingAgreementStore(std::exception_ptr error)
      : error_(std::move(error)) {}

  void barrier(const std::string&, int64_t, const std::chrono::milliseconds&)
      override {
    std::rethrow_exception(error_);
  }

 private:
  std::exception_ptr error_;
};

TEST(StoreExchangeTest, LocalFailureSurvivesAgreementFailure) {
  auto local_error =
      std::make_exception_ptr(std::runtime_error("import failed"));
  auto store = c10::make_intrusive<FailingAgreementStore>(
      std::make_exception_ptr(std::runtime_error("agreement failed")));
  StoreExchange exchange("failed-agreement");
  testing::internal::CaptureStderr();
  try {
    exchange.barrier_with_error(store, 2, local_error);
    ADD_FAILURE() << "the local exception was not rethrown";
  } catch (...) {
    EXPECT_EQ(std::current_exception(), local_error);
  }
  const auto diagnostics = testing::internal::GetCapturedStderr();
  const auto local = diagnostics.find("peer setup failed: import failed");
  const auto agreement =
      diagnostics.find("peer setup agreement also failed: agreement failed");
  EXPECT_NE(local, std::string::npos);
  EXPECT_NE(agreement, std::string::npos);
  EXPECT_LT(local, agreement);
}

TEST(StoreExchangeTest, HealthyRankPreservesAgreementFailure) {
  auto error = std::make_exception_ptr(std::runtime_error("agreement failed"));
  auto store = c10::make_intrusive<FailingAgreementStore>(error);
  StoreExchange exchange("failed-agreement");
  try {
    exchange.barrier_with_error(store, 2, nullptr);
    ADD_FAILURE() << "the agreement exception was not rethrown";
  } catch (...) {
    EXPECT_EQ(std::current_exception(), error);
  }
}

TEST(StoreExchangeTest, NonStandardLocalFailureSurvivesAgreementFailure) {
  auto local_error = std::make_exception_ptr(17);
  auto store = c10::make_intrusive<FailingAgreementStore>(
      std::make_exception_ptr(std::runtime_error("agreement failed")));
  StoreExchange exchange("failed-agreement");
  try {
    exchange.barrier_with_error(store, 2, local_error);
    ADD_FAILURE() << "the local exception was not rethrown";
  } catch (...) {
    EXPECT_EQ(std::current_exception(), local_error);
  }
}

TEST(StoreExchangeTest, SuccessfulAgreementPreservesLocalFailure) {
  auto error = std::make_exception_ptr(std::runtime_error("import failed"));
  auto store = c10::make_intrusive<HashStore>();
  StoreExchange exchange("failed-import");
  try {
    exchange.barrier_with_error(store, 1, error);
    ADD_FAILURE() << "the local exception was not rethrown";
  } catch (...) {
    EXPECT_EQ(std::current_exception(), error);
  }
}

TEST(StoreExchangeTest, MissingParticipantDoesNotReplaceLocalFailure) {
  auto error = std::make_exception_ptr(std::runtime_error("import failed"));
  auto store = c10::make_intrusive<HashStore>();
  store->setTimeout(std::chrono::milliseconds(50));
  StoreExchange exchange("missing-peer-after-import-failure");
  try {
    exchange.barrier_with_error(store, 2, error);
    ADD_FAILURE() << "the local exception was not rethrown";
  } catch (...) {
    EXPECT_EQ(std::current_exception(), error);
  }
}

TEST(PeerOwnershipTest, SuccessfulHandoffDisarmsRollback) {
  bool mapped = false;
  int cleanups = 0;
  const auto owner = map_and_own_peer(
      [&] { mapped = true; },
      [&] {
        EXPECT_TRUE(mapped);
        return std::make_unique<int>(11);
      },
      [&] { ++cleanups; });
  EXPECT_EQ(*owner, 11);
  EXPECT_EQ(cleanups, 0);
}

TEST(PeerOwnershipTest, MappingFailureRollsBackExactlyOnce) {
  int cleanups = 0;
  int owner_calls = 0;
  const auto error = std::make_exception_ptr(std::runtime_error("map failed"));
  EXPECT_THROW(
      map_and_own_peer(
          [&] { std::rethrow_exception(error); },
          [&] {
            ++owner_calls;
            return std::make_unique<int>(11);
          },
          [&] { ++cleanups; }),
      std::runtime_error);
  EXPECT_EQ(cleanups, 1);
  EXPECT_EQ(owner_calls, 0);
}

TEST(PeerOwnershipTest, OwnerAllocationFailureRollsBackExactlyOnce) {
  bool mapped = false;
  int cleanups = 0;
  const auto error = std::make_exception_ptr(std::bad_alloc());
  EXPECT_THROW(
      map_and_own_peer(
          [&] { mapped = true; },
          [&]() -> std::unique_ptr<int> {
            EXPECT_TRUE(mapped);
            std::rethrow_exception(error);
          },
          [&] { ++cleanups; }),
      std::bad_alloc);
  EXPECT_EQ(cleanups, 1);
}

TEST(PeerFileDescriptorTest, ClosesAllPeerSlotsAndPreservesLocalSlot) {
  std::array<std::array<int, 2>, 3> pipes{{{-1, -1}, {-1, -1}, {-1, -1}}};
  const auto cleanup = c10::make_scope_exit([&] {
    for (const auto& pipe_fds : pipes) {
      for (const auto fd : pipe_fds) {
        if (fd >= 0) {
          close(fd);
        }
      }
    }
  });
  for (auto& pipe_fds : pipes) {
    ASSERT_EQ(pipe(pipe_fds.data()), 0);
  }
  const std::vector<int> received{pipes[0][0], pipes[1][0], pipes[2][0]};
  std::array<int, 3> close_attempts{};
  close_peer_fds(received, 1, [&](int fd) {
    const auto found = std::find(received.begin(), received.end(), fd);
    ASSERT_NE(found, received.end());
    ++close_attempts[found - received.begin()];
    EXPECT_EQ(close(fd), 0);
  });
  EXPECT_EQ(close_attempts, (std::array<int, 3>{1, 0, 1}));
  struct stat state;
  EXPECT_EQ(fstat(received[1], &state), 0);
  for (const auto peer : {0, 2}) {
    errno = 0;
    EXPECT_EQ(fstat(received[peer], &state), -1);
    EXPECT_EQ(errno, EBADF);
    pipes[peer][0] = -1;
  }
}

TEST(StoreExchangeTest, MissingParticipantTimesOut) {
  auto store = c10::make_intrusive<HashStore>();
  store->setTimeout(std::chrono::milliseconds(50));
  StoreExchange exchange("missing-peer");
  EXPECT_THROW(exchange.barrier_with_failure(store, 2, true), c10::Error);
}

} // namespace
} // namespace c10d::symmetric_memory
