#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <vector>

#include <torch/csrc/inductor/aoti_runtime/model_pool.h>

namespace torch::aot_inductor {
namespace {

// Stands in for AOTInductorModel. start() puts work "on the device"; the test
// completes it with finish() or fail().
class FakeModel {
 public:
  bool is_finished() {
    std::lock_guard lk(mutex_);
    return state_ == State::kIdle;
  }

  void wait_for_completion() {
    std::unique_lock lk(mutex_);
    cv_.wait(lk, [this] { return state_ != State::kRunning; });
    const bool failed = state_ == State::kFailed;
    state_ = State::kIdle;
    AOTI_RUNTIME_CHECK(!failed, "device failure");
  }

  void start() {
    set_state(State::kRunning);
  }
  void finish() {
    set_state(State::kIdle);
  }
  void fail() {
    set_state(State::kFailed);
  }

 private:
  enum class State { kIdle, kRunning, kFailed };

  void set_state(State state) {
    {
      std::lock_guard lk(mutex_);
      state_ = state;
    }
    cv_.notify_all();
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  State state_{State::kIdle};
};

using Stream = void*;
using Pool = ModelPool<FakeModel, Stream>;

// Only checkout() creates leases, so no caller can return a model to the pool
// without having checked it out.
static_assert(
    !std::is_constructible_v<Pool::Lease, Pool&, FakeModel*>,
    "Lease's constructor must stay private");

struct TestPool {
  explicit TestPool(size_t num_models) {
    pool.reserve(num_models);
    for (size_t i = 0; i < num_models; ++i) {
      models.push_back(&pool.add_model(std::make_unique<FakeModel>()));
    }
  }

  Pool pool;
  // The pool owns the models; these point to them in index order.
  std::vector<FakeModel*> models;
};

constexpr auto kBlocked = std::chrono::milliseconds(100);

// Pool bugs tend to show up as hangs; fail loudly, naming the test, instead of
// waiting for the test runner's timeout.
template <typename T>
T wait_or_abort(std::future<T>& future) {
  if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
    const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
    std::cerr << "ModelPool test " << (test ? test->name() : "(none)")
              << " timed out, likely a deadlock\n";
    std::abort();
  }
  return future.get();
}

Pool::Lease checkout(Pool& pool, Stream stream = nullptr) {
  auto future =
      std::async(std::launch::async, [&] { return pool.checkout(stream); });
  return wait_or_abort(future);
}

FakeModel* model_of(const Pool::Lease& lease) {
  return lease.operator->();
}

// Mirrors a run() that reached the device: the model goes back in flight.
FakeModel* start_run(Pool& pool, Stream stream = nullptr) {
  auto lease = checkout(pool, stream);
  lease->start();
  FakeModel* model = model_of(lease);
  lease.mark_in_flight();
  return model;
}

// Mirrors a run() that throws after launching device work: the lease is dropped
// while that work may still be running.
void fail_run(Pool::Lease lease) {
  lease->start();
  AOTI_RUNTIME_CHECK(false, "run failed");
}

TEST(ModelPoolTest, WaitsForOldestInFlightRun) {
  TestPool t(2);
  FakeModel* oldest = start_run(t.pool);
  FakeModel* newest = start_run(t.pool);
  auto waiter =
      std::async(std::launch::async, [&] { return t.pool.checkout(); });
  EXPECT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
  oldest->finish();
  EXPECT_EQ(model_of(wait_or_abort(waiter)), oldest);
  newest->finish();
}

// A run that throws drops its lease, which returns the model to the available
// pool without waiting for the device work the run launched: a checkout waiting
// for the pool's only model gets it while that work is still running.
TEST(ModelPoolTest, FailedRunWakesWaitingCheckout) {
  TestPool t(1);
  auto failed_run = checkout(t.pool);
  auto waiter =
      std::async(std::launch::async, [&] { return t.pool.checkout(); });
  EXPECT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
  EXPECT_THROW(fail_run(std::move(failed_run)), std::runtime_error);
  EXPECT_EQ(model_of(wait_or_abort(waiter)), t.models[0]);
}

TEST(ModelPoolTest, LeaseMarksInFlightOnlyOnce) {
  TestPool t(1);
  auto lease = checkout(t.pool);
  lease.mark_in_flight();
  EXPECT_THROW(lease.mark_in_flight(), std::runtime_error);
  EXPECT_THROW(lease.operator->(), std::runtime_error);
}

TEST(ModelPoolTest, MovedFromLeaseDoesNotReturnModel) {
  TestPool t(1);
  std::optional<Pool::Lease> holder;
  {
    auto lease = checkout(t.pool);
    holder.emplace(std::move(lease));
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THROW(lease.operator->(), std::runtime_error);
  }
  auto waiter =
      std::async(std::launch::async, [&] { return t.pool.checkout(); });
  EXPECT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
  holder.reset();
  wait_or_abort(waiter);
}

// The pool supports stream affinity only on device runtimes, and
// CMakeLists.txt compiles this file with the build's device macro.
#if defined(USE_CUDA) || defined(USE_XPU)

Stream stream_handle(uintptr_t id) {
  return reinterpret_cast<Stream>(id);
}

// A failed run clears its model's stream binding, so the stream's next call
// may choose another model.
TEST(ModelPoolTest, FailedRunClearsStreamBinding) {
  TestPool t(1);
  t.pool.set_use_stream_affinity(true);
  start_run(t.pool, stream_handle(1))->finish();
  EXPECT_EQ(t.pool.get_stream_affinity_model_index(stream_handle(1)), 0);
  // Dropping the lease without mark_in_flight() is a failed run.
  checkout(t.pool, stream_handle(1));
  EXPECT_EQ(
      t.pool.get_stream_affinity_model_index(stream_handle(1)),
      AOTI_STREAM_AFFINITY_UNBOUND);
}

#endif

} // namespace
} // namespace torch::aot_inductor
