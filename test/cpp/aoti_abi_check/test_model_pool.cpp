#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
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

  int64_t index_of(const FakeModel* model) const {
    return std::find(models.begin(), models.end(), model) - models.begin();
  }

  Pool pool;
  // The pool owns the models; these point to them in index order.
  std::vector<FakeModel*> models;
};

Stream stream_handle(uintptr_t id) {
  return reinterpret_cast<Stream>(id);
}

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

// Mirrors a run() that reached the device: the model goes back in flight.
FakeModel* start_run(Pool& pool, Stream stream = nullptr) {
  auto lease = checkout(pool, stream);
  lease->start();
  FakeModel* model = &*lease;
  lease.mark_in_flight();
  return model;
}

TEST(ModelPoolTest, ReusesMostRecentlyReturnedModel) {
  TestPool t(3);
  FakeModel* model = &*checkout(t.pool);
  EXPECT_EQ(model, t.models[2]);
  EXPECT_EQ(&*checkout(t.pool), model);
}

TEST(ModelPoolTest, PrefersFinishedRunOverWaiting) {
  TestPool t(2);
  start_run(t.pool);
  FakeModel* finished = start_run(t.pool);
  finished->finish();
  EXPECT_EQ(&*checkout(t.pool), finished);
}

TEST(ModelPoolTest, ReusesMostRecentFinishedRun) {
  TestPool t(2);
  FakeModel* older = start_run(t.pool);
  FakeModel* newer = start_run(t.pool);
  older->finish();
  newer->finish();
  EXPECT_EQ(&*checkout(t.pool), newer);
}

TEST(ModelPoolTest, WaitsForOldestInFlightRun) {
  TestPool t(2);
  FakeModel* oldest = start_run(t.pool);
  FakeModel* newest = start_run(t.pool);
  auto waiter =
      std::async(std::launch::async, [&] { return t.pool.checkout(); });
  EXPECT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
  oldest->finish();
  EXPECT_EQ(&*wait_or_abort(waiter), oldest);
  newest->finish();
}

TEST(ModelPoolTest, FailedRunWakesWaitingCheckout) {
  TestPool t(1);
  std::optional<Pool::Lease> failed_run(checkout(t.pool));
  FakeModel* model = &**failed_run;
  auto waiter =
      std::async(std::launch::async, [&] { return t.pool.checkout(); });
  EXPECT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
  failed_run.reset();
  EXPECT_EQ(&*wait_or_abort(waiter), model);
}

TEST(ModelPoolTest, DeviceFailureWhileWaitingReturnsModel) {
  TestPool t(1);
  FakeModel* model = start_run(t.pool);
  auto waiter =
      std::async(std::launch::async, [&] { return t.pool.checkout(); });
  EXPECT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
  model->fail();
  EXPECT_THROW(wait_or_abort(waiter), std::runtime_error);
  EXPECT_EQ(&*checkout(t.pool), model);
}

TEST(ModelPoolTest, DrainWaitsForEveryInFlightRun) {
  TestPool t(2);
  FakeModel* older = start_run(t.pool);
  FakeModel* newer = start_run(t.pool);
  auto drained = std::async(std::launch::async, [&] { t.pool.drain(); });
  EXPECT_EQ(drained.wait_for(kBlocked), std::future_status::timeout);
  newer->finish();
  EXPECT_EQ(drained.wait_for(kBlocked), std::future_status::timeout);
  older->finish();
  wait_or_abort(drained);
  // Reconfiguring affinity requires every model to be available.
  EXPECT_NO_THROW(t.pool.set_use_stream_affinity(false));
}

TEST(ModelPoolTest, ForEachModelRequiresQuiescentPool) {
  TestPool t(2);
  std::vector<FakeModel*> visited;
  t.pool.for_each_model([&](FakeModel& model) { visited.push_back(&model); });
  EXPECT_EQ(visited, t.models);
  FakeModel* in_flight = start_run(t.pool);
  EXPECT_THROW(t.pool.for_each_model([](FakeModel&) {}), std::runtime_error);
  in_flight->finish();
  auto drained = std::async(std::launch::async, [&] { t.pool.drain(); });
  wait_or_abort(drained);
  {
    auto lease = checkout(t.pool);
    EXPECT_THROW(t.pool.for_each_model([](FakeModel&) {}), std::runtime_error);
  }
  EXPECT_NO_THROW(t.pool.for_each_model([](FakeModel&) {}));
}

TEST(ModelPoolTest, LeaseMarksInFlightOnlyOnce) {
  TestPool t(1);
  auto lease = checkout(t.pool);
  lease.mark_in_flight();
  EXPECT_THROW(lease.mark_in_flight(), std::runtime_error);
  EXPECT_THROW(lease.operator->(), std::runtime_error);
  EXPECT_THROW((void)*lease, std::runtime_error);
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

// Every checkout must return a model that no other thread holds and whose
// device work has finished, and with stream affinity, threads sharing a stream
// must not hold models at the same time.
int concurrent_checkout_violations(bool use_stream_affinity) {
  constexpr size_t kNumModels = 3;
  constexpr int kNumThreads = 8;
  constexpr int kNumIters = 500;
  constexpr int kNumStreams = 4;
  TestPool t(kNumModels);
  t.pool.set_use_stream_affinity(use_stream_affinity);
  std::vector<std::atomic<bool>> held(kNumModels);
  std::vector<std::atomic<int>> stream_holders(kNumStreams + 1);
  std::atomic<int> violations{0};
  std::atomic<bool> stop{false};

  std::thread device([&] {
    while (!stop.load()) {
      for (FakeModel* model : t.models) {
        model->finish();
      }
      std::this_thread::yield();
    }
  });

  std::vector<std::thread> workers;
  for (int tid = 0; tid < kNumThreads; ++tid) {
    workers.emplace_back([&, tid] {
      const int stream_id = use_stream_affinity ? 1 + tid % kNumStreams : 0;
      Stream stream = use_stream_affinity ? stream_handle(stream_id) : nullptr;
      for (int i = 0; i < kNumIters; ++i) {
        auto lease = t.pool.checkout(stream);
        auto idx = t.index_of(&*lease);
        if (held[idx].exchange(true) || !lease->is_finished() ||
            (use_stream_affinity && stream_holders[stream_id]++ != 0)) {
          violations++;
        }
        held[idx] = false;
        if (use_stream_affinity) {
          stream_holders[stream_id]--;
        }
        // Every seventh run fails before reaching the device.
        if (i % 7 != 0) {
          lease->start();
          lease.mark_in_flight();
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  stop = true;
  device.join();
  return violations.load();
}

// The stress test is the likeliest to deadlock, so it runs under the same
// watchdog as every blocking call instead of hanging the binary in join().
void run_concurrent_checkouts(bool use_stream_affinity) {
  auto violations = std::async(
      std::launch::async, concurrent_checkout_violations, use_stream_affinity);
  EXPECT_EQ(wait_or_abort(violations), 0);
}

TEST(ModelPoolTest, ConcurrentCheckoutsNeverShareModel) {
  run_concurrent_checkouts(/*use_stream_affinity=*/false);
}

// The pool supports stream affinity only on device runtimes, and
// CMakeLists.txt compiles this file with the build's device macro.
#if defined(USE_CUDA) || defined(USE_XPU)

TEST(ModelPoolTest, StreamAffinityKeepsModelPerStream) {
  TestPool t(2);
  t.pool.set_use_stream_affinity(true);
  FakeModel* first = start_run(t.pool, stream_handle(1));
  FakeModel* second = start_run(t.pool, stream_handle(2));
  EXPECT_NE(first, second);
  first->finish();
  second->finish();
  EXPECT_EQ(start_run(t.pool, stream_handle(1)), first);
  EXPECT_EQ(
      t.pool.get_stream_affinity_model_index(stream_handle(1)),
      t.index_of(first));
  EXPECT_EQ(
      t.pool.get_stream_affinity_model_index(stream_handle(2)),
      t.index_of(second));
}

TEST(ModelPoolTest, StreamAffinityRebindsWhenStreamsOutnumberModels) {
  TestPool t(1);
  t.pool.set_use_stream_affinity(true);
  FakeModel* model = start_run(t.pool, stream_handle(1));
  model->finish();
  EXPECT_EQ(start_run(t.pool, stream_handle(2)), model);
  EXPECT_EQ(
      t.pool.get_stream_affinity_model_index(stream_handle(1)),
      AOTI_STREAM_AFFINITY_UNBOUND);
  EXPECT_EQ(t.pool.get_stream_affinity_model_index(stream_handle(2)), 0);
}

// A failed run clears its model's stream binding, so the stream's next call
// may choose another model. The run can fail before reaching the device, or its
// device work can fail while a later call waits for it.
TEST(ModelPoolTest, FailedRunClearsStreamBinding) {
  {
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
  // The waiting call is stream 1's next one, or stream 2's, which finds no
  // model and waits on the oldest in-flight run.
  for (uintptr_t waiter : {1, 2}) {
    SCOPED_TRACE(::testing::Message() << "stream " << waiter << " waits");
    TestPool t(1);
    t.pool.set_use_stream_affinity(true);
    FakeModel* model = start_run(t.pool, stream_handle(1));
    model->fail();
    EXPECT_THROW(checkout(t.pool, stream_handle(waiter)), std::runtime_error);
    EXPECT_EQ(
        t.pool.get_stream_affinity_model_index(stream_handle(1)),
        AOTI_STREAM_AFFINITY_UNBOUND);
    EXPECT_EQ(&*checkout(t.pool), model);
  }
}

// When every available model is bound, a checkout without a stream borrows
// one. The binding survives: the owning stream waits for the borrower, then for
// the borrowed run's device work, and gets its model back.
TEST(ModelPoolTest, BorrowedModelKeepsStreamBinding) {
  TestPool t(1);
  t.pool.set_use_stream_affinity(true);
  FakeModel* model = start_run(t.pool, stream_handle(1));
  model->finish();
  auto borrowed = checkout(t.pool);
  EXPECT_EQ(&*borrowed, model);
  EXPECT_EQ(t.pool.get_stream_affinity_model_index(stream_handle(1)), 0);
  auto owner = std::async(
      std::launch::async, [&] { return t.pool.checkout(stream_handle(1)); });
  EXPECT_EQ(owner.wait_for(kBlocked), std::future_status::timeout);
  borrowed->start();
  borrowed.mark_in_flight();
  // The borrowed run is still on the device.
  EXPECT_EQ(owner.wait_for(kBlocked), std::future_status::timeout);
  model->finish();
  auto owned = wait_or_abort(owner);
  EXPECT_EQ(&*owned, model);
  EXPECT_EQ(t.pool.get_stream_affinity_model_index(stream_handle(1)), 0);
}

TEST(ModelPoolTest, SameStreamCallsSerialize) {
  TestPool t(2);
  t.pool.set_use_stream_affinity(true);
  std::optional<Pool::Lease> first(checkout(t.pool, stream_handle(1)));
  FakeModel* model = &**first;
  auto same_stream = std::async(
      std::launch::async, [&] { return t.pool.checkout(stream_handle(1)); });
  auto other_stream = std::async(
      std::launch::async, [&] { return t.pool.checkout(stream_handle(2)); });
  EXPECT_NE(&*wait_or_abort(other_stream), model);
  EXPECT_EQ(same_stream.wait_for(kBlocked), std::future_status::timeout);
  (*first)->start();
  first->mark_in_flight();
  // The first run is still on the device.
  EXPECT_EQ(same_stream.wait_for(kBlocked), std::future_status::timeout);
  model->finish();
  EXPECT_EQ(&*wait_or_abort(same_stream), model);
}

// Both calls find no model bound to the stream and wait on a different
// in-flight run; they must still end up sharing one model.
TEST(ModelPoolTest, SameStreamFirstUsesShareOneModel) {
  TestPool t(2);
  t.pool.set_use_stream_affinity(true);
  FakeModel* older = start_run(t.pool);
  FakeModel* newer = start_run(t.pool);
  auto first = std::async(
      std::launch::async, [&] { return t.pool.checkout(stream_handle(1)); });
  EXPECT_EQ(first.wait_for(kBlocked), std::future_status::timeout);
  auto second = std::async(
      std::launch::async, [&] { return t.pool.checkout(stream_handle(1)); });
  EXPECT_EQ(second.wait_for(kBlocked), std::future_status::timeout);
  newer->finish();
  auto second_lease = wait_or_abort(second);
  EXPECT_EQ(&*second_lease, newer);
  older->finish();
  EXPECT_EQ(first.wait_for(kBlocked), std::future_status::timeout);
  second_lease.mark_in_flight();
  EXPECT_EQ(&*wait_or_abort(first), newer);
}

// Returning the only model makes the pool quiescent, so affinity can be
// enabled while a checkout still waits; the checkout must follow the new mode.
// Reconfiguration has to win the pool mutex against the waking checkout, which
// is up to the scheduler, so the scenario repeats until it does and the test
// skips if it never does.
TEST(ModelPoolTest, CheckoutFollowsAffinityEnabledWhileWaiting) {
  for (int attempt = 0; attempt < 50; ++attempt) {
    TestPool t(1);
    std::optional<Pool::Lease> held(checkout(t.pool));
    auto waiter = std::async(
        std::launch::async, [&] { return t.pool.checkout(stream_handle(1)); });
    EXPECT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
    held.reset();
    bool enabled = true;
    try {
      t.pool.set_use_stream_affinity(true);
    } catch (const std::runtime_error&) {
      enabled = false;
    }
    auto lease = wait_or_abort(waiter);
    if (enabled) {
      EXPECT_EQ(t.pool.get_stream_affinity_model_index(stream_handle(1)), 0);
      return;
    }
  }
  GTEST_SKIP() << "Reconfiguration never won the pool mutex while the "
                  "checkout waited, so the interleaving was not exercised";
}

TEST(ModelPoolTest, StreamAffinityReportsIndexSentinels) {
  TestPool t(1);
  EXPECT_EQ(
      t.pool.get_stream_affinity_model_index(stream_handle(1)),
      AOTI_STREAM_AFFINITY_DISABLED);
  t.pool.set_use_stream_affinity(true);
  EXPECT_EQ(
      t.pool.get_stream_affinity_model_index(stream_handle(1)),
      AOTI_STREAM_AFFINITY_UNBOUND);
  EXPECT_EQ(
      t.pool.get_stream_affinity_model_index(nullptr),
      AOTI_STREAM_AFFINITY_UNBOUND);
}

TEST(ModelPoolTest, StreamAffinityReconfiguresOnlyWhileIdle) {
  TestPool t(1);
  {
    auto lease = checkout(t.pool);
    EXPECT_THROW(t.pool.set_use_stream_affinity(true), std::runtime_error);
    lease->start();
    lease.mark_in_flight();
  }
  EXPECT_THROW(t.pool.set_use_stream_affinity(true), std::runtime_error);
  t.models[0]->finish();
  // Reclaiming the finished run makes the pool idle again.
  checkout(t.pool);
  t.pool.set_use_stream_affinity(true);
  EXPECT_TRUE(t.pool.use_stream_affinity());
}

TEST(ModelPoolTest, ConcurrentCheckoutsNeverShareModelWithStreamAffinity) {
  run_concurrent_checkouts(/*use_stream_affinity=*/true);
}

#else

TEST(ModelPoolTest, StreamAffinityRequiresDeviceRuntime) {
  TestPool t(1);
  EXPECT_THROW(t.pool.set_use_stream_affinity(true), std::runtime_error);
  EXPECT_NO_THROW(t.pool.set_use_stream_affinity(false));
  EXPECT_EQ(
      t.pool.get_stream_affinity_model_index(stream_handle(1)),
      AOTI_STREAM_AFFINITY_DISABLED);
}

#endif

} // namespace
} // namespace torch::aot_inductor
