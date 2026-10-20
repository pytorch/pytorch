#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
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

struct TestPool {
  explicit TestPool(size_t num_models) : models(num_models) {
    pool.reserve(num_models);
    for (auto& model : models) {
      pool.add_model(&model);
    }
  }

  int64_t index_of(const FakeModel* model) const {
    return model - models.data();
  }

  std::vector<FakeModel> models;
  Pool pool;
};

Stream stream_handle(uintptr_t id) {
  return reinterpret_cast<Stream>(id);
}

constexpr auto kBlocked = std::chrono::milliseconds(100);

// Pool bugs tend to show up as hangs; fail loudly instead of waiting for the
// test runner's timeout.
template <typename T>
T wait_or_abort(std::future<T>& future) {
  if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
    std::cerr << "ModelPool test timed out, likely a deadlock\n";
    std::abort();
  }
  return future.get();
}

FakeModel* checkout(Pool& pool, Stream stream = nullptr) {
  auto future = std::async(
      std::launch::async, [&] { return pool.get_available_model(stream); });
  return wait_or_abort(future);
}

// Mirrors a run() that reached the device: the model goes back as pending.
FakeModel* start_run(Pool& pool, Stream stream = nullptr) {
  FakeModel* model = checkout(pool, stream);
  model->start();
  pool.return_model_to_pending(model);
  return model;
}

TEST(ModelPoolTest, ReusesMostRecentlyReturnedModel) {
  TestPool t(3);
  FakeModel* model = checkout(t.pool);
  EXPECT_EQ(model, &t.models[2]);
  t.pool.return_model_to_available(model);
  EXPECT_EQ(checkout(t.pool), model);
}

TEST(ModelPoolTest, PrefersFinishedRunOverWaiting) {
  TestPool t(2);
  start_run(t.pool);
  FakeModel* finished = start_run(t.pool);
  finished->finish();
  EXPECT_EQ(checkout(t.pool), finished);
}

TEST(ModelPoolTest, WaitsForOldestPendingRun) {
  TestPool t(2);
  FakeModel* oldest = start_run(t.pool);
  FakeModel* newest = start_run(t.pool);
  auto waiter = std::async(
      std::launch::async, [&] { return t.pool.get_available_model(); });
  EXPECT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
  oldest->finish();
  EXPECT_EQ(wait_or_abort(waiter), oldest);
  newest->finish();
}

TEST(ModelPoolTest, FailedRunWakesWaitingCheckout) {
  TestPool t(1);
  FakeModel* model = checkout(t.pool);
  auto waiter = std::async(
      std::launch::async, [&] { return t.pool.get_available_model(); });
  EXPECT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
  t.pool.return_model_to_available(model);
  EXPECT_EQ(wait_or_abort(waiter), model);
}

TEST(ModelPoolTest, DeviceFailureWhileWaitingReturnsModel) {
  TestPool t(1);
  FakeModel* model = start_run(t.pool);
  auto waiter = std::async(
      std::launch::async, [&] { return t.pool.get_available_model(); });
  EXPECT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
  model->fail();
  EXPECT_THROW(wait_or_abort(waiter), std::runtime_error);
  EXPECT_EQ(checkout(t.pool), model);
}

// Every checkout must return a model that no other thread holds and whose
// device work has finished.
void run_concurrent_checkouts(bool use_stream_affinity) {
  constexpr size_t kNumModels = 3;
  constexpr int kNumThreads = 8;
  constexpr int kNumIters = 500;
  TestPool t(kNumModels);
  t.pool.set_use_stream_affinity(use_stream_affinity);
  std::vector<std::atomic<bool>> held(kNumModels);
  std::atomic<int> violations{0};
  std::atomic<bool> stop{false};

  std::thread device([&] {
    while (!stop.load()) {
      for (auto& model : t.models) {
        model.finish();
      }
      std::this_thread::yield();
    }
  });

  std::vector<std::thread> workers;
  for (int tid = 0; tid < kNumThreads; ++tid) {
    workers.emplace_back([&, tid] {
      Stream stream =
          use_stream_affinity ? stream_handle(1 + tid % 4) : nullptr;
      for (int i = 0; i < kNumIters; ++i) {
        auto stream_lock = t.pool.acquire_stream_lock(stream);
        FakeModel* model = t.pool.get_available_model(stream);
        auto idx = t.index_of(model);
        if (held[idx].exchange(true) || !model->is_finished()) {
          violations++;
        }
        held[idx] = false;
        if (i % 7 == 0) {
          t.pool.return_model_to_available(model);
        } else {
          model->start();
          t.pool.return_model_to_pending(model);
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  stop = true;
  device.join();
  EXPECT_EQ(violations.load(), 0);
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

TEST(ModelPoolTest, StreamLockSerializesCallsOnSameStream) {
  TestPool t(2);
  t.pool.set_use_stream_affinity(true);
  std::future<void> same_stream;
  {
    auto held = t.pool.acquire_stream_lock(stream_handle(1));
    same_stream = std::async(std::launch::async, [&] {
      auto lock = t.pool.acquire_stream_lock(stream_handle(1));
    });
    auto other_stream = std::async(std::launch::async, [&] {
      auto lock = t.pool.acquire_stream_lock(stream_handle(2));
    });
    wait_or_abort(other_stream);
    EXPECT_EQ(same_stream.wait_for(kBlocked), std::future_status::timeout);
  }
  wait_or_abort(same_stream);
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
  FakeModel* model = checkout(t.pool);
  EXPECT_THROW(t.pool.set_use_stream_affinity(true), std::runtime_error);
  model->start();
  t.pool.return_model_to_pending(model);
  EXPECT_THROW(t.pool.set_use_stream_affinity(true), std::runtime_error);
  model->finish();
  // Reclaiming the finished run makes the pool idle again.
  t.pool.return_model_to_available(checkout(t.pool));
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
