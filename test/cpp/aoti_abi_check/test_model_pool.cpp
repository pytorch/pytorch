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
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <torch/csrc/inductor/aoti_runtime/model_pool.h>

namespace torch::aot_inductor {
namespace {

// Pool bugs tend to show up as hangs; fail loudly, naming the test, instead of
// waiting for the test runner's timeout.
constexpr auto kHang = std::chrono::seconds(30);

[[noreturn]] void abort_on_hang() {
  const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
  std::cerr << "ModelPool test " << (test ? test->name() : "(none)")
            << " timed out, likely a deadlock\n";
  std::abort();
}

// Stands in for AOTInductorModel. start() puts work "on the device"; the test
// completes it with finish() or fail().
class FakeModel {
 public:
  bool is_finished() {
    std::lock_guard lk(mutex_);
    AOTI_RUNTIME_CHECK(!status_query_fails_, "status query failed");
    return state_ == State::kIdle;
  }

  void wait_for_completion() {
    std::unique_lock lk(mutex_);
    waited_on_ = true;
    cv_.notify_all();
    cv_.wait(lk, [this] { return state_ != State::kRunning; });
    const bool failed = state_ == State::kFailed;
    state_ = State::kIdle;
    AOTI_RUNTIME_CHECK(!failed, "device failure");
  }

  // Returns once a thread is waiting for this model's device work.
  void wait_until_waited_on() {
    std::unique_lock lk(mutex_);
    if (!cv_.wait_for(lk, kHang, [this] { return waited_on_; })) {
      abort_on_hang();
    }
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
  // Makes is_finished() throw, as AOTInductorModel's does on a CUDA error,
  // until the next state change.
  void fail_status_query() {
    std::lock_guard lk(mutex_);
    status_query_fails_ = true;
  }

 private:
  enum class State { kIdle, kRunning, kFailed };

  void set_state(State state) {
    {
      std::lock_guard lk(mutex_);
      state_ = state;
      status_query_fails_ = false;
    }
    cv_.notify_all();
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  State state_{State::kIdle};
  bool waited_on_{false};
  bool status_query_fails_{false};
};

using Stream = void*;
using Pool = ModelPool<FakeModel, Stream>;

// Only checkout() creates leases: the constructor is private and takes the
// pool's private slot type, so a caller can neither forge a lease nor wrap a
// bare model in one.
static_assert(
    !std::is_constructible_v<Pool::Lease, Pool&, FakeModel*>,
    "A lease must not be constructible from a bare model");

template <typename T, typename = void>
constexpr bool kHasArrow = false;
template <typename T>
constexpr bool
    kHasArrow<T, std::void_t<decltype(std::declval<T>().operator->())>> = true;

static_assert(kHasArrow<Pool::Lease&>);
static_assert(kHasArrow<const Pool::Lease&>);
static_assert(
    !kHasArrow<Pool::Lease>,
    "A temporary lease must not give access to its model");
static_assert(
    !kHasArrow<const Pool::Lease>,
    "A const temporary lease must not give access to its model");

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

template <typename T>
T wait_or_abort(std::future<T>& future) {
  if (future.wait_for(kHang) != std::future_status::ready) {
    abort_on_hang();
  }
  return future.get();
}

// Returns the message of the std::runtime_error fn throws, as every
// AOTI_RUNTIME_CHECK does, or "" if it throws none.
template <typename Fn>
std::string runtime_error_of(Fn&& fn) {
  try {
    fn();
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

[[nodiscard]] Pool::Lease checkout(Pool& pool, Stream stream = nullptr) {
  auto future =
      std::async(std::launch::async, [&] { return pool.checkout(stream); });
  return wait_or_abort(future);
}

// Starts a checkout that should block on the pool's condition variable. Nothing
// signals from inside that wait, so this checks only that the checkout hasn't
// returned within kBlocked of its thread calling it.
constexpr auto kBlocked = std::chrono::milliseconds(100);

std::future<Pool::Lease> blocked_checkout(Pool& pool) {
  std::promise<void> entering;
  auto entered = entering.get_future();
  auto lease = std::async(
      std::launch::async, [&pool, entering = std::move(entering)]() mutable {
        entering.set_value();
        return pool.checkout();
      });
  entered.wait();
  EXPECT_EQ(lease.wait_for(kBlocked), std::future_status::timeout);
  return lease;
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
  // Aborts if the checkout waits on another run.
  oldest->wait_until_waited_on();
  oldest->finish();
  EXPECT_EQ(model_of(wait_or_abort(waiter)), oldest);
  newest->finish();
}

// A checkout that finds no available model reclaims every finished run, oldest
// first, so it reuses the most recently launched finished run.
TEST(ModelPoolTest, ReusesNewestFinishedRun) {
  TestPool t(3);
  FakeModel* oldest = start_run(t.pool);
  FakeModel* middle = start_run(t.pool);
  FakeModel* newest = start_run(t.pool);
  oldest->finish();
  middle->finish();
  EXPECT_EQ(model_of(checkout(t.pool)), middle);
  newest->finish();
}

// A checkout whose reclaim stops partway, because a model's status query
// throws, leaves every model in exactly one state: none is handed out twice.
TEST(ModelPoolTest, FailedStatusQueryKeepsPoolConsistent) {
  TestPool t(3);
  FakeModel* oldest = start_run(t.pool);
  FakeModel* middle = start_run(t.pool);
  FakeModel* newest = start_run(t.pool);
  oldest->finish();
  newest->fail_status_query();
  EXPECT_EQ(
      runtime_error_of([&] { (void)checkout(t.pool); }), "status query failed");
  middle->finish();
  newest->finish();
  std::vector<Pool::Lease> leases;
  std::set<FakeModel*> held;
  for (size_t i = 0; i < t.models.size(); ++i) {
    leases.push_back(checkout(t.pool));
    held.insert(model_of(leases.back()));
  }
  EXPECT_EQ(held.size(), t.models.size());
  auto waiter = blocked_checkout(t.pool);
  leases.clear();
  wait_or_abort(waiter);
}

// A run that throws drops its lease, which returns the model to the available
// pool without waiting for the device work the run launched: a checkout waiting
// for the pool's only model gets it while that work is still running.
TEST(ModelPoolTest, FailedRunWakesWaitingCheckout) {
  TestPool t(1);
  auto failed_run = checkout(t.pool);
  auto waiter = blocked_checkout(t.pool);
  // On another thread, so that the test aborts instead of hanging if dropping
  // the lease waits for the run's device work, which never finishes here.
  auto failing =
      std::async(std::launch::async, [&] { fail_run(std::move(failed_run)); });
  EXPECT_EQ(runtime_error_of([&] { wait_or_abort(failing); }), "run failed");
  EXPECT_EQ(model_of(wait_or_abort(waiter)), t.models[0]);
}

// A device failure found while waiting for an in-flight model returns the model
// to the available pool, so the next checkout gets it.
TEST(ModelPoolTest, DeviceFailureReturnsModelToPool) {
  TestPool t(1);
  start_run(t.pool)->fail();
  EXPECT_EQ(
      runtime_error_of([&] { (void)checkout(t.pool); }), "device failure");
  EXPECT_EQ(model_of(checkout(t.pool)), t.models[0]);
}

TEST(ModelPoolTest, LeaseMarksInFlightOnlyOnce) {
  TestPool t(1);
  auto lease = checkout(t.pool);
  lease.mark_in_flight();
  EXPECT_EQ(
      runtime_error_of([&] { lease.mark_in_flight(); }),
      "The lease no longer holds a model");
  EXPECT_EQ(
      runtime_error_of([&] { (void)lease.operator->(); }),
      "The lease no longer holds a model");
}

TEST(ModelPoolTest, MovedFromLeaseDoesNotReturnModel) {
  TestPool t(1);
  std::optional<Pool::Lease> holder;
  {
    auto lease = checkout(t.pool);
    holder.emplace(std::move(lease));
    EXPECT_EQ(
        // NOLINTNEXTLINE(bugprone-use-after-move)
        runtime_error_of([&] { (void)lease.operator->(); }),
        "The lease no longer holds a model");
  }
  auto waiter = blocked_checkout(t.pool);
  holder.reset();
  wait_or_abort(waiter);
}

// The pool supports stream affinity only on device runtimes, and
// CMakeLists.txt compiles this file with the build's device macro. A ROCm build
// hipifies model_pool.h, turning its USE_CUDA into USE_ROCM, but not this file.
#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_XPU)

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
  {
    // Dropping the lease without mark_in_flight() is a failed run.
    auto lease = checkout(t.pool, stream_handle(1));
  }
  EXPECT_EQ(
      t.pool.get_stream_affinity_model_index(stream_handle(1)),
      AOTI_STREAM_AFFINITY_UNBOUND);
}

// A device failure found while a stream waits for its bound model clears the
// binding and returns the model to the available pool.
TEST(ModelPoolTest, DeviceFailureClearsStreamBinding) {
  TestPool t(1);
  t.pool.set_use_stream_affinity(true);
  start_run(t.pool, stream_handle(1))->fail();
  EXPECT_EQ(
      runtime_error_of([&] { (void)checkout(t.pool, stream_handle(1)); }),
      "device failure");
  EXPECT_EQ(
      t.pool.get_stream_affinity_model_index(stream_handle(1)),
      AOTI_STREAM_AFFINITY_UNBOUND);
  EXPECT_EQ(model_of(checkout(t.pool, stream_handle(1))), t.models[0]);
}

#endif

} // namespace
} // namespace torch::aot_inductor
