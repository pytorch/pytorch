#include <condition_variable>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_set>

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/orchestration/observer.h>

// Reproducer for a cross-session use-after-free in the global RecordFunction
// callback path. The narrow write-section guard keeps a callback's critical
// section from overlapping teardown, but it intentionally does not span the op
// body (that would deadlock on the GIL). So an op can:
//
//   1. Enter in session A.
//   2. Have A torn down (freeing its RecordQueue event) while in-flight.
//   3. Exit while a LATER session B is active.
//
// The gate is open for B, so the exit is not skipped and onFunctionExitGlobal
// dereferences A's freed event. We tag each ObserverContext with the session
// generation and drop the exit when it no longer matches.
//
// at::RecordFunction's destructor runs the captured exit callback even after
// A's callback was removed.

namespace {
using torch::profiler::impl::ActivityType;
using torch::profiler::impl::ExperimentalConfig;
using torch::profiler::impl::ProfilerConfig;
using torch::profiler::impl::ProfilerState;

ProfilerConfig makeGlobalCallbackCpuConfig() {
  ExperimentalConfig experimental_config;
  experimental_config.profile_all_threads = true;
  return ProfilerConfig(
      ProfilerState::KINETO,
      /*report_input_shapes=*/false,
      /*profile_memory=*/false,
      /*with_stack=*/false,
      /*with_flops=*/false,
      /*with_modules=*/false,
      experimental_config);
}

void enableGlobalProfiler() {
  const auto config = makeGlobalCallbackCpuConfig();
  const std::set<ActivityType> activities{ActivityType::CPU};
  const std::unordered_set<at::RecordScope> scopes{at::RecordScope::FUNCTION};
  torch::autograd::profiler::prepareProfiler(config, activities);
  torch::autograd::profiler::enableProfiler(config, activities, scopes);
}
} // namespace

TEST(
    ProfilerGlobalCallbackStaleSessionTest,
    ExitAfterSessionTornDownIsDropped) {
  at::clearCallbacks();

  std::mutex mtx;
  std::condition_variable cv;
  bool worker_entered = false; // guarded by mtx
  bool release_worker = false; // guarded by mtx

  enableGlobalProfiler(); // session A

  std::thread worker([&]() {
    at::RecordFunction guard(at::RecordScope::FUNCTION);
    guard.before("stale_probe_op"); // onFunctionEnterGlobal: ctx tagged gen(A)
    {
      std::unique_lock<std::mutex> lock(mtx);
      worker_entered = true;
      cv.notify_all();
      cv.wait(lock, [&] { return release_worker; });
    }
    // guard destructor -> onFunctionExitGlobal, now during session B
  });

  {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] { return worker_entered; });
  }

  // Tear down A (frees its RecordQueue, including the in-flight op's event),
  // then bring up a fresh session B so the pending exit lands with the gate
  // open.
  torch::autograd::profiler::disableProfiler();
  enableGlobalProfiler(); // session B

  // Release the op: its exit runs in B with a ctx tagged gen(A). This should
  // be safe because of generation checks.
  {
    std::lock_guard<std::mutex> lock(mtx);
    release_worker = true;
  }
  cv.notify_all();
  worker.join();

  torch::autograd::profiler::disableProfiler();
  EXPECT_FALSE(at::hasGlobalCallbacks());
}
