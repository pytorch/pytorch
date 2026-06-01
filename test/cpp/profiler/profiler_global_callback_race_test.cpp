#include <algorithm>
#include <atomic>
#include <set>
#include <thread>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/orchestration/observer.h>

// Reproducer for a use-after-free in the profiler's *global* RecordFunction
// callback path.
//
// When the profiler installs global callbacks (ProfilerState::KINETO_ONDEMAND,
// or ProfilerState::KINETO with ExperimentalConfig.profile_all_threads=true),
// enableProfiler() registers the callback via at::addGlobalCallback, so it
// fires on every thread -- including long-lived worker threads (e.g. the
// DataLoader pin_memory thread that crashed in production). onFunctionEnter
// fetches the profiler state through a raw, non-owning pointer and then
// dereferences it:
//
//   auto state_ptr = KinetoThreadLocalState::get(/*global=*/true);
//   if (!state_ptr) return nullptr;
//   return state_ptr->recordQueue.getSubqueue()->begin_op(fn);  // <-- UAF
//
// disableProfiler() (running on another thread) pops and frees that state
// without waiting for in-flight global callbacks to finish, so the worker
// thread can dereference freed memory between the null check and begin_op.
//
// We drive the path with ProfilerState::KINETO + profile_all_threads=true --
// exactly the configuration torch.profiler uses when profile_all_threads is
// requested, which is what crashed in production. This installs the global
// RecordFunction callback (instantiating onFunctionEnter<true>/
// onFunctionExit<true>) and tears it down through disableProfiler().
//
// The test continuously dispatches ATen ops on worker threads while the main
// thread repeatedly enables and disables the profiler. On unfixed code this
// segfaults / trips ASAN heap-use-after-free, usually within a few hundred
// iterations. The pass condition is simply that the process runs to completion
// without crashing.

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

} // namespace

TEST(ProfilerGlobalCallbackRaceTest, DisableDuringConcurrentDispatch) {
  at::clearCallbacks();

  constexpr int kIterations = 2000;
  const unsigned int kNumWorkers =
      std::max(3u, std::min(8u, std::thread::hardware_concurrency()));

  std::atomic<bool> stop{false};
  std::atomic<unsigned int> workers_ready{0};

  std::vector<std::thread> workers;
  workers.reserve(kNumWorkers);
  for (unsigned int i = 0; i < kNumWorkers; ++i) {
    workers.emplace_back([&]() {
      auto a = at::ones({8});
      auto b = at::ones({8});
      workers_ready.fetch_add(1);
      // Continuously dispatch an op that runs RecordFunction (FUNCTION scope),
      // so the global profiler callback fires on this thread while it is alive.
      while (!stop.load(std::memory_order_relaxed)) {
        // at::add dispatches through the dispatcher and runs RecordFunction;
        // the call crosses the library boundary so it is not optimized away.
        auto c = a + b;
        (void)c;
      }
    });
  }

  // Wait for all workers to be actively dispatching before we start toggling.
  while (workers_ready.load() < kNumWorkers) {
    std::this_thread::yield();
  }

  const std::set<ActivityType> activities{ActivityType::CPU};
  const std::unordered_set<at::RecordScope> scopes{at::RecordScope::FUNCTION};
  for (int i = 0; i < kIterations; ++i) {
    const auto config = makeGlobalCallbackCpuConfig();
    torch::autograd::profiler::prepareProfiler(config, activities);
    torch::autograd::profiler::enableProfiler(config, activities, scopes);
    // Keep the enabled window short so disable overlaps with callbacks that
    // are in flight on the worker threads -- this is the race window.
    torch::autograd::profiler::disableProfiler();
  }

  stop.store(true);
  for (auto& worker : workers) {
    worker.join();
  }

  // After a clean teardown there must be no dangling global callback.
  EXPECT_FALSE(at::hasGlobalCallbacks());
}
