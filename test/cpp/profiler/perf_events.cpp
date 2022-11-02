
#include <gtest/gtest.h>

#include <torch/csrc/profiler/events.h>
#include <torch/csrc/profiler/perf.h>

double calc_pi() {
  volatile double pi = 1.0;
  for (int i = 3; i < 100000; i += 2) {
    pi += (((i + 1) >> 1) % 2) ? 1.0 / i : -1.0 / i;
  }
  return pi * 4.0;
}

TEST(ProfilerTest, LinuxPerf) {
  torch::profiler::impl::linux_perf::PerfProfiler profiler;

  std::vector<std::string> standard_events(
      std::begin(torch::profiler::ProfilerPerfEvents),
      std::end(torch::profiler::ProfilerPerfEvents));
  torch::profiler::perf_counters_t counters;
  counters.resize(standard_events.size(), 0);

  // Use try..catch HACK to check TORCH_CHECK because we don't yet fail
  // gracefully if the syscall were to fail
  try {
    profiler.Configure(standard_events);

    profiler.Enable();
    auto pi = calc_pi();
    profiler.Disable(counters);
  } catch (const c10::Error&) {
    // Bail here if something bad happened during the profiling, we don't want
    // to make the test fail
    return;
  } catch (...) {
    // something else went wrong - this should be reported
    ASSERT_EQ(0, 1);
  }

  // Should have counted something if worked, so lets test that
  // And if it not supported the counters should be zeros.
#if defined(__ANDROID__) || defined(__linux__)
  for (auto counter : counters) {
    ASSERT_GT(counter, 0);
  }
#else /* __ANDROID__ || __linux__ */
  for (auto counter : counters) {
    ASSERT_EQ(counter, 0);
  }
#endif /* __ANDROID__ || __linux__ */
}
