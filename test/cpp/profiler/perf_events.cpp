
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
    calc_pi();
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

TEST(ProfilerTest, LinuxPerfNestedDepth) {
  torch::profiler::impl::linux_perf::PerfProfiler profiler;

  // Only monotonically increasing events will work
  std::vector<std::string> standard_events(
      std::begin(torch::profiler::ProfilerPerfEvents),
      std::end(torch::profiler::ProfilerPerfEvents));

  torch::profiler::perf_counters_t counters_A;
  torch::profiler::perf_counters_t counters_B;
  torch::profiler::perf_counters_t counters_C;

  counters_A.resize(standard_events.size(), 0);
  counters_B.resize(standard_events.size(), 0);
  counters_C.resize(standard_events.size(), 0);

  // Use try..catch HACK to check TORCH_CHECK because we don't yet fail
  // gracefully if the syscall were to fail
  try {
    profiler.Configure(standard_events);

    // * = work kernel calc_pi()
    //
    // A --*---+              +--*-- A
    //         |              |
    //         |              |
    //       B +-*--+    +--*-+ B
    //              |    |
    //              |    |
    //            C +-*--+ C
    //

    profiler.Enable();
    calc_pi();

    profiler.Enable();
    calc_pi();

    profiler.Enable();
    calc_pi();
    profiler.Disable(counters_C);

    calc_pi();
    profiler.Disable(counters_B);

    calc_pi();
    profiler.Disable(counters_A);
  } catch (const c10::Error&) {
    // Bail here if something bad happened during the profiling, we don't want
    // to make the test fail
    return;
  } catch (...) {
    // something else went wrong - this should be reported
    ASSERT_EQ(0, 1);
  }

// for each counter, assert A > B > C
#if defined(__ANDROID__) || defined(__linux__)
  for (auto i = 0; i < standard_events.size(); ++i) {
    ASSERT_GT(counters_A[i], counters_B[i]);
    ASSERT_GT(counters_A[i], counters_C[i]);
    ASSERT_GT(counters_B[i], counters_C[i]);
    ASSERT_GT(counters_A[i], counters_B[i] + counters_C[i]);
  }
#else /* __ANDROID__ || __linux__ */
  for (auto i = 0; i < standard_events.size(); ++i) {
    ASSERT_EQ(counters_A[i], 0);
    ASSERT_EQ(counters_B[i], 0);
    ASSERT_EQ(counters_C[i], 0);
  }
#endif /* __ANDROID__ || __linux__ */
}

TEST(ProfilerTest, LinuxPerfNestedMultiple) {
  torch::profiler::impl::linux_perf::PerfProfiler profiler;

  // Only monotonically increasing events will work
  std::vector<std::string> standard_events(
      std::begin(torch::profiler::ProfilerPerfEvents),
      std::end(torch::profiler::ProfilerPerfEvents));

  torch::profiler::perf_counters_t counters_A;
  torch::profiler::perf_counters_t counters_B;
  torch::profiler::perf_counters_t counters_C;

  counters_A.resize(standard_events.size(), 0);
  counters_B.resize(standard_events.size(), 0);
  counters_C.resize(standard_events.size(), 0);

  // Use try..catch HACK to check TORCH_CHECK because we don't yet fail
  // gracefully if the syscall were to fail
  try {
    profiler.Configure(standard_events);

    // * = work kernel calc_pi()
    //
    // A --*---+    +---*----+    +--*-- A
    //         |    |        |    |
    //         |    |        |    |
    //      B  +-**-+ B    C +-*--+ C

    profiler.Enable();
    calc_pi();

    profiler.Enable();
    calc_pi();
    calc_pi();
    profiler.Disable(counters_B);

    calc_pi();

    profiler.Enable();
    calc_pi();
    profiler.Disable(counters_C);

    calc_pi();
    profiler.Disable(counters_A);
  } catch (const c10::Error&) {
    // Bail here if something bad happened during the profiling, we don't want
    // to make the test fail
    return;
  } catch (...) {
    // something else went wrong - this should be reported
    ASSERT_EQ(0, 1);
  }

// for each counter, assert A > B > C
#if defined(__ANDROID__) || defined(__linux__)
  for (auto i = 0; i < standard_events.size(); ++i) {
    ASSERT_GT(counters_A[i], counters_B[i]);
    ASSERT_GT(counters_A[i], counters_C[i]);
    ASSERT_GT(counters_B[i], counters_C[i]);
    ASSERT_GT(counters_A[i], counters_B[i] + counters_C[i]);
  }
#else /* __ANDROID__ || __linux__ */
  for (auto i = 0; i < standard_events.size(); ++i) {
    ASSERT_EQ(counters_A[i], 0);
    ASSERT_EQ(counters_B[i], 0);
    ASSERT_EQ(counters_C[i], 0);
  }
#endif /* __ANDROID__ || __linux__ */
}

TEST(ProfilerTest, LinuxPerfNestedSingle) {
  torch::profiler::impl::linux_perf::PerfProfiler profiler;

  // Only monotonically increasing events will work
  std::vector<std::string> standard_events(
      std::begin(torch::profiler::ProfilerPerfEvents),
      std::end(torch::profiler::ProfilerPerfEvents));

  torch::profiler::perf_counters_t counters_A;
  torch::profiler::perf_counters_t counters_B;
  torch::profiler::perf_counters_t counters_C;

  counters_A.resize(standard_events.size(), 0);
  counters_B.resize(standard_events.size(), 0);
  counters_C.resize(standard_events.size(), 0);

  // Use try..catch HACK to check TORCH_CHECK because we don't yet fail
  // gracefully if the syscall were to fail
  try {
    profiler.Configure(standard_events);

    profiler.Enable();
    profiler.Enable();
    profiler.Enable();
    calc_pi();
    profiler.Disable(counters_C);
    profiler.Disable(counters_B);
    profiler.Disable(counters_A);
  } catch (const c10::Error&) {
    // Bail here if something bad happened during the profiling, we don't want
    // to make the test fail
    return;
  } catch (...) {
    // something else went wrong - this should be reported
    ASSERT_EQ(0, 1);
  }

// for each counter, assert A > B > C
#if defined(__ANDROID__) || defined(__linux__)
  for (auto i = 0; i < standard_events.size(); ++i) {
    ASSERT_GE(counters_A[i], counters_B[i]);
    ASSERT_GE(counters_A[i], counters_C[i]);
    ASSERT_GE(counters_B[i], counters_C[i]);
  }
#else /* __ANDROID__ || __linux__ */
  for (auto i = 0; i < standard_events.size(); ++i) {
    ASSERT_EQ(counters_A[i], 0);
    ASSERT_EQ(counters_B[i], 0);
    ASSERT_EQ(counters_C[i], 0);
  }
#endif /* __ANDROID__ || __linux__ */
}
