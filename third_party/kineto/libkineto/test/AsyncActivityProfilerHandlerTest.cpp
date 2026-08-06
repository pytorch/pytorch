/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <cstdlib>

#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <mutex>

#include "include/Config.h"
#include "include/libkineto.h"
#include "src/AsyncActivityProfilerHandler.h"
#include "src/GenericActivityProfiler.h"

#include "src/Logger.h"
#include "test/TestUtils.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;
using namespace libkineto::test;

// Several tests step to timestamps derived from the warmup and duration values
// they pass in the config, using named constants shared between the two so they
// cannot drift apart. startTime leads now by the warmup period plus a
// one-second buffer: canStart() requires at least ACTIVITIES_WARMUP_PERIOD_SECS
// of lead time, and the buffer absorbs the millisecond truncation in the
// PROFILE_START_TIME round-trip (a bare now + warmup can round just under and
// be rejected). Steps meant to fall after collection add a second past
// startTime + ACTIVITIES_DURATION_SECS.

// Subclass that lets us control isGpuCollectionStopped() for testing
// the buffer-overflow-during-warmup path without needing real CUPTI.
class MockGpuProfiler : public GenericActivityProfiler {
 public:
  MockGpuProfiler() : GenericActivityProfiler(/*cpuOnly=*/false) {}

  void setGpuCollectionStopped(bool stopped) {
    gpuStopped_ = stopped;
  }

 protected:
  bool isGpuCollectionStopped() const override {
    return gpuStopped_;
  }

 private:
  bool gpuStopped_{false};
};

// Records ClientInterface callbacks so tests can assert the handler drives the
// registered client. Counters are atomic because the memory-snapshot path
// invokes them from a background thread.
class MockClientInterface : public libkineto::ClientInterface {
 public:
  void init() override {}
  void prepare(
      bool /*unused*/,
      bool /*unused*/,
      bool /*unused*/,
      bool /*unused*/,
      bool /*unused*/) override {}
  void start() override {
    ++startCount;
  }
  void stop() override {
    ++stopCount;
  }
  void start_memory_profile() override {
    ++memoryStartCount;
  }
  void stop_memory_profile() override {
    ++memoryStopCount;
    // set_value() throws std::future_error if the promise is already satisfied,
    // which on this background thread would terminate the test process. Signal
    // exactly once so a repeated call surfaces via memoryStopCount instead of
    // throwing.
    if (!memoryStopSignaled_.exchange(true)) {
      memoryStopPromise_.set_value();
    }
  }
  void export_memory_profile(const std::string& path) override {
    ++memoryExportCount;
    std::scoped_lock guard(mutex_);
    exportedPath_ = path;
  }

  std::string exportedPath() {
    std::scoped_lock guard(mutex_);
    return exportedPath_;
  }

  // Fulfilled when stop_memory_profile() runs, i.e. the memory loop finished.
  std::future<void> memoryStopFuture() {
    return memoryStopPromise_.get_future();
  }

  std::atomic<int> startCount{0};
  std::atomic<int> stopCount{0};
  std::atomic<int> memoryStartCount{0};
  std::atomic<int> memoryStopCount{0};
  std::atomic<int> memoryExportCount{0};

 private:
  std::mutex mutex_;
  std::string exportedPath_;
  std::promise<void> memoryStopPromise_;
  std::atomic<bool> memoryStopSignaled_{false};
};

// Registers a mock client on the process-global libkineto::api() for a test,
// saving and restoring whatever client was registered before so the fixture
// never clobbers another test's client regardless of binary composition.
class AsyncClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    priorClient_ = libkineto::api().client();
    libkineto::api().registerClient(&client_);
  }
  void TearDown() override {
    libkineto::api().registerClient(priorClient_);
  }

  MockClientInterface client_;
  libkineto::ClientInterface* priorClient_ = nullptr;
};

TEST(AsyncActivityProfilerHandler, AsyncTrace) {
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;

  constexpr int kWarmupSecs = 5;
  constexpr int kDurationSecs = 1;
  int iter = 0;
  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  bool success = cfg.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = {}
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
      kWarmupSecs,
      kDurationSecs,
      traceFile.path(),
      duration_cast<milliseconds>(startTime.time_since_epoch()).count()));

  EXPECT_TRUE(success);
  EXPECT_FALSE(handler.isAsyncActive());

  handler.configure(cfg, now);

  EXPECT_TRUE(handler.isAsyncActive());

  // fast forward in time and we have reached the startTime
  now = startTime;

  // Warmup
  handler.performRunLoopStep(now, now);

  // End of the collection window: startTime + duration.
  auto next = startTime + seconds(kDurationSecs);

  // Iteration-based steps should have no effect on timestamp-based config
  while (++iter < 20) {
    handler.performRunLoopStep(now, now, iter);
  }

  // Terminate collection
  handler.performRunLoopStep(next, next);

  EXPECT_TRUE(handler.isAsyncActive());

  // Past the window, to drive the finalize (ProcessTrace) steps.
  auto nextnext = next + seconds(kDurationSecs);

  while (++iter < 40) {
    handler.performRunLoopStep(next, next, iter);
  }

  EXPECT_TRUE(handler.isAsyncActive());

  handler.performRunLoopStep(nextnext, nextnext);
  handler.performRunLoopStep(nextnext, nextnext);

  EXPECT_FALSE(handler.isAsyncActive());

  auto logFile = logUrlToPath(cfg.activitiesLogUrl());
  checkTracefile(logFile.c_str());
}

TEST(AsyncActivityProfilerHandler, AsyncTraceUsingIter) {
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  auto runIterTest = [&](int start_iter, int warmup_iters, int trace_iters) {
    LOG(INFO) << "Async Trace Test: start_iteration = " << start_iter
              << " warmup iterations = " << warmup_iters
              << " trace iterations = " << trace_iters;

    GenericActivityProfiler profiler(/*cpu only*/ true);
    AsyncActivityProfilerHandler handler(profiler);

    auto traceFile = createTempTraceFile("libkineto_test", ".json");

    Config cfg;

    int iter = 0;
    auto now = system_clock::now();

    bool success = cfg.parse(fmt::format(
        R"CFG(
      PROFILE_START_ITERATION = {}
      ACTIVITIES_WARMUP_ITERATIONS={}
      ACTIVITIES_ITERATIONS={}
      ACTIVITIES_DURATION_SECS = 1
      ACTIVITIES_LOG_FILE = {}
    )CFG",
        start_iter,
        warmup_iters,
        trace_iters,
        traceFile.path()));

    EXPECT_TRUE(success);
    EXPECT_FALSE(handler.isAsyncActive());

    while (iter < (start_iter - warmup_iters)) {
      iter++;
    }

    handler.configure(cfg, now);
    EXPECT_TRUE(handler.isAsyncActive());

    now += seconds(10);
    auto next = now + milliseconds(1000);

    handler.performRunLoopStep(now, next);
    EXPECT_TRUE(handler.isAsyncActive());

    while (iter < start_iter) {
      handler.performRunLoopStep(now, next, iter++);
    }

    while (iter < (start_iter + trace_iters)) {
      handler.performRunLoopStep(now, next, iter++);
    }

    if (iter >= (start_iter + trace_iters)) {
      handler.performRunLoopStep(now, next, iter++);
    }
    EXPECT_TRUE(handler.isAsyncActive());

    auto nextnext = next + milliseconds(1000);
    handler.performRunLoopStep(nextnext, nextnext);
    handler.ensureCollectTraceDone();
    handler.performRunLoopStep(nextnext, nextnext);

    EXPECT_FALSE(handler.isAsyncActive());

    auto logFile = logUrlToPath(cfg.activitiesLogUrl());
    checkTracefile(logFile.c_str());
  };

  runIterTest(50, 5, 10);
  runIterTest(0, 0, 2);
  runIterTest(0, 5, 5);
}

TEST(AsyncActivityProfilerHandler, MetadataJsonFormatingTest) {
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  setenv("PT_PROFILER_JOB_NAME", "test_training_job", 1);
  setenv("PT_PROFILER_JOB_VERSION", "2", 1);
  setenv("PT_PROFILER_JOB_ATTEMPT_INDEX", "5", 1);

  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;

  constexpr int kWarmupSecs = 1;
  constexpr int kDurationSecs = 1;
  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  bool success = cfg.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = {}
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
      kWarmupSecs,
      kDurationSecs,
      traceFile.path(),
      duration_cast<milliseconds>(startTime.time_since_epoch()).count()));

  EXPECT_TRUE(success);
  EXPECT_FALSE(handler.isAsyncActive());

  handler.configure(cfg, now);

  EXPECT_TRUE(handler.isAsyncActive());

  std::string keyPrefix = "TEST_METADATA_";
  profiler.addMetadata(keyPrefix + "NORMAL", "\"metadata value\"");
  profiler.addMetadata(keyPrefix + "NEWLINE", "\"metadata \nvalue\"");
  profiler.addMetadata(keyPrefix + "BACKSLASH", R"("/test/metadata\path")");

  // End of the collection window, then a step past it to finalize.
  auto next = startTime + seconds(kDurationSecs);
  auto after = next + seconds(kDurationSecs);

  handler.performRunLoopStep(startTime, startTime);
  EXPECT_TRUE(handler.isAsyncActive());

  handler.performRunLoopStep(next, next);
  EXPECT_TRUE(handler.isAsyncActive());

  handler.performRunLoopStep(after, after);
  EXPECT_FALSE(handler.isAsyncActive());

#ifdef __linux__
  auto logFile = logUrlToPath(cfg.activitiesLogUrl());
  std::ifstream file(logFile);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open the trace JSON file.");
  }
  std::string jsonStr(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  nlohmann::json jsonData = nlohmann::json::parse(jsonStr);

  EXPECT_EQ(3, countSubstrings(jsonStr, keyPrefix));
  EXPECT_EQ(2, countSubstrings(jsonStr, "metadata value"));
  EXPECT_EQ(1, countSubstrings(jsonStr, "/test/metadata/path"));

  EXPECT_EQ(
      jsonData["PT_PROFILER_JOB_NAME"].get<std::string>(), "test_training_job");
  EXPECT_EQ(jsonData["PT_PROFILER_JOB_VERSION"].get<std::string>(), "2");
  EXPECT_EQ(jsonData["PT_PROFILER_JOB_ATTEMPT_INDEX"].get<std::string>(), "5");
#endif

  unsetenv("PT_PROFILER_JOB_NAME");
  unsetenv("PT_PROFILER_JOB_VERSION");
  unsetenv("PT_PROFILER_JOB_ATTEMPT_INDEX");
}

TEST(AsyncActivityProfilerHandler, Cancel) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  // Cancel when inactive is a no-op
  EXPECT_FALSE(handler.isAsyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isAsyncActive());

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;
  constexpr int kWarmupSecs = 5;
  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  bool success = cfg.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
      kWarmupSecs,
      traceFile.path(),
      duration_cast<milliseconds>(startTime.time_since_epoch()).count()));
  EXPECT_TRUE(success);

  // Cancel during Warmup
  handler.configure(cfg, now);
  EXPECT_TRUE(handler.isAsyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isAsyncActive());

  // Stays inactive on subsequent steps
  handler.performRunLoopStep(startTime, startTime);
  EXPECT_FALSE(handler.isAsyncActive());

  // Cancel during CollectTrace
  handler.configure(cfg, now);
  EXPECT_TRUE(handler.isAsyncActive());
  now = startTime;
  handler.performRunLoopStep(now, now);
  EXPECT_TRUE(handler.isAsyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isAsyncActive());
}

TEST(AsyncActivityProfilerHandler, FinalizesPendingTraceOnTeardown) {
  // Destroying a scheduled handler with a collected-but-unprocessed trace must
  // let the profiler loop finalize it, not drop it.
  GenericActivityProfiler profiler(/*cpu only*/ true);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;

  bool success = cfg.parse(fmt::format(
      R"CFG(
    PROFILE_START_ITERATION = 1
    ACTIVITIES_WARMUP_ITERATIONS = 0
    ACTIVITIES_ITERATIONS = 1
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
      traceFile.path()));
  EXPECT_TRUE(success);

  {
    AsyncActivityProfilerHandler handler(profiler);
    handler.step();
    EXPECT_FALSE(handler.isAsyncActive());

    handler.scheduleTrace(cfg);

    // Activate the scheduled config and enter CollectTrace.
    handler.step();
    EXPECT_TRUE(handler.isAsyncActive());

    // CollectTrace -> ProcessTrace asynchronously. Application step()
    // deliberately does not finalize ProcessTrace; the loop thread should do
    // that while exiting.
    handler.step();
    handler.ensureCollectTraceDone();
    EXPECT_TRUE(handler.isAsyncActive());

    // handler goes out of scope here with a collected trace still pending.
  }

  // The pending trace must have been finalized during teardown.
  auto logFile = logUrlToPath(cfg.activitiesLogUrl());
  checkTracefile(logFile.c_str());
}

TEST(AsyncActivityProfilerHandler, BufferSizeLimitDuringWarmup) {
  MockGpuProfiler profiler;
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;
  constexpr int kWarmupSecs = 5;
  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  bool success = cfg.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
    ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB = 3
  )CFG",
      kWarmupSecs,
      traceFile.path(),
      duration_cast<milliseconds>(startTime.time_since_epoch()).count()));
  EXPECT_TRUE(success);

  handler.configure(cfg, now);
  EXPECT_TRUE(handler.isAsyncActive());

  // Simulate GPU buffer overflow
  profiler.setGpuCollectionStopped(true);

  // During warmup, the handler should detect GPU collection stopped
  // and transition back to WaitForRequest
  now = startTime;
  handler.performRunLoopStep(now, now);
  EXPECT_FALSE(handler.isAsyncActive());
}

// An iteration-based request that arrives before the application has begun
// counting iterations (iterationCount_ == -1) cannot be honored unless it also
// carries a duration. With no duration it must be dropped, not scheduled.
TEST(AsyncActivityProfilerHandler, IterationRequestWithoutStepIsRejected) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;
  bool success = cfg.parse(fmt::format(
      R"CFG(
    PROFILE_START_ITERATION = 5
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 2
    ACTIVITIES_DURATION_SECS = 0
    ACTIVITIES_LOG_FILE = {}
  )CFG",
      traceFile.path()));
  ASSERT_TRUE(success);
  ASSERT_TRUE(cfg.hasProfileStartIteration());

  // No step() has advanced the iteration count, so the request is dropped
  // synchronously.
  EXPECT_FALSE(handler.scheduleTrace(cfg));
  EXPECT_FALSE(handler.isAsyncActive());
}

// The same iteration-based request is accepted when it carries a duration: the
// handler falls back to duration/timestamp scheduling instead of rejecting it.
// This is the path daemon configs take when the application is not reporting
// iterations.
TEST(
    AsyncActivityProfilerHandler,
    IterationRequestWithDurationFallsBackToTimestamp) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;
  bool success = cfg.parse(fmt::format(
      R"CFG(
    PROFILE_START_ITERATION = 5
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 2
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
      traceFile.path()));
  ASSERT_TRUE(success);
  ASSERT_TRUE(cfg.hasProfileStartIteration());

  // Accepted via the duration fallback (contrast the no-duration case above,
  // which is rejected). scheduleTrace() reports the decision synchronously.
  EXPECT_TRUE(handler.scheduleTrace(cfg));
}

// Only one on-demand request may be pending at a time. A second request that
// arrives while the first is still pending is dropped.
TEST(AsyncActivityProfilerHandler, SecondRequestWhilePendingIsRejected) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto acceptedFile = createTempTraceFile("libkineto_test_accepted", ".json");
  auto rejectedFile = createTempTraceFile("libkineto_test_rejected", ".json");

  // Start far in the future so neither request activates during the test; the
  // accept/reject decision is made synchronously inside scheduleTrace().
  auto startMs = duration_cast<milliseconds>(
                     (system_clock::now() + seconds(3600)).time_since_epoch())
                     .count();

  Config accepted;
  ASSERT_TRUE(accepted.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 0
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
      acceptedFile.path(),
      startMs)));

  Config rejected;
  ASSERT_TRUE(rejected.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 0
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
      rejectedFile.path(),
      startMs)));

  // The first request is accepted; a second arriving while it is still pending
  // is rejected.
  EXPECT_TRUE(handler.scheduleTrace(accepted));
  EXPECT_FALSE(handler.scheduleTrace(rejected));
}

// configure() must refuse a request whose start time has already passed rather
// than entering warmup for a trace that can never start on time. The start time
// is kept within the max request age so the config still parses.
TEST(AsyncActivityProfilerHandler, ConfigureRejectsStartTimeInThePast) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  auto now = system_clock::now();
  auto startTime = now - seconds(5);

  Config cfg;
  bool success = cfg.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 5
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
      traceFile.path(),
      duration_cast<milliseconds>(startTime.time_since_epoch()).count()));
  ASSERT_TRUE(success);

  handler.configure(cfg, now);
  EXPECT_FALSE(handler.isAsyncActive());
}

// cancel() must tear down cleanly when the trace has finished collecting and is
// waiting to be processed (RunloopState::ProcessTrace), returning the handler
// to the idle state.
TEST(AsyncActivityProfilerHandler, CancelDuringProcessTrace) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  constexpr int kWarmupSecs = 5;
  constexpr int kDurationSecs = 1;
  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  Config cfg;
  bool success = cfg.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = {}
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
      kWarmupSecs,
      kDurationSecs,
      traceFile.path(),
      duration_cast<milliseconds>(startTime.time_since_epoch()).count()));
  ASSERT_TRUE(success);

  handler.configure(cfg, now);
  EXPECT_TRUE(handler.isAsyncActive());

  // Warmup -> CollectTrace at the start time.
  now = startTime;
  handler.performRunLoopStep(now, now);

  // CollectTrace -> ProcessTrace once the collection window closes (a second
  // past startTime + duration). Driving with the default currentIter (-1)
  // collects inline, so the state settles on ProcessTrace.
  auto afterEnd = startTime + seconds(kDurationSecs + 1);
  handler.performRunLoopStep(afterEnd, afterEnd);
  EXPECT_TRUE(handler.isAsyncActive());

  handler.cancel();
  EXPECT_FALSE(handler.isAsyncActive());
}

// acceptConfig() only schedules when the config actually enables the activity
// profiler; a config with activities disabled is a no-op.
TEST(
    AsyncActivityProfilerHandler,
    AcceptConfigIgnoresDisabledActivityProfiler) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  Config cfg;
  ASSERT_TRUE(cfg.parse("ACTIVITIES_ENABLED = false"));
  ASSERT_FALSE(cfg.activityProfilerEnabled());

  // acceptConfig() reports that it declined the disabled config, rather than
  // silently doing nothing.
  EXPECT_FALSE(handler.acceptConfig(cfg));
  EXPECT_FALSE(handler.isAsyncActive());
}

// acceptConfig() schedules and reports acceptance when the config enables the
// activity profiler. Paired with the disabled case above, this pins that the
// return value reflects the config rather than being constant.
TEST(
    AsyncActivityProfilerHandler,
    AcceptConfigSchedulesEnabledActivityProfiler) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  // Far-future start so the request stays scheduled without activating during
  // the test.
  auto startMs = duration_cast<milliseconds>(
                     (system_clock::now() + seconds(3600)).time_since_epoch())
                     .count();

  Config cfg;
  ASSERT_TRUE(cfg.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = 0
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
      traceFile.path(),
      startMs)));
  ASSERT_TRUE(cfg.activityProfilerEnabled());

  EXPECT_TRUE(handler.acceptConfig(cfg));
}

// Collection start and end drive the registered client exactly once each:
// start() when warmup completes and collection begins, and stop() when
// collection finishes (the profiler stops the client while collecting the
// final trace).
TEST_F(AsyncClientTest, StartsAndStopsClientAroundCollection) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  constexpr int kWarmupSecs = 5;
  constexpr int kDurationSecs = 1;

  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  Config cfg;
  ASSERT_TRUE(cfg.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = {}
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
      kWarmupSecs,
      kDurationSecs,
      traceFile.path(),
      duration_cast<milliseconds>(startTime.time_since_epoch()).count())));

  handler.configure(cfg, now);
  EXPECT_EQ(client_.startCount.load(), 0);

  // Warmup -> CollectTrace starts the client.
  now = startTime;
  handler.performRunLoopStep(now, now);
  EXPECT_EQ(client_.startCount.load(), 1);
  EXPECT_EQ(client_.stopCount.load(), 0);

  // A second past the end of the collection window (startTime + duration).
  auto afterEnd = startTime + seconds(kDurationSecs + 1);

  // The run loop advances one state per call. First tick: the collection window
  // has closed, so it collects the trace (where the profiler stops the client)
  // and moves CollectTrace -> ProcessTrace. Second tick: it finalizes the
  // trace, moving ProcessTrace -> WaitForRequest (idle). Same timestamps --
  // collection has already ended, so we only pump the state machine, not
  // advance time.
  handler.performRunLoopStep(afterEnd, afterEnd);
  handler.performRunLoopStep(afterEnd, afterEnd);
  EXPECT_FALSE(handler.isAsyncActive());
  EXPECT_EQ(client_.startCount.load(), 1);
  EXPECT_EQ(client_.stopCount.load(), 1);
}

// Cancelling an in-progress collection stops the registered client.
TEST_F(AsyncClientTest, CancelStopsClient) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  AsyncActivityProfilerHandler handler(profiler);

  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  constexpr int kWarmupSecs = 5;
  auto now = system_clock::now();
  auto startTime = now + seconds(kWarmupSecs + 1);

  Config cfg;
  ASSERT_TRUE(cfg.parse(fmt::format(
      R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG",
      kWarmupSecs,
      traceFile.path(),
      duration_cast<milliseconds>(startTime.time_since_epoch()).count())));

  handler.configure(cfg, now);
  now = startTime;
  handler.performRunLoopStep(now, now); // Warmup -> CollectTrace
  ASSERT_EQ(client_.startCount.load(), 1);

  handler.cancel();
  EXPECT_FALSE(handler.isAsyncActive());
  EXPECT_EQ(client_.stopCount.load(), 1);
}

// A memory-profiling request drives the client's memory hooks in order and
// exports to the configured path. This exercises the CollectMemorySnapshot
// state and memoryProfilerLoop, which run on a background thread.
TEST_F(AsyncClientTest, MemoryProfileRequestDrivesClientHooks) {
  GenericActivityProfiler profiler(/*cpu only*/ true);

  auto traceFile = createTempTraceFile("libkineto_test_memory", ".json");

  Config cfg;
  // PROFILE_MEMORY resets the log file to a default, so ACTIVITIES_LOG_FILE
  // must follow it to control the export path.
  ASSERT_TRUE(cfg.parse(fmt::format(
      R"CFG(
    PROFILE_MEMORY = true
    PROFILE_MEMORY_DURATION_MSECS = 50
    ACTIVITIES_LOG_FILE = {}
  )CFG",
      traceFile.path())));
  ASSERT_TRUE(cfg.memoryProfilerEnabled());

  auto memoryStopped = client_.memoryStopFuture();
  {
    // The memory loop runs on a background thread; block on its final client
    // call (stop_memory_profile) via a future instead of polling. Destroying
    // the handler at the end of this scope joins that thread before we read the
    // client below, so the assertions and fixture teardown cannot race the
    // still-running loop.
    AsyncActivityProfilerHandler handler(profiler);
    EXPECT_TRUE(handler.scheduleTrace(cfg));
    ASSERT_EQ(memoryStopped.wait_for(seconds(15)), std::future_status::ready);
  }

  EXPECT_EQ(client_.memoryStartCount.load(), 1);
  EXPECT_EQ(client_.memoryExportCount.load(), 1);
  EXPECT_EQ(client_.exportedPath(), traceFile.path());
}
