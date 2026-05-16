// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>

#include <c10/util/Flags.h>
#include <gtest/gtest.h>
#include <torch/csrc/distributed/c10d/debug.h>
#include <torch/csrc/distributed/c10d/logging.h>

C10_DECLARE_int(caffe2_log_level);
C10_DECLARE_int32(minloglevel);
C10_DECLARE_bool(logtostderr);

namespace {

// fmt probe: format() increments the counter every time the fmt library
// formats this value. Because formatLogMessage is only invoked inside a
// path whose severity gate passes, counting formatter invocations tells
// us how many times a given path fired.
struct FmtHitCounter {
  size_t* count;
};

} // namespace

template <>
struct fmt::formatter<FmtHitCounter> {
  constexpr auto parse(format_parse_context& ctx) const {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const FmtHitCounter& h, FormatContext& ctx) const {
    ++(*h.count);
    return fmt::format_to(ctx.out(), "x");
  }
};

namespace {

// RAII helper: set c10d log levels for the duration of a test, then restore.
// caffe2_log_level uses lower = more verbose (0=INFO, 1=WARNING, 2=ERROR);
// setDebugLevel gates Debug/Trace on top of that (see isLogLevelEnabled).
class LogLevelGuard {
 public:
  LogLevelGuard(int caffe2_level, c10d::DebugLevel debug)
      : prev_caffe2_(FLAGS_caffe2_log_level), prev_debug_(c10d::debug_level()) {
    FLAGS_caffe2_log_level = caffe2_level;
    c10d::setDebugLevel(debug);
  }

  ~LogLevelGuard() {
    FLAGS_caffe2_log_level = prev_caffe2_;
    c10d::setDebugLevel(prev_debug_);
  }

  LogLevelGuard(const LogLevelGuard&) = delete;
  LogLevelGuard& operator=(const LogLevelGuard&) = delete;

 private:
  int prev_caffe2_;
  c10d::DebugLevel prev_debug_;
};

// RAII helper: route logs to stderr (mirrors c10::ShowLogInfoToStderr) and
// restore the prior values on destruction. Under glog these flags control
// the sink; under non-glog they exist as backward-compat shims and the
// save/restore is a no-op for actual log routing.
class StderrLoggingGuard {
 public:
  StderrLoggingGuard()
      : prev_logtostderr_(FLAGS_logtostderr),
        prev_minloglevel_(FLAGS_minloglevel) {
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = std::min(FLAGS_minloglevel, 0);
  }

  ~StderrLoggingGuard() {
    FLAGS_logtostderr = prev_logtostderr_;
    FLAGS_minloglevel = prev_minloglevel_;
  }

  StderrLoggingGuard(const StderrLoggingGuard&) = delete;
  StderrLoggingGuard& operator=(const StderrLoggingGuard&) = delete;

 private:
  bool prev_logtostderr_;
  int prev_minloglevel_;
};

size_t countOccurrences(const std::string& s, std::string_view target) {
  size_t n = 0;
  size_t pos = 0;
  while ((pos = s.find(target, pos)) != std::string::npos) {
    ++n;
    pos += target.size();
  }
  return n;
}

} // namespace

// With DEBUG disabled and WARNING enabled, only the Nth,2Nth,... calls produce
// a format invocation — the "else" path calls C10D_DEBUG which is skipped
// at its severity gate before evaluating formatLogMessage.
TEST(C10DWarningEveryNElseDebug, WarningPathOnlyWhenDebugDisabled) {
  // caffe2_log_level=1 -> WARNING enabled, INFO/DEBUG/TRACE disabled.
  LogLevelGuard guard(/*caffe2_level=*/1, c10d::DebugLevel::Off);

  size_t hits = 0;
  FmtHitCounter h{&hits};

  for (int i = 0; i < 12; ++i) {
    C10D_WARNING_EVERY_N_ELSE_DEBUG(3, "hit {}", h);
  }

  // 12 calls, N=3 -> warning logs on 3,6,9,12 (4 times). Debug path runs on
  // the other 8 but its severity gate is off, so formatLogMessage isn't called.
  EXPECT_EQ(hits, 4u);
}

// Distinguish the warning and debug paths in a single run by capturing stderr.
// Each path emits a different prefix ("[c10d]" for warnings vs "[c10d - debug]"
// for debug), so we can count each by occurrence search.
TEST(C10DWarningEveryNElseDebug, StderrShowsBothPaths) {
  StderrLoggingGuard stderr_guard;
  LogLevelGuard guard(/*caffe2_level=*/0, c10d::DebugLevel::Info);

  testing::internal::CaptureStderr();
  for (int i = 0; i < 12; ++i) {
    C10D_WARNING_EVERY_N_ELSE_DEBUG(3, "msg");
  }
  std::string output = testing::internal::GetCapturedStderr();

  size_t debug_lines = countOccurrences(output, "[c10d - debug]");
  size_t c10d_lines = countOccurrences(output, "[c10d");
  size_t warning_lines = c10d_lines - debug_lines;

  EXPECT_EQ(warning_lines, 4u); // calls 3, 6, 9, 12
  EXPECT_EQ(debug_lines, 8u); // calls 1, 2, 4, 5, 7, 8, 10, 11
}

// Two different call sites must have independent counters (static inside do-
// while block is per-expansion).
TEST(C10DWarningEveryNElseDebug, PerCallSiteCounter) {
  LogLevelGuard guard(/*caffe2_level=*/1, c10d::DebugLevel::Off);

  size_t hits_a = 0;
  size_t hits_b = 0;
  FmtHitCounter ha{&hits_a};
  FmtHitCounter hb{&hits_b};

  for (int i = 0; i < 6; ++i) {
    C10D_WARNING_EVERY_N_ELSE_DEBUG(2, "a {}", ha);
  }

  for (int i = 0; i < 3; ++i) {
    C10D_WARNING_EVERY_N_ELSE_DEBUG(2, "b {}", hb);
  }

  EXPECT_EQ(hits_a, 3u); // warnings on calls 2, 4, 6
  EXPECT_EQ(hits_b, 1u); // warning on call 2
}

// N=1 means the warning path logs on every call.
TEST(C10DWarningEveryNElseDebug, N1AllWarnings) {
  LogLevelGuard guard(/*caffe2_level=*/1, c10d::DebugLevel::Off);

  size_t hits = 0;
  FmtHitCounter h{&hits};

  for (int i = 0; i < 5; ++i) {
    C10D_WARNING_EVERY_N_ELSE_DEBUG(1, "x {}", h);
  }

  EXPECT_EQ(hits, 5u);
}

// Dangling-else guard: the do-while wrapper means the macro is a single
// statement and an outer else binds to an outer if, not into the macro.
TEST(C10DWarningEveryNElseDebug, NoDanglingElse) {
  LogLevelGuard guard(/*caffe2_level=*/1, c10d::DebugLevel::Off);

  size_t hits = 0;
  FmtHitCounter h{&hits};
  bool taken_else = false;

  if (false)
    C10D_WARNING_EVERY_N_ELSE_DEBUG(1, "x {}", h);
  else
    taken_else = true;

  EXPECT_TRUE(taken_else);
  EXPECT_EQ(hits, 0u);
}
