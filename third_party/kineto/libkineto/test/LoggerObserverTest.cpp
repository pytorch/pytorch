/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "LoggerCollector.h"
#include "include/libkineto.h"
#include "src/ActivityProfilerController.h"
#include "src/Logger.h"

using namespace KINETO_NAMESPACE;

#if !USE_GOOGLE_LOG

constexpr char InfoTestStr[] = "Checking LOG(INFO)";
constexpr char WarningTestStr[] = "Checking LOG(WARNING)";
constexpr char ErrorTestStr[] = "Checking LOG(ERROR)";

TEST(LoggerObserverTest, SingleCollectorObserver) {
  // Add a LoggerObserverCollector to collect all logs during the trace.
  std::unique_ptr<LoggerCollector> lCollector =
      std::make_unique<LoggerCollector>();
  Logger::addLoggerObserver(lCollector.get());

  LOG(INFO) << InfoTestStr;
  LOG(WARNING) << WarningTestStr;
  LOG(ERROR) << ErrorTestStr;

  auto LoggerMD = lCollector->extractCollectorMetadata();
  EXPECT_TRUE(
      LoggerMD[LoggerOutputType::INFO][0].find(InfoTestStr) !=
      std::string::npos);
  EXPECT_TRUE(
      LoggerMD[LoggerOutputType::WARNING][0].find(WarningTestStr) !=
      std::string::npos);
  EXPECT_TRUE(
      LoggerMD[LoggerOutputType::ERROR][0].find(ErrorTestStr) !=
      std::string::npos);

  Logger::removeLoggerObserver(lCollector.get());
}

#define NUM_OF_MESSAGES_FOR_EACH_TYPE 10
#define NUM_OF_WRITE_THREADS 200

// Writes NUM_OF_MESSAGES_FOR_EACH_TYPE messages for each INFO, WARNING, and
// ERROR.
void* writeSeveralMessages() {
  for (int i = 0; i < NUM_OF_MESSAGES_FOR_EACH_TYPE; i++) {
    LOG(INFO) << InfoTestStr;
    LOG(WARNING) << WarningTestStr;
    LOG(ERROR) << ErrorTestStr;
  }
  return nullptr;
}

TEST(LoggerObserverTest, FourCollectorObserver) {
  // There shouldn't be too many CUPTIActivityProfilers active at the same time.
  std::unique_ptr<LoggerCollector> lc1 = std::make_unique<LoggerCollector>();
  std::unique_ptr<LoggerCollector> lc2 = std::make_unique<LoggerCollector>();
  std::unique_ptr<LoggerCollector> lc3 = std::make_unique<LoggerCollector>();
  std::unique_ptr<LoggerCollector> lc4 = std::make_unique<LoggerCollector>();
  Logger::addLoggerObserver(lc1.get());
  Logger::addLoggerObserver(lc2.get());
  Logger::addLoggerObserver(lc3.get());
  Logger::addLoggerObserver(lc4.get());

  // Launch NUM_OF_WRITE_THREADS threads writing several messages.
  std::vector<std::thread> ListOfThreads;
  for (int i = 0; i < NUM_OF_WRITE_THREADS; i++) {
    ListOfThreads.emplace_back(writeSeveralMessages);
  }

  // Wait for all threads to finish.
  for (auto& thread : ListOfThreads) {
    thread.join();
  }

  auto lc1MD = lc1->extractCollectorMetadata();
  int InfoCount = 0, WarnCount = 0, ErrorCount = 0;
  for (auto& md : lc1MD) {
    InfoCount += md.first == LoggerOutputType::INFO ? md.second.size() : 0;
    WarnCount += md.first == LoggerOutputType::WARNING ? md.second.size() : 0;
    ErrorCount += md.first == LoggerOutputType::ERROR ? md.second.size() : 0;
  }

  EXPECT_EQ(InfoCount, NUM_OF_WRITE_THREADS * NUM_OF_MESSAGES_FOR_EACH_TYPE);
  EXPECT_EQ(WarnCount, NUM_OF_WRITE_THREADS * NUM_OF_MESSAGES_FOR_EACH_TYPE);
  EXPECT_EQ(ErrorCount, NUM_OF_WRITE_THREADS * NUM_OF_MESSAGES_FOR_EACH_TYPE);

  Logger::removeLoggerObserver(lc1.get());
  Logger::removeLoggerObserver(lc2.get());
  Logger::removeLoggerObserver(lc3.get());
  Logger::removeLoggerObserver(lc4.get());
}

TEST(LoggerObserverTest, AddAndGetLoggerCollector) {
  auto baseline = ActivityProfilerController::getLoggerCollectors().size();

  // Register a logger collector via the static API.
  ActivityProfilerController::addLoggerCollectorFactory(
      []() { return std::make_shared<LoggerCollector>(); });

  auto collectors = ActivityProfilerController::getLoggerCollectors();
  EXPECT_EQ(collectors.size(), baseline + 1);

  // The returned collector should be a valid LoggerCollector that can
  // receive log messages when manually registered as an observer.
  auto& collector = collectors[baseline];
  Logger::addLoggerObserver(collector.get());

  LOG(INFO) << InfoTestStr;
  LOG(WARNING) << WarningTestStr;
  LOG(ERROR) << ErrorTestStr;

  auto metadata = collector->extractCollectorMetadata();
  EXPECT_TRUE(
      metadata[LoggerOutputType::INFO][0].find(InfoTestStr) !=
      std::string::npos);
  EXPECT_TRUE(
      metadata[LoggerOutputType::WARNING][0].find(WarningTestStr) !=
      std::string::npos);
  EXPECT_TRUE(
      metadata[LoggerOutputType::ERROR][0].find(ErrorTestStr) !=
      std::string::npos);

  Logger::removeLoggerObserver(collector.get());
}

TEST(LoggerObserverTest, MultipleLoggerCollectors) {
  auto baseline = ActivityProfilerController::getLoggerCollectors().size();

  // Register two more collectors.
  ActivityProfilerController::addLoggerCollectorFactory(
      []() { return std::make_shared<LoggerCollector>(); });
  ActivityProfilerController::addLoggerCollectorFactory(
      []() { return std::make_shared<LoggerCollector>(); });

  auto collectors = ActivityProfilerController::getLoggerCollectors();
  EXPECT_EQ(collectors.size(), baseline + 2);

  // Both collectors should independently receive log messages.
  auto& c1 = collectors[baseline];
  auto& c2 = collectors[baseline + 1];
  Logger::addLoggerObserver(c1.get());
  Logger::addLoggerObserver(c2.get());

  LOG(INFO) << InfoTestStr;

  auto md1 = c1->extractCollectorMetadata();
  auto md2 = c2->extractCollectorMetadata();
  EXPECT_EQ(md1[LoggerOutputType::INFO].size(), 1);
  EXPECT_EQ(md2[LoggerOutputType::INFO].size(), 1);

  Logger::removeLoggerObserver(c1.get());
  Logger::removeLoggerObserver(c2.get());
}

// Minimal observer that, unlike LoggerCollector, retains STAGE messages so we
// can verify UST_LOGGER_MARK_COMPLETED behavior.
class CapturingObserver : public ILoggerObserver {
 public:
  void write(const std::string& message, LoggerOutputType ot) override {
    messages[ot].push_back(message);
  }
  const std::map<LoggerOutputType, std::vector<std::string>>
  extractCollectorMetadata() override {
    return messages;
  }
  void reset() override {
    messages.clear();
  }
  void addDevice([[maybe_unused]] const int64_t device) override {}
  void setTraceDurationMS([[maybe_unused]] const int64_t duration) override {}
  void addEventCount([[maybe_unused]] const int64_t count) override {}
  void addDestination([[maybe_unused]] const std::string& dest) override {}
  void addMetadata(
      [[maybe_unused]] const std::string& key,
      [[maybe_unused]] const std::string& value) override {}

  std::map<LoggerOutputType, std::vector<std::string>> messages;
};

// Regression test for UST_LOGGER_MARK_COMPLETED and related UST logger
// completion behavior. USDT_LOGGER_EMIT_MESSAGE previously expanded through the
// unqualified `LOG`/`LOG_IS_ON` macros, which glog's <glog/logging.h> redefines
// (LOG -> COMPACT_GOOGLE_LOG_##sev, LOG_IS_ON -> google::GLOG_##sev >= ...).
// Including both headers in the same translation unit produced
// COMPACT_GOOGLE_LOG_libkineto / google::GLOG_libkineto -- a compile error.
//
// Rather than taking on a build-time dep on glog (which the OSS CMake build
// does not link), simulate glog's LOG and LOG_IS_ON macros locally. If a
// future change reintroduces a bare `LOG(...)` or `LOG_IS_ON(...)` token at
// any UST/USDT macro site, the preprocessor will expand it to
// `COMPACT_GOOGLE_LOG_libkineto::...` / `google::GLOG_libkineto::...` here
// and the test will fail to compile. The `#pragma push_macro/pop_macro` pairs
// scope the redefinition so the surrounding tests still see libkineto's own
// LOG/LOG_IS_ON.
#pragma push_macro("LOG")
#pragma push_macro("LOG_IS_ON")
#undef LOG
#undef LOG_IS_ON
#define COMPACT_GOOGLE_LOG_INFO 0
#define COMPACT_GOOGLE_LOG_WARNING 0
#define COMPACT_GOOGLE_LOG_ERROR 0
#define COMPACT_GOOGLE_LOG_FATAL 0
#define LOG(severity) COMPACT_GOOGLE_LOG_##severity
#define LOG_IS_ON(severity) (google::GLOG_##severity >= 0)

TEST(LoggerObserverTest, MacrosImmuneToGlogLogCollision) {
  auto observer = std::make_unique<CapturingObserver>();
  Logger::addLoggerObserver(observer.get());

  UST_LOGGER_MARK_COMPLETED("test_stage");
  USDT_LOGGER_EMIT_MESSAGE("test_msg");
  USDT_EMIT_START_TRACE();
  USDT_EMIT_STOP_TRACE();

  const auto md = observer->extractCollectorMetadata();
  const auto stageIt = md.find(LoggerOutputType::STAGE);
  ASSERT_NE(stageIt, md.end());
  EXPECT_EQ(stageIt->second.size(), 1u);
  EXPECT_NE(stageIt->second[0].find("test_stage"), std::string::npos);

  const auto usdtIt = md.find(LoggerOutputType::USDT);
  ASSERT_NE(usdtIt, md.end());
  // One from USDT_LOGGER_EMIT_MESSAGE, plus one each from start/stop trace.
  EXPECT_EQ(usdtIt->second.size(), 3u);

  Logger::removeLoggerObserver(observer.get());
}

#pragma pop_macro("LOG_IS_ON")
#pragma pop_macro("LOG")

// Regression test for the second arm of the fix: the macros must
// continue to honor Logger::severityLevel(). suppressLibkinetoLogMessages()
// in init.cpp raises the threshold to ERROR, which historically suppressed
// STAGE output (STAGE=3 < ERROR=4) via LOG_IF. Since the macros no longer
// expand through LOG_IF, the gating must be re-implemented in the macro
// expansion itself.
//
// USDT (=5) is the highest severity value in the enum, so it is never
// suppressed by setSeverityLevel(ERROR); to verify USDT is also gated we set
// the threshold one step above USDT.
TEST(LoggerObserverTest, UstMacrosRespectSeverityThreshold) {
  auto observer = std::make_unique<CapturingObserver>();
  Logger::addLoggerObserver(observer.get());

  const int originalLevel = Logger::severityLevel();

  // STAGE (3) < ERROR (4): UST_LOGGER_MARK_COMPLETED must be suppressed.
  Logger::setSeverityLevel(LoggerOutputType::ERROR);
  UST_LOGGER_MARK_COMPLETED("suppressed_stage");
  auto md = observer->extractCollectorMetadata();
  EXPECT_EQ(md[LoggerOutputType::STAGE].size(), 0u);

  // USDT (5) is the highest severity; to suppress it the threshold must be
  // higher than USDT itself.
  Logger::setSeverityLevel(LoggerOutputType::USDT + 1);
  USDT_LOGGER_EMIT_MESSAGE("suppressed_usdt");
  md = observer->extractCollectorMetadata();
  EXPECT_EQ(md[LoggerOutputType::USDT].size(), 0u);

  // Lower the threshold and confirm the macros now emit messages.
  Logger::setSeverityLevel(LoggerOutputType::VERBOSE);
  UST_LOGGER_MARK_COMPLETED("emitted_stage");
  USDT_LOGGER_EMIT_MESSAGE("emitted_usdt");

  md = observer->extractCollectorMetadata();
  ASSERT_EQ(md[LoggerOutputType::STAGE].size(), 1u);
  EXPECT_NE(
      md[LoggerOutputType::STAGE][0].find("emitted_stage"), std::string::npos);
  ASSERT_EQ(md[LoggerOutputType::USDT].size(), 1u);
  EXPECT_NE(
      md[LoggerOutputType::USDT][0].find("emitted_usdt"), std::string::npos);

  Logger::setSeverityLevel(originalLevel);
  Logger::removeLoggerObserver(observer.get());
}

#endif // !USE_GOOGLE_LOG

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
