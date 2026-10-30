/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "include/libkineto.h"
#include "include/output_base.h"
#include "src/ActivityBuffers.h"
#include "src/ActivityLoggerFactory.h"
#include "src/ActivityProfilerController.h"
#include "src/Logger.h"

using namespace KINETO_NAMESPACE;

class MockActivityLogger : public libkineto::ActivityLogger {
 public:
  explicit MockActivityLogger(const std::string& url) : url_(url) {}

  void handleDeviceInfo(const libkineto::DeviceInfo&, int64_t) override {}
  void handleResourceInfo(const libkineto::ResourceInfo&, int64_t) override {}
  void handleOverheadInfo(
      const libkineto::ActivityLogger::OverheadInfo&,
      int64_t) override {}
  void handleTraceSpan(const libkineto::TraceSpan&) override {}
  void handleActivity(const libkineto::ITraceActivity&) override {}
  void handleGenericActivity(const libkineto::GenericTraceActivity&) override {}
  void handleTraceStart(
      const std::unordered_map<std::string, std::string>&,
      const std::string&) override {}
  void finalizeTrace(
      const libkineto::Config&,
      std::unique_ptr<libkineto::ActivityBuffers>,
      int64_t) override {}
  void finalizeMemoryTrace(const std::string&, const libkineto::Config&)
      override {}

  const std::string& getUrl() const {
    return url_;
  }

 private:
  std::string url_;
};

class CountingLogger : public libkineto::ActivityLogger {
 public:
  explicit CountingLogger(std::shared_ptr<int> counter)
      : counter_(std::move(counter)) {
    (*counter_)++;
  }

  void handleDeviceInfo(const libkineto::DeviceInfo&, int64_t) override {}
  void handleResourceInfo(const libkineto::ResourceInfo&, int64_t) override {}
  void handleOverheadInfo(
      const libkineto::ActivityLogger::OverheadInfo&,
      int64_t) override {}
  void handleTraceSpan(const libkineto::TraceSpan&) override {}
  void handleActivity(const libkineto::ITraceActivity&) override {}
  void handleGenericActivity(const libkineto::GenericTraceActivity&) override {}
  void handleTraceStart(
      const std::unordered_map<std::string, std::string>&,
      const std::string&) override {}
  void finalizeTrace(
      const libkineto::Config&,
      std::unique_ptr<libkineto::ActivityBuffers>,
      int64_t) override {}
  void finalizeMemoryTrace(const std::string&, const libkineto::Config&)
      override {}

 private:
  std::shared_ptr<int> counter_;
};

class WarningObserver : public libkineto::ILoggerObserver {
 public:
  void write(const std::string& message, libkineto::LoggerOutputType ot)
      override {
    if (ot == libkineto::LoggerOutputType::WARNING) {
      warnings_.push_back(message);
    }
  }
  const std::map<libkineto::LoggerOutputType, std::vector<std::string>>
  extractCollectorMetadata() override {
    return {};
  }
  void reset() override {
    warnings_.clear();
  }
  void addDevice(int64_t) override {}
  void setTraceDurationMS(int64_t) override {}
  void addEventCount(int64_t) override {}
  void addDestination(const std::string&) override {}
  void addMetadata(const std::string&, const std::string&) override {}

  const std::vector<std::string>& warnings() const {
    return warnings_;
  }

 private:
  std::vector<std::string> warnings_;
};

// Basic public API functionality
TEST(RegisterLoggerFactoryTest, BasicPublicAPI) {
  libkineto::registerLoggerFactory(
      "basic_api_proto", [](const std::string& path) {
        return std::make_unique<MockActivityLogger>(path);
      });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();
  auto logger = factory.makeLogger("basic_api_proto:///tmp/trace.log");

  ASSERT_NE(logger, nullptr);
  auto* mockLogger = dynamic_cast<MockActivityLogger*>(logger.get());
  ASSERT_NE(mockLogger, nullptr);
  EXPECT_EQ(mockLogger->getUrl(), "/tmp/trace.log");
}

// Protocol case insensitivity
TEST(RegisterLoggerFactoryTest, ProtocolCaseInsensitive) {
  libkineto::registerLoggerFactory("CaseProto", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>(path);
  });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  auto logger1 = factory.makeLogger("caseproto:///path1");
  ASSERT_NE(logger1, nullptr);
  auto* mock1 = dynamic_cast<MockActivityLogger*>(logger1.get());
  EXPECT_EQ(mock1->getUrl(), "/path1");

  auto logger2 = factory.makeLogger("CASEPROTO:///path2");
  ASSERT_NE(logger2, nullptr);
  auto* mock2 = dynamic_cast<MockActivityLogger*>(logger2.get());
  EXPECT_EQ(mock2->getUrl(), "/path2");

  auto logger3 = factory.makeLogger("CaseProto:///path3");
  ASSERT_NE(logger3, nullptr);
  auto* mock3 = dynamic_cast<MockActivityLogger*>(logger3.get());
  EXPECT_EQ(mock3->getUrl(), "/path3");
}

// Unregistered protocol throws
TEST(RegisterLoggerFactoryTest, UnregisteredProtocolThrows) {
  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  EXPECT_THROW(
      factory.makeLogger("nonexistent:///path"), std::invalid_argument);

  try {
    factory.makeLogger("unknown:///path");
    FAIL() << "Expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    std::string msg(e.what());
    EXPECT_TRUE(
        msg.find("unknown") != std::string::npos ||
        msg.find("protocol") != std::string::npos);
  }
}

// Protocol overwriting behavior and warning
TEST(RegisterLoggerFactoryTest, OverwriteProtocolWarning) {
  auto counter1 = std::make_shared<int>(0);
  auto counter2 = std::make_shared<int>(0);

  libkineto::registerLoggerFactory(
      "overwrite_warning_proto", [counter1](const std::string&) {
        return std::make_unique<CountingLogger>(counter1);
      });

  WarningObserver observer;
  libkineto::Logger::addLoggerObserver(&observer);

  libkineto::registerLoggerFactory(
      "overwrite_warning_proto", [counter2](const std::string&) {
        return std::make_unique<CountingLogger>(counter2);
      });

  libkineto::Logger::removeLoggerObserver(&observer);

  // Assert the overwrite warning was emitted without pinning the exact count;
  // the observer captures any WARNING logged during the second registration.
  const auto& warnings = observer.warnings();
  EXPECT_TRUE(std::any_of(
      warnings.begin(), warnings.end(), [](const std::string& warning) {
        return warning.find("Overwriting") != std::string::npos &&
            warning.find("overwrite_warning_proto") != std::string::npos;
      }));

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();
  auto logger = factory.makeLogger("overwrite_warning_proto:///path");

  EXPECT_EQ(*counter1, 0);
  EXPECT_EQ(*counter2, 1);
  ASSERT_NE(logger, nullptr);
}

// Built-in "file" protocol remains functional
TEST(RegisterLoggerFactoryTest, BuiltInFileProtocolStillWorks) {
  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  const std::string tempDir = testing::TempDir();
  auto logger1 = factory.makeLogger("file://" + tempDir + "trace1.json");
  ASSERT_NE(logger1, nullptr);

  libkineto::registerLoggerFactory(
      "file_custom_proto", [](const std::string& path) {
        return std::make_unique<MockActivityLogger>(
            "file_custom_proto:" + path);
      });

  auto customLogger1 = factory.makeLogger("file_custom_proto:///path");

  ASSERT_NE(customLogger1, nullptr);

  auto logger2 = factory.makeLogger("file://" + tempDir + "trace2.json");
  ASSERT_NE(logger2, nullptr);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
