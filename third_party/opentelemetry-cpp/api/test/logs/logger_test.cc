// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <vector>

#include "opentelemetry/common/timestamp.h"
#include "opentelemetry/logs/logger.h"
#include "opentelemetry/logs/provider.h"
#include "opentelemetry/nostd/shared_ptr.h"

using opentelemetry::logs::EventId;
using opentelemetry::logs::Logger;
using opentelemetry::logs::LoggerProvider;
using opentelemetry::logs::Provider;
using opentelemetry::logs::Severity;
using opentelemetry::nostd::shared_ptr;
using opentelemetry::nostd::span;
using opentelemetry::nostd::string_view;
namespace common = opentelemetry::common;
namespace nostd  = opentelemetry::nostd;
namespace trace  = opentelemetry::trace;

// Check that the default logger is a noop logger instance
TEST(Logger, GetLoggerDefault)
{
  auto lp = Provider::GetLoggerProvider();
  const std::string schema_url{"https://opentelemetry.io/schemas/1.11.0"};
  auto logger = lp->GetLogger("TestLogger", "opentelelemtry_library", "", schema_url);
  auto name   = logger->GetName();
  EXPECT_NE(nullptr, logger);
  EXPECT_EQ(name, "noop logger");
}

// Test the two additional overloads for GetLogger()
TEST(Logger, GetNoopLoggerNameWithArgs)
{
  auto lp = Provider::GetLoggerProvider();

  const std::string schema_url{"https://opentelemetry.io/schemas/1.11.0"};
  lp->GetLogger("NoopLoggerWithArgs", "opentelelemtry_library", "", schema_url);

  lp->GetLogger("NoopLoggerWithOptions", "opentelelemtry_library", "", schema_url);
}

// Test the EmitLogRecord() overloads
TEST(Logger, LogMethodOverloads)
{
  auto lp = Provider::GetLoggerProvider();
  const std::string schema_url{"https://opentelemetry.io/schemas/1.11.0"};
  auto logger = lp->GetLogger("TestLogger", "opentelelemtry_library", "", schema_url);

  EventId trace_event_id{0x1, "TraceEventId"};
  EventId debug_event_id{0x2, "DebugEventId"};
  EventId info_event_id{0x3, "InfoEventId"};
  EventId warn_event_id{0x4, "WarnEventId"};
  EventId error_event_id{0x5, "ErrorEventId"};
  EventId fatal_event_id{0x6, "FatalEventId"};

  // Create a map to test the logs with
  std::map<std::string, std::string> m = {{"key1", "value1"}};

  // EmitLogRecord overloads
  logger->EmitLogRecord(Severity::kTrace, "Test log message");
  logger->EmitLogRecord(Severity::kInfo, "Test log message");
  logger->EmitLogRecord(Severity::kDebug, m);
  logger->EmitLogRecord(Severity::kWarn, "Logging a map", m);
  logger->EmitLogRecord(Severity::kError,
                        opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->EmitLogRecord(Severity::kFatal, "Logging an initializer list",
                        opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->EmitLogRecord(Severity::kDebug, opentelemetry::common::MakeAttributes(m));
  logger->EmitLogRecord(Severity::kDebug,
                        common::KeyValueIterableView<std::map<std::string, std::string>>(m));
  std::pair<nostd::string_view, common::AttributeValue> array[] = {{"key1", "value1"}};
  logger->EmitLogRecord(Severity::kDebug, opentelemetry::common::MakeAttributes(array));
  std::vector<std::pair<std::string, std::string>> vec = {{"key1", "value1"}};
  logger->EmitLogRecord(Severity::kDebug, opentelemetry::common::MakeAttributes(vec));

  // Severity methods
  logger->Trace("Test log message");
  logger->Trace("Test log message", m);
  logger->Trace("Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Trace(m);
  logger->Trace(opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Trace(trace_event_id, "Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Trace(trace_event_id.id_, "Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));

  logger->Debug("Test log message");
  logger->Debug("Test log message", m);
  logger->Debug("Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Debug(m);
  logger->Debug(opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Debug(debug_event_id, "Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Debug(debug_event_id.id_, "Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));

  logger->Info("Test log message");
  logger->Info("Test log message", m);
  logger->Info("Test log message",
               opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Info(m);
  logger->Info(opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Info(info_event_id, "Test log message",
               opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Info(info_event_id.id_, "Test log message",
               opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));

  logger->Warn("Test log message");
  logger->Warn("Test log message", m);
  logger->Warn("Test log message",
               opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Warn(m);
  logger->Warn(opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Warn(warn_event_id, "Test log message",
               opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Warn(warn_event_id.id_, "Test log message",
               opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));

  logger->Error("Test log message");
  logger->Error("Test log message", m);
  logger->Error("Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Error(m);
  logger->Error(opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Error(error_event_id, "Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Error(error_event_id.id_, "Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));

  logger->Fatal("Test log message");
  logger->Fatal("Test log message", m);
  logger->Fatal("Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Fatal(m);
  logger->Fatal(opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Fatal(fatal_event_id, "Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  logger->Fatal(fatal_event_id.id_, "Test log message",
                opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
}

TEST(Logger, EventLogMethodOverloads)
{
  auto lp = Provider::GetLoggerProvider();
  const std::string schema_url{"https://opentelemetry.io/schemas/1.11.0"};
  auto logger = lp->GetLogger("TestLogger", "opentelelemtry_library", "", schema_url);

  auto elp          = Provider::GetEventLoggerProvider();
  auto event_logger = elp->CreateEventLogger(logger, "otel-cpp.test");

  std::map<std::string, std::string> m = {{"key1", "value1"}};

  event_logger->EmitEvent("event name", Severity::kTrace, "Test log message");
  event_logger->EmitEvent("event name", Severity::kInfo, "Test log message");
  event_logger->EmitEvent("event name", Severity::kDebug, m);
  event_logger->EmitEvent("event name", Severity::kWarn, "Logging a map", m);
  event_logger->EmitEvent(
      "event name", Severity::kError,
      opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  event_logger->EmitEvent(
      "event name", Severity::kFatal, "Logging an initializer list",
      opentelemetry::common::MakeAttributes({{"key1", "value 1"}, {"key2", 2}}));
  event_logger->EmitEvent("event name", Severity::kDebug, opentelemetry::common::MakeAttributes(m));
  event_logger->EmitEvent("event name", Severity::kDebug,
                          common::KeyValueIterableView<std::map<std::string, std::string>>(m));
  std::pair<nostd::string_view, common::AttributeValue> array[] = {{"key1", "value1"}};
  event_logger->EmitEvent("event name", Severity::kDebug,
                          opentelemetry::common::MakeAttributes(array));
  std::vector<std::pair<std::string, std::string>> vec = {{"key1", "value1"}};
  event_logger->EmitEvent("event name", Severity::kDebug,
                          opentelemetry::common::MakeAttributes(vec));
}

// Define a basic Logger class
class TestLogger : public Logger
{
  const nostd::string_view GetName() noexcept override { return "test logger"; }

  nostd::unique_ptr<opentelemetry::logs::LogRecord> CreateLogRecord() noexcept override
  {
    return nullptr;
  }

  using Logger::EmitLogRecord;

  void EmitLogRecord(nostd::unique_ptr<opentelemetry::logs::LogRecord> &&) noexcept override {}
};

// Define a basic LoggerProvider class that returns an instance of the logger class defined above
class TestProvider : public LoggerProvider
{
  nostd::shared_ptr<Logger> GetLogger(nostd::string_view /* logger_name */,
                                      nostd::string_view /* library_name */,
                                      nostd::string_view /* library_version */,
                                      nostd::string_view /* schema_url */,
                                      const common::KeyValueIterable & /* attributes */) override
  {
    return nostd::shared_ptr<Logger>(new TestLogger());
  }
};

TEST(Logger, PushLoggerImplementation)
{
  // Push the new loggerprovider class into the global singleton
  auto test_provider = shared_ptr<LoggerProvider>(new TestProvider());
  Provider::SetLoggerProvider(test_provider);

  auto lp = Provider::GetLoggerProvider();

  // Check that the implementation was pushed by calling TestLogger's GetName()
  nostd::string_view schema_url{"https://opentelemetry.io/schemas/1.11.0"};
  auto logger = lp->GetLogger("TestLogger", "opentelelemtry_library", "", schema_url);
  ASSERT_EQ("test logger", logger->GetName());
}
