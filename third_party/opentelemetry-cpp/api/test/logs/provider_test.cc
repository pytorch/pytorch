// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>

#include "opentelemetry/logs/provider.h"
#include "opentelemetry/nostd/shared_ptr.h"

using opentelemetry::logs::EventLogger;
using opentelemetry::logs::EventLoggerProvider;
using opentelemetry::logs::Logger;
using opentelemetry::logs::LoggerProvider;
using opentelemetry::logs::Provider;
using opentelemetry::nostd::shared_ptr;
namespace nostd = opentelemetry::nostd;

class TestProvider : public LoggerProvider
{
  nostd::shared_ptr<Logger> GetLogger(
      nostd::string_view /* logger_name */,
      nostd::string_view /* library_name */,
      nostd::string_view /* library_version */,
      nostd::string_view /* schema_url */,
      const opentelemetry::common::KeyValueIterable & /* attributes */) override
  {
    return shared_ptr<Logger>(nullptr);
  }
};

TEST(Provider, GetLoggerProviderDefault)
{
  auto tf = Provider::GetLoggerProvider();
  EXPECT_NE(nullptr, tf);
}

TEST(Provider, SetLoggerProvider)
{
  auto tf = shared_ptr<LoggerProvider>(new TestProvider());
  Provider::SetLoggerProvider(tf);
  ASSERT_EQ(tf, Provider::GetLoggerProvider());
}

TEST(Provider, MultipleLoggerProviders)
{
  auto tf = shared_ptr<LoggerProvider>(new TestProvider());
  Provider::SetLoggerProvider(tf);
  auto tf2 = shared_ptr<LoggerProvider>(new TestProvider());
  Provider::SetLoggerProvider(tf2);

  ASSERT_NE(Provider::GetLoggerProvider(), tf);
}

TEST(Provider, GetLogger)
{
  auto tf = shared_ptr<LoggerProvider>(new TestProvider());
  // tests GetLogger(name, version, schema)
  const std::string schema_url{"https://opentelemetry.io/schemas/1.2.0"};
  auto logger = tf->GetLogger("logger1", "opentelelemtry_library", "", schema_url);
  EXPECT_EQ(nullptr, logger);

  // tests GetLogger(name, arguments)

  auto logger2 = tf->GetLogger("logger2", "opentelelemtry_library", "", schema_url);
  EXPECT_EQ(nullptr, logger2);
}

class TestEventLoggerProvider : public EventLoggerProvider
{
public:
  nostd::shared_ptr<EventLogger> CreateEventLogger(
      nostd::shared_ptr<Logger> /*delegate_logger*/,
      nostd::string_view /*event_domain*/) noexcept override
  {
    return nostd::shared_ptr<EventLogger>(nullptr);
  }
};

TEST(Provider, GetEventLoggerProviderDefault)
{
  auto tf = Provider::GetEventLoggerProvider();
  EXPECT_NE(nullptr, tf);
}

TEST(Provider, SetEventLoggerProvider)
{
  auto tf = nostd::shared_ptr<EventLoggerProvider>(new TestEventLoggerProvider());
  Provider::SetEventLoggerProvider(tf);
  ASSERT_EQ(tf, Provider::GetEventLoggerProvider());
}

TEST(Provider, MultipleEventLoggerProviders)
{
  auto tf = nostd::shared_ptr<EventLoggerProvider>(new TestEventLoggerProvider());
  Provider::SetEventLoggerProvider(tf);
  auto tf2 = nostd::shared_ptr<EventLoggerProvider>(new TestEventLoggerProvider());
  Provider::SetEventLoggerProvider(tf2);

  ASSERT_NE(Provider::GetEventLoggerProvider(), tf);
}

TEST(Provider, CreateEventLogger)
{
  auto tf     = nostd::shared_ptr<TestEventLoggerProvider>(new TestEventLoggerProvider());
  auto logger = tf->CreateEventLogger(nostd::shared_ptr<Logger>(nullptr), "domain");

  EXPECT_EQ(nullptr, logger);
}
