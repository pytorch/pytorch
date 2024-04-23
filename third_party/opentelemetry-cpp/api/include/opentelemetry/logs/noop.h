// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Please refer to provider.h for documentation on how to obtain a Logger object.
//
// This file is part of the internal implementation of OpenTelemetry. Nothing in this file should be
// used directly. Please refer to logger.h for documentation on these interfaces.

#include "opentelemetry/logs/event_logger.h"
#include "opentelemetry/logs/event_logger_provider.h"
#include "opentelemetry/logs/logger.h"
#include "opentelemetry/logs/logger_provider.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/unique_ptr.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace common
{
class KeyValueIterable;
}  // namespace common

namespace logs
{
/**
 * No-op implementation of Logger. This class should not be used directly. It should only be
 * instantiated using a LoggerProvider's GetLogger() call.
 */
class NoopLogger final : public Logger
{
public:
  const nostd::string_view GetName() noexcept override { return "noop logger"; }

  nostd::unique_ptr<LogRecord> CreateLogRecord() noexcept override { return nullptr; }

  using Logger::EmitLogRecord;

  void EmitLogRecord(nostd::unique_ptr<LogRecord> &&) noexcept override {}
};

/**
 * No-op implementation of a LoggerProvider.
 */
class NoopLoggerProvider final : public LoggerProvider
{
public:
  NoopLoggerProvider() : logger_{nostd::shared_ptr<NoopLogger>(new NoopLogger())} {}

  nostd::shared_ptr<Logger> GetLogger(nostd::string_view /* logger_name */,
                                      nostd::string_view /* library_name */,
                                      nostd::string_view /* library_version */,
                                      nostd::string_view /* schema_url */,
                                      const common::KeyValueIterable & /* attributes */) override
  {
    return logger_;
  }

private:
  nostd::shared_ptr<Logger> logger_;
};

class NoopEventLogger final : public EventLogger
{
public:
  NoopEventLogger() : logger_{nostd::shared_ptr<NoopLogger>(new NoopLogger())} {}

  const nostd::string_view GetName() noexcept override { return "noop event logger"; }

  nostd::shared_ptr<Logger> GetDelegateLogger() noexcept override { return logger_; }

  void EmitEvent(nostd::string_view, nostd::unique_ptr<LogRecord> &&) noexcept override {}

private:
  nostd::shared_ptr<Logger> logger_;
};

/**
 * No-op implementation of a EventLoggerProvider.
 */
class NoopEventLoggerProvider final : public EventLoggerProvider
{
public:
  NoopEventLoggerProvider() : event_logger_{nostd::shared_ptr<EventLogger>(new NoopEventLogger())}
  {}

  nostd::shared_ptr<EventLogger> CreateEventLogger(
      nostd::shared_ptr<Logger> /*delegate_logger*/,
      nostd::string_view /*event_domain*/) noexcept override
  {
    return event_logger_;
  }

private:
  nostd::shared_ptr<EventLogger> event_logger_;
};

}  // namespace logs
OPENTELEMETRY_END_NAMESPACE
