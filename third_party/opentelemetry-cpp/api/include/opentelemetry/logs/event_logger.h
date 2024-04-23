// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/logs/log_record.h"
#include "opentelemetry/logs/logger.h"
#include "opentelemetry/logs/logger_type_traits.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/unique_ptr.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace logs
{
/**
 * Handles event log record creation.
 **/
class EventLogger
{
public:
  virtual ~EventLogger() = default;

  /* Returns the name of the logger */
  virtual const nostd::string_view GetName() noexcept = 0;

  /* Returns the delegate logger of this event logger */
  virtual nostd::shared_ptr<Logger> GetDelegateLogger() noexcept = 0;

  /**
   * Emit a event Log Record object
   *
   * @param event_name Event name
   * @param log_record Log record
   */
  virtual void EmitEvent(nostd::string_view event_name,
                         nostd::unique_ptr<LogRecord> &&log_record) noexcept = 0;

  /**
   * Emit a event Log Record object with arguments
   *
   * @param event_name Event name
   * @tparam args Arguments which can be used to set data of log record by type.
   *  Severity                                -> severity, severity_text
   *  string_view                             -> body
   *  AttributeValue                          -> body
   *  SpanContext                             -> span_id,tace_id and trace_flags
   *  SpanId                                  -> span_id
   *  TraceId                                 -> tace_id
   *  TraceFlags                              -> trace_flags
   *  SystemTimestamp                         -> timestamp
   *  system_clock::time_point                -> timestamp
   *  KeyValueIterable                        -> attributes
   *  Key value iterable container            -> attributes
   *  span<pair<string_view, AttributeValue>> -> attributes(return type of MakeAttributes)
   */
  template <class... ArgumentType>
  void EmitEvent(nostd::string_view event_name, ArgumentType &&... args)
  {
    nostd::shared_ptr<Logger> delegate_logger = GetDelegateLogger();
    if (!delegate_logger)
    {
      return;
    }
    nostd::unique_ptr<LogRecord> log_record = delegate_logger->CreateLogRecord();
    if (!log_record)
    {
      return;
    }

    IgnoreTraitResult(
        detail::LogRecordSetterTrait<typename std::decay<ArgumentType>::type>::template Set(
            log_record.get(), std::forward<ArgumentType>(args))...);

    EmitEvent(event_name, std::move(log_record));
  }

private:
  template <class... ValueType>
  void IgnoreTraitResult(ValueType &&...)
  {}
};
}  // namespace logs
OPENTELEMETRY_END_NAMESPACE
