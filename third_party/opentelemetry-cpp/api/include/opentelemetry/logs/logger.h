// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/logs/logger_type_traits.h"
#include "opentelemetry/logs/severity.h"
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

class EventId;
class LogRecord;

/**
 * Handles log record creation.
 **/
class Logger
{
public:
  virtual ~Logger() = default;

  /* Returns the name of the logger */
  virtual const nostd::string_view GetName() noexcept = 0;

  /**
   * Create a Log Record object
   *
   * @return nostd::unique_ptr<LogRecord>
   */
  virtual nostd::unique_ptr<LogRecord> CreateLogRecord() noexcept = 0;

  /**
   * Emit a Log Record object
   *
   * @param log_record
   */
  virtual void EmitLogRecord(nostd::unique_ptr<LogRecord> &&log_record) noexcept = 0;

  /**
   * Emit a Log Record object with arguments
   *
   * @param log_record Log record
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
  void EmitLogRecord(nostd::unique_ptr<LogRecord> &&log_record, ArgumentType &&... args)
  {
    if (!log_record)
    {
      return;
    }

    IgnoreTraitResult(
        detail::LogRecordSetterTrait<typename std::decay<ArgumentType>::type>::template Set(
            log_record.get(), std::forward<ArgumentType>(args))...);

    EmitLogRecord(std::move(log_record));
  }

  /**
   * Emit a Log Record object with arguments
   *
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
  void EmitLogRecord(ArgumentType &&... args)
  {
    nostd::unique_ptr<LogRecord> log_record = CreateLogRecord();
    if (!log_record)
    {
      return;
    }

    EmitLogRecord(std::move(log_record), std::forward<ArgumentType>(args)...);
  }

  /**
   * Writes a log with a severity of trace.
   * @tparam args Arguments which can be used to set data of log record by type.
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
  void Trace(ArgumentType &&... args) noexcept
  {
    static_assert(
        !detail::LogRecordHasType<Severity, typename std::decay<ArgumentType>::type...>::value,
        "Severity is already set.");
    this->EmitLogRecord(Severity::kTrace, std::forward<ArgumentType>(args)...);
  }

  /**
   * Writes a log with a severity of debug.
   * @tparam args Arguments which can be used to set data of log record by type.
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
  void Debug(ArgumentType &&... args) noexcept
  {
    static_assert(
        !detail::LogRecordHasType<Severity, typename std::decay<ArgumentType>::type...>::value,
        "Severity is already set.");
    this->EmitLogRecord(Severity::kDebug, std::forward<ArgumentType>(args)...);
  }

  /**
   * Writes a log with a severity of info.
   * @tparam args Arguments which can be used to set data of log record by type.
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
  void Info(ArgumentType &&... args) noexcept
  {
    static_assert(
        !detail::LogRecordHasType<Severity, typename std::decay<ArgumentType>::type...>::value,
        "Severity is already set.");
    this->EmitLogRecord(Severity::kInfo, std::forward<ArgumentType>(args)...);
  }

  /**
   * Writes a log with a severity of warn.
   * @tparam args Arguments which can be used to set data of log record by type.
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
  void Warn(ArgumentType &&... args) noexcept
  {
    static_assert(
        !detail::LogRecordHasType<Severity, typename std::decay<ArgumentType>::type...>::value,
        "Severity is already set.");
    this->EmitLogRecord(Severity::kWarn, std::forward<ArgumentType>(args)...);
  }

  /**
   * Writes a log with a severity of error.
   * @tparam args Arguments which can be used to set data of log record by type.
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
  void Error(ArgumentType &&... args) noexcept
  {
    static_assert(
        !detail::LogRecordHasType<Severity, typename std::decay<ArgumentType>::type...>::value,
        "Severity is already set.");
    this->EmitLogRecord(Severity::kError, std::forward<ArgumentType>(args)...);
  }

  /**
   * Writes a log with a severity of fatal.
   * @tparam args Arguments which can be used to set data of log record by type.
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
  void Fatal(ArgumentType &&... args) noexcept
  {
    static_assert(
        !detail::LogRecordHasType<Severity, typename std::decay<ArgumentType>::type...>::value,
        "Severity is already set.");
    this->EmitLogRecord(Severity::kFatal, std::forward<ArgumentType>(args)...);
  }

  //
  // OpenTelemetry C++ user-facing Logs API
  //

  inline bool Enabled(Severity severity, const EventId &event_id) const noexcept
  {
    OPENTELEMETRY_LIKELY_IF(Enabled(severity) == false) { return false; }
    return EnabledImplementation(severity, event_id);
  }

  inline bool Enabled(Severity severity, int64_t event_id) const noexcept
  {
    OPENTELEMETRY_LIKELY_IF(Enabled(severity) == false) { return false; }
    return EnabledImplementation(severity, event_id);
  }

  inline bool Enabled(Severity severity) const noexcept
  {
    return static_cast<uint8_t>(severity) >= OPENTELEMETRY_ATOMIC_READ_8(&minimum_severity_);
  }

  /**
   * Log an event
   *
   * @severity severity of the log
   * @event_id event identifier of the log
   * @format an utf-8 string following https://messagetemplates.org/
   * @attributes key value pairs of the log
   */
  virtual void Log(Severity severity,
                   const EventId &event_id,
                   nostd::string_view format,
                   const common::KeyValueIterable &attributes) noexcept
  {
    this->EmitLogRecord(severity, event_id, format, attributes);
  }

  virtual void Log(Severity severity,
                   int64_t event_id,
                   nostd::string_view format,
                   const common::KeyValueIterable &attributes) noexcept
  {
    this->EmitLogRecord(severity, EventId{event_id}, format, attributes);
  }

  virtual void Log(Severity severity,
                   nostd::string_view format,
                   const common::KeyValueIterable &attributes) noexcept
  {
    this->EmitLogRecord(severity, format, attributes);
  }

  virtual void Log(Severity severity, nostd::string_view message) noexcept
  {
    this->EmitLogRecord(severity, message);
  }

  // Convenient wrappers based on virtual methods Log().
  // https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/logs/data-model.md#field-severitynumber

  inline void Trace(const EventId &event_id,
                    nostd::string_view format,
                    const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kTrace, event_id, format, attributes);
  }

  inline void Trace(int64_t event_id,
                    nostd::string_view format,
                    const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kTrace, EventId{event_id}, format, attributes);
  }

  inline void Trace(nostd::string_view format, const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kTrace, format, attributes);
  }

  inline void Trace(nostd::string_view message) noexcept { this->Log(Severity::kTrace, message); }

  inline void Debug(const EventId &event_id,
                    nostd::string_view format,
                    const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kDebug, event_id, format, attributes);
  }

  inline void Debug(int64_t event_id,
                    nostd::string_view format,
                    const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kDebug, EventId{event_id}, format, attributes);
  }

  inline void Debug(nostd::string_view format, const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kDebug, format, attributes);
  }

  inline void Debug(nostd::string_view message) noexcept { this->Log(Severity::kDebug, message); }

  inline void Info(const EventId &event_id,
                   nostd::string_view format,
                   const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kInfo, event_id, format, attributes);
  }

  inline void Info(int64_t event_id,
                   nostd::string_view format,
                   const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kInfo, EventId{event_id}, format, attributes);
  }

  inline void Info(nostd::string_view format, const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kInfo, format, attributes);
  }

  inline void Info(nostd::string_view message) noexcept { this->Log(Severity::kInfo, message); }

  inline void Warn(const EventId &event_id,
                   nostd::string_view format,
                   const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kWarn, event_id, format, attributes);
  }

  inline void Warn(int64_t event_id,
                   nostd::string_view format,
                   const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kWarn, EventId{event_id}, format, attributes);
  }

  inline void Warn(nostd::string_view format, const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kWarn, format, attributes);
  }

  inline void Warn(nostd::string_view message) noexcept { this->Log(Severity::kWarn, message); }

  inline void Error(const EventId &event_id,
                    nostd::string_view format,
                    const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kError, event_id, format, attributes);
  }

  inline void Error(int64_t event_id,
                    nostd::string_view format,
                    const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kError, EventId{event_id}, format, attributes);
  }

  inline void Error(nostd::string_view format, const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kError, format, attributes);
  }

  inline void Error(nostd::string_view message) noexcept { this->Log(Severity::kError, message); }

  inline void Fatal(const EventId &event_id,
                    nostd::string_view format,
                    const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kFatal, event_id, format, attributes);
  }

  inline void Fatal(int64_t event_id,
                    nostd::string_view format,
                    const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kFatal, EventId{event_id}, format, attributes);
  }

  inline void Fatal(nostd::string_view format, const common::KeyValueIterable &attributes) noexcept
  {
    this->Log(Severity::kFatal, format, attributes);
  }

  inline void Fatal(nostd::string_view message) noexcept { this->Log(Severity::kFatal, message); }

  //
  // End of OpenTelemetry C++ user-facing Log API.
  //

protected:
  // TODO: discuss with community about naming for internal methods.
  virtual bool EnabledImplementation(Severity /*severity*/,
                                     const EventId & /*event_id*/) const noexcept
  {
    return false;
  }

  virtual bool EnabledImplementation(Severity /*severity*/, int64_t /*event_id*/) const noexcept
  {
    return false;
  }

  void SetMinimumSeverity(uint8_t severity_or_max) noexcept
  {
    OPENTELEMETRY_ATOMIC_WRITE_8(&minimum_severity_, severity_or_max);
  }

private:
  template <class... ValueType>
  void IgnoreTraitResult(ValueType &&...)
  {}

  //
  // minimum_severity_ can be updated concurrently by multiple threads/cores, so race condition on
  // read/write should be handled. And std::atomic can not be used here because it is not ABI
  // compatible for OpenTelemetry C++ API.
  //
  mutable uint8_t minimum_severity_{kMaxSeverity};
};
}  // namespace logs
OPENTELEMETRY_END_NAMESPACE
