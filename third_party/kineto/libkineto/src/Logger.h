/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>

#define LIBKINETO_DBG_STREAM std::cerr

#if USE_GOOGLE_LOG

#include <glog/logging.h>

#define SET_LOG_SEVERITY_LEVEL(level)
#define SET_LOG_VERBOSITY_LEVEL(level, modules)
#define LOGGER_OBSERVER_ADD_DEVICE(device)
#define LOGGER_OBSERVER_ADD_EVENT_COUNT(count)
#define LOGGER_OBSERVER_SET_TRACE_DURATION_MS(duration)
#define LOGGER_OBSERVER_SET_TRACE_ID(tid)
#define LOGGER_OBSERVER_SET_GROUP_TRACE_ID(gtid)
#define LOGGER_OBSERVER_ADD_DESTINATION(dest)
#define LOGGER_OBSERVER_SET_TRIGGER_ON_DEMAND()
#define LOGGER_OBSERVER_ADD_METADATA(key, value)
#define LOGGER_OBSERVER_ADD_PERSISTENT_METADATA(key, value)
#define LOGGER_OBSERVER_RESET()
#define LOGGER_OBSERVER_WRITE_STAGE_CANCELLATION( \
    trace_id, group_trace_id, reason)
#define UST_LOGGER_MARK_COMPLETED(stage)
#define UST_LOGGER_STAGE_SCOPE(stage)
#define USDT_LOGGER_EMIT_MESSAGE(usdt_type)
#define USDT_EMIT_START_TRACE()
#define USDT_EMIT_STOP_TRACE()

#else // !USE_GOOGLE_LOG
#include <stdio.h>
#include <atomic>
#include <map>
#include <mutex>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ILoggerObserver.h"

#ifdef _MSC_VER
// unset a predefined ERROR (windows)
#undef ERROR
#endif // _MSC_VER

namespace KINETO_NAMESPACE {
void get_local_time(const time_t* time, struct tm* tm_result);

class Logger {
 public:
  Logger(int severity, int line, const char* filePath, int errnum = 0);
  ~Logger();

  inline std::ostream& stream() {
    return buf_;
  }

  static inline void setSeverityLevel(int level) {
    severityLevel_ = level;
  }

  static inline int severityLevel() {
    return severityLevel_;
  }

  static inline void setVerboseLogLevel(int level) {
    verboseLogLevel_ = level;
  }

  static inline int verboseLogLevel() {
    return verboseLogLevel_;
  }

  // This is constexpr so that the hash for a file name is computed at compile
  // time when used in the VLOG macros.
  // This way, there is no string comparison for matching VLOG modules,
  // only a comparison of pre-computed hashes.
  // No fancy hashing needed here. It's pretty inefficient (one character
  // at a time) but the strings are not large and it's not in the critical path.
  static constexpr uint64_t rol(uint64_t val, int amount) {
    return val << amount | val >> (63 - amount);
  }
  static constexpr uint64_t hash(const char* s) {
    uint64_t hash = hash_rec(s, 0);
    return hash & rol(0x41a0240682483014ull, hash & 63);
  }
  static constexpr uint64_t hash_rec(const char* s, int off) {
    // Random constants!
    return (!s[off] ? 57ull : (hash_rec(s, off + 1) * 293) ^ s[off]);
  }
  static constexpr const char* basename(const char* s, int off = 0) {
    return !s[off]      ? s
        : s[off] == '/' ? basename(&s[off + 1])
                        : basename(s, off + 1);
  }

  static void setVerboseLogModules(const std::vector<std::string>& modules);

  static inline uint64_t verboseLogModules() {
    return verboseLogModules_;
  }

  static void clearLoggerObservers() {
    std::lock_guard<std::mutex> g(loggerObserversMutex());
    loggerObservers().clear();
  }

  static void addLoggerObserver(ILoggerObserver* observer);

  static void removeLoggerObserver(ILoggerObserver* observer);

  static void addLoggerObserverDevice(int64_t device);

  static void addLoggerObserverEventCount(int64_t count);

  static void setLoggerObserverTraceDurationMS(int64_t duration);

  static void setLoggerObserverTraceID(const std::string& tid);

  static void setLoggerObserverGroupTraceID(const std::string& gtid);

  static void addLoggerObserverDestination(const std::string& dest);

  static void setLoggerObserverOnDemand();

  static void resetLoggerObservers();

  static void addLoggerObserverAddMetadata(
      const std::string& key,
      const std::string& value);

  static void addLoggerObserverPersistentMetadata(
      const std::string& key,
      const std::string& value);

  static void writeStageCancellation(
      const std::string& trace_id,
      const std::string& group_trace_id,
      const std::string& reason);

 private:
  std::stringstream buf_;
  std::ostream& out_;
  int errnum_;
  int messageSeverity_;
  static std::atomic_int severityLevel_;
  static std::atomic_int verboseLogLevel_;
  static std::atomic<uint64_t> verboseLogModules_;
  static std::set<ILoggerObserver*>& loggerObservers() {
    static auto* inst = new std::set<ILoggerObserver*>();
    return *inst;
  }
  static std::mutex& loggerObserversMutex() {
    static auto* loggerObserversMutex = new std::mutex();
    return *loggerObserversMutex;
  }
};

class VoidLogger {
 public:
  VoidLogger() {}
  void operator&(std::ostream&) {}
};

// RAII helper used to ensure a UST stage row is emitted on every exit from a
// scope. Bucketed LOG(ERROR) / LOG(WARNING) calls within the scope are picked
// up by the emitted row as usual.
class USTLoggerStageGuard {
 public:
  explicit USTLoggerStageGuard(const std::string& stage) : stage_(stage) {}
  ~USTLoggerStageGuard();

  USTLoggerStageGuard(const USTLoggerStageGuard&) = delete;
  USTLoggerStageGuard& operator=(const USTLoggerStageGuard&) = delete;
  USTLoggerStageGuard(USTLoggerStageGuard&&) = delete;
  USTLoggerStageGuard& operator=(USTLoggerStageGuard&&) = delete;

 private:
  std::string stage_;
};

} // namespace KINETO_NAMESPACE

#ifdef LOG // Undefine in case these are already defined (quite likely)
#undef LOG
#undef LOG_IS_ON
#undef LOG_IF
#undef LOG_EVERY_N
#undef LOG_IF_EVERY_N
#undef DLOG
#undef DLOG_IF
#undef VLOG
#undef VLOG_IF
#undef VLOG_EVERY_N
#undef VLOG_IS_ON
#undef DVLOG
#undef LOG_FIRST_N
#undef CHECK
#undef DCHECK
#undef DCHECK_EQ
#undef PLOG
#undef PCHECK
#undef LOG_OCCURRENCES
#endif

#define LOG_IS_ON(severity) (severity >= libkineto::Logger::severityLevel())

#define LOG_IF(severity, condition)                                 \
  !(LOG_IS_ON(severity) && (condition)) ? (void)0                   \
                                        : libkineto::VoidLogger() & \
          libkineto::Logger(severity, __LINE__, __FILE__).stream()

#define LOG(severity) LOG_IF(severity, true)

#define LOCAL_VARNAME_CONCAT(name, suffix) _##name##suffix##_

#define LOCAL_VARNAME(name) LOCAL_VARNAME_CONCAT(name, __LINE__)

#define LOG_OCCURRENCES LOCAL_VARNAME(log_count)

#define LOG_EVERY_N(severity, rate)               \
  static int LOG_OCCURRENCES = 0;                 \
  LOG_IF(severity, LOG_OCCURRENCES++ % rate == 0) \
      << "(x" << LOG_OCCURRENCES << ") "

#define LOG_FIRST_N(severity, threshold)          \
  static int LOG_OCCURRENCES = 0;                 \
  LOG_IF(severity, LOG_OCCURRENCES++ < threshold) \
      << "(x" << LOG_OCCURRENCES << ") "

template <uint64_t n>
struct __to_constant__ {
  static const uint64_t val = n;
};
#define FILENAME_HASH                      \
  __to_constant__<libkineto::Logger::hash( \
      libkineto::Logger::basename(__FILE__))>::val
#define VLOG_IS_ON(verbosity)                           \
  (libkineto::Logger::verboseLogLevel() >= verbosity && \
   (libkineto::Logger::verboseLogModules() & FILENAME_HASH) == FILENAME_HASH)

#define VLOG_IF(verbosity, condition) \
  LOG_IF(VERBOSE, VLOG_IS_ON(verbosity) && (condition))

#define VLOG(verbosity) VLOG_IF(verbosity, true)

#define VLOG_EVERY_N(verbosity, rate)               \
  static int LOG_OCCURRENCES = 0;                   \
  VLOG_IF(verbosity, LOG_OCCURRENCES++ % rate == 0) \
      << "(x" << LOG_OCCURRENCES << ") "

#define PLOG(severity) \
  libkineto::Logger(severity, __LINE__, __FILE__, errno).stream()

#define SET_LOG_SEVERITY_LEVEL(level) libkineto::Logger::setSeverityLevel(level)

#define SET_LOG_VERBOSITY_LEVEL(level, modules) \
  libkineto::Logger::setVerboseLogLevel(level); \
  libkineto::Logger::setVerboseLogModules(modules)

// Logging the set of devices the trace is collect on.
#define LOGGER_OBSERVER_ADD_DEVICE(device_count) \
  libkineto::Logger::addLoggerObserverDevice(device_count)

// Incrementing the number of events collected by this trace.
#define LOGGER_OBSERVER_ADD_EVENT_COUNT(count) \
  libkineto::Logger::addLoggerObserverEventCount(count)

// Record duration of trace in milliseconds.
#define LOGGER_OBSERVER_SET_TRACE_DURATION_MS(duration) \
  libkineto::Logger::setLoggerObserverTraceDurationMS(duration)

// Record the trace id when given.
#define LOGGER_OBSERVER_SET_TRACE_ID(tid) \
  libkineto::Logger::setLoggerObserverTraceID(tid)

// Record the group trace id when given.
#define LOGGER_OBSERVER_SET_GROUP_TRACE_ID(gtid) \
  libkineto::Logger::setLoggerObserverGroupTraceID(gtid)

// Log the set of destinations the trace is sent to.
#define LOGGER_OBSERVER_ADD_DESTINATION(dest) \
  libkineto::Logger::addLoggerObserverDestination(dest)

// Record this was triggered by On-Demand.
#define LOGGER_OBSERVER_SET_TRIGGER_ON_DEMAND() \
  libkineto::Logger::setLoggerObserverOnDemand()

// Reset all logger observers to a clean per-trace state.
#define LOGGER_OBSERVER_RESET() libkineto::Logger::resetLoggerObservers()

// Record that an on-demand trace request was rejected.
#define LOGGER_OBSERVER_WRITE_STAGE_CANCELLATION( \
    trace_id, group_trace_id, reason)             \
  libkineto::Logger::writeStageCancellation(trace_id, group_trace_id, reason)

// Record this was triggered by On-Demand.
#define LOGGER_OBSERVER_ADD_METADATA(key, value) \
  libkineto::Logger::addLoggerObserverAddMetadata(key, value)

// Metadata that is constant for the process lifetime and survives reset().
#define LOGGER_OBSERVER_ADD_PERSISTENT_METADATA(key, value) \
  libkineto::Logger::addLoggerObserverPersistentMetadata(key, value)

// UST Logger Semantics to describe when a stage is complete.
// Use libkineto::Logger directly instead of the LOG/LOG_IS_ON macros, which
// can be redefined by glog in translation units that include both Logger.h
// and glog/logging.h. Inline the severity gate so messages are still
// suppressed when libkineto's severity threshold is above the message
// severity.
#define UST_LOGGER_MARK_COMPLETED(stage)                                      \
  !(libkineto::LoggerOutputType::STAGE >= libkineto::Logger::severityLevel()) \
      ? (void)0                                                               \
      : libkineto::VoidLogger() &                                             \
          libkineto::Logger(                                                  \
              libkineto::LoggerOutputType::STAGE, __LINE__, __FILE__)         \
                  .stream()                                                   \
              << "Completed Stage: " << stage

// RAII helper that fires UST_LOGGER_MARK_COMPLETED(stage) on scope exit.
#define UST_LOGGER_STAGE_SCOPE(stage) \
  libkineto::USTLoggerStageGuard LOCAL_VARNAME(ust_stage_guard)(stage)

#define USDT_LOGGER_EMIT_MESSAGE(usdt_type)                                  \
  !(libkineto::LoggerOutputType::USDT >= libkineto::Logger::severityLevel()) \
      ? (void)0                                                              \
      : libkineto::VoidLogger() &                                            \
          libkineto::Logger(                                                 \
              libkineto::LoggerOutputType::USDT, __LINE__, __FILE__)         \
                  .stream()                                                  \
              << usdt_type
#define USDT_EMIT_START_TRACE() USDT_LOGGER_EMIT_MESSAGE("profiler_start")
#define USDT_EMIT_STOP_TRACE() USDT_LOGGER_EMIT_MESSAGE("profiler_stop")

#endif // USE_GOOGLE_LOG
