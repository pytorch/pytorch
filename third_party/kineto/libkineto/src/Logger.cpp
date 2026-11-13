/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "Logger.h"
#include "ILoggerObserver.h"

#ifndef USE_GOOGLE_LOG

#include <chrono>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include "ThreadUtil.h"

namespace KINETO_NAMESPACE {

std::atomic_int Logger::severityLevel_{VERBOSE};
std::atomic_int Logger::verboseLogLevel_{-1};
std::atomic<uint64_t> Logger::verboseLogModules_{~0ull};

void get_local_time(const time_t* time, struct tm* tm_result) {
#ifdef _WIN32
  // Windows-specific code
  localtime_s(tm_result, time);
#else
  // Linux-specific code
  localtime_r(time, tm_result);
#endif
}

Logger::Logger(int severity, int line, const char* filePath, int errnum)
    : out_(LIBKINETO_DBG_STREAM), errnum_(errnum), messageSeverity_(severity) {
  buf_ << toString((LoggerOutputType)severity) << ":";

  const auto tt =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm_result;
  get_local_time(&tt, &tm_result);
  const char* file = std::strrchr(filePath, '/');
  fmt::print(
      buf_,
      "{:%Y-%m-%d %H:%M:%S} {}:{} {}:{}] ",
      tm_result,
      processId(false),
      systemThreadId(false),
      (file ? file + 1 : filePath),
      line);
}

Logger::~Logger() {
#ifdef __linux__
  if (errnum_ != 0) {
    thread_local char buf[1024];
    buf_ << " : " << strerror_r(errnum_, buf, sizeof(buf));
  }
#endif

  {
    std::lock_guard<std::mutex> guard(loggerObserversMutex());
    for (auto* observer : loggerObservers()) {
      // Output to observers. Current Severity helps keep track of which bucket
      // the output goes.
      if (observer) {
        observer->write(buf_.str(), (LoggerOutputType)messageSeverity_);
      }
    }
  }

  // Finally, print to terminal or console.
  out_ << buf_.str() << std::endl;
}

void Logger::setVerboseLogModules(const std::vector<std::string>& modules) {
  uint64_t mask = 0;
  if (modules.empty()) {
    mask = ~0ull;
  } else {
    for (const std::string& name : modules) {
      mask |= hash(name.c_str());
    }
  }
  verboseLogModules_ = mask;
}

void Logger::addLoggerObserver(ILoggerObserver* observer) {
  if (observer == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  loggerObservers().insert(observer);
}

void Logger::removeLoggerObserver(ILoggerObserver* observer) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  loggerObservers().erase(observer);
}

void Logger::addLoggerObserverDevice(int64_t device) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->addDevice(device);
  }
}

void Logger::addLoggerObserverEventCount(int64_t count) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->addEventCount(count);
  }
}

void Logger::setLoggerObserverTraceDurationMS(int64_t duration) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->setTraceDurationMS(duration);
  }
}

void Logger::setLoggerObserverTraceID(const std::string& tid) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->setTraceID(tid);
  }
}

void Logger::setLoggerObserverGroupTraceID(const std::string& gtid) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->setGroupTraceID(gtid);
  }
}

void Logger::addLoggerObserverDestination(const std::string& dest) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->addDestination(dest);
  }
}

void Logger::setLoggerObserverOnDemand() {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->setTriggerOnDemand();
  }
}

void Logger::resetLoggerObservers() {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->reset();
  }
}

void Logger::addLoggerObserverAddMetadata(
    const std::string& key,
    const std::string& value) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->addMetadata(key, value);
  }
}

void Logger::addLoggerObserverPersistentMetadata(
    const std::string& key,
    const std::string& value) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->addPersistentMetadata(key, value);
  }
}

void Logger::writeStageCancellation(
    const std::string& trace_id,
    const std::string& group_trace_id,
    const std::string& reason) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->writeStageCancellation(trace_id, group_trace_id, reason);
  }
}

USTLoggerStageGuard::~USTLoggerStageGuard() {
  try {
    UST_LOGGER_MARK_COMPLETED(stage_);
  } catch (...) {
  }
}

} // namespace KINETO_NAMESPACE

#endif // USE_GOOGLE_LOG
