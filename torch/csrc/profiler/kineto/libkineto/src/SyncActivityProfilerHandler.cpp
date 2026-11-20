/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SyncActivityProfilerHandler.h"

#include <chrono>

#include "ActivityProfilerController.h"
#include "ActivityTrace.h"
#include "Config.h"
#include "GenericActivityProfiler.h"
#include "Logger.h"
#include "libkineto.h"
#include "output_membuf.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

SyncActivityProfilerHandler::SyncActivityProfilerHandler(
    GenericActivityProfiler& profiler)
    : profiler_(profiler) {}

void SyncActivityProfilerHandler::prepareTrace(const Config& config) {
  auto now = std::chrono::system_clock::now();
  active_ = true;

  LOGGER_OBSERVER_RESET();
  profiler_.configure(config, now);
}

void SyncActivityProfilerHandler::startTrace() {
  UST_LOGGER_MARK_COMPLETED(kWarmUpStage);
  USDT_EMIT_START_TRACE();
  profiler_.startTrace(std::chrono::system_clock::now());
}

std::unique_ptr<ActivityTraceInterface> SyncActivityProfilerHandler::
    stopTrace() {
  profiler_.stopTrace(std::chrono::system_clock::now());
  USDT_EMIT_STOP_TRACE();
  UST_LOGGER_MARK_COMPLETED(kCollectionStage);
  auto logger = std::make_unique<MemoryTraceLogger>(profiler_.config());
  profiler_.processTrace(*logger);
  // Will follow up with another patch for logging URLs when ActivityTrace
  // is moved.

  profiler_.reset();
  active_ = false;
  return std::make_unique<ActivityTrace>(
      std::move(logger), ActivityProfilerController::loggerFactory());
}

void SyncActivityProfilerHandler::cancel() {
  if (!active_) {
    return;
  }

  LOG(ERROR) << "Cancelling current trace request in order to start "
             << "higher priority synchronous request";
  UST_LOGGER_MARK_COMPLETED(kCancellationStage);

  if (libkineto::api().client() != nullptr) {
    libkineto::api().client()->stop();
  }
  profiler_.cancelTrace(std::chrono::system_clock::now());
  active_ = false;
}

void SyncActivityProfilerHandler::toggleCollectionDynamic(const bool enable) {
  profiler_.toggleCollectionDynamic(enable);
}

} // namespace KINETO_NAMESPACE
