/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiScopeProfilerSession.h"

namespace KINETO_NAMESPACE {

XpuptiScopeProfilerSession::XpuptiScopeProfilerSession(
    XpuptiActivityApi& xpti,
    const std::string& name,
    const libkineto::Config& config,
    const std::set<ActivityType>& activity_types)
    : XpuptiActivityProfilerSession(xpti, name, config, activity_types) {
  scopeProfilerEnabled_ =
      activity_types.count(ActivityType::XPU_SCOPE_PROFILER) > 0;
  if (scopeProfilerEnabled_) {
    xptiScopeProf_.enableScopeProfiler(*config_);
  }
}

XpuptiScopeProfilerSession::~XpuptiScopeProfilerSession() {
  if (scopeProfilerEnabled_) {
    xptiScopeProf_.disableScopeProfiler();
  }
}

void XpuptiScopeProfilerSession::start() {
  XpuptiActivityProfilerSession::start();
  if (scopeProfilerEnabled_) {
    xptiScopeProf_.startScopeActivity();
  }
}

void XpuptiScopeProfilerSession::stop() {
  if (scopeProfilerEnabled_) {
    xptiScopeProf_.stopScopeActivity();
  }
  XpuptiActivityProfilerSession::stop();
}

void XpuptiScopeProfilerSession::toggleCollectionDynamic(const bool enable) {
  XpuptiActivityProfilerSession::toggleCollectionDynamic(enable);
  if (scopeProfilerEnabled_) {
    if (enable) {
      xptiScopeProf_.startScopeActivity();
    } else {
      xptiScopeProf_.stopScopeActivity();
    }
  }
}

void XpuptiScopeProfilerSession::processTrace(ActivityLogger& logger) {
  XpuptiActivityProfilerSession::processTrace(logger);
  if (scopeProfilerEnabled_) {
    xptiScopeProf_.processScopeTrace(
        [this, &logger](
            const pti_metrics_scope_record_t* record,
            const pti_metrics_scope_record_metadata_t& metadata) -> void {
          handleScopeRecord(record, metadata, logger);
        });
  }
}

} // namespace KINETO_NAMESPACE
