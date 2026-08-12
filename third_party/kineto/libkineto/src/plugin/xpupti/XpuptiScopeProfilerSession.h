/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "XpuptiActivityProfilerSession.h"
#include "XpuptiScopeProfilerApi.h"

#include <pti/pti_metrics_scope.h>

namespace KINETO_NAMESPACE {

class XpuptiScopeProfilerApi;

class XpuptiScopeProfilerSession : public XpuptiActivityProfilerSession {
 public:
  XpuptiScopeProfilerSession(
      XpuptiActivityApi& xpti,
      const std::string& name,
      const libkineto::Config& config,
      const std::set<ActivityType>& activity_types);

  XpuptiScopeProfilerSession(const XpuptiScopeProfilerSession&) = delete;
  XpuptiScopeProfilerSession& operator=(const XpuptiScopeProfilerSession&) =
      delete;

  ~XpuptiScopeProfilerSession();
  void start();
  void stop();
  void toggleCollectionDynamic(const bool enable);

  void processTrace(ActivityLogger& logger) override;

  void handleScopeRecord(
      const pti_metrics_scope_record_t* record,
      const pti_metrics_scope_record_metadata_t& metadata,
      ActivityLogger& logger);

 private:
  XpuptiScopeProfilerApi xptiScopeProf_;
  bool scopeProfilerEnabled_{false};
};

} // namespace KINETO_NAMESPACE
