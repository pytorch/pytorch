/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#ifdef HAS_ROCTRACER

#include <atomic>
#include <functional>
#include <set>

#include "RocprofLogger.h"

#include "ActivityType.h"
#include "GenericTraceActivity.h"

class RocprofLogger;

namespace KINETO_NAMESPACE {

using namespace libkineto;

class RocprofActivityApi {
 public:
  enum CorrelationFlowType { Default, User };

  RocprofActivityApi();
  RocprofActivityApi(const RocprofActivityApi&) = delete;
  RocprofActivityApi& operator=(const RocprofActivityApi&) = delete;

  virtual ~RocprofActivityApi();

  static RocprofActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableActivities(const std::set<ActivityType>& selected_activities);
  void disableActivities(const std::set<ActivityType>& selected_activities);
  void flushActivities();
  void clearActivities();
  void teardownContext() {}
  void setTimeOffset(timestamp_t toffset);
  void setMaxEvents(uint32_t maxEvents);

  virtual int processActivities(
      std::function<void(const rocprofBase*)> handler,
      std::function<void(uint64_t, uint64_t, RocLogger::CorrelationDomain)>
          correlationHandler);

  void setMaxBufferSize(int64_t size);

  std::atomic_bool stopCollection{false};

 private:
  bool registered_{false};
  timestamp_t toffset_{0};

  // Enabled Activity Filters
  uint32_t activityMask_{0};
  uint32_t activityMaskSnapshot_{0};
  bool isLogged(libkineto::ActivityType atype) const;

  RocprofLogger* d;
};

} // namespace KINETO_NAMESPACE
#endif
