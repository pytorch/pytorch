//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "src/plugin/xpupti/XpuptiActivityProfilerSession.h"

#include <gtest/gtest.h>

namespace KN = KINETO_NAMESPACE;

// Exercises the predicate that decides which records start a CPU->GPU ("ac2g")
// flow arrow. Runs without a GPU: only the static predicate is queried, no PTI
// collection is performed.

TEST(XpuptiFlowCorrelationTest, RuntimeRecordsStartFlow) {
  // Host runtime records (kernel launches, memcpy/fill enqueues) are the source
  // of the CPU->GPU arrow. The runtime view is filtered to work-submitting APIs
  // by ptiViewEnableRuntimeApiClass(PTI_API_CLASS_GPU_OPERATION_CORE).
  EXPECT_TRUE(KN::XpuptiActivityProfilerSession::startsFlow(
      KN::ActivityType::XPU_RUNTIME));
}

TEST(XpuptiFlowCorrelationTest, DriverRecordsDoNotStartFlow) {
  // Driver (ze*) records share the runtime record's correlation id; if they
  // also started a flow the trace would have a duplicate flow start for that
  // id (Perfetto flow_duplicate_id). They must only be flow ends.
  EXPECT_FALSE(KN::XpuptiActivityProfilerSession::startsFlow(
      KN::ActivityType::XPU_DRIVER));
}

TEST(XpuptiFlowCorrelationTest, DeviceRecordsDoNotStartFlow) {
  // GPU-side activities are flow destinations, never sources.
  EXPECT_FALSE(KN::XpuptiActivityProfilerSession::startsFlow(
      KN::ActivityType::CONCURRENT_KERNEL));
  EXPECT_FALSE(KN::XpuptiActivityProfilerSession::startsFlow(
      KN::ActivityType::GPU_MEMCPY));
  EXPECT_FALSE(KN::XpuptiActivityProfilerSession::startsFlow(
      KN::ActivityType::GPU_MEMSET));
}
