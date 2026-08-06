/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <set>
#include <string>

namespace libkineto {

// Note : All activity types are not enabled by default. Please add them
// at correct position in the enum
enum class ActivityType {
  // Activity types enabled by default
  CPU_OP = 0, // cpu side ops
  USER_ANNOTATION = 1,
  GPU_USER_ANNOTATION = 2,
  GPU_MEMCPY = 3,
  GPU_MEMSET = 4,
  CONCURRENT_KERNEL = 5, // on-device kernels
  EXTERNAL_CORRELATION = 6,
  CUDA_RUNTIME = 7, // host side cuda runtime events
  CUDA_DRIVER = 8, // host side cuda driver events
  CPU_INSTANT_EVENT = 9, // host side point-like events
  PYTHON_FUNCTION = 10,
  OVERHEAD = 11, // CUPTI induced overhead events sampled from its overhead API.
  MTIA_RUNTIME = 12, // host side MTIA runtime events
  MTIA_CCP_EVENTS = 13, // MTIA ondevice CCP events
  MTIA_INSIGHT = 14, // MTIA Insight Events
  CUDA_SYNC = 15, // synchronization events between runtime and kernels
  CUDA_EVENT = 16, // CUDA event activities (cudaEventRecord, etc.)
  MTIA_COUNTERS = 17, // MTIA hardware counter events (HBM, cache, DPE, SFU)

  // Optional Activity types
  GLOW_RUNTIME = 18, // host side glow runtime events
  // Reserved and no longer produced: the CUPTI range profiler that emitted this
  // activity was removed. The value stays fixed so existing traces and other
  // tools that share this serialized integer keep deserializing correctly.
  CUDA_PROFILER_RANGE = 19,
  HPU_OP = 20, // HPU host side runtime event
  XPU_RUNTIME = 21, // host side xpu runtime events
  XPU_DRIVER = 22, // host side xpu driver events
  COLLECTIVE_COMM = 23, // collective communication

  // PRIVATEUSE1 Activity types are used for custom backends.
  // The corresponding device type is `DeviceType::PrivateUse1` in PyTorch.
  PRIVATEUSE1_RUNTIME = 24, // host side privateUse1 runtime events
  PRIVATEUSE1_DRIVER = 25, // host side privateUse1 driver events

  XPU_SCOPE_PROFILER = 26, // XPUPTI Profiler scope for performance metrics
  XPU_SYNC = 27, // XPU synchronization events

  ENUM_COUNT =
      28, // This is to add buffer and not used for any profiling logic. Add
  // your new type before it.
  OPTIONAL_ACTIVITY_TYPE_START = GLOW_RUNTIME,
};

// Return an array of all activity types except COUNT
constexpr int activityTypeCount = (int)ActivityType::ENUM_COUNT;
constexpr int defaultActivityTypeCount =
    (int)ActivityType::OPTIONAL_ACTIVITY_TYPE_START;

// These definitions are not part of the public Kineto API. They are inlined
// here because some build configurations include this header
// without linking libkineto, and toString() must resolve at compile time.
struct _ActivityTypeName {
  const char* name;
  ActivityType type;
};

inline constexpr std::array<_ActivityTypeName, activityTypeCount + 1>
    _activityTypeNames{{
        {"cpu_op", ActivityType::CPU_OP},
        {"user_annotation", ActivityType::USER_ANNOTATION},
        {"gpu_user_annotation", ActivityType::GPU_USER_ANNOTATION},
        {"gpu_memcpy", ActivityType::GPU_MEMCPY},
        {"gpu_memset", ActivityType::GPU_MEMSET},
        {"kernel", ActivityType::CONCURRENT_KERNEL},
        {"external_correlation", ActivityType::EXTERNAL_CORRELATION},
        {"cuda_runtime", ActivityType::CUDA_RUNTIME},
        {"cuda_driver", ActivityType::CUDA_DRIVER},
        {"cpu_instant_event", ActivityType::CPU_INSTANT_EVENT},
        {"python_function", ActivityType::PYTHON_FUNCTION},
        {"overhead", ActivityType::OVERHEAD},
        {"mtia_runtime", ActivityType::MTIA_RUNTIME},
        {"mtia_ccp_events", ActivityType::MTIA_CCP_EVENTS},
        {"mtia_insight", ActivityType::MTIA_INSIGHT},
        {"cuda_sync", ActivityType::CUDA_SYNC},
        {"cuda_event", ActivityType::CUDA_EVENT},
        {"mtia_counters", ActivityType::MTIA_COUNTERS},
        {"glow_runtime", ActivityType::GLOW_RUNTIME},
        {"cuda_profiler_range", ActivityType::CUDA_PROFILER_RANGE},
        {"hpu_op", ActivityType::HPU_OP},
        {"xpu_runtime", ActivityType::XPU_RUNTIME},
        {"xpu_driver", ActivityType::XPU_DRIVER},
        {"collective_comm", ActivityType::COLLECTIVE_COMM},
        {"privateuse1_runtime", ActivityType::PRIVATEUSE1_RUNTIME},
        {"privateuse1_driver", ActivityType::PRIVATEUSE1_DRIVER},
        {"xpu_scope_profiler", ActivityType::XPU_SCOPE_PROFILER},
        {"xpu_sync", ActivityType::XPU_SYNC},
        {"ENUM_COUNT", ActivityType::ENUM_COUNT},
    }};

inline const char* toString(ActivityType t) {
  return _activityTypeNames[static_cast<int>(t)].name;
}

ActivityType toActivityType(const std::string& str);

std::array<ActivityType, activityTypeCount> activityTypes();
std::array<ActivityType, defaultActivityTypeCount> defaultActivityTypes();

} // namespace libkineto
