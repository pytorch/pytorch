/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "GenericTraceActivity.h"
#include "ITraceActivity.h"
#include "MetadataFieldCatalog.h"
#include "RocprofLogger.h"
#include "ThreadUtil.h"

#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/fwd.h>

namespace libkineto {
class ActivityLogger;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;
struct TraceSpan;

// These classes wrap the various Rocprof activity types
// into subclasses of ITraceActivity so that they can all be accessed
// using the ITraceActivity interface and logged via ActivityLogger.

// Abstract base class, templated on Rocprof activity type
template <class T>
struct RocprofActivity : public ITraceActivity {
  explicit RocprofActivity(const T* activity, const ITraceActivity* linked)
      : activity_(*activity), linked_(linked) {}
  // Our stored timestamps (from rocprof and generated) are in CLOCK_MONOTONIC
  // domain (in ns). Convert the timestamps.
  int64_t timestamp() const override {
    return activity_.begin;
  }
  int64_t duration() const override {
    return activity_.end - activity_.begin;
  }
  int64_t correlationId() const override {
    return 0;
  }
  int32_t getThreadId() const override {
    return 0;
  }
  const ITraceActivity* linkedActivity() const override {
    return linked_;
  }
  int flowType() const override {
    return kLinkAsyncCpuGpu;
  }
  int64_t flowId() const override {
    return correlationId();
  }
  const T& raw() const {
    return activity_;
  }
  const TraceSpan* traceSpan() const override {
    return nullptr;
  }
  const std::string getMetadataValue(const std::string& key) const override {
    auto it = metadata_.find(key);
    if (it != metadata_.end()) {
      return it->second;
    }
    return "";
  }

 protected:
  const T& activity_;
  const ITraceActivity* linked_{nullptr};
  std::unordered_map<std::string, std::string> metadata_;
};

// rocprofAsyncRow - Rocprof GPU activities
struct GpuActivity : public RocprofActivity<rocprofAsyncRow> {
  explicit GpuActivity(
      const rocprofAsyncRow* activity,
      const ITraceActivity* linked)
      : RocprofActivity(activity, linked) {
    switch (activity_.domain) {
      case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
        type_ = ActivityType::GPU_MEMCPY;
        break;
      case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
      default:
        type_ = ActivityType::CONCURRENT_KERNEL;
        break;
    }
  }
  int64_t correlationId() const override {
    return activity_.id;
  }
  int64_t deviceId() const override {
    return activity_.device;
  }
  int64_t resourceId() const override {
    return activity_.stream != 0 ? activity_.stream : activity_.queue;
  }
  ActivityType type() const override {
    return type_;
  };
  bool flowStart() const override {
    return false;
  }
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  void visitTypedMetadata(ITypedMetadataVisitor& visitor) const override;

  // Add small buffer to fix visual error created by
  // https://github.com/ROCm/rocprof/issues/105 Once this is resolved we can
  // use ifdef to handle having this buffer or not based on version
  int64_t timestamp() const override {
    return activity_.begin + 1;
  }
  int64_t duration() const override {
    return activity_.end - (activity_.begin + 1);
  }

 private:
  ActivityType type_;
};

// rocprofRow, rocprofKernelRow, rocprofCopyRow, rocprofMallocRow -
// Rocprof runtime activities
template <class T>
struct RuntimeActivity : public RocprofActivity<T> {
  explicit RuntimeActivity(const T* activity, const ITraceActivity* linked)
      : RocprofActivity<T>(activity, linked) {}
  int64_t correlationId() const override {
    return raw().id;
  }
  int64_t deviceId() const override {
    return raw().pid;
  }
  int64_t resourceId() const override {
    return raw().tid;
  }
  ActivityType type() const override {
    return ActivityType::CUDA_RUNTIME;
  }
  bool flowStart() const override;
  const std::string name() const override {
    return RocprofLogger::opString(
        ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, raw().cid);
  }
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  void visitTypedMetadata(ITypedMetadataVisitor& visitor) const override;
  const T& raw() const {
    return RocprofActivity<T>::raw();
  }
};

} // namespace KINETO_NAMESPACE

// Include the implementation detail of this header file.
// The *_inl.h helps separate header interface from implementation details.
#include "RocprofActivity_inl.h"
