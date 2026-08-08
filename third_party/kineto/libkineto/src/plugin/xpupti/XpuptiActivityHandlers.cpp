/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityProfilerSession.h"
#include "output_json.h"

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace KINETO_NAMESPACE {

// =========== Session Private Methods ============= //
void XpuptiActivityProfilerSession::removeCorrelatedPtiActivities(
    const ITraceActivity* act1) {
  correlatedPtiActivities_.erase(act1->correlationId());
}

void XpuptiActivityProfilerSession::checkTimestampOrder(
    const ITraceActivity* act1) {
  auto [it, inserted] =
      correlatedPtiActivities_.insert({act1->correlationId(), act1});
  if (inserted) {
    return;
  }

  const ITraceActivity* act2 = it->second;
  if (act2->type() == ActivityType::XPU_RUNTIME) {
    std::swap(act1, act2);
  }
  if (act1->timestamp() > act2->timestamp()) {
    std::string err_msg;
    err_msg += "GPU op timestamp (" + std::to_string(act2->timestamp());
    err_msg += ") < runtime timestamp (" + std::to_string(act1->timestamp());
    err_msg += ") by " + std::to_string(act1->timestamp() - act2->timestamp());
    err_msg += "us Name: " + act2->name();
    err_msg += " Device: " + std::to_string(act2->deviceId());
    err_msg += " Queue: " + std::to_string(act2->resourceId());
    errors_.push_back(err_msg);
  }
}

inline bool XpuptiActivityProfilerSession::outOfRange(
    const ITraceActivity* act) {
  bool outOfRange = act->timestamp() < captureWindowStartTime_ ||
      (act->timestamp() + act->duration()) > captureWindowEndTime_;
  if (outOfRange) {
    std::string err_msg;
    err_msg += "TraceActivity outside of profiling window: " + act->name();
    err_msg += " (" + std::to_string(act->timestamp());
    err_msg += " < " + std::to_string(captureWindowStartTime_);
    err_msg += " or " + std::to_string(act->timestamp() + act->duration());
    err_msg += " > " + std::to_string(captureWindowEndTime_);
    errors_.push_back(err_msg);
  }
  return outOfRange;
}

const ITraceActivity* XpuptiActivityProfilerSession::linkedActivity(
    int32_t correlationId,
    const std::unordered_map<int64_t, int64_t>& correlationMap) {
  const auto& it = correlationMap.find(correlationId);
  if (it != correlationMap.end()) {
    return cpuActivity_(it->second);
  }
  return nullptr;
}

template <class ze_handle_type>
inline std::string handleToHexString(ze_handle_type handle) {
  return fmt::format("0x{:016x}", reinterpret_cast<uintptr_t>(handle));
}

inline void XpuptiActivityProfilerSession::handleCorrelationActivity(
    const pti_view_record_external_correlation* correlation) {
  switch (correlation->_external_kind) {
    case PTI_VIEW_EXTERNAL_KIND_CUSTOM_0:
      cpuCorrelationMap_[correlation->_correlation_id] =
          correlation->_external_id;
      break;
    case PTI_VIEW_EXTERNAL_KIND_CUSTOM_1:
      userCorrelationMap_[correlation->_correlation_id] =
          correlation->_external_id;
      break;
    default:
      errors_.push_back(
          "Invalid PTI External Correlation activity sent to handlePtiActivity");
  }
}

inline std::string memcpyName(
    pti_view_memcpy_type kind,
    pti_view_memory_type src,
    pti_view_memory_type dst) {
  return fmt::format(
      "Memcpy {} ({} -> {})",
      ptiViewMemcpyTypeToString(kind),
      ptiViewMemoryTypeToString(src),
      ptiViewMemoryTypeToString(dst));
}

inline std::string memsetName(pti_view_memory_type type, uint64_t val) {
  return fmt::format("Memset ({} -> {})", val, ptiViewMemoryTypeToString(type));
}

template <class pti_view_memory_record_type>
inline std::string bandwidth(pti_view_memory_record_type* activity) {
  auto duration = activity->_end_timestamp - activity->_start_timestamp;
  auto bytes = activity->_bytes;
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

void XpuptiActivityProfilerSession::addResouceInfo(
    int32_t device_id,
    int32_t sycl_queue_id) {
  if (std::find_if(
          resourceInfo_.begin(),
          resourceInfo_.end(),
          [device_id, sycl_queue_id](std::pair<int32_t, int32_t> pair) {
            return (pair.first == device_id) && (pair.second == sycl_queue_id);
          }) == resourceInfo_.end()) {
    resourceInfo_.emplace_back(device_id, sycl_queue_id);
  }
}

template <class T>
inline std::string formatTimeLikeOutputJson(T time) {
  return fmt::format("{}.{:03}", time / 1000, abs(time) % 1000);
}

inline int64_t signedFromUnsignedDiff(uint64_t time, uint64_t time_ref) {
  if (time >= time_ref) {
    return static_cast<int64_t>(time - time_ref);
  } else {
    return -static_cast<int64_t>(time_ref - time);
  }
}

static void addTimestampMetadata(
    GenericTraceActivity* trace_activity,
    std::string&& label,
    uint64_t time,
    uint64_t time_ref) {
  trace_activity->addMetadataQuoted(
      label, formatTimeLikeOutputJson(transToRelativeTime(time)));
  label += "_rel_to_start";
  trace_activity->addMetadataQuoted(
      label, formatTimeLikeOutputJson(signedFromUnsignedDiff(time, time_ref)));
}

template <class pti_view_memory_record_type>
void XpuptiActivityProfilerSession::handleRuntimeKernelMemcpyMemsetActivities(
    ActivityType activityType,
    const pti_view_memory_record_type* activity,
    ActivityLogger& logger) {
  constexpr bool handleRuntimeActivities =
      std::is_same_v<pti_view_memory_record_type, pti_view_record_api_t>;
  constexpr bool handleKernelActivities =
      std::is_same_v<pti_view_memory_record_type, pti_view_record_kernel>;
  constexpr bool handleMemcpyActivities =
      std::is_same_v<pti_view_memory_record_type, pti_view_record_memory_copy>;
  constexpr bool handleMemsetActivities =
      std::is_same_v<pti_view_memory_record_type, pti_view_record_memory_fill>;

  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;

  if constexpr (handleRuntimeActivities) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span, activityType, getApiName(activity));
  } else if constexpr (handleKernelActivities) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span, activityType, std::string(activity->_name));
  } else if constexpr (handleMemcpyActivities) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span,
        activityType,
        memcpyName(
            activity->_memcpy_type, activity->_mem_src, activity->_mem_dst));
  } else if constexpr (handleMemsetActivities) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span,
        activityType,
        memsetName(activity->_mem_type, activity->_value_for_set));
  }

  auto& trace_activity = traceBuffer_.activities.back();

  trace_activity->startTime = activity->_start_timestamp;
  trace_activity->endTime = activity->_end_timestamp;
  trace_activity->threadId = activity->_thread_id;
  trace_activity->flow.id = activity->_correlation_id;
  trace_activity->flow.type = libkineto::kLinkAsyncCpuGpu;

  trace_activity->id = activity->_correlation_id;
  trace_activity->linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);
  trace_activity->addMetadata("correlation", activity->_correlation_id);

  if constexpr (handleRuntimeActivities) {
    trace_activity->device = activity->_process_id;
    trace_activity->resource = activity->_thread_id;
    trace_activity->flow.start = startsFlow(activityType);
  } else {
    trace_activity->device = getDeviceIdxFromUUID(activity->_device_uuid);
    trace_activity->resource = activity->_sycl_queue_id;
    trace_activity->flow.start = 0;

    if constexpr (handleKernelActivities) {
      kernelActivities_[activity->_kernel_id].emplace(
          trace_activity->startTime,
          trace_activity->endTime,
          trace_activity->device,
          trace_activity->resource);
    }

    addResouceInfo(trace_activity->device, trace_activity->resource);
  }

  if constexpr (handleMemcpyActivities || handleMemsetActivities) {
    trace_activity->addMetadataQuoted("l0 call", std::string(activity->_name));
  }

  if constexpr (!handleRuntimeActivities) {
    addTimestampMetadata(
        trace_activity.get(),
        "appended",
        activity->_append_timestamp,
        activity->_start_timestamp);
    addTimestampMetadata(
        trace_activity.get(),
        "submitted",
        activity->_submit_timestamp,
        activity->_start_timestamp);
    if constexpr (handleKernelActivities) {
      addTimestampMetadata(
          trace_activity.get(),
          "sycl_task_begin",
          activity->_sycl_task_begin_timestamp,
          activity->_start_timestamp);
      addTimestampMetadata(
          trace_activity.get(),
          "sycl_enqk_begin",
          activity->_sycl_enqk_begin_timestamp,
          activity->_start_timestamp);
    }
    trace_activity->addMetadata("device", trace_activity->deviceId());
    trace_activity->addMetadataQuoted(
        "context", handleToHexString(activity->_context_handle));
    trace_activity->addMetadata("sycl queue", activity->_sycl_queue_id);
    trace_activity->addMetadataQuoted(
        "l0 queue", handleToHexString(activity->_queue_handle));
  }

  if constexpr (handleKernelActivities) {
    if (activity->_source_file_name) {
      trace_activity->addMetadataQuoted(
          "source_file_name", activity->_source_file_name);
      trace_activity->addMetadata(
          "source_line_number", activity->_source_line_number);
    }
    trace_activity->addMetadata("kernel_id", activity->_kernel_id);
    trace_activity->addMetadata("sycl_node_id", activity->_sycl_node_id);
    trace_activity->addMetadata(
        "sycl_invocation_id", activity->_sycl_invocation_id);
  } else if constexpr (handleMemcpyActivities || handleMemsetActivities) {
    trace_activity->addMetadata("memory opration id", activity->_mem_op_id);
    trace_activity->addMetadata("bytes", activity->_bytes);
    trace_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));
  }

  checkTimestampOrder(trace_activity.get());
  if (outOfRange(trace_activity.get())) {
    traceBuffer_.span.opCount -= 1;
    traceBuffer_.gpuOpCount -= 1;
    removeCorrelatedPtiActivities(trace_activity.get());
    traceBuffer_.activities.pop_back();
    return;
  }
  trace_activity->log(logger);

  // GPU_USER_ANNOTATION activities are synthetic spans that bracket all
  // GPU work (kernels/memcpies) on a given (device, stream) pair that was
  // enqueued while a user correlation ID was active.  We only do this for
  // actual GPU activities (handleRuntimeActivities == false); CPU-side
  // runtime events are skipped because they carry no device/resource info.
  // For each (device, stream, user_external_id) key we either expand the
  // existing annotation to cover the new activity's time range, or create a
  // fresh GenericTraceActivity linked back to the CPU op.  The annotations
  // are flushed to the logger at the end of processTrace().
  if constexpr (!handleRuntimeActivities) {
    if (activity_types_.count(ActivityType::GPU_USER_ANNOTATION)) {
      auto userIt = userCorrelationMap_.find(activity->_correlation_id);
      if (userIt != userCorrelationMap_.end() && cpuActivity_) {
        const int64_t user_external_id = userIt->second;
        const int32_t dev = trace_activity->device;
        const int32_t res = trace_activity->resource;
        auto key = std::make_tuple(dev, res, user_external_id);
        auto annIt = userAnnotationsByStream_.find(key);
        if (annIt != userAnnotationsByStream_.end()) {
          GenericTraceActivity* ua = annIt->second;
          ua->startTime = std::min(ua->startTime, trace_activity->startTime);
          ua->endTime = std::max(ua->endTime, trace_activity->endTime);
        } else if (
            const ITraceActivity* cpu_act = cpuActivity_(user_external_id)) {
          traceBuffer_.emplace_activity(
              traceBuffer_.span,
              ActivityType::GPU_USER_ANNOTATION,
              cpu_act->name());
          auto& ua = traceBuffer_.activities.back();
          ua->startTime = trace_activity->startTime;
          ua->endTime = trace_activity->endTime;
          ua->device = dev;
          ua->resource = res;
          ua->id = user_external_id;
          ua->threadId = trace_activity->threadId;
          ua->linked = cpu_act;
          userAnnotationsByStream_.emplace(key, ua.get());
        }
      }
    }
  }
}

void XpuptiActivityProfilerSession::handleCommunicationActivity(
    const pti_view_record_comms* activity,
    ActivityLogger& logger) {
  const auto& activity_record = *activity;
  const std::string activity_name{activity_record._name};
  const std::string xccl_prefix{"xccl::"};
  const auto record_name = xccl_prefix + activity_name;

  traceBuffer_.span.opCount += 1;
  traceBuffer_.emplace_activity(
      traceBuffer_.span, ActivityType::COLLECTIVE_COMM, record_name);
  auto& comms_activity = *(traceBuffer_.activities.back());

  comms_activity.startTime = activity_record._start_timestamp;
  comms_activity.endTime = activity_record._end_timestamp;
  comms_activity.device = activity_record._process_id;
  comms_activity.resource = activity_record._thread_id;
  comms_activity.threadId = activity_record._thread_id;

  comms_activity.addMetadata(
      "Communicator_id", activity_record._communicator_id);

  if (outOfRange(&comms_activity)) {
    traceBuffer_.span.opCount -= 1;
    traceBuffer_.activities.pop_back();
    return;
  }

  comms_activity.log(logger);
}

namespace {
// Map a PTI overhead kind to a human-readable name aligned with CUPTI's
// overheadKindString() (see cupti_strings.cpp), so XPU and CUDA traces use the
// same vocabulary. PTI's own ptiViewOverheadKindToString() returns raw enum
// spellings (e.g. "BUFFER_FLUSH", "BUFFER_TIME") which do not match CUDA.
// PTI_VIEW_OVERHEAD_KIND_TIME aggregates the time PTI spends inside
// the Level Zero calls it injects to instrument the workload,
// which is the same notion as CUPTI's "Instrumentation" overhead.
const char* overheadKindString(pti_view_overhead_kind kind) {
  switch (kind) {
    case PTI_VIEW_OVERHEAD_KIND_UNKNOWN:
      return "Unknown";
    case PTI_VIEW_OVERHEAD_KIND_RESOURCE:
      return "Resource";
    case PTI_VIEW_OVERHEAD_KIND_BUFFER_FLUSH:
      return "Buffer Flush";
    case PTI_VIEW_OVERHEAD_KIND_DRIVER:
      return "Driver";
    case PTI_VIEW_OVERHEAD_KIND_TIME:
      return "Instrumentation";
    default:
      return "Unknown";
  }
}

std::string getStringFromSynchronizationType(
    const pti_view_synchronization_type& synchronization_type) {
  using pv_st = pti_view_synchronization_type;
  static const std::unordered_map<pv_st, std::string> name_map{
      {pv_st::PTI_VIEW_SYNCHRONIZATION_TYPE_UNKNOWN, "UNKNOWN"},
      {pv_st::PTI_VIEW_SYNCHRONIZATION_TYPE_GPU_BARRIER_EXECUTION,
       "GPU_BARRIER_EXECUTION"},
      {pv_st::PTI_VIEW_SYNCHRONIZATION_TYPE_GPU_BARRIER_MEMORY,
       "GPU_BARRIER_MEMORY"},
      {pv_st::PTI_VIEW_SYNCHRONIZATION_TYPE_HOST_FENCE, "HOST_FENCE"},
      {pv_st::PTI_VIEW_SYNCHRONIZATION_TYPE_HOST_EVENT, "HOST_EVENT"},
      {pv_st::PTI_VIEW_SYNCHRONIZATION_TYPE_HOST_COMMAND_LIST,
       "HOST_COMMAND_LIST"},
      {pv_st::PTI_VIEW_SYNCHRONIZATION_TYPE_HOST_COMMAND_QUEUE,
       "HOST_COMMAND_QUEUE"},
  };

  const auto& name_string = name_map.find(synchronization_type);
  if (name_string == name_map.end()) {
    const std::string error_message =
        "404: Not found string literal for this synchronization type: " +
        std::to_string(synchronization_type);
    return error_message;
  }
  return name_string->second;
}
} // namespace

void XpuptiActivityProfilerSession::handleSynchronizationActivity(
    const pti_view_record_synchronization* activity,
    ActivityLogger& logger) {
  const auto& activity_record = *activity;
  const auto record_name = getApiName(activity);

  const bool isGpuSync = activity_record._synch_type ==
          PTI_VIEW_SYNCHRONIZATION_TYPE_GPU_BARRIER_EXECUTION ||
      activity_record._synch_type ==
          PTI_VIEW_SYNCHRONIZATION_TYPE_GPU_BARRIER_MEMORY;

  traceBuffer_.span.opCount += 1;
  if (isGpuSync) {
    traceBuffer_.gpuOpCount += 1;
  }
  traceBuffer_.emplace_activity(
      traceBuffer_.span, ActivityType::XPU_SYNC, record_name);
  auto& synchronization_activity = *(traceBuffer_.activities.back());

  synchronization_activity.startTime = activity_record._start_timestamp;
  synchronization_activity.endTime = activity_record._end_timestamp;
  synchronization_activity.device = -1;
  synchronization_activity.resource = activity_record._thread_id;
  synchronization_activity.threadId = activity_record._thread_id;

  synchronization_activity.id = activity->_correlation_id;
  synchronization_activity.linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);
  synchronization_activity.addMetadata(
      "correlation", activity_record._correlation_id);

  synchronization_activity.addMetadataQuoted(
      "Type", getStringFromSynchronizationType(activity_record._synch_type));
  synchronization_activity.addMetadataQuoted(
      "Context_handle", handleToHexString(activity_record._context_handle));
  synchronization_activity.addMetadataQuoted(
      "Queue_handle", handleToHexString(activity_record._queue_handle));
  synchronization_activity.addMetadataQuoted(
      "Event_handle", handleToHexString(activity_record._event_handle));
  synchronization_activity.addMetadata(
      "Number_wait_events", activity_record._number_wait_events);
  synchronization_activity.addMetadata(
      "Return_code", activity_record._return_code);

  if (outOfRange(&synchronization_activity)) {
    traceBuffer_.span.opCount -= 1;
    if (isGpuSync) {
      traceBuffer_.gpuOpCount -= 1;
    }
    removeCorrelatedPtiActivities(&synchronization_activity);
    traceBuffer_.activities.pop_back();
    return;
  }

  synchronization_activity.log(logger);
}

void XpuptiActivityProfilerSession::handleOverheadActivity(
    const pti_view_record_overhead* activity,
    ActivityLogger& logger) {
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      ActivityType::OVERHEAD,
      overheadKindString(activity->_overhead_kind));
  auto& overhead_activity = traceBuffer_.activities.back();
  overhead_activity->startTime = activity->_overhead_start_timestamp_ns;
  overhead_activity->endTime = activity->_overhead_end_timestamp_ns;
  overhead_activity->device = -1;
  overhead_activity->resource = activity->_overhead_thread_id;
  overhead_activity->threadId = activity->_overhead_thread_id;
  overhead_activity->addMetadata(
      "overhead cost", activity->_overhead_duration_ns);
  // Occupancy is the share of the observation window [start, end] that was
  // actually spent in overhead: _overhead_duration_ns is cumulative and may be
  // smaller than the window. Guard on the divisor (duration()) directly so the
  // check holds regardless of PTI's start/end/duration relationship -- PTI may
  // emit point-like records (start == end), e.g. on platforms with a coarse
  // clock.
  const auto occupancy = overhead_activity->duration() > 0
      ? activity->_overhead_duration_ns * 100 / overhead_activity->duration()
      : 0;
  overhead_activity->addMetadataQuoted(
      "overhead occupancy", fmt::format("{}%", occupancy));
  overhead_activity->addMetadata("overhead count", activity->_overhead_count);

  if (!outOfRange(overhead_activity.get())) {
    overhead_activity->log(logger);
  }
}

void XpuptiActivityProfilerSession::handlePtiActivity(
    const pti_view_record_base* record,
    ActivityLogger& logger) {
  switch (record->_view_kind) {
    case PTI_VIEW_EXTERNAL_CORRELATION:
      handleCorrelationActivity(
          reinterpret_cast<const pti_view_record_external_correlation*>(
              record));
      break;
    case PTI_VIEW_RUNTIME_API:
      handleRuntimeKernelMemcpyMemsetActivities(
          ActivityType::XPU_RUNTIME,
          reinterpret_cast<const pti_view_record_api_t*>(record),
          logger);
      break;
    case PTI_VIEW_DRIVER_API:
      handleRuntimeKernelMemcpyMemsetActivities(
          ActivityType::XPU_DRIVER,
          reinterpret_cast<const pti_view_record_api_t*>(record),
          logger);
      break;
    case PTI_VIEW_DEVICE_GPU_KERNEL:
      handleRuntimeKernelMemcpyMemsetActivities(
          ActivityType::CONCURRENT_KERNEL,
          reinterpret_cast<const pti_view_record_kernel*>(record),
          logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_COPY:
      handleRuntimeKernelMemcpyMemsetActivities(
          ActivityType::GPU_MEMCPY,
          reinterpret_cast<const pti_view_record_memory_copy*>(record),
          logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_FILL:
      handleRuntimeKernelMemcpyMemsetActivities(
          ActivityType::GPU_MEMSET,
          reinterpret_cast<const pti_view_record_memory_fill*>(record),
          logger);
      break;
    case PTI_VIEW_COLLECTION_OVERHEAD:
      handleOverheadActivity(
          reinterpret_cast<const pti_view_record_overhead*>(record), logger);
      break;
    case PTI_VIEW_DEVICE_SYNCHRONIZATION:
      handleSynchronizationActivity(
          reinterpret_cast<const pti_view_record_synchronization*>(record),
          logger);
      break;
    case PTI_VIEW_COMMUNICATION:
      handleCommunicationActivity(
          reinterpret_cast<const pti_view_record_comms*>(record), logger);
      break;
    default:
      errors_.push_back(
          "Unexpected activity type: " + std::to_string(record->_view_kind));
      break;
  }
}

} // namespace KINETO_NAMESPACE
