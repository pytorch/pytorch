#pragma once

#include <string>

#include "XPUProfilerMacros.h"

namespace c10::xpu {

template <class T>
struct XPUActivity : public libkineto::ITraceActivity {
  explicit XPUActivity(const T* activity, const itrace_t* linked)
      : activity_(*activity), linked_(linked) {}
  int64_t timestamp() const override;
  int64_t duration() const override;
  int64_t correlationId() const override {
    return 0;
  }
  int32_t getThreadId() const override {
    return 0;
  }
  const itrace_t* linkedActivity() const override {
    return linked_;
  }
  int flowType() const override {
    return libkineto::kLinkAsyncCpuGpu;
  }
  int flowId() const override {
    return correlationId();
  }
  const T& raw() const {
    return activity_;
  }
  const libkineto::TraceSpan* traceSpan() const override {
    return nullptr;
  }

 protected:
  const T& activity_;
  const itrace_t* linked_{nullptr};
};

// Pti_ActivityAPI - PTI-sdk runtime activities
struct RuntimeActivity : public XPUActivity<pti_view_record_sycl_runtime> {
  explicit RuntimeActivity(
      const pti_view_record_sycl_runtime* activity,
      const itrace_t* linked,
      int32_t threadId)
      : XPUActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override {
    return activity_._correlation_id;
  }
  int64_t deviceId() const override {
    return libkineto::processId();
  }
  int64_t resourceId() const override {
    return threadId_;
  }
  act_t type() const override {
    return act_t::XPU_RUNTIME;
  }
  bool flowStart() const override;
  const std::string name() const override {
    return activity_._name;
  }
  void log(logger_t& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// Pti_ActivityAPI - PTI-sdk overhead activities
struct OverheadActivity : public XPUActivity<pti_view_record_overhead> {
  explicit OverheadActivity(
      const pti_view_record_overhead* activity,
      const itrace_t* linked,
      int32_t threadId = 0)
      : XPUActivity(activity, linked), threadId_(threadId) {}
  // TODO: Update this with PID ordering
  int64_t deviceId() const override {
    return -1;
  }
  int64_t resourceId() const override {
    return threadId_;
  }
  act_t type() const override {
    return act_t::OVERHEAD;
  }
  bool flowStart() const override;
  const std::string name() const override;
  void log(logger_t& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// Base class for GPU activities.
// Can also be instantiated directly.
template <class T>
struct GpuActivity : public XPUActivity<T> {
  explicit GpuActivity(const T* activity, const itrace_t* linked)
      : XPUActivity<T>(activity, linked) {}
  int64_t correlationId() const override {
    return raw()._correlation_id;
  }
  int64_t deviceId() const override {
    // FIXME: how to return _device_uuid as int64_t
    return (int64_t)raw()._device_uuid[0];
  }
  int64_t resourceId() const override {
    return (int64_t)raw()._queue_handle;
  }
  act_t type() const override;
  bool flowStart() const override {
    return false;
  }
  const std::string name() const override;
  void log(logger_t& logger) const override;
  const std::string metadataJson() const override;
  const T& raw() const {
    return XPUActivity<T>::raw();
  }
};

template struct GpuActivity<pti_view_record_memory_copy>;
template struct GpuActivity<pti_view_record_memory_fill>;
template struct GpuActivity<pti_view_record_kernel>;

} // namespace xpu
