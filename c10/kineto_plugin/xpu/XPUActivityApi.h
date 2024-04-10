#pragma once

#include "XPUActivityBuffer.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <set>

#include "XPUProfilerMacros.h"

namespace c10::kineto_plugin::xpu {

using Pti_Activity = pti_view_record_base;

class XPUActivityApi {
 public:
  enum CorrelationFlowType { Default, User };

  XPUActivityApi() = default;
  XPUActivityApi(const XPUActivityApi&) = delete;
  XPUActivityApi& operator=(const XPUActivityApi&) = delete;

  virtual ~XPUActivityApi() {}

  static XPUActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enablePtiActivities(const std::set<act_t>& selected_activities);
  void disablePtiActivities(const std::set<act_t>& selected_activities);
  void clearActivities();

  virtual std::unique_ptr<XPUActivityBufferMap> activityBuffers();

  virtual const std::pair<int, int> processActivities(
      XPUActivityBufferMap&,
      std::function<void(const Pti_Activity*)> handler);

  void setMaxBufferSize(int size);
  // void setDeviceBufferSize(size_t size);
  // void setDeviceBufferPoolLimit(size_t limit);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

 private:
  int maxGpuBufferCount_{0};
  XPUActivityBufferMap allocatedGpuTraceBuffers_;
  std::unique_ptr<XPUActivityBufferMap> readyGpuTraceBuffers_;
  std::mutex mutex_;
  std::atomic<uint32_t> tracingEnabled_{0};
  bool externalCorrelationEnabled_{false};

  int processActivitiesForBuffer(
      uint8_t* buf,
      size_t validSize,
      std::function<void(const Pti_Activity*)> handler);
  static void bufferRequestedTrampoline(uint8_t** buffer, size_t* size);
  static void bufferCompletedTrampoline(
      uint8_t* buffer,
      size_t size,
      size_t validSize);

 protected:
  void bufferRequested(uint8_t** buffer, size_t* size);
  void bufferCompleted(uint8_t* buffer, size_t size, size_t validSize);
};

} // namespace xpu 
