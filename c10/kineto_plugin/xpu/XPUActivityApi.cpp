#include "XPUActivityApi.h"

#include <assert.h>
#include <chrono>
#include <mutex>
#include <thread>

using namespace std::chrono;

namespace c10::kineto_plugin::xpu {

constexpr size_t kBufSize(4 * 1024 * 1024);

XPUActivityApi& XPUActivityApi::singleton() {
  static XPUActivityApi instance;
  return instance;
}

void XPUActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch (type) {
    case Default:
      ptiViewPushExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_0, id);
      break;
    case User:
      ptiViewPushExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_1, id);
  }
}

void XPUActivityApi::popCorrelationID(CorrelationFlowType type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch (type) {
    case Default:
      ptiViewPopExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_0, nullptr);
      break;
    case User:
      ptiViewPopExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_1, nullptr);
  }
}

static bool nextActivityRecord(
    uint8_t* buffer,
    size_t valid_size,
    Pti_Activity*& record) {
  pti_result status = ptiViewGetNextRecord(buffer, valid_size, &record);
  if (status != pti_result::PTI_SUCCESS) {
    record = nullptr;
  }
  return record != nullptr;
}

void XPUActivityApi::setMaxBufferSize(int size) {
  maxGpuBufferCount_ = 1 + size / kBufSize;
}

void XPUActivityApi::bufferRequestedTrampoline(uint8_t** buffer, size_t* size) {
  singleton().bufferRequested(buffer, size);
}

void XPUActivityApi::bufferRequested(uint8_t** buffer, size_t* size) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (allocatedGpuTraceBuffers_.size() >= (size_t)maxGpuBufferCount_) {
    stopCollection = true;
  }

  auto buf = std::make_unique<XPUActivityBuffer>(kBufSize);
  *buffer = buf->data();
  *size = kBufSize;

  allocatedGpuTraceBuffers_[*buffer] = std::move(buf);
}

std::unique_ptr<XPUActivityBufferMap> XPUActivityApi::activityBuffers() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGpuTraceBuffers_.empty()) {
      return nullptr;
    }
  }

  time_point<system_clock> t1;
  ptiFlushAllViews();
  std::lock_guard<std::mutex> guard(mutex_);
  return std::move(readyGpuTraceBuffers_);
}

int XPUActivityApi::processActivitiesForBuffer(
    uint8_t* buf,
    size_t validSize,
    std::function<void(const Pti_Activity*)> handler) {
  int count = 0;
  if (buf && validSize) {
    Pti_Activity* record{nullptr};
    while (nextActivityRecord(buf, validSize, record)) {
      handler(record);
      ++count;
    }
  }
  return count;
}

const std::pair<int, int> XPUActivityApi::processActivities(
    XPUActivityBufferMap& buffers,
    std::function<void(const Pti_Activity*)> handler) {
  std::pair<int, int> res{0, 0};
  for (auto& pair : buffers) {
    auto& buf = pair.second;
    res.first += processActivitiesForBuffer(buf->data(), buf->size(), handler);
    res.second += buf->size();
  }
  return res;
}

void XPUActivityApi::clearActivities() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGpuTraceBuffers_.empty()) {
      return;
    }
  }
  ptiFlushAllViews();
  std::lock_guard<std::mutex> guard(mutex_);
  readyGpuTraceBuffers_ = nullptr;
}

void XPUActivityApi::bufferCompletedTrampoline(
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  singleton().bufferCompleted(buffer, size, validSize);
}

void XPUActivityApi::bufferCompleted(
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = allocatedGpuTraceBuffers_.find(buffer);

  if (!readyGpuTraceBuffers_) {
    readyGpuTraceBuffers_ = std::make_unique<XPUActivityBufferMap>();
  }
  it->second->setSize(validSize);
  (*readyGpuTraceBuffers_)[it->first] = std::move(it->second);
  allocatedGpuTraceBuffers_.erase(it);
}

void XPUActivityApi::enablePtiActivities(
    const std::set<act_t>& selected_activities) {
  ptiViewSetCallbacks(bufferRequestedTrampoline, bufferCompletedTrampoline);

  externalCorrelationEnabled_ = false;
  for (const auto& activity : selected_activities) {
    if (activity == act_t::GPU_MEMCPY) {
      ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_COPY);
    }
    if (activity == act_t::GPU_MEMSET) {
      ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_FILL);
    }
    if (activity == act_t::CONCURRENT_KERNEL) {
      ptiViewEnable(PTI_VIEW_DEVICE_GPU_KERNEL);
    }
    if (activity == act_t::EXTERNAL_CORRELATION) {
      ptiViewEnable(PTI_VIEW_EXTERNAL_CORRELATION);
      externalCorrelationEnabled_ = true;
    }
    if (activity == act_t::XPU_RUNTIME) {
      ptiViewEnable(PTI_VIEW_SYCL_RUNTIME_CALLS);
    }
    if (activity == act_t::OVERHEAD) {
      ptiViewEnable(PTI_VIEW_COLLECTION_OVERHEAD);
    }
  }

  tracingEnabled_ = 1;

  stopCollection = false;
}

void XPUActivityApi::disablePtiActivities(
    const std::set<act_t>& selected_activities) {
  for (const auto& activity : selected_activities) {
    if (activity == act_t::GPU_MEMCPY) {
      ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_COPY);
    }
    if (activity == act_t::GPU_MEMSET) {
      ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_FILL);
    }
    if (activity == act_t::CONCURRENT_KERNEL) {
      ptiViewDisable(PTI_VIEW_DEVICE_GPU_KERNEL);
    }
    if (activity == act_t::EXTERNAL_CORRELATION) {
      ptiViewDisable(PTI_VIEW_EXTERNAL_CORRELATION);
    }
    if (activity == act_t::XPU_RUNTIME) {
      ptiViewDisable(PTI_VIEW_SYCL_RUNTIME_CALLS);
    }
    if (activity == act_t::OVERHEAD) {
      ptiViewDisable(PTI_VIEW_COLLECTION_OVERHEAD);
    }
  }
  externalCorrelationEnabled_ = false;
}

} // namespace xpu
