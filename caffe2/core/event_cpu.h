#include "caffe2/core/event.h"

#include <atomic>
#include <condition_variable>

namespace caffe2 {

struct CPUEventWrapper {
  explicit CPUEventWrapper(const DeviceOption& option)
      : status_(EventStatus::EVENT_INITIALIZED) {
    CAFFE_ENFORCE(
        option.device_type() == PROTO_CPU ||
            option.device_type() == PROTO_MKLDNN ||
            option.device_type() == PROTO_IDEEP,
        "Expected CPU/MKLDNN/IDEEP device type");
  }
  ~CPUEventWrapper() {}

  std::mutex mutex_;
  std::condition_variable cv_completed_;
  std::atomic<int> status_;
  std::string err_msg_;
  std::vector<EventCallbackFunction> callbacks_;
};

void EventCreateCPU(const DeviceOption& option, Event* event);

void EventRecordCPU(
    Event* event,
    const void* /* unused */,
    const char* err_msg);

void EventFinishCPU(const Event* event);

void EventWaitCPUCPU(const Event* event, void* /* context */);

EventStatus EventQueryCPU(const Event* event);

const std::string& EventErrorMessageCPU(const Event* event);

void EventSetFinishedCPU(const Event* event, const char* err_msg);

bool EventCanScheduleCPU(const Event*, const Event*);

void EventResetCPU(Event*);

} // namespace caffe2
