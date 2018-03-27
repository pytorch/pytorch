#include "caffe2/core/event.h"
#include "caffe2/core/operator.h"

#include <atomic>

namespace caffe2 {

struct CPUEventWrapper {
  explicit CPUEventWrapper(const DeviceOption& option)
      : status_(EventStatus::EVENT_INITIALIZED) {
    CAFFE_ENFORCE(
        option.device_type() == CPU || option.device_type() == MKLDNN,
        "Expected CPU/MKLDNN device type");
  }
  ~CPUEventWrapper() {}

  std::mutex mutex_;
  std::condition_variable cv_completed_;
  std::atomic<int> status_;
  std::string err_msg_;
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
