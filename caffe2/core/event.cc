#include "caffe2/core/event_cpu.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
TORCH_API EventCreateFunction Event::event_creator_[MaxDeviceTypes];
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
TORCH_API EventRecordFunction Event::event_recorder_[MaxDeviceTypes];
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
TORCH_API EventWaitFunction
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    Event::event_waiter_[MaxDeviceTypes][MaxDeviceTypes];
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
TORCH_API EventFinishFunction Event::event_finisher_[MaxDeviceTypes];

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
TORCH_API EventQueryFunction Event::event_querier_[MaxDeviceTypes];
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
TORCH_API EventErrorMessageFunction
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    Event::event_err_msg_getter_[MaxDeviceTypes];
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
TORCH_API EventSetFinishedFunction
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    Event::event_finished_setter_[MaxDeviceTypes];
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
TORCH_API EventResetFunction Event::event_resetter_[MaxDeviceTypes];
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
TORCH_API EventSetCallbackFunction
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    Event::event_callback_setter_[MaxDeviceTypes];

namespace {
const std::string kNoError = "No error";
}

void EventCreateCPU(const DeviceOption& option, Event* event) {
  event->event_ = std::make_shared<CPUEventWrapper>(option);
}

void EventRecordCPU(
    Event* event,
    const void* /* unused */,
    const char* err_msg) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);

  // Possible state changes:
  //  INITIALIZED -> SCHEDULED or SUCCESS/FAILED
  //  SCHEDULED -> SUCCESS/FAILED
  //  SUCCESS/FAILED - terminal, no further changes to status_/err_msg_

  CAFFE_ENFORCE(
      wrapper->status_ != EventStatus::EVENT_SCHEDULED,
      "Calling Record multiple times");

  // Event might be in SUCCESS/FAILED state in case an op has
  // finished async execution part first
  if (wrapper->status_ == EventStatus::EVENT_INITIALIZED) {
    if (!err_msg) {
      wrapper->status_ = EventStatus::EVENT_SCHEDULED;
    } else {
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::EVENT_FAILED;
      wrapper->cv_completed_.notify_all();
    }
  }
}

void EventFinishCPU(const Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  while (wrapper->status_ != EventStatus::EVENT_SUCCESS &&
         wrapper->status_ != EventStatus::EVENT_FAILED) {
    wrapper->cv_completed_.wait(lock);
  }
}

void EventWaitCPUCPU(const Event* event, void* /* context */) {
  EventFinishCPU(event);
}

EventStatus EventQueryCPU(const Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  return static_cast<EventStatus>(wrapper->status_.load());
}

const std::string& EventErrorMessageCPU(const Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  if (wrapper->status_ == EventStatus::EVENT_FAILED) {
    // Failed is a terminal state, not synchronizing,
    // err_msg_ should not be changed anymore
    return wrapper->err_msg_;
  } else {
    return kNoError;
  }
}

void EventSetFinishedCPU(const Event* event, const char* err_msg) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);

  if (wrapper->status_ == EventStatus::EVENT_FAILED) {
    LOG(WARNING) << "SetFinished called on a finished event. "
                 << "Most likely caused by an external cancellation. "
                 << "old message: " << wrapper->err_msg_ << ", "
                 << "new message: " << err_msg;
    return;
  }

  CAFFE_ENFORCE(
      wrapper->status_ == EventStatus::EVENT_INITIALIZED ||
          wrapper->status_ == EventStatus::EVENT_SCHEDULED,
      "Calling SetFinished on finished event");

  if (!err_msg) {
    wrapper->status_ = EventStatus::EVENT_SUCCESS;
  } else {
    wrapper->err_msg_ = err_msg;
    wrapper->status_ = EventStatus::EVENT_FAILED;
  }

  for (auto& callback : wrapper->callbacks_) {
    callback();
  }

  wrapper->cv_completed_.notify_all();
}

void EventSetCallbackCPU(Event* event, EventCallbackFunction callback) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);

  wrapper->callbacks_.push_back(callback);
  if (wrapper->status_ == EventStatus::EVENT_SUCCESS ||
      wrapper->status_ == EventStatus::EVENT_FAILED) {
    callback();
  }
}

void EventResetCPU(Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  wrapper->status_ = EventStatus::EVENT_INITIALIZED;
  wrapper->err_msg_ = "";
  wrapper->callbacks_.clear();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_CREATE_FUNCTION(CPU, EventCreateCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_RECORD_FUNCTION(CPU, EventRecordCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_WAIT_FUNCTION(CPU, CPU, EventWaitCPUCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_FINISH_FUNCTION(CPU, EventFinishCPU);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_QUERY_FUNCTION(CPU, EventQueryCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(CPU, EventErrorMessageCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_SET_FINISHED_FUNCTION(CPU, EventSetFinishedCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_RESET_FUNCTION(CPU, EventResetCPU);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_SET_CALLBACK_FUNCTION(CPU, EventSetCallbackCPU);

} // namespace caffe2
