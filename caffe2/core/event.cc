/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/event_cpu.h"

namespace caffe2 {

EventCreateFunction* Event::event_creator() {
  static EventCreateFunction event_creator_[MaxDeviceTypes];
  return event_creator_;
}
EventRecordFunction* Event::event_recorder() {
  static EventRecordFunction event_recorder_[MaxDeviceTypes];
  return event_recorder_;
}
Event::EWFMatrix Event::event_waiter() {
  static EventWaitFunction event_waiter_[MaxDeviceTypes][MaxDeviceTypes];
  return event_waiter_;
}
EventFinishFunction* Event::event_finisher() {
  static EventFinishFunction event_finisher_[MaxDeviceTypes];
  return event_finisher_;
}
EventQueryFunction* Event::event_querier() {
  static EventQueryFunction event_querier_[MaxDeviceTypes];
  return event_querier_;
}
EventErrorMessageFunction* Event::event_err_msg_getter() {
  static EventErrorMessageFunction event_err_msg_getter_[MaxDeviceTypes];
  return event_err_msg_getter_;
}
EventSetFinishedFunction* Event::event_finished_setter() {
  static EventSetFinishedFunction event_finished_setter_[MaxDeviceTypes];
  return event_finished_setter_;
}
EventResetFunction* Event::event_resetter() {
  static EventResetFunction event_resetter_[MaxDeviceTypes];
  return event_resetter_;
}

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
      wrapper->status_ == EventStatus::EVENT_INITIALIZED,
      "Calling Record multiple times");

  if (!err_msg) {
    wrapper->status_ = EventStatus::EVENT_SCHEDULED;
  } else {
    wrapper->err_msg_ = err_msg;
    wrapper->status_ = EventStatus::EVENT_FAILED;
    wrapper->cv_completed_.notify_all();
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
  wrapper->cv_completed_.notify_all();
}

void EventResetCPU(Event* event) {
  auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  wrapper->status_ = EventStatus::EVENT_INITIALIZED;
  wrapper->err_msg_ = "";
}

REGISTER_EVENT_CREATE_FUNCTION(CPU, EventCreateCPU);
REGISTER_EVENT_RECORD_FUNCTION(CPU, EventRecordCPU);
REGISTER_EVENT_WAIT_FUNCTION(CPU, CPU, EventWaitCPUCPU);
REGISTER_EVENT_FINISH_FUNCTION(CPU, EventFinishCPU);

REGISTER_EVENT_QUERY_FUNCTION(CPU, EventQueryCPU);
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(CPU, EventErrorMessageCPU);
REGISTER_EVENT_SET_FINISHED_FUNCTION(CPU, EventSetFinishedCPU);
REGISTER_EVENT_RESET_FUNCTION(CPU, EventResetCPU);

} // namespace caffe2
