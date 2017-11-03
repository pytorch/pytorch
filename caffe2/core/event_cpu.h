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
