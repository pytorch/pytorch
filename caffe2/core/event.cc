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
#include "caffe2/core/context.h"

namespace caffe2 {

EventCreateFunction Event::event_creator_[MaxDeviceTypes];
EventRecordFunction Event::event_recorder_[MaxDeviceTypes];
EventWaitFunction Event::event_waiter_[MaxDeviceTypes][MaxDeviceTypes];
EventFinishFunction Event::event_finisher_[MaxDeviceTypes];

// For CPU devices, Event is essentially a no-op since they are all synchronous.
void EventCreateCPU(const DeviceOption& /* unused */, Event* /* unused */) {}
void EventRecordCPU(const void* /* unused */, Event* /* unused */) {}
void EventWaitCPUCPU(const Event* /* unused */, void* /* unused */) {}
void EventFinishCPU(const Event* /* unused */) {}

REGISTER_EVENT_CREATE_FUNCTION(CPU, EventCreateCPU);
REGISTER_EVENT_RECORD_FUNCTION(CPU, EventRecordCPU);
REGISTER_EVENT_WAIT_FUNCTION(CPU, CPU, EventWaitCPUCPU);
REGISTER_EVENT_FINISH_FUNCTION(CPU, EventFinishCPU);

} // namespace caffe2
