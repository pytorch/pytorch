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

#include <ideep_pin_singletons.hpp>
#include <caffe2/core/operator.h>
#include <caffe2/proto/caffe2.pb.h>
#include <caffe2/core/event_cpu.h>

namespace caffe2 {

CAFFE_KNOWN_TYPE(ideep::tensor);

CAFFE_DEFINE_REGISTRY(
    IDEEPOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

CAFFE_REGISTER_DEVICE_TYPE(DeviceType::IDEEP, IDEEPOperatorRegistry);

REGISTER_EVENT_CREATE_FUNCTION(IDEEP, EventCreateCPU);
REGISTER_EVENT_RECORD_FUNCTION(IDEEP, EventRecordCPU);
REGISTER_EVENT_WAIT_FUNCTION(IDEEP, IDEEP, EventWaitCPUCPU);
REGISTER_EVENT_WAIT_FUNCTION(IDEEP, CPU, EventWaitCPUCPU);
REGISTER_EVENT_WAIT_FUNCTION(CPU, IDEEP, EventWaitCPUCPU);
REGISTER_EVENT_FINISH_FUNCTION(IDEEP, EventFinishCPU);
REGISTER_EVENT_QUERY_FUNCTION(IDEEP, EventQueryCPU);
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(IDEEP, EventErrorMessageCPU);
REGISTER_EVENT_SET_FINISHED_FUNCTION(IDEEP, EventSetFinishedCPU);
REGISTER_EVENT_RESET_FUNCTION(IDEEP, EventResetCPU);

} // namespace caffe2
