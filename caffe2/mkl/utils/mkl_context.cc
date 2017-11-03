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

//#include "caffe2/mkl/utils/mkl_context.h"

#include "caffe2/core/event_cpu.h"

namespace caffe2 {

// MKL events are the same as CPU events

REGISTER_EVENT_CREATE_FUNCTION(MKLDNN, EventCreateCPU);
REGISTER_EVENT_RECORD_FUNCTION(MKLDNN, EventRecordCPU);
REGISTER_EVENT_WAIT_FUNCTION(MKLDNN, MKLDNN, EventWaitCPUCPU);
REGISTER_EVENT_WAIT_FUNCTION(MKLDNN, CPU, EventWaitCPUCPU);
REGISTER_EVENT_WAIT_FUNCTION(CPU, MKLDNN, EventWaitCPUCPU);
REGISTER_EVENT_FINISH_FUNCTION(MKLDNN, EventFinishCPU);

REGISTER_EVENT_QUERY_FUNCTION(MKLDNN, EventQueryCPU);
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(MKLDNN, EventErrorMessageCPU);
REGISTER_EVENT_SET_FINISHED_FUNCTION(MKLDNN, EventSetFinishedCPU);
REGISTER_EVENT_RESET_FUNCTION(MKLDNN, EventResetCPU);

} // namespace caffe2
