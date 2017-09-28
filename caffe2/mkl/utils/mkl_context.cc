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

#include "caffe2/mkl/utils/mkl_context.h"

#include "caffe2/core/event.h"

namespace caffe2 {

// For MKLDNN devices, Event is essentially a no-op since they are all
// synchronous.
void EventCreateMKLDNN(const DeviceOption& /* unused */, Event* /* unused */) {}
void EventRecordMKLDNN(const void* /* unused */, Event* /* unused */) {}
void EventWaitMKLDNNMKLDNN(const Event* /* unused */, void* /* unused */) {}
void EventFinishMKLDNN(const Event* /* unused */) {}

REGISTER_EVENT_CREATE_FUNCTION(MKLDNN, EventCreateMKLDNN);
REGISTER_EVENT_RECORD_FUNCTION(MKLDNN, EventRecordMKLDNN);
REGISTER_EVENT_WAIT_FUNCTION(MKLDNN, MKLDNN, EventWaitMKLDNNMKLDNN);
REGISTER_EVENT_FINISH_FUNCTION(MKLDNN, EventFinishMKLDNN);

} // namespace caffe2
