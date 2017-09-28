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

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/event.h"

namespace caffe2 {

struct CudaEventWrapper {
  CudaEventWrapper(const DeviceOption& option) {
    CAFFE_ENFORCE(option.device_type(), CUDA);
    cuda_gpu_id_ = option.cuda_gpu_id();
    DeviceGuard g(cuda_gpu_id_);
    CUDA_ENFORCE(cudaEventCreate(
        &cuda_event_, cudaEventDefault | cudaEventDisableTiming));
  }
  ~CudaEventWrapper() {
    DeviceGuard g(cuda_gpu_id_);
    CUDA_CHECK(cudaEventDestroy(cuda_event_));
  }
  cudaEvent_t cuda_event_;
  int cuda_gpu_id_;
};

void EventCreateCUDA(const DeviceOption& option, Event* event) {
  event->event_.reset(new CudaEventWrapper(option), [](void* ptr) {
    delete static_cast<CudaEventWrapper*>(ptr);
  });
}

void EventRecordCUDA(const void* context, Event* event) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  // When recording, one needs to make sure that the current gpu id is correct.
  // TODO(jiayq): move the enforce logic to the caller?
  CAFFE_ENFORCE_EQ(
      CaffeCudaGetDevice(),
      wrapper->cuda_gpu_id_,
      "When you call EventRecordCUDA, your current device should be the same "
      "as the device specified by the event.");
  CAFFE_ENFORCE_EQ(
      CaffeCudaGetDevice(),
      static_cast<const CUDAContext*>(context)->cuda_gpu_id());
  CUDA_ENFORCE(cudaEventRecord(
      wrapper->cuda_event_,
      static_cast<const CUDAContext*>(context)->cuda_stream()));
}

void EventFinishCUDA(const Event* event) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  DeviceGuard g(wrapper->cuda_gpu_id_);
  CUDA_ENFORCE(cudaEventSynchronize(wrapper->cuda_event_));
}

// Both waiter and event are cuda.
void EventWaitCUDACUDA(const Event* event, void* context) {
  CAFFE_ENFORCE_EQ(
      CaffeCudaGetDevice(),
      static_cast<const CUDAContext*>(context)->cuda_gpu_id());
  CUDA_CHECK(cudaStreamWaitEvent(
      static_cast<CUDAContext*>(context)->cuda_stream(),
      static_cast<CudaEventWrapper*>(event->event_.get())->cuda_event_,
      0));
}

// Waiter is CPU, event is cuda - in this case call cudaEventSynchronize.
void EventWaitCPUCUDA(const Event* event, void* context) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  DeviceGuard g(wrapper->cuda_gpu_id_);
  CUDA_ENFORCE(cudaEventSynchronize(wrapper->cuda_event_));
}

// Waiter is cuda, event is CPU - in this case do nothing as cpu is always sync.
void EventWaitCUDACPU(const Event* event, void* context) {}
REGISTER_EVENT_CREATE_FUNCTION(CUDA, EventCreateCUDA);
REGISTER_EVENT_RECORD_FUNCTION(CUDA, EventRecordCUDA);
REGISTER_EVENT_WAIT_FUNCTION(CUDA, CUDA, EventWaitCUDACUDA);
REGISTER_EVENT_WAIT_FUNCTION(CPU, CUDA, EventWaitCPUCUDA);
REGISTER_EVENT_WAIT_FUNCTION(CUDA, CPU, EventWaitCUDACPU);
REGISTER_EVENT_FINISH_FUNCTION(CUDA, EventFinishCUDA);

} // namespace caffe2
