/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_nccl.h"

#include <algorithm>

#include "gloo/cuda_private.h"

namespace gloo {
namespace nccl {

template <typename T>
NCCLContext<T>::NCCLContext(
    int device,
    cudaStream_t stream,
    std::vector<NCCLElement<T>>&& elements,
    int root)
    : masterDevice(device),
      masterStream(stream),
      root(root),
      elements(std::move(elements)) {
  std::vector<int> devices;
  devices.reserve(this->elements.size());
  for (auto& el : this->elements) {
    GLOO_ENFORCE(
      // Performing a linear search given small set of devices
      std::find(devices.begin(), devices.end(), el.device) == devices.end(),
      "NCCL elements must map to unique devices");
    devices.push_back(el.device);
  }
  {
    // Initialze comms. Synchronize with conflicting CUDA and NCCL operations.
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    comms.resize(this->elements.size());
    NCCL_CHECK(ncclCommInitAll(comms.data(), devices.size(), devices.data()));
  }
  // Allocate the events and streams
  events.resize(this->elements.size());
  for (auto i = 0; i < this->elements.size(); i++) {
    CudaDeviceScope scope(this->elements[i].device);
    CUDA_CHECK(cudaEventCreateWithFlags(
        &events[i], cudaEventDefault | cudaEventDisableTiming));
  }
  CUDA_CHECK(cudaEventCreateWithFlags(
      &masterEvent, cudaEventDefault | cudaEventDisableTiming));
}

template <typename T>
NCCLContext<T>::NCCLContext(NCCLContext&& other) noexcept
  : masterDevice(other.masterDevice),
    masterEvent(other.masterEvent),
    masterStream(other.masterStream),
    root(other.root),
    elements(std::move(other.elements)),
    comms(std::move(other.comms)),
    events(std::move(other.events)) {
  // Nullify fields that would otherwise be destructed
  other.masterEvent = nullptr;
}

template <typename T>
NCCLContext<T>::~NCCLContext() {
  if (masterEvent != nullptr) {
    CudaDeviceScope scope(masterDevice);
    // Make sure outstanding operations are complete. If the event
    // hasn't been queued this call will return immediately.
    CUDA_CHECK(cudaEventSynchronize(masterEvent));
    CUDA_CHECK(cudaEventDestroy(masterEvent));
  }
  for (auto i = 0; i < elements.size(); ++i) {
    CudaDeviceScope scope(elements[i].device);
    CUDA_CHECK(cudaEventDestroy(events[i]));
    {
      // Synchronize memory allocation with NCCL operations
      std::lock_guard<std::mutex> lock(CudaShared::getMutex());
      ncclCommDestroy(comms[i]);
    }
  }
}

template <typename T>
class ncclTypeWrapper;

template <>
class ncclTypeWrapper<float> {
 public:
  static const ncclDataType_t type = ncclFloat;
};

template <typename T>
void NCCLOp<T>::wait() {
  CudaDeviceScope scope(context_.masterDevice);
  CUDA_CHECK(cudaEventSynchronize(context_.masterEvent));
}

template <typename T>
template <typename F>
void NCCLOp<T>::runNCCL(F&& f) {
  // Record an event on the master stream and wait on it in each of the child
  // streams to ensure both are synchronized.
  {
    CudaDeviceScope scope(context_.masterDevice);
    CUDA_CHECK(
        cudaEventRecord(context_.masterEvent, context_.masterStream));
  }

  // Kick off the NCCL operation on each device
  {
    // Synchronize memory allocation with NCCL operations
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());

    const auto& elements = context_.elements;
    for (auto i = 0; i < elements.size(); i++) {
      const auto& element = elements[i];
      const auto& comm = context_.comms[i];
      const auto& event = context_.events[i];
      const auto& srcStream = element.src.getStream();
      const auto& dstStream = element.dst.getStream();
      // Synchronize the source and destination with the master stream.
      CudaDeviceScope scope(element.device);
      CUDA_CHECK(cudaStreamWaitEvent(srcStream, context_.masterEvent, 0));
      if (srcStream != dstStream) {
        CUDA_CHECK(cudaStreamWaitEvent(dstStream, context_.masterEvent, 0));
      }
      // Run the operation
      f(element, comm, dstStream);
      CUDA_CHECK(cudaEventRecord(event, dstStream));
    }
  }

  // Synchronize with the master stream.
  CudaDeviceScope scope(context_.masterDevice);
  for (auto& event : context_.events) {
    CUDA_CHECK(cudaStreamWaitEvent(context_.masterStream, event, 0));
  }
  // Record an event on the master stream to signal end of the operation.
  CUDA_CHECK(
      cudaEventRecord(context_.masterEvent, context_.masterStream));
}

template <typename T>
void ReduceOp<T>::runAsync() {
  const auto root = this->context_.root;
  this->runNCCL([root](
      const NCCLElement<T>& element, ncclComm_t comm, cudaStream_t stream) {
    NCCL_CHECK(ncclReduce(
        *element.src,
        *element.dst,
        element.count,
        ncclTypeWrapper<T>::type,
        ncclSum,
        root,
        comm,
        stream));
  });
}

template <typename T>
void BroadcastOp<T>::runAsync() {
  const auto root = this->context_.root;
  this->runNCCL([root](
      const NCCLElement<T>& element, ncclComm_t comm, cudaStream_t stream) {
    NCCL_CHECK(ncclBcast(
        *element.dst,
        element.count,
        ncclTypeWrapper<T>::type,
        root,
        comm,
        stream));
  });
}

template class NCCLContext<float>;
template class NCCLOp<float>;
template class ReduceOp<float>;
template class BroadcastOp<float>;

} // namespace nccl
} // namespace gloo
