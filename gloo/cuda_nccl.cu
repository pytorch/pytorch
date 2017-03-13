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

// Allocate a set of per-device streams used to serialize NCCL op scheduling.
// These ensure concurrent NCCL ops are not interleaved across devices (i.e.,
// through priority scheduling), resulting in deadlock. Use a function-scope
// static to avoid SIOF with the CUDA runtime.
static CudaDeviceStreams& getNcclStreams() {
  static CudaDeviceStreams ncclStreams;
  return ncclStreams;
}

template <typename T>
NCCLContext<T>::NCCLContext(std::vector<NCCLElement<T>>&& elements, int root)
    : root(root), elements(std::move(elements)) {
  std::vector<int> devices;
  const size_t numElements = this->elements.size();
  devices.reserve(numElements);
  for (auto& el : this->elements) {
    GLOO_ENFORCE(
        // Performing a linear search given small set of devices
        std::find(devices.begin(), devices.end(), el.device) == devices.end(),
        "NCCL elements must map to unique devices");
    devices.push_back(el.device);
  }
  {
    // Initialze comms. Synchronize with conflicting CUDA and NCCL operations.
    comms.resize(numElements);
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    NCCL_CHECK(ncclCommInitAll(comms.data(), devices.size(), devices.data()));
  }
  // Allocate events to synchronize source, destination, and NCCL streams
  ncclEvents.resize(numElements);
  for (auto i = 0; i < numElements; i++) {
    CudaDeviceScope scope(this->elements[i].device);
    CUDA_CHECK(cudaEventCreateWithFlags(
        &ncclEvents[i], cudaEventDefault | cudaEventDisableTiming));
  }
}

template <typename T>
NCCLContext<T>::~NCCLContext() {
  for (auto i = 0; i < elements.size(); ++i) {
    CudaDeviceScope scope(elements[i].device);
    CUDA_CHECK(cudaEventDestroy(ncclEvents[i]));
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
  auto& elements = context_.elements;
  for (auto i = 0; i < elements.size(); ++i) {
    CudaDeviceScope scope(elements[i].device);
    elements[i].dst.wait();
  }
}

template <typename T>
template <typename F>
void NCCLOp<T>::runNCCL(F&& f) {
  const auto& elements = context_.elements;
  const auto& comms = context_.comms;
  const auto& ncclEvents = context_.ncclEvents;

  // Synchronize memory allocation with NCCL operations
  std::lock_guard<std::mutex> lock(CudaShared::getMutex());

  // Kick off the NCCL operation on each device
  for (auto i = 0; i < elements.size(); i++) {
    const auto& element = elements[i];
    const auto& srcStream = element.src.getStream();
    const auto& dstStream = element.dst.getStream();
    const auto& ncclStream = getNcclStreams()[element.device];
    const auto& srcEvent = element.src.getEvent();
    const auto& dstEvent = element.src.getEvent();

    CudaDeviceScope scope(element.device);
    // Synchronize the source and destination with the NCCL stream. Record
    // events in the source and destination streams, and wait on these in the
    // NCCL streams.
    CUDA_CHECK(cudaEventRecord(srcEvent, srcStream));
    CUDA_CHECK(cudaStreamWaitEvent(ncclStream, srcEvent, 0));
    if (srcStream != dstStream) {
      CUDA_CHECK(cudaEventRecord(dstEvent, dstStream));
      CUDA_CHECK(cudaStreamWaitEvent(ncclStream, dstEvent, 0));
    }
    // Run the operation
    f(element, comms[i], ncclStream);
    // Record an event in the NCCL stream signaling the operation is complete.
    // Synchronize with the destination stream.
    CUDA_CHECK(cudaEventRecord(ncclEvents[i], ncclStream));
    CUDA_CHECK(cudaStreamWaitEvent(dstStream, ncclEvents[i], 0));
    CUDA_CHECK(cudaEventRecord(dstEvent, dstStream));
  }
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
