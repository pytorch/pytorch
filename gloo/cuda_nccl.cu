/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_nccl.h"

#include "gloo/cuda_private.h"

namespace gloo {
namespace nccl {

NCCLContext::NCCLContext(
    int device,
    cudaStream_t stream,
    std::vector<NCCLElement>&& elements,
    int root)
    : masterDevice_(device),
      masterStream_(stream),
      root_(root),
      elements_(elements) {
  std::vector<int> devices;
  devices.reserve(elements_.size());
  for (auto el : elements_) {
    devices.push_back(el.device);
  }
  {
    // Initialze comms. Synchronize with conflicting CUDA and NCCL operations.
    std::lock_guard<std::mutex> lock(gCudaMutex);
    comms_.resize(elements_.size());
    NCCL_CHECK(ncclCommInitAll(comms_.data(), devices.size(), devices.data()));
  }
  // Allocate the events and streams
  events_.resize(elements_.size());
  for (auto i = 0; i < elements_.size(); i++) {
    CudaDeviceScope scope(elements_[i].device);
    CUDA_CHECK(cudaEventCreateWithFlags(
        &events_[i], cudaEventDefault | cudaEventDisableTiming));
  }
  CUDA_CHECK(cudaEventCreateWithFlags(
      &masterEvent_, cudaEventDefault | cudaEventDisableTiming));
}

NCCLContext::NCCLContext(NCCLContext&& other) noexcept
  : masterDevice_(other.masterDevice_),
    masterEvent_(other.masterEvent_),
    masterStream_(other.masterStream_),
    root_(other.root_),
    elements_(std::move(other.elements_)),
    comms_(std::move(other.comms_)),
    events_(std::move(other.events_)) {
  // Nullify fields that would otherwise be destructed
  other.masterEvent_ = nullptr;
}

NCCLContext::~NCCLContext() {
  for (auto i = 0; i < elements_.size(); ++i) {
    CudaDeviceScope scope(elements_[i].device);
    CUDA_CHECK(cudaEventDestroy(events_[i]));
    {
      // Synchronize memory allocation with NCCL operations
      std::lock_guard<std::mutex> lock(gCudaMutex);
      ncclCommDestroy(comms_[i]);
    }
  }
  if (masterEvent_ != nullptr) {
    CudaDeviceScope scope(masterDevice_);
    CUDA_CHECK(cudaEventDestroy(masterEvent_));
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
  CudaDeviceScope scope(context_.masterDevice_);
  CUDA_CHECK(cudaEventSynchronize(context_.masterEvent_));
}

template <typename T>
template <typename F>
void NCCLOp<T>::runNCCL(F&& f) {
  // Record an event on the master stream and wait on it in each of the child
  // streams to ensure both are synchronized.
  {
    CudaDeviceScope scope(context_.masterDevice_);
    CUDA_CHECK(
        cudaEventRecord(context_.masterEvent_, context_.masterStream_));
  }

  // Kick off the NCCL operation on each device
  {
    // Synchronize memory allocation with NCCL operations
    std::lock_guard<std::mutex> lock(gCudaMutex);

    const auto& elements = context_.elements_;
    for (auto i = 0; i < elements.size(); i++) {
      const auto& element = elements[i];
      const auto& comm = context_.comms_[i];
      const auto& event = context_.events_[i];
      const auto& stream = element.stream;
      // Synchronize with the master stream
      CudaDeviceScope scope(element.device);
      CUDA_CHECK(cudaStreamWaitEvent(stream, context_.masterEvent_, 0));
      // Run the operation
      f(element, comm, stream);
      CUDA_CHECK(cudaEventRecord(event, stream));
    }
  }

  // Synchronize with the master stream.
  CudaDeviceScope scope(context_.masterDevice_);
  for (auto& event : context_.events_) {
    CUDA_CHECK(cudaStreamWaitEvent(context_.masterStream_, event, 0));
  }
  // Record an event on the master stream to signal end of the operation.
  CUDA_CHECK(
      cudaEventRecord(context_.masterEvent_, context_.masterStream_));
}

template <typename T>
void ReduceOp<T>::runAsync() {
  const auto root = this->context_.root_;
  this->runNCCL([root](
      const NCCLElement& element, ncclComm_t comm, cudaStream_t stream) {
    NCCL_CHECK(ncclReduce(
        element.src,
        element.dst,
        element.length,
        ncclTypeWrapper<T>::type,
        ncclSum,
        root,
        comm,
        stream));
  });
}

template <typename T>
void BroadcastOp<T>::runAsync() {
  const auto root = this->context_.root_;
  this->runNCCL([root](
      const NCCLElement& element, ncclComm_t comm, cudaStream_t stream) {
    NCCL_CHECK(ncclBcast(
        element.dst,
        element.length,
        ncclTypeWrapper<T>::type,
        root,
        comm,
        stream));
  });
}

template class NCCLOp<float>;
template class ReduceOp<float>;
template class BroadcastOp<float>;

} // namespace nccl
} // namespace gloo
