/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/nccl/nccl.h"

#include <algorithm>
#include <unordered_map>

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
class NCCLContext {
 public:
  NCCLContext(const std::vector<int>& devices) : devices(devices) {
    // Initialze comms. Synchronize with conflicting CUDA and NCCL operations.
    comms.resize(devices.size());
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    NCCL_CHECK(ncclCommInitAll(comms.data(), devices.size(), devices.data()));
  }
  ~NCCLContext() {
    for (auto i = 0; i < devices.size(); ++i) {
      CudaDeviceScope scope(devices[i]);
      {
        // Synchronize memory allocation with NCCL operations
        std::lock_guard<std::mutex> lock(CudaShared::getMutex());
        ncclCommDestroy(comms[i]);
      }
    }
  }

  // Instances cannot be copied or copy-assigned
  NCCLContext(const NCCLContext&) = delete;
  NCCLContext& operator=(const NCCLContext&) = delete;

  const std::vector<int> devices;
  std::vector<ncclComm_t> comms;
};

// Initializing NCCL communications is expensive. Allocate context as needed per
// unique device set and cache for reuse.
template <typename T>
static std::shared_ptr<NCCLContext<T>> getNcclContext(
    const NCCLExecution<T>& ex) {
  static std::unordered_map<std::string, std::shared_ptr<NCCLContext<T>>>
      contexts;
  const auto key = ex.getKey();
  {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if (!contexts[key]) {
      contexts[key] = std::make_shared<NCCLContext<T>>(ex.getDevices());
    }
  }
  const auto context = contexts[key];
  GLOO_ENFORCE_NE(context.get(), (void*)nullptr);
  return context;
}

template <typename T>
NCCLExecution<T>::NCCLExecution(std::vector<NCCLElement<T>>&& elements)
    : elements(std::move(elements)) {
  // Allocate events to synchronize source, destination, and NCCL streams
  ncclEvents.resize(this->elements.size());
  for (auto i = 0; i < this->elements.size(); i++) {
    CudaDeviceScope scope(this->elements[i].device);
    CUDA_CHECK(cudaEventCreateWithFlags(
        &ncclEvents[i], cudaEventDefault | cudaEventDisableTiming));
  }
}

template <typename T>
NCCLExecution<T>::~NCCLExecution() {
  for (auto i = 0; i < this->elements.size(); i++) {
    CudaDeviceScope scope(this->elements[i].device);
    CUDA_CHECK(cudaEventDestroy(ncclEvents[i]));
  }
}

template <typename T>
std::vector<int> NCCLExecution<T>::getDevices() const {
  std::vector<int> result;
  result.reserve(elements.size());
  for (const auto& el : elements) {
    GLOO_ENFORCE(
        // Performing a linear search given small set of devices
        std::find(result.begin(), result.end(), el.device) == result.end(),
        "NCCL elements must map to unique devices");
    result.push_back(el.device);
  }
  return result;
}

template <typename T>
std::string NCCLExecution<T>::getKey() const {
  // Construct a key representing the order-dependent devices in this NCCL
  // execution. This is used to index into the NCCL context map and allows an
  // implicit association between elements[i].device and NCCLContext::comms[i]
  std::string result;
  for (const auto& el : elements) {
    result += std::to_string(el.device) + ",";
  }
  return result;
}

template <typename T>
class ncclTypeWrapper;

template <>
class ncclTypeWrapper<float> {
 public:
  static const ncclDataType_t type = ncclFloat;
};

template <typename T>
NCCLOp<T>::NCCLOp(NCCLExecution<T>&& execution)
    : execution_(std::move(execution)), context_(getNcclContext(execution_)) {}

template <typename T>
void NCCLOp<T>::wait() {
  auto& elements = execution_.elements;
  for (auto i = 0; i < elements.size(); ++i) {
    CudaDeviceScope scope(elements[i].device);
    elements[i].dst.wait();
  }
}

template <typename T>
template <typename F>
void NCCLOp<T>::runNCCL(F&& f) {
  const auto& elements = execution_.elements;
  const auto& ncclEvents = execution_.ncclEvents;
  const auto& comms = context_->comms;

  // Synchronize memory allocation with NCCL operations
  std::lock_guard<std::mutex> lock(CudaShared::getMutex());

  // Kick off the NCCL operation on each device
  for (auto i = 0; i < elements.size(); i++) {
    const auto& element = elements[i];
    const auto& srcStream = element.src.getStream();
    const auto& dstStream = element.dst.getStream();
    const auto& ncclStream = getNcclStreams()[element.device];
    const auto& srcEvent = element.src.getEvent();
    const auto& dstEvent = element.dst.getEvent();

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
  const int root = root_;
  this->runNCCL([root](
      const NCCLElement<T>& element, ncclComm_t comm, cudaStream_t stream) {
    NCCL_CHECK(ncclReduce(
        *element.src,
        *element.dst,
        element.src.getCount(),
        ncclTypeWrapper<T>::type,
        ncclSum,
        root,
        comm,
        stream));
  });
}

template <typename T>
void AllreduceOp<T>::runAsync() {
  this->runNCCL([](
      const NCCLElement<T>& element, ncclComm_t comm, cudaStream_t stream) {
    NCCL_CHECK(ncclAllReduce(
        *element.src,
        *element.dst,
        element.src.getCount(),
        ncclTypeWrapper<T>::type,
        ncclSum,
        comm,
        stream));
  });
}

template <typename T>
void ReduceScatterOp<T>::runAsync() {
  this->runNCCL([](
      const NCCLElement<T>& element, ncclComm_t comm, cudaStream_t stream) {
    NCCL_CHECK(ncclReduceScatter(
        *element.src,
        *element.dst,
        element.dst.getCount(),
        ncclTypeWrapper<T>::type,
        ncclSum,
        comm,
        stream));
  });
}

template <typename T>
void BroadcastOp<T>::runAsync() {
  const int root = root_;
  this->runNCCL([root](
      const NCCLElement<T>& element, ncclComm_t comm, cudaStream_t stream) {
    NCCL_CHECK(ncclBcast(
        *element.dst,
        element.dst.getCount(),
        ncclTypeWrapper<T>::type,
        root,
        comm,
        stream));
  });
}

template <typename T>
void AllgatherOp<T>::runAsync() {
  this->runNCCL([](
      const NCCLElement<T>& element, ncclComm_t comm, cudaStream_t stream) {
    NCCL_CHECK(ncclAllGather(
        *element.src,
        element.src.getCount(),
        ncclTypeWrapper<T>::type,
        *element.dst,
        comm,
        stream));
  });
}

template class NCCLExecution<float>;
template class NCCLContext<float>;
template class NCCLOp<float>;

template class ReduceOp<float>;
template class AllreduceOp<float>;
template class ReduceScatterOp<float>;
template class BroadcastOp<float>;
template class AllgatherOp<float>;

} // namespace nccl
} // namespace gloo
