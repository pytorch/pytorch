/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_allreduce_ring.h"

#include <string.h>

#include "gloo/cuda.h"

namespace gloo {

template <typename T>
struct CudaAllreduceRing<T>::ptr {
  T* device;
  T* host;

  // GPU ID the device pointer lives on
  int deviceId;

  // Kick off memcpy's on non-default stream with high priority.
  // Use events to wait for memcpy's to complete on host side.
  cudaStream_t stream;
  cudaEvent_t event;
};

template <typename T>
CudaAllreduceRing<T>::CudaAllreduceRing(
  const std::shared_ptr<Context>& context,
  std::vector<T*> ptrs,
  int count)
    : Allreduce<T>(context, nullptr),
    count_(count),
    bytes_(count * sizeof(T)),
    leftPair_(this->getLeftPair()),
    rightPair_(this->getRightPair()) {
  for (int i = 0; i < ptrs.size(); i++) {
    struct ptr tmp;
    tmp.device = ptrs[i];
    tmp.deviceId = getGPUIDForPointer(ptrs[i]);
    CUDA_CHECK(cudaMallocHost(&tmp.host, bytes_));

    int loPri, hiPri;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&loPri, &hiPri));
    CUDA_CHECK(cudaStreamCreateWithPriority(
                 &tmp.stream, cudaStreamNonBlocking, hiPri));
    CUDA_CHECK(cudaEventCreateWithFlags(
                 &tmp.event, cudaEventDefault | cudaEventDisableTiming));
    ptrs_.push_back(tmp);
  }

  inbox_ = static_cast<T*>(malloc(bytes_));
  outbox_ = static_cast<T*>(malloc(bytes_));

  // Buffer to send to (rank+1).
  sendDataBuf_ = rightPair_->createSendBuffer(0, outbox_, bytes_);

  // Buffer that (rank-1) writes to.
  recvDataBuf_ = leftPair_->createRecvBuffer(0, inbox_, bytes_);

  // Dummy buffers for localized barrier.
  // Before sending to the right, we only need to know that the node
  // on the right is done using the inbox that's about to be written
  // into. No need for a global barrier.
  sendNotificationBuf_ =
    leftPair_->createSendBuffer(1, &dummy_, sizeof(dummy_));
  recvNotificationBuf_ =
    rightPair_->createRecvBuffer(1, &dummy_, sizeof(dummy_));
}

template <typename T>
CudaAllreduceRing<T>::~CudaAllreduceRing() {
  CudaDeviceGuard guard;

  for (auto& ptr : ptrs_) {
    CUDA_CHECK(cudaSetDevice(ptr.deviceId));
    CUDA_CHECK(cudaStreamDestroy(ptr.stream));
    CUDA_CHECK(cudaEventDestroy(ptr.event));
    CUDA_CHECK(cudaFreeHost(ptr.host));
  }
  if (inbox_ != nullptr) {
    free(inbox_);
  }
  if (outbox_ != nullptr) {
    free(outbox_);
  }
}

template <typename T>
void CudaAllreduceRing<T>::run() {
  CudaDeviceGuard guard;

  // Asynchronously copy all device buffers to host
  for (auto& ptr : ptrs_) {
    CUDA_CHECK(cudaSetDevice(ptr.deviceId));
    CUDA_CHECK(cudaMemcpyAsync(
                 ptr.host,
                 ptr.device,
                 bytes_,
                 cudaMemcpyDeviceToHost,
                 ptr.stream));
    CUDA_CHECK(cudaEventRecord(ptr.event, ptr.stream));
  }

  // Reduce specified pointers into ptrs_[0]
  CUDA_CHECK(cudaEventSynchronize(ptrs_[0].event));
  for (int i = 1; i < ptrs_.size(); i++) {
    CUDA_CHECK(cudaEventSynchronize(ptrs_[i].event));
    this->fn_(ptrs_[0].host, ptrs_[i].host, count_);
  }

  // Intialize outbox with locally reduced values
  memcpy(outbox_, ptrs_[0].host, bytes_);

  int numRounds = this->contextSize_ - 1;
  for (int round = 0; round < numRounds; round++) {
    // Initiate write to inbox of node on the right
    sendDataBuf_->send();

    // Wait for inbox write from node on the left
    recvDataBuf_->waitRecv();

    // Reduce
    this->fn_(ptrs_[0].host, inbox_, count_);

    // Wait for outbox write to complete
    sendDataBuf_->waitSend();

    // Prepare for next round if necessary
    if (round < (numRounds - 1)) {
      memcpy(outbox_, inbox_, bytes_);
    }

    // Send notification to node on the left that
    // this node is ready for an inbox write.
    sendNotificationBuf_->send();

    // Wait for notification from node on the right
    recvNotificationBuf_->waitRecv();
  }

  // Asynchronously copy result buffer to all device buffers
  for (auto& ptr : ptrs_) {
    CUDA_CHECK(cudaSetDevice(ptr.deviceId));
    CUDA_CHECK(cudaMemcpyAsync(
                 ptr.device,
                 ptrs_[0].host,
                 bytes_,
                 cudaMemcpyHostToDevice,
                 ptr.stream));
    CUDA_CHECK(cudaEventRecord(ptr.event, ptr.stream));
  }

  // Wait for memcpy's to complete
  for (auto& ptr : ptrs_) {
    CUDA_CHECK(cudaEventSynchronize(ptr.event));
  }
}

// Instantiate template
template class CudaAllreduceRing<float>;

} // namespace gloo
