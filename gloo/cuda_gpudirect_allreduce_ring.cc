/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_gpudirect_allreduce_ring.h"

#include <string.h>

#include "gloo/cuda_private.h"
#include "gloo/nccl/nccl.h"

namespace gloo {

template <typename T>
CudaGPUDirectAllreduceRing<T>::CudaGPUDirectAllreduceRing(
  const std::shared_ptr<Context>& context,
  const std::vector<T*>& ptrs,
  int count,
  const std::vector<cudaStream_t>& streams)
    : Allreduce<T>(context, CudaReductionFunction<T>::sum),
      count_(count),
      bytes_(count_ * sizeof(T)),
      synchronizeDeviceOutputs_(streams.size() == 0),
      fn_(CudaReductionFunction<T>::toCudaReductionFunction(Allreduce<T>::fn_)),
      leftPair_(this->getLeftPair()),
      rightPair_(this->getRightPair()) {
  auto newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }

  for (auto i = 0; i < ptrs.size(); i++) {
    if (newStream) {
      devicePtrs_.push_back(
          CudaDevicePointer<T>::create(ptrs[i], count_));
    } else {
      devicePtrs_.push_back(
          CudaDevicePointer<T>::create(ptrs[i], count_, streams[i]));
    }
  }

  // Create NCCL elements for each device pointer if necessary
  if (devicePtrs_.size() > 1) {
    std::vector<nccl::NCCLElement<T>> reduceElements;
    std::vector<nccl::NCCLElement<T>> broadcastElements;
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      const auto ptr = *devicePtrs_[i];
      const auto stream = devicePtrs_[i].getStream();
      reduceElements.push_back(nccl::NCCLElement<T>(
          CudaDevicePointer<T>::create(ptr, count_, stream),
          CudaDevicePointer<T>::create(ptr, count_, stream)));
      broadcastElements.push_back(nccl::NCCLElement<T>(
          CudaDevicePointer<T>::create(ptr, count_, stream),
          CudaDevicePointer<T>::create(ptr, count_, stream)));
    }

    localReduceOp_ = std::make_unique<nccl::ReduceOp<T> >(
      nccl::NCCLExecution<T>(std::move(reduceElements)),
      ReductionFunction<T>::sum,
      devicePtrs_[0].getDeviceID());
    localBroadcastOp_ = std::make_unique<nccl::BroadcastOp<T> >(
      nccl::NCCLExecution<T>(std::move(reduceElements)),
      devicePtrs_[0].getDeviceID());
  }

  // Allocate inbox/outbox on device of first pointer
  {
    CudaDeviceScope scope(devicePtrs_[0].getDeviceID());
    const auto& stream = devicePtrs_[0].getStream();
    T *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes_));
    inbox_ = std::move(CudaDevicePointer<T>::create(ptr, count_, stream));
    CUDA_CHECK(cudaMalloc(&ptr, bytes_));
    outbox_ = std::move(CudaDevicePointer<T>::create(ptr, count_, stream));
  }

  auto slot = this->context_->nextSlot();

  // Buffer to send to (rank+1).
  sendDataBuf_ = rightPair_->createSendBuffer(slot, *outbox_, bytes_);

  // Buffer that (rank-1) writes to.
  recvDataBuf_ = leftPair_->createRecvBuffer(slot, *inbox_, bytes_);

  // Dummy buffers for localized barrier.
  // Before sending to the right, we only need to know that the node
  // on the right is done using the inbox that's about to be written
  // into. No need for a global barrier.
  auto notificationSlot = this->context_->nextSlot();
  sendNotificationBuf_ =
    leftPair_->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));
  recvNotificationBuf_ =
    rightPair_->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));
}

template <typename T>
CudaGPUDirectAllreduceRing<T>::~CudaGPUDirectAllreduceRing() {
  if (*inbox_ != nullptr) {
    CudaDeviceScope scope(inbox_.getDeviceID());
    CUDA_CHECK(cudaFree(*inbox_));
  }
  if (*outbox_ != nullptr) {
    CudaDeviceScope scope(outbox_.getDeviceID());
    CUDA_CHECK(cudaFree(*outbox_));
  }
}

template <typename T>
void CudaGPUDirectAllreduceRing<T>::run() {
  CudaDeviceGuard guard;

  // Kick off local reduction if necessary
  if (localReduceOp_) {
    localReduceOp_->runAsync();
  }

  // Initialize outbox with locally reduced values.
  auto& root = devicePtrs_[0];
  root.copyToDeviceAsync(*outbox_);
  root.wait();

  int numRounds = this->contextSize_ - 1;
  for (int round = 0; round < numRounds; round++) {
    // Initiate write to inbox of node on the right
    sendDataBuf_->send();

    // Wait for inbox write from node on the left
    recvDataBuf_->waitRecv();

    // Reduce
    fn_->callAsync(*root, *inbox_, count_, root.getStream());

    // Wait for outbox write to complete
    sendDataBuf_->waitSend();

    // Prepare for next round if necessary
    if (round < (numRounds - 1)) {
      outbox_.copyFromDeviceAsync(*inbox_);
      outbox_.wait();
    }

    // Wait for reduction to complete
    root.wait();

    // Send notification to node on the left that
    // this node is ready for an inbox write.
    sendNotificationBuf_->send();

    // Wait for notification from node on the right
    recvNotificationBuf_->waitRecv();
  }

  // Kick off local broadcast if necessary
  if (localBroadcastOp_) {
    localBroadcastOp_->runAsync();
    if (synchronizeDeviceOutputs_) {
      localBroadcastOp_->wait();
    }
  }
}

// Instantiate template
template class CudaGPUDirectAllreduceRing<float>;

} // namespace gloo
