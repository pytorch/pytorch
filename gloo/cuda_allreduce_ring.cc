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

#include "gloo/cuda_private.h"

namespace gloo {

template <typename T>
CudaAllreduceRing<T>::CudaAllreduceRing(
  const std::shared_ptr<Context>& context,
  const std::vector<T*>& ptrs,
  int count,
  const std::vector<cudaStream_t>& streams)
    : Allreduce<T>(context, nullptr),
      count_(count),
      bytes_(count * sizeof(T)),
      synchronizeDeviceOutputs_(streams.size() == 0),
      leftPair_(this->getLeftPair()),
      rightPair_(this->getRightPair()) {
  auto newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }

  hostPtrs_.resize(ptrs.size());
  for (int i = 0; i < ptrs.size(); i++) {
    if (newStream) {
      devicePtrs_.push_back(
          CudaDevicePointer<T>::create(ptrs[i], count_));
    } else {
      devicePtrs_.push_back(
          CudaDevicePointer<T>::create(ptrs[i], count_, streams[i]));
    }
    CUDA_CHECK(cudaMallocHost(&hostPtrs_[i], bytes_));
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
  for (auto& hostPtr : hostPtrs_) {
    CUDA_CHECK(cudaFreeHost(hostPtr));
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
  for (int i = 0; i < devicePtrs_.size(); i++) {
    devicePtrs_[i].copyToHostAsync(hostPtrs_[i]);
  }

  // Reduce specified pointers into hostPtrs_[0]
  devicePtrs_[0].waitAsync();
  for (int i = 1; i < devicePtrs_.size(); i++) {
    devicePtrs_[i].waitAsync();
    this->fn_(hostPtrs_[0], hostPtrs_[i], count_);
  }

  // Intialize outbox with locally reduced values
  memcpy(outbox_, hostPtrs_[0], bytes_);

  int numRounds = this->contextSize_ - 1;
  for (int round = 0; round < numRounds; round++) {
    // Initiate write to inbox of node on the right
    sendDataBuf_->send();

    // Wait for inbox write from node on the left
    recvDataBuf_->waitRecv();

    // Reduce
    this->fn_(hostPtrs_[0], inbox_, count_);

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
  for (int i = 0; i < devicePtrs_.size(); i++) {
    devicePtrs_[i].copyFromHostAsync(hostPtrs_[0]);
  }

  // If running synchronously, wait for memcpy's to complete
  if (synchronizeDeviceOutputs_) {
    for (int i = 0; i < devicePtrs_.size(); i++) {
      devicePtrs_[i].waitAsync();
    }
  }
}

// Instantiate template
template class CudaAllreduceRing<float>;

} // namespace gloo
