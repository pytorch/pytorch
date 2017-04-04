/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_broadcast_one_to_all.h"

#include "gloo/common/logging.h"
#include "gloo/cuda_private.h"
#include "gloo/nccl/nccl.h"

namespace gloo {

template <typename T>
CudaBroadcastOneToAll<T>::CudaBroadcastOneToAll(
    const std::shared_ptr<Context>& context,
    const std::vector<T*>& ptrs,
    int count,
    int rootRank,
    int rootPointerRank,
    const std::vector<cudaStream_t>& streams)
    : Algorithm(context),
      hostPtr_(nullptr),
      count_(count),
      bytes_(count * sizeof(T)),
      rootRank_(rootRank),
      rootPointerRank_(rootPointerRank),
      synchronizeDeviceOutputs_(streams.size() == 0) {
  GLOO_ENFORCE_GE(rootRank_, 0);
  GLOO_ENFORCE_LT(rootRank_, contextSize_);

  auto newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }

  // Create CUDA device pointers
  for (auto i = 0; i < ptrs.size(); i++) {
    if (newStream) {
      devicePtrs_.push_back(
        CudaDevicePointer<T>::create(ptrs[i], count_));
    } else {
      devicePtrs_.push_back(
        CudaDevicePointer<T>::create(ptrs[i], count_, streams[i]));
    }
  }

  // Allocate host side buffer if we need to communicate
  if (contextSize_ > 1) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaMallocHost(&hostPtr_, bytes_));
  }

  // Setup pairs/buffers for sender/receivers
  if (contextSize_ > 1) {
    auto slot = context_->nextSlot();
    if (contextRank_ == rootRank_) {
      for (int i = 0; i < contextSize_; i++) {
        if (i == contextRank_) {
          continue;
        }

        auto& pair = context_->getPair(i);
        sendDataBuffers_.push_back(
          pair->createSendBuffer(slot, hostPtr_, bytes_));
      }
    } else {
      auto& rootPair = context_->getPair(rootRank_);
      recvDataBuffer_ = rootPair->createRecvBuffer(slot, hostPtr_, bytes_);
    }
  }

  // Setup local broadcast if needed
  if (devicePtrs_.size() > 1) {
    localBroadcastOp_ =
      cudaDeviceBroadcast(devicePtrs_, devicePtrs_[0], 0, count_);
  }
}

template <typename T>
CudaBroadcastOneToAll<T>::~CudaBroadcastOneToAll() {
  std::lock_guard<std::mutex> lock(CudaShared::getMutex());
  if (hostPtr_ != nullptr) {
    CUDA_CHECK(cudaFreeHost(hostPtr_));
  }
}

template <typename T>
void CudaBroadcastOneToAll<T>::run() {
  if (contextSize_ == 1) {
    if (localBroadcastOp_) {
      localBroadcastOp_->runAsync();
      if (synchronizeDeviceOutputs_) {
        localBroadcastOp_->wait();
      }
    }
    return;
  }

  if (contextRank_ == rootRank_) {
    // Copy device buffer to host
    devicePtrs_[rootPointerRank_].copyToHostAsync(hostPtr_);
    devicePtrs_[rootPointerRank_].wait();

    // Fire off all send operations concurrently
    for (auto& buf : sendDataBuffers_) {
      buf->send();
    }

    // Broadcast locally while sends are happening
    if (localBroadcastOp_) {
      localBroadcastOp_->runAsync();
      if (synchronizeDeviceOutputs_) {
        localBroadcastOp_->wait();
      }
    }

    // Wait for all send operations to complete
    for (auto& buf : sendDataBuffers_) {
      buf->waitSend();
    }
  } else {
    // Wait on buffer
    recvDataBuffer_->waitRecv();
    // Copy host buffer to device
    devicePtrs_[rootPointerRank_].copyFromHostAsync(hostPtr_);

    // Broadcast locally after receiving from root
    if (localBroadcastOp_) {
      // Since broadcast synchronizes on root pointer, there is no
      // need to explicity wait for the memcpy to complete.
      localBroadcastOp_->runAsync();
      if (synchronizeDeviceOutputs_) {
        localBroadcastOp_->wait();
      }
    } else {
      // Wait for memcpy to complete
      if (synchronizeDeviceOutputs_) {
        devicePtrs_[rootPointerRank_].wait();
      }
    }
  }
}

// Instantiate template
template class CudaBroadcastOneToAll<float>;

} // namespace gloo
