/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_broadcast_one_to_all.h"

#include "gloo/cuda_private.h"

namespace gloo {

template <typename T>
CudaBroadcastOneToAll<T>::CudaBroadcastOneToAll(
    const std::shared_ptr<Context>& context,
    T* ptr,
    int count,
    int rootRank)
    : Broadcast<T>(context, rootRank),
      devicePtr_(CudaDevicePointer<T>::createWithNewStream(ptr, count)) {
  auto bytes = count * sizeof(T);
  CUDA_CHECK(cudaMallocHost(&hostPtr_, bytes));
  if (this->contextRank_ == this->rootRank_) {
    for (int i = 0; i < this->contextSize_; i++) {
      if (i == this->contextRank_) {
        continue;
      }

      auto& pair = this->context_->getPair(i);
      sendDataBuffers_.push_back(
          pair->createSendBuffer(0, hostPtr_, bytes));
    }
  } else {
    auto& rootPair = this->context_->getPair(this->rootRank_);
    recvDataBuffer_ = rootPair->createRecvBuffer(0, hostPtr_, bytes);
  }
}

template <typename T>
CudaBroadcastOneToAll<T>::~CudaBroadcastOneToAll() {
  CUDA_CHECK(cudaFreeHost(hostPtr_));
}

template <typename T>
void CudaBroadcastOneToAll<T>::run() {
  if (this->contextRank_ == this->rootRank_) {
    // Copy device buffer to host
    devicePtr_.copyToHostAsync(hostPtr_);
    devicePtr_.waitAsync();
    // Fire off all send operations concurrently
    for (auto& buf : sendDataBuffers_) {
      buf->send();
    }
    // Wait for all send operations to complete
    for (auto& buf : sendDataBuffers_) {
      buf->waitSend();
    }
  } else {
    // Wait on buffer
    recvDataBuffer_->waitRecv();
    // Copy host buffer to device
    devicePtr_.copyFromHostAsync(hostPtr_);
    devicePtr_.waitAsync();
  }
}

// Instantiate template
template class CudaBroadcastOneToAll<float>;

} // namespace gloo
