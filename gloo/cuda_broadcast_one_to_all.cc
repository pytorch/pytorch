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
    T* dataPtr,
    int dataSize,
    int rootRank)
    : Broadcast<T>(context, rootRank),
      dataPtr_(dataPtr),
      dataSize_(dataSize),
      dataSizeBytes_(dataSize * sizeof(T)),
      deviceId_(getGPUIDForPointer(dataPtr)) {
  CUDA_CHECK(cudaMallocHost(&hostPtr_, dataSizeBytes_));
  if (this->contextRank_ == this->rootRank_) {
    for (int i = 0; i < this->contextSize_; i++) {
      if (i == this->contextRank_) {
        continue;
      }

      auto& pair = this->context_->getPair(i);
      sendDataBuffers_.push_back(
          pair->createSendBuffer(0, hostPtr_, dataSizeBytes_));
    }
  } else {
    auto& rootPair = this->context_->getPair(this->rootRank_);
    recvDataBuffer_ = rootPair->createRecvBuffer(0, hostPtr_, dataSizeBytes_);
  }
}

template <typename T>
CudaBroadcastOneToAll<T>::~CudaBroadcastOneToAll() {
  CUDA_CHECK(cudaFreeHost(hostPtr_));
}

template <typename T>
void CudaBroadcastOneToAll<T>::run() {
  CudaDeviceGuard guard;
  CUDA_CHECK(cudaSetDevice(deviceId_));
  if (this->contextRank_ == this->rootRank_) {
    // Copy device buffer to host
    CUDA_CHECK(cudaMemcpy(
                hostPtr_,
                dataPtr_,
                dataSizeBytes_,
                cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaMemcpy(
                dataPtr_,
                hostPtr_,
                dataSizeBytes_,
                cudaMemcpyHostToDevice));
  }
}

// Instantiate template
template class CudaBroadcastOneToAll<float>;

} // namespace gloo
