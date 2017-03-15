/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_broadcast_one_to_all.h"

#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/cuda_private.h"
#include "gloo/cuda_nccl.h"

namespace gloo {

template <typename T>
struct CudaBroadcastOneToAll<T>::LocalBroadcast {
  LocalBroadcast(
      const std::vector<CudaDevicePointer<T> >& devicePtrs,
      int count,
      int rootPointerRank) {
    std::vector<nccl::NCCLElement<T>> elements;
    for (auto& devicePtr : devicePtrs) {
      GLOO_ENFORCE_EQ(count, devicePtr.getCount());
      const auto ptr = *devicePtr;
      const auto stream = devicePtr.getStream();
      nccl::NCCLElement<T> element(
        CudaDevicePointer<T>::create(ptr, count, stream),
        CudaDevicePointer<T>::create(ptr, count, stream));
      elements.push_back(std::move(element));
    }

    auto& rootDevicePtr = devicePtrs[rootPointerRank];
    nccl::NCCLExecution<T> execution(
      std::move(elements),
      rootDevicePtr.getDeviceID());

    broadcastOp.reset(new nccl::BroadcastOp<T>(std::move(execution)));
  }

  void runAsync() {
    broadcastOp->runAsync();
  }

  void wait() {
    broadcastOp->wait();
  }

  std::unique_ptr<nccl::BroadcastOp<T> > broadcastOp;
};

template <typename T>
CudaBroadcastOneToAll<T>::CudaBroadcastOneToAll(
    const std::shared_ptr<Context>& context,
    const std::vector<T*>& ptrs,
    int count,
    int rootRank,
    int rootPointerRank,
    const std::vector<cudaStream_t>& streams)
    : Broadcast<T>(context, rootRank, rootPointerRank),
      hostPtr_(nullptr),
      count_(count),
      bytes_(count * sizeof(T)),
      synchronizeDeviceOutputs_(streams.size() == 0) {
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
  if (this->contextSize_ > 1) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaMallocHost(&hostPtr_, bytes_));
  }

  // Setup pairs/buffers for sender/receivers
  if (this->contextSize_ > 1) {
    auto slot = this->context_->nextSlot();
    if (this->contextRank_ == this->rootRank_) {
      for (int i = 0; i < this->contextSize_; i++) {
        if (i == this->contextRank_) {
          continue;
        }

        auto& pair = this->context_->getPair(i);
        sendDataBuffers_.push_back(
          pair->createSendBuffer(slot, hostPtr_, bytes_));
      }
    } else {
      auto& rootPair = this->context_->getPair(this->rootRank_);
      recvDataBuffer_ = rootPair->createRecvBuffer(slot, hostPtr_, bytes_);
    }
  }

  // Setup local broadcast if needed
  if (devicePtrs_.size() > 1) {
    localBroadcast_ =
      make_unique<LocalBroadcast>(devicePtrs_, count_, rootPointerRank);
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
  if (this->contextSize_ == 1) {
    if (localBroadcast_) {
      localBroadcast_->runAsync();
      if (synchronizeDeviceOutputs_) {
        localBroadcast_->wait();
      }
    }
    return;
  }

  if (this->contextRank_ == this->rootRank_) {
    // Copy device buffer to host
    devicePtrs_[this->getRootPointerRank()].copyToHostAsync(hostPtr_);
    devicePtrs_[this->getRootPointerRank()].wait();

    // Fire off all send operations concurrently
    for (auto& buf : sendDataBuffers_) {
      buf->send();
    }

    // Broadcast locally while sends are happening
    if (localBroadcast_) {
      localBroadcast_->runAsync();
      if (synchronizeDeviceOutputs_) {
        localBroadcast_->wait();
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
    devicePtrs_[this->getRootPointerRank()].copyFromHostAsync(hostPtr_);

    // Broadcast locally after receiving from root
    if (localBroadcast_) {
      // Since broadcast synchronizes on root pointer, there is no
      // need to explicity wait for the memcpy to complete.
      localBroadcast_->runAsync();
      if (synchronizeDeviceOutputs_) {
        localBroadcast_->wait();
      }
    } else {
      // Wait for memcpy to complete
      if (synchronizeDeviceOutputs_) {
        devicePtrs_[this->getRootPointerRank()].wait();
      }
    }
  }
}

// Instantiate template
template class CudaBroadcastOneToAll<float>;

} // namespace gloo
