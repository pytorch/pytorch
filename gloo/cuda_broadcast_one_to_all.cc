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

template <typename T, typename W>
CudaBroadcastOneToAll<T, W>::CudaBroadcastOneToAll(
    const std::shared_ptr<Context>& context,
    const std::vector<T*>& ptrs,
    int count,
    int rootRank,
    int rootPointerRank,
    const std::vector<cudaStream_t>& streams)
    : Algorithm(context),
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

  for (auto i = 0; i < ptrs.size(); i++) {
    auto ptr = CudaDevicePointer<T>::create(ptrs[i], count_);
    if (newStream) {
      streams_.push_back(CudaStream(ptr.getDeviceID()));
    } else {
      streams_.push_back(CudaStream(ptr.getDeviceID(), streams[i]));
    }
    devicePtrs_.push_back(std::move(ptr));
  }

  // Workspace specific initialization (see below)
  init();

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
          pair->createSendBuffer(slot, *scratch_, bytes_));
      }
    } else {
      auto& rootPair = context_->getPair(rootRank_);
      recvDataBuffer_ = rootPair->createRecvBuffer(slot, *scratch_, bytes_);
    }
  }

  // Setup local broadcast if needed
  if (devicePtrs_.size() > 1) {
    localBroadcastOp_ =
      cudaDeviceBroadcast(streams_, devicePtrs_, devicePtrs_[0], 0, count_);
  }
}

template <typename T, typename W>
void CudaBroadcastOneToAll<T, W>::run() {
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
    CudaStream& stream = streams_[rootPointerRank_];

    // Copy device buffer to host
    stream.copyAsync(scratch_, devicePtrs_[rootPointerRank_]);
    stream.wait();

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
    CudaStream& stream = streams_[rootPointerRank_];

    // Wait on buffer
    recvDataBuffer_->waitRecv();

    // Copy host buffer to device
    stream.copyAsync(devicePtrs_[rootPointerRank_], scratch_);

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
        stream.wait();
      }
    }
  }
}

template <typename T, typename W>
template <typename U>
void CudaBroadcastOneToAll<T, W>::init(
    typename std::enable_if<std::is_same<U, CudaHostWorkspace<T> >::value,
                            typename U::Pointer>::type*) {
  // Allocate host side buffer if we need to communicate
  if (contextSize_ > 1) {
    // Since broadcast transmits from/to a buffer in system memory, the
    // scratch space is a new host side buffer.
    scratch_ = W::Pointer::alloc(count_);
  }
}

// Instantiate template
template class CudaBroadcastOneToAll<float, CudaHostWorkspace<float>>;

} // namespace gloo
