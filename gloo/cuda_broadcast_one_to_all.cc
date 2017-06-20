/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_broadcast_one_to_all.h"

#include "gloo/cuda_collectives_device.h"
#include "gloo/cuda_collectives_host.h"
#include "gloo/cuda_private.h"

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
      sender_.resize(contextSize_);
      for (auto i = 0; i < contextSize_; i++) {
        if (i == contextRank_) {
          continue;
        }

        sender_[i] = make_unique<forSender>();
        auto& pair = context_->getPair(i);
        sender_[i]->clearToSendBuffer = pair->createRecvBuffer(
            slot, &sender_[i]->dummy, sizeof(sender_[i]->dummy));
        sender_[i]->sendBuffer = pair->createSendBuffer(
            slot, *scratch_, bytes_);
      }
    } else {
      receiver_ = make_unique<forReceiver>();
      auto& rootPair = context_->getPair(rootRank_);
      receiver_->clearToSendBuffer = rootPair->createSendBuffer(
          slot, &receiver_->dummy, sizeof(receiver_->dummy));
      receiver_->recvBuffer = rootPair->createRecvBuffer(
          slot, *scratch_, bytes_);
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

    // Fire off send operations after receiving clear to send
    for (auto i = 0; i < contextSize_; i++) {
      if (i == contextRank_) {
        continue;
      }
      sender_[i]->clearToSendBuffer->waitRecv();
      sender_[i]->sendBuffer->send();
    }

    // Broadcast locally while sends are happening
    if (localBroadcastOp_) {
      localBroadcastOp_->runAsync();
      if (synchronizeDeviceOutputs_) {
        localBroadcastOp_->wait();
      }
    }

    // Wait for all send operations to complete
    for (auto i = 0; i < contextSize_; i++) {
      if (i == contextRank_) {
        continue;
      }
      sender_[i]->sendBuffer->waitSend();
    }
  } else {
    CudaStream& stream = streams_[rootPointerRank_];
    // Ensure previous H2D copy is complete before notifying the sender
    // NOTE: this only waits for last copyAsync, not for the whole stream
    stream.wait();

    receiver_->clearToSendBuffer->send();
    receiver_->recvBuffer->waitRecv();

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

// Instantiate templates
#define INSTANTIATE_TEMPLATE(T)                                         \
template class CudaBroadcastOneToAll<T, CudaHostWorkspace<T> >;


INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo
