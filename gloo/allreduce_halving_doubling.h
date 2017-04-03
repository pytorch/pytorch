/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "gloo/algorithm.h"
#include "gloo/common/error.h"
#include "gloo/context.h"

namespace gloo {

template <typename T>
class AllreduceHalvingDoubling : public Algorithm {
 public:
  AllreduceHalvingDoubling(
      const std::shared_ptr<Context>& context,
      const std::vector<T*> ptrs,
      const int count,
      const ReductionFunction<T>* fn = ReductionFunction<T>::sum)
      : Algorithm(context),
        ptrs_(ptrs),
        count_(count),
        bytes_(count_ * sizeof(T)),
        chunks_(this->contextSize_),
        chunkSize_((count_ + chunks_ - 1) / chunks_),
        chunkBytes_(chunkSize_ * sizeof(T)),
        steps_(log2(this->contextSize_)),
        fn_(fn),
        recvBuf_(count_),
        sendOffsets_(steps_),
        recvOffsets_(steps_),
        sendDataBufs_(steps_),
        recvDataBufs_(steps_) {
    if ((this->contextSize_ & (this->contextSize_ - 1)) != 0) {
      throw ::gloo::Exception(
          "allreduce_halving_doubling does not support non-power-of-2 "
          "number of processes yet");
    }

    size_t bitmask = 1;
    size_t stepChunkSize = chunkSize_ << (steps_ - 1);
    size_t stepChunkBytes = stepChunkSize * sizeof(T);
    size_t sendOffset = 0;
    size_t recvOffset = 0;
    size_t bufferOffset = 0; // offset into recvBuf_
    for (int i = 0; i < steps_; i++) {
      const auto destRank = (this->context_->rank) ^ bitmask;
      commPairs_.push_back(this->context_->getPair(destRank));
      const auto slot = this->context_->nextSlot();
      sendOffsets_[i] = sendOffset + ((destRank & bitmask) ? stepChunkSize : 0);
      recvOffsets_[i] =
        recvOffset + ((this->context_->rank & bitmask) ? stepChunkSize : 0);
      if (sendOffsets_[i] < count_) {
        sendDataBufs_[i] =
          commPairs_[i].get()->createSendBuffer(slot, ptrs_[0], bytes_);
      }
      if (recvOffsets_[i] < count_) {
        recvDataBufs_[i] = commPairs_[i].get()->createRecvBuffer(
          slot, &recvBuf_[bufferOffset], stepChunkBytes);
      }
      bufferOffset += stepChunkSize;
      if (this->context_->rank & bitmask) {
        sendOffset += stepChunkSize;
        recvOffset += stepChunkSize;
      }
      bitmask <<= 1;
      stepChunkSize >>= 1;
      stepChunkBytes >>= 1;

      auto notificationSlot = this->context_->nextSlot();
      sendNotificationBufs_.push_back(commPairs_[i].get()->createSendBuffer(
          notificationSlot, &dummy_, sizeof(dummy_)));
      recvNotificationBufs_.push_back(commPairs_[i].get()->createRecvBuffer(
          notificationSlot, &dummy_, sizeof(dummy_)));
    }
  }

  void run() {
    for (int i = 1; i < ptrs_.size(); i++) {
      fn_->call(ptrs_[0], ptrs_[i], count_);
    }
    size_t bufferOffset = 0;
    size_t numItems = chunkSize_ << (steps_ -1 );
    size_t numSending;
    size_t numReceiving;

    // Reduce-scatter
    for (int i = 0; i < steps_; i++) {
      if (sendOffsets_[i] < count_) {
        numSending = sendOffsets_[i] + numItems > count_ ?
          count_ - sendOffsets_[i] : numItems;
        sendDataBufs_[i]->send(
          sendOffsets_[i] * sizeof(T), numSending * sizeof(T));
      }
      if (recvOffsets_[i] < count_) {
        recvDataBufs_[i]->waitRecv();
        numReceiving = recvOffsets_[i] + numItems > count_
            ? count_ - recvOffsets_[i]
            : numItems;
        fn_->call(
            &ptrs_[0][recvOffsets_[i]], &recvBuf_[bufferOffset], numReceiving);
      }
      bufferOffset += numItems;
      sendNotificationBufs_[i]->send();
      numItems >>= 1;
    }

    numItems = chunkSize_;

    // Allgather
    for (int i = steps_ - 1; i >= 0; i--) {
      // verify that destination rank has received and processed this rank's
      // message during the reduce-scatter phase
      recvNotificationBufs_[i]->waitRecv();
      if (recvOffsets_[i] < count_) {
        numSending = recvOffsets_[i] + numItems > count_ ?
          count_ - recvOffsets_[i] : numItems;
        sendDataBufs_[i]->send(
          recvOffsets_[i] * sizeof(T), numSending * sizeof(T));
      }
      bufferOffset -= numItems;
      if (sendOffsets_[i] < count_) {
        recvDataBufs_[i]->waitRecv();
        numReceiving = sendOffsets_[i] + numItems > count_ ?
          count_ - sendOffsets_[i] : numItems;
        memcpy(
          &ptrs_[0][sendOffsets_[i]],
          &recvBuf_[bufferOffset],
          numReceiving * sizeof(T));
      }
      numItems <<= 1;
    }

    // Broadcast ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      memcpy(ptrs_[i], ptrs_[0], bytes_);
    }
  }

 protected:
  std::vector<T*> ptrs_;
  const int count_;
  const int bytes_;
  const size_t chunks_;
  const size_t chunkSize_;
  const size_t chunkBytes_;
  const size_t steps_;
  const ReductionFunction<T>* fn_;

  // buffer where data is received prior to being reduced
  std::vector<T> recvBuf_;

  // offsets into the data buffer from which to send during the reduce-scatter
  // these become the offsets at which the process receives during the allgather
  // indexed by step
  std::vector<size_t> sendOffsets_;

  // offsets at which data is reduced during the reduce-scatter and sent from in
  // the allgather
  std::vector<size_t> recvOffsets_;

  std::vector<std::reference_wrapper<std::unique_ptr<transport::Pair>>>
      commPairs_;

  std::vector<std::unique_ptr<transport::Buffer>> sendDataBufs_;
  std::vector<std::unique_ptr<transport::Buffer>> recvDataBufs_;

  int dummy_;
  std::vector<std::unique_ptr<transport::Buffer>> sendNotificationBufs_;
  std::vector<std::unique_ptr<transport::Buffer>> recvNotificationBufs_;
};

} // namespace gloo
