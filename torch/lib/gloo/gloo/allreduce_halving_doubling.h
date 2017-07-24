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

namespace {
// returns the last n bits of ctr reversed
uint32_t reverseLastNBits(uint32_t ctr, uint32_t n) {
  uint32_t bitMask = 1;
  uint32_t reversed = 0;
  while (bitMask < (static_cast<uint32_t>(1) << n)) {
    reversed <<= 1;
    if (ctr & bitMask) {
      reversed |= 1;
    }
    bitMask <<= 1;
  }
  return reversed;
}
}

template <typename T>
class AllreduceHalvingDoubling : public Algorithm {
  void initBinaryBlocks() {
    uint32_t offset = this->contextSize_;
    uint32_t blockSize = 1;
    uint32_t currentBlockSize = 0;
    uint32_t prevBlockSize = 0;
    do {
      if (this->contextSize_ & blockSize) {
        prevBlockSize = currentBlockSize;
        currentBlockSize = blockSize;
        offset -= blockSize;
        if (myBinaryBlockSize_ != 0) {
          nextLargerBlockSize_ = currentBlockSize;
          break;
        }
        if (offset <= this->context_->rank) {
          offsetToMyBinaryBlock_ = offset;
          myBinaryBlockSize_ = currentBlockSize;
          nextSmallerBlockSize_ = prevBlockSize;
        }
      }
      blockSize <<= 1;
    } while (offset != 0);

    stepsWithinBlock_ = log2(myBinaryBlockSize_);
    rankInBinaryBlock_ = this->context_->rank % myBinaryBlockSize_;
  }

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
        steps_(log2(this->contextSize_)),
        chunks_(1 << steps_),
        chunkSize_((count_ + chunks_ - 1) / chunks_),
        chunkBytes_(chunkSize_ * sizeof(T)),
        fn_(fn),
        recvBuf_(chunkSize_ << steps_),
        sendOffsets_(steps_),
        recvOffsets_(steps_),
        sendCounts_(steps_, 0),
        recvCounts_(steps_, 0),
        sendCountToLargerBlock_(0),
        offsetToMyBinaryBlock_(0),
        myBinaryBlockSize_(0),
        stepsWithinBlock_(0),
        rankInBinaryBlock_(0),
        nextSmallerBlockSize_(0),
        nextLargerBlockSize_(0) {
    initBinaryBlocks();
    sendDataBufs_.reserve(stepsWithinBlock_);
    recvDataBufs_.reserve(stepsWithinBlock_);
    // Reserve max needed number of context slots. Up to 2 slots per process
    // pair are needed (one for regular sends and one for notifications). For
    // simplicity, the same mapping is used on all processes so that the slots
    // trivially match across processes
    slotOffset_ = this->context_->nextSlot(
        2 * this->contextSize_ * (this->contextSize_ - 1));

    size_t bitmask = 1;
    size_t stepChunkSize = chunkSize_ << (steps_ - 1);
    size_t stepChunkBytes = stepChunkSize * sizeof(T);
    size_t sendOffset = 0;
    size_t recvOffset = 0;
    size_t bufferOffset = 0; // offset into recvBuf_
    for (int i = 0; i < stepsWithinBlock_; i++) {
      const int destRank = (this->context_->rank) ^ bitmask;
      auto& pair = this->context_->getPair(destRank);
      sendOffsets_[i] = sendOffset + ((destRank & bitmask) ? stepChunkSize : 0);
      recvOffsets_[i] =
          recvOffset + ((this->context_->rank & bitmask) ? stepChunkSize : 0);
      if (sendOffsets_[i] < count_) {
        // specifies number of elements to send in each step
        if (sendOffsets_[i] + stepChunkSize > count_) {
          sendCounts_[i] = count_ - sendOffsets_[i];
        } else {
          sendCounts_[i] = stepChunkSize;
        }
      }
      int myRank = this->context_->rank;
      auto slot = slotOffset_ +
          2 * (std::min(myRank, destRank) * this->contextSize_ +
               std::max(myRank, destRank));
      sendDataBufs_.push_back(pair->createSendBuffer(slot, ptrs_[0], bytes_));
      if (recvOffsets_[i] < count_) {
        // specifies number of elements received in each step
        if (recvOffsets_[i] + stepChunkSize > count_) {
          recvCounts_[i] = count_ - recvOffsets_[i];
        } else {
          recvCounts_[i] = stepChunkSize;
        }
      }
      recvDataBufs_.push_back(
          pair->createRecvBuffer(
              slot, &recvBuf_[bufferOffset], stepChunkBytes));
      bufferOffset += stepChunkSize;
      if (this->context_->rank & bitmask) {
        sendOffset += stepChunkSize;
        recvOffset += stepChunkSize;
      }
      bitmask <<= 1;
      stepChunkSize >>= 1;
      stepChunkBytes >>= 1;

      ++slot;
      sendNotificationBufs_.push_back(
          pair->createSendBuffer(slot, &dummy_, sizeof(dummy_)));
      recvNotificationBufs_.push_back(
          pair->createRecvBuffer(slot, &dummy_, sizeof(dummy_)));
    }

    if (nextSmallerBlockSize_ != 0) {
      const auto offsetToSmallerBlock =
          offsetToMyBinaryBlock_ + myBinaryBlockSize_;
      const int destRank =
          offsetToSmallerBlock + rankInBinaryBlock_ % nextSmallerBlockSize_;
      auto& destPair = this->context_->getPair(destRank);
      const auto myRank = this->context_->rank;
      const auto slot = slotOffset_ +
          2 * (std::min(myRank, destRank) * this->contextSize_ +
               std::max(myRank, destRank));
      smallerBlockSendDataBuf_ = destPair->createSendBuffer(
          slot, ptrs_[0], bytes_);
      const auto itemCount = recvCounts_[stepsWithinBlock_ - 1];
      if (itemCount > 0) {
        smallerBlockRecvDataBuf_ = destPair->createRecvBuffer(
            slot, &recvBuf_[bufferOffset], itemCount * sizeof(T));
      }
    }
    if (nextLargerBlockSize_ != 0) {
      // Due to the design decision of sending large messages to nearby ranks,
      // after the reduce-scatter the reduced chunks end up in an order
      // according to the reversed bit pattern of each proc's rank within the
      // block. So, instead of ranks 0, 1, 2, ... 7 having blocks A, B, C, D, E,
      // F, G, H etc. what you get is A, E, C, G, B, F, D, H. Taking this
      // example further, if there is also a smaller binary block of size 2
      // (with the reduced blocks A - D, E - H), rank 0 within the smaller block
      // will need to send chunks of its buffer to ranks 0, 4, 2, 6 within the
      // larger block (in that order) and rank 1 will send to 1, 5, 3, 7. Within
      // the reversed bit patterns, this communication is actually 0 to [0, 1,
      // 2, 3] and 1 to [4, 5, 6, 7].
      const auto offsetToLargerBlock =
          offsetToMyBinaryBlock_ - nextLargerBlockSize_;
      const auto numSendsAndReceivesToLargerBlock =
          nextLargerBlockSize_ / myBinaryBlockSize_;
      const auto totalItemsToSend =
          stepsWithinBlock_ > 0 ? recvCounts_[stepsWithinBlock_ - 1] : count_;
      sendCountToLargerBlock_ = stepChunkSize >>
          (static_cast<size_t>(log2(numSendsAndReceivesToLargerBlock)) - 1);
      auto srcOrdinal =
          reverseLastNBits(rankInBinaryBlock_, log2(myBinaryBlockSize_));
      auto destOrdinal = srcOrdinal * numSendsAndReceivesToLargerBlock;
      for (int i = 0; i < numSendsAndReceivesToLargerBlock; i++) {
        const int destRank = offsetToLargerBlock +
            reverseLastNBits(destOrdinal, log2(nextLargerBlockSize_));
        auto& destPair = this->context_->getPair(destRank);
        const auto myRank = this->context_->rank;
        const auto slot = slotOffset_ +
            2 * (std::min(myRank, destRank) * this->contextSize_ +
                 std::max(myRank, destRank));
        largerBlockSendDataBufs_.push_back(
            destPair->createSendBuffer(slot, ptrs[0], bytes_));
        if (sendCountToLargerBlock_ * i < totalItemsToSend) {
          const auto toSend = std::min(
              sendCountToLargerBlock_,
              totalItemsToSend - sendCountToLargerBlock_ * i);
          largerBlockRecvDataBufs_.push_back(
              destPair->createRecvBuffer(
                  slot, &recvBuf_[bufferOffset], toSend * sizeof(T)));
          bufferOffset += toSend;
        }
        destOrdinal++;
      }
    }
  }

  void run() {
    size_t bufferOffset = 0;
    size_t numItems =
        stepsWithinBlock_ > 0 ? chunkSize_ << (steps_ - 1) : count_;

    for (int i = 1; i < ptrs_.size(); i++) {
      fn_->call(ptrs_[0], ptrs_[i], count_);
    }

    // Reduce-scatter
    for (int i = 0; i < stepsWithinBlock_; i++) {
      if (sendOffsets_[i] < count_) {
        sendDataBufs_[i]->send(
            sendOffsets_[i] * sizeof(T), sendCounts_[i] * sizeof(T));
      }
      if (recvOffsets_[i] < count_) {
        recvDataBufs_[i]->waitRecv();
        fn_->call(
            &ptrs_[0][recvOffsets_[i]],
            &recvBuf_[bufferOffset],
            recvCounts_[i]);
      }
      bufferOffset += numItems;
      sendNotificationBufs_[i]->send();
      numItems >>= 1;
    }

    // Communication across binary blocks for non-power-of-two number of
    // processes

    // receive from smaller block
    // data sizes same as in the last step of intrablock reduce-scatter above
    if (nextSmallerBlockSize_ != 0 && smallerBlockRecvDataBuf_ != nullptr) {
      smallerBlockRecvDataBuf_->waitRecv();
      fn_->call(
          &ptrs_[0][recvOffsets_[stepsWithinBlock_ - 1]],
          &recvBuf_[bufferOffset],
          recvCounts_[stepsWithinBlock_ - 1]);
    }

    const auto totalItemsToSend =
        stepsWithinBlock_ > 0 ? recvCounts_[stepsWithinBlock_ - 1] : count_;
    if (nextLargerBlockSize_ != 0 && totalItemsToSend != 0) {
      // scatter to larger block
      const auto offset =
          stepsWithinBlock_ > 0 ? recvOffsets_[stepsWithinBlock_ - 1] : 0;
      const auto numSendsAndReceivesToLargerBlock =
          nextLargerBlockSize_ / myBinaryBlockSize_;
      for (int i = 0; i < numSendsAndReceivesToLargerBlock; i++) {
        if (sendCountToLargerBlock_ * i < totalItemsToSend) {
          largerBlockSendDataBufs_[i]->send(
              (offset + i * sendCountToLargerBlock_) * sizeof(T),
              std::min(
                  sendCountToLargerBlock_,
                  totalItemsToSend - sendCountToLargerBlock_ * i) *
                  sizeof(T));
        }
      }
      // no notification is needed because the forward and backward messages
      // across blocks are serialized in relation to each other

      // receive from larger blocks
      for (int i = 0; i < numSendsAndReceivesToLargerBlock; i++) {
        if (sendCountToLargerBlock_ * i < totalItemsToSend) {
          largerBlockRecvDataBufs_[i]->waitRecv();
        }
      }
      memcpy(
          &ptrs_[0][offset],
          &recvBuf_[bufferOffset],
          totalItemsToSend * sizeof(T));
    }

    // Send to smaller block (technically the beginning of allgather)
    bool sentToSmallerBlock = false;
    if (nextSmallerBlockSize_ != 0) {
      if (recvOffsets_[stepsWithinBlock_ - 1] < count_) {
        sentToSmallerBlock = true;
        smallerBlockSendDataBuf_->send(
            recvOffsets_[stepsWithinBlock_ - 1] * sizeof(T),
            recvCounts_[stepsWithinBlock_ - 1] * sizeof(T));
      }
    }

    // Allgather
    numItems = chunkSize_ << (steps_ - stepsWithinBlock_);
    for (int i = stepsWithinBlock_ - 1; i >= 0; i--) {
      // verify that destination rank has received and processed this rank's
      // message during the reduce-scatter phase
      recvNotificationBufs_[i]->waitRecv();
      if (recvOffsets_[i] < count_) {
        sendDataBufs_[i]->send(
            recvOffsets_[i] * sizeof(T), recvCounts_[i] * sizeof(T));
      }
      bufferOffset -= numItems;
      if (sendOffsets_[i] < count_) {
        recvDataBufs_[i]->waitRecv();
        memcpy(
            &ptrs_[0][sendOffsets_[i]],
            &recvBuf_[bufferOffset],
            sendCounts_[i] * sizeof(T));
      }
      numItems <<= 1;

      // Send notification to the pair we just received from that
      // we're done dealing with the receive buffer.
      sendNotificationBufs_[i]->send();
    }

    // Broadcast ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      memcpy(ptrs_[i], ptrs_[0], bytes_);
    }

    // Wait for notifications from our peers within the block to make
    // sure we can send data immediately without risking overwriting
    // data in its receive buffer before it consumed that data.
    for (int i = stepsWithinBlock_ - 1; i >= 0; i--) {
      recvNotificationBufs_[i]->waitRecv();
    }

    // We have to be sure the send to the smaller block (if any) has
    // completed before returning. If we don't, the buffer contents may
    // be modified by our caller.
    if (sentToSmallerBlock) {
      smallerBlockSendDataBuf_->waitSend();
    }
  }

 protected:
  std::vector<T*> ptrs_;
  const int count_;
  const int bytes_;
  const size_t steps_;
  const size_t chunks_;
  const size_t chunkSize_;
  const size_t chunkBytes_;
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

  std::vector<std::unique_ptr<transport::Buffer>> sendDataBufs_;
  std::vector<std::unique_ptr<transport::Buffer>> recvDataBufs_;

  std::unique_ptr<transport::Buffer> smallerBlockSendDataBuf_;
  std::unique_ptr<transport::Buffer> smallerBlockRecvDataBuf_;

  std::vector<std::unique_ptr<transport::Buffer>> largerBlockSendDataBufs_;
  std::vector<std::unique_ptr<transport::Buffer>> largerBlockRecvDataBufs_;

  std::vector<size_t> sendCounts_;
  std::vector<size_t> recvCounts_;
  size_t sendCountToLargerBlock_;

  int dummy_;
  std::vector<std::unique_ptr<transport::Buffer>> sendNotificationBufs_;
  std::vector<std::unique_ptr<transport::Buffer>> recvNotificationBufs_;

  // for non-power-of-two number of processes, partition the processes into
  // binary blocks and keep track of which block each process is in, as well as
  // the adjoining larger and smaller blocks (with which communication will be
  // required)
  uint32_t offsetToMyBinaryBlock_;
  uint32_t myBinaryBlockSize_;
  uint32_t stepsWithinBlock_;
  uint32_t rankInBinaryBlock_;
  uint32_t nextSmallerBlockSize_;
  uint32_t nextLargerBlockSize_;

  int slotOffset_;
};

} // namespace gloo
