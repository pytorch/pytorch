/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <stddef.h>
#include <string.h>

#include "gloo/algorithm.h"
#include "gloo/context.h"

namespace gloo {

// AllgatherRing is similar to MPI_Allgather where all processes receive the
// buffers (inPtrs) from all other processes.
// The caller needs to pass a preallocated receive buffer (outPtr) of size equal
// to the context size x the total size of the send buffers (inPtrs) where the
// send buffers of the process with rank = k will be written to
// outPtr[k * number of input buffers * count] consecutively.
template <typename T>
class AllgatherRing : public Algorithm {
 public:
  AllgatherRing(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& inPtrs,
      T* outPtr,
      int count)
      : Algorithm(context),
        inPtrs_(inPtrs),
        outPtr_(outPtr),
        count_(count),
        bytes_(count * sizeof(T)),
        inputStride_(count_ * inPtrs_.size()),
        leftPair_(this->getLeftPair()),
        rightPair_(this->getRightPair()) {
    auto slot = this->context_->nextSlot();

    sendDataBuf_ = rightPair_->createSendBuffer(
        slot, outPtr_, inPtrs_.size() * context_->size * bytes_);
    recvDataBuf_ = leftPair_->createRecvBuffer(
        slot, outPtr_, inPtrs_.size() * context_->size * bytes_);

    auto notificationSlot = this->context_->nextSlot();
    sendNotificationBuf_ =
        leftPair_->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));
    recvNotificationBuf_ =
        rightPair_->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));
  }

  virtual ~AllgatherRing() {}

  void run() {
    const int rank = this->contextRank_;
    const int numRounds = this->contextSize_ - 1;

    // Copy local buffers.
    for (int i = 0; i < inPtrs_.size(); i++) {
      memcpy(outPtr_ + rank * inputStride_ + i * count_, inPtrs_[i], bytes_);
    }

    // We send input buffers in order.
    for (int i = 0; i < inPtrs_.size(); i++) {
      // We start every iteration by sending local buffer.
      int inRank = rank;
      for (int round = 0; round < numRounds; round++) {
        const int sendOffset = inRank * inputStride_ + i * count_;
        sendDataBuf_->send(
            sendOffset * sizeof(T), bytes_, sendOffset * sizeof(T));
        recvDataBuf_->waitRecv();

        // Nodes receive data from the left node in every round and forward it
        // to the right node.
        inRank = (numRounds - round + rank) % this->contextSize_;

        // Send notification to node on the left that this node is ready for an
        // inbox write.
        sendNotificationBuf_->send();

        // Wait for notification from node on the right.
        recvNotificationBuf_->waitRecv();
      }
    }
  }

 private:
  const std::vector<T*> inPtrs_;
  T* outPtr_;
  const int count_;
  const int bytes_;
  const int inputStride_;

  std::unique_ptr<transport::Pair>& leftPair_;
  std::unique_ptr<transport::Pair>& rightPair_;

  std::unique_ptr<transport::Buffer> sendDataBuf_;
  std::unique_ptr<transport::Buffer> recvDataBuf_;

  int dummy_;

  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;
};

}  // namespace gloo
