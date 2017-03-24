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
// The caller needs to pass the receive buffers as a vector of memory pointers
// (outPtrs) of size equal to the context size where the send buffers of the
// process with rank = k will be written to outPtrs[k] contiguously.
template <typename T>
class AllgatherRing : public Algorithm {
 public:
  AllgatherRing(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& inPtrs,
      std::vector<T*> outPtrs,
      int count)
      : Algorithm(context),
        inPtrs_(inPtrs),
        outPtrs_(outPtrs),
        count_(count),
        bytes_(count * sizeof(T)),
        leftPair_(this->getLeftPair()),
        rightPair_(this->getRightPair()) {
    inbox_ = static_cast<T*>(malloc(bytes_));
    outbox_ = static_cast<T*>(malloc(bytes_));

    auto slot = this->context_->nextSlot();

    sendDataBuf_ = rightPair_->createSendBuffer(slot, outbox_, bytes_);
    recvDataBuf_ = leftPair_->createRecvBuffer(slot, inbox_, bytes_);

    auto notificationSlot = this->context_->nextSlot();
    sendNotificationBuf_ =
        leftPair_->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));
    recvNotificationBuf_ =
        rightPair_->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));
  }

  virtual ~AllgatherRing() {
    if (inbox_ != nullptr) {
      free(inbox_);
    }

    if (outbox_ != nullptr) {
      free(outbox_);
    }
  }

  void run() {
    const int rank = this->contextRank_;
    const int numRounds = this->contextSize_ - 1;

    // Copy local buffer.
    for (int i = 0; i < inPtrs_.size(); i++) {
      memcpy(outPtrs_[rank] + i * count_, inPtrs_[i], bytes_);
    }

    // We send input buffers in order.
    for (int i = 0; i < inPtrs_.size(); i++) {
      memcpy(outbox_, inPtrs_[i], bytes_);
      for (int round = 0; round < numRounds; round++) {
        // Send data in the outbox buffer and wait to receive from left.
        sendDataBuf_->send();
        recvDataBuf_->waitRecv();

        // Nodes receive data from the left node in every round and forward it
        // to the right node.
        int inRank = (numRounds - round + rank) % this->contextSize_;

        // Copy received buffer inplace.
        memcpy(outPtrs_[inRank] + i * count_, inbox_, bytes_);

        // Forward received buffer to the right.
        if (round < (numRounds - 1)) {
          memcpy(outbox_, inbox_, bytes_);
        }

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
  std::vector<T*> outPtrs_;
  const int count_;
  const int bytes_;

  std::unique_ptr<transport::Pair>& leftPair_;
  std::unique_ptr<transport::Pair>& rightPair_;

  T* inbox_;
  T* outbox_;

  std::unique_ptr<transport::Buffer> sendDataBuf_;
  std::unique_ptr<transport::Buffer> recvDataBuf_;

  int dummy_;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;
};

}  // namespace gloo
