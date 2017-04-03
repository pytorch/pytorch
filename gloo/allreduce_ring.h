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

template <typename T>
class AllreduceRing : public Algorithm {
 public:
  AllreduceRing(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const ReductionFunction<T>* fn = ReductionFunction<T>::sum)
      : Algorithm(context),
        ptrs_(ptrs),
        count_(count),
        bytes_(count_ * sizeof(T)),
        fn_(fn),
        leftPair_(this->getLeftPair()),
        rightPair_(this->getRightPair()) {
    inbox_ = static_cast<T*>(malloc(bytes_));
    outbox_ = static_cast<T*>(malloc(bytes_));

    auto slot = this->context_->nextSlot();

    // Buffer to send to (rank+1).
    sendDataBuf_ = rightPair_->createSendBuffer(slot, outbox_, bytes_);

    // Buffer that (rank-1) writes to.
    recvDataBuf_ = leftPair_->createRecvBuffer(slot, inbox_, bytes_);

    // Dummy buffers for localized barrier.
    // Before sending to the right, we only need to know that the node
    // on the right is done using the inbox that's about to be written
    // into. No need for a global barrier.
    auto notificationSlot = this->context_->nextSlot();
    sendNotificationBuf_ =
      leftPair_->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));
    recvNotificationBuf_ =
      rightPair_->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));
  }

  virtual ~AllreduceRing() {
    if (inbox_ != nullptr) {
      free(inbox_);
    }
    if (outbox_ != nullptr) {
      free(outbox_);
    }
  }

  void run() {
    // Reduce specified pointers into ptrs_[0]
    for (int i = 1; i < ptrs_.size(); i++) {
      fn_->call(ptrs_[0], ptrs_[i], count_);
    }

    // Intialize outbox with locally reduced values
    memcpy(outbox_, ptrs_[0], bytes_);

    int numRounds = this->contextSize_ - 1;
    for (int round = 0; round < numRounds; round++) {
      // Initiate write to inbox of node on the right
      sendDataBuf_->send();

      // Wait for inbox write from node on the left
      recvDataBuf_->waitRecv();

      // Reduce
      fn_->call(ptrs_[0], inbox_, count_);

      // Wait for outbox write to complete
      sendDataBuf_->waitSend();

      // Prepare for next round if necessary
      if (round < (numRounds - 1)) {
        memcpy(outbox_, inbox_, bytes_);
      }

      // Send notification to node on the left that
      // this node is ready for an inbox write.
      sendNotificationBuf_->send();

      // Wait for notification from node on the right
      recvNotificationBuf_->waitRecv();
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
  const ReductionFunction<T>* fn_;

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

} // namespace gloo
