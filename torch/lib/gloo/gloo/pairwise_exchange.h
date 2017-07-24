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

#include "gloo/algorithm.h"
#include "gloo/context.h"

namespace gloo {

class PairwiseExchange: public Algorithm {
 public:
  explicit PairwiseExchange(
      const std::shared_ptr<Context>& context,
      const int numBytes, const int numDestinations)
      : Algorithm(context),
        numDestinations_(numDestinations),
        bytesPerMsg_(numBytes / numDestinations_),
        sendBufferData_(new char[numBytes]),
        recvBufferData_(new char[numBytes]) {
    assert(this->contextSize_ % 2 == 0);
    assert(bytesPerMsg_ > 0);
    assert(
        numDestinations_ > 0 && numDestinations_ <= log2(this->contextSize_));
    // Processes communicate bidirectionally in pairs
    size_t bitmask = 1;
    for (int i = 0; i < numDestinations_; i++) {
      auto slot = this->context_->nextSlot();
      int destination =
        this->context_->rank ^ bitmask;
      const auto& pair = this->getPair(destination);
      sendBuffers_.push_back(pair->createSendBuffer(
          slot, &sendBufferData_.get()[i * bytesPerMsg_], bytesPerMsg_));
      recvBuffers_.push_back(pair->createRecvBuffer(
          slot, &recvBufferData_.get()[i * bytesPerMsg_], bytesPerMsg_));
      slot = this->context_->nextSlot();
      sendNotificationBufs_.push_back(
          pair->createSendBuffer(slot, &dummy_, sizeof(dummy_)));
      recvNotificationBufs_.push_back(
          pair->createRecvBuffer(slot, &dummy_, sizeof(dummy_)));
      bitmask <<= 1;
    }
  }

  void run() {
    for (int i = 0; i < numDestinations_; i++) {
      sendBuffers_[i]->send();
      recvBuffers_[i]->waitRecv();
      sendNotificationBufs_[i]->send();
      recvNotificationBufs_[i]->waitRecv();
    }
  }

 protected:
  const int numDestinations_;
  const int bytesPerMsg_;
  std::unique_ptr<char> sendBufferData_;
  std::unique_ptr<char> recvBufferData_;
  std::vector<std::unique_ptr<transport::Buffer>> sendBuffers_;
  std::vector<std::unique_ptr<transport::Buffer>> recvBuffers_;
  int dummy_;
  std::vector<std::unique_ptr<transport::Buffer>> sendNotificationBufs_;
  std::vector<std::unique_ptr<transport::Buffer>> recvNotificationBufs_;
};

} // namespace gloo
