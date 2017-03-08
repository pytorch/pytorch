/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/barrier.h"

namespace gloo {

class BarrierAllToAll : public Barrier {
 public:
  explicit BarrierAllToAll(const std::shared_ptr<Context>& context)
      : Barrier(context) {
    // Create send/recv buffers for every peer
    auto slot = this->context_->nextSlot();
    for (auto i = 0; i < this->contextSize_; i++) {
      // Skip self
      if (i == this->contextRank_) {
        continue;
      }

      auto& pair = this->getPair(i);
      auto sdata = std::unique_ptr<int>(new int);
      auto sbuf = pair->createSendBuffer(slot, sdata.get(), sizeof(int));
      sendBuffersData_.push_back(std::move(sdata));
      sendBuffers_.push_back(std::move(sbuf));
      auto rdata = std::unique_ptr<int>(new int);
      auto rbuf = pair->createRecvBuffer(slot, rdata.get(), sizeof(int));
      recvBuffersData_.push_back(std::move(rdata));
      recvBuffers_.push_back(std::move(rbuf));
    }
  }

  void run() {
    // Notify peers
    for (auto& buffer : sendBuffers_) {
      buffer->send();
    }
    // Wait for notification from peers
    for (auto& buffer : recvBuffers_) {
      buffer->waitRecv();
    }
  }

 protected:
  std::vector<std::unique_ptr<int>> sendBuffersData_;
  std::vector<std::unique_ptr<transport::Buffer>> sendBuffers_;
  std::vector<std::unique_ptr<int>> recvBuffersData_;
  std::vector<std::unique_ptr<transport::Buffer>> recvBuffers_;
};

} // namespace gloo
