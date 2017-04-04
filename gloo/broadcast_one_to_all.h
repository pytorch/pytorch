/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cstring>
#include <vector>

#include "gloo/algorithm.h"
#include "gloo/common/logging.h"

namespace gloo {

template <typename T>
class BroadcastOneToAll : public Algorithm {
 public:
  BroadcastOneToAll(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      int count,
      int rootRank = 0,
      int rootPointerRank = 0)
      : Algorithm(context),
        ptrs_(ptrs),
        count_(count),
        bytes_(count * sizeof(T)),
        rootRank_(rootRank),
        rootPointerRank_(rootPointerRank) {
    GLOO_ENFORCE_GE(rootRank_, 0);
    GLOO_ENFORCE_LT(rootRank_, contextSize_);
    GLOO_ENFORCE_GE(rootPointerRank_, 0);
    GLOO_ENFORCE_LT(rootPointerRank_, ptrs_.size());

    // Setup pairs/buffers for sender/receivers
    if (contextSize_ > 1) {
      auto ptr = ptrs_[rootPointerRank_];
      auto slot = context_->nextSlot();
      if (contextRank_ == rootRank_) {
        for (auto i = 0; i < contextSize_; i++) {
          if (i == contextRank_) {
            continue;
          }

          auto& pair = context_->getPair(i);
          sendDataBuffers_.push_back(
            pair->createSendBuffer(slot, ptr, bytes_));
        }
      } else {
        auto& rootPair = context_->getPair(rootRank_);
        recvDataBuffer_ = rootPair->createRecvBuffer(slot, ptr, bytes_);
      }
    }
  }

  void run() {
    if (contextSize_ == 1) {
      broadcastLocally();
      return;
    }

    if (contextRank_ == rootRank_) {
      // Fire off all send operations concurrently
      for (auto& buf : sendDataBuffers_) {
        buf->send();
      }
      // Broadcast locally while sends are happening
      broadcastLocally();
      // Wait for all send operations to complete
      for (auto& buf : sendDataBuffers_) {
        buf->waitSend();
      }
    } else {
      recvDataBuffer_->waitRecv();
      // Broadcast locally after receiving from root
      broadcastLocally();
    }
  }

 protected:
  // Broadcast from root pointer to other pointers
  void broadcastLocally() {
    for (auto i = 0; i < ptrs_.size(); i++) {
      if (i == rootPointerRank_) {
        continue;
      }

      memcpy(ptrs_[i], ptrs_[rootPointerRank_], bytes_);
    }
  }

  std::vector<T*> ptrs_;
  const int count_;
  const int bytes_;
  const int rootRank_;
  const int rootPointerRank_;

  // For the sender (root)
  std::vector<std::unique_ptr<transport::Buffer>> sendDataBuffers_;

  // For all receivers
  std::unique_ptr<transport::Buffer> recvDataBuffer_;
};

} // namespace gloo
