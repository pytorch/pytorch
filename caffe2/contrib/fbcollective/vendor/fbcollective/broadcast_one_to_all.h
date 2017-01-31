#pragma once

#include <vector>

#include "fbcollective/broadcast.h"

namespace fbcollective {

template <typename T>
class BroadcastOneToAll : public Broadcast<T> {
 public:
  BroadcastOneToAll(
      const std::shared_ptr<Context>& context,
      T* dataPtr,
      int dataSize,
      int rootRank = 0)
      : Broadcast<T>(context, rootRank),
        dataPtr_(dataPtr),
        dataSize_(dataSize),
        dataSizeBytes_(dataSize * sizeof(T)) {
    if (this->contextRank_ == this->rootRank_) {
      for (int i = 0; i < this->contextSize_; i++) {
        if (i == this->contextRank_) {
          continue;
        }

        auto& pair = this->context_->getPair(i);
        sendDataBuffers_.push_back(
            pair->createSendBuffer(0, dataPtr_, dataSizeBytes_));
      }
    } else {
      auto& rootPair = this->context_->getPair(this->rootRank_);
      recvDataBuffer_ = rootPair->createRecvBuffer(0, dataPtr_, dataSizeBytes_);
    }
  }

  void Run() {
    if (this->contextRank_ == this->rootRank_) {
      // Fire off all send operations concurrently
      for (auto& buf : sendDataBuffers_) {
        buf->send();
      }
      // Wait for all send operations to complete
      for (auto& buf : sendDataBuffers_) {
        buf->waitSend();
      }
    } else {
      recvDataBuffer_->waitRecv();
    }
  }

 protected:
  T* dataPtr_;
  const int dataSize_;
  const int dataSizeBytes_;

  // For the sender (root)
  std::vector<std::unique_ptr<transport::Buffer>> sendDataBuffers_;

  // For all receivers
  std::unique_ptr<transport::Buffer> recvDataBuffer_;
};

} // namespace fbcollective
