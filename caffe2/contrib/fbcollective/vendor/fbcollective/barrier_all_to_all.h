#pragma once

#include "fbcollective/barrier.h"

namespace fbcollective {

class BarrierAllToAll : public Barrier {
 public:
  explicit BarrierAllToAll(const std::shared_ptr<Context>& context)
      : Barrier(context) {
    // Create send/recv buffers for every peer
    for (int i = 0; i < this->contextSize_; i++) {
      // Skip self
      if (i == this->contextRank_) {
        continue;
      }

      auto& pair = this->getPair(i);

      auto sendBufData = std::unique_ptr<int>(new int);
      auto sendBuf = pair->createSendBuffer(0, sendBufData.get(), sizeof(int));
      sendBuffersData_.push_back(std::move(sendBufData));
      sendBuffers_.push_back(std::move(sendBuf));

      auto recvBufData = std::unique_ptr<int>(new int);
      auto recvBuf = pair->createRecvBuffer(0, recvBufData.get(), sizeof(int));
      recvBuffersData_.push_back(std::move(recvBufData));
      recvBuffers_.push_back(std::move(recvBuf));
    }
  }

  void Run() {
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
  std::vector<std::unique_ptr<transport::Buffer>> sendBuffers_;
  std::vector<std::unique_ptr<int>> sendBuffersData_;
  std::vector<std::unique_ptr<transport::Buffer>> recvBuffers_;
  std::vector<std::unique_ptr<int>> recvBuffersData_;
};

} // namespace fbcollective
