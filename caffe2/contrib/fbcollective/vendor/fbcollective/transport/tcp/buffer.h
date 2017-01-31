#pragma once

#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>

#include "fbcollective/transport/buffer.h"
#include "fbcollective/transport/tcp/device.h"
#include "fbcollective/transport/tcp/pair.h"

namespace fbcollective {
namespace transport {
namespace tcp {

class Buffer : public ::fbcollective::transport::Buffer {
 public:
  virtual ~Buffer();

  void handleRecvCompletion();
  void handleSendCompletion();

  virtual void send(size_t offset, size_t length) override;

  virtual void waitRecv() override;
  virtual void waitSend() override;

 protected:
  // May only be constructed from helper function in pair.cc
  explicit Buffer(Pair* pair, int slot, void* ptr, size_t size);

  Pair* pair_;

  std::mutex m_;
  std::condition_variable recvCv_;
  std::condition_variable sendCv_;

  int recvCompletions_;
  int sendCompletions_;

  friend class Pair;
};

} // namespace tcp
} // namespace transport
} // namespace fbcollective
