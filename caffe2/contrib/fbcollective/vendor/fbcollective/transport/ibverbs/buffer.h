#pragma once

#include <condition_variable>
#include <map>
#include <mutex>

#include <infiniband/verbs.h>

#include "fbcollective/transport/buffer.h"
#include "fbcollective/transport/ibverbs/device.h"
#include "fbcollective/transport/ibverbs/pair.h"

namespace fbcollective {
namespace transport {
namespace ibverbs {

class Buffer : public ::fbcollective::transport::Buffer, public Handler {
 public:
  virtual ~Buffer();

  virtual void send(size_t offset, size_t length) override;

  virtual void waitRecv() override;
  virtual void waitSend() override;

  virtual void handleCompletion(struct ibv_wc* wc) override;

 protected:
  // May only be constructed from helper function in pair.cc
  Buffer(Pair* pair, int slot, void* ptr, size_t size);

  Pair* pair_;

  struct ibv_mr* mr_;

  std::mutex m_;
  std::condition_variable recvCv_;
  std::condition_variable sendCv_;

  int recvCompletions_;
  int sendCompletions_;

  friend class Pair;
};

} // namespace ibverbs
} // namespace transport
} // namespace fbcollective
