/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <map>
#include <mutex>

#include <infiniband/verbs.h>

#include "gloo/transport/buffer.h"
#include "gloo/transport/ibverbs/device.h"
#include "gloo/transport/ibverbs/pair.h"

namespace gloo {
namespace transport {
namespace ibverbs {

class Buffer : public ::gloo::transport::Buffer {
 public:
  virtual ~Buffer();

  virtual void send(size_t offset, size_t length, size_t roffset = 0) override;

  virtual void waitRecv() override;
  virtual void waitSend() override;

  void handleCompletion(struct ibv_wc* wc);

  void signalError(const std::exception_ptr& ex);
  void checkErrorState();

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
  std::atomic<int> sendPending_;

  std::exception_ptr ex_;

  friend class Pair;
};

} // namespace ibverbs
} // namespace transport
} // namespace gloo
