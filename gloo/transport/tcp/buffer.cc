/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/tcp/buffer.h"

#include <string.h>

#include "gloo/common/logging.h"

namespace gloo {
namespace transport {
namespace tcp {

Buffer::Buffer(Pair* pair, int slot, void* ptr, size_t size)
    : ::gloo::transport::Buffer(slot, ptr, size),
      pair_(pair),
      recvCompletions_(0),
      sendCompletions_(0) {}

Buffer::~Buffer() {
  pair_->unregisterBuffer(this);
}

void Buffer::handleRecvCompletion() {
  std::lock_guard<std::mutex> lock(m_);
  recvCompletions_++;
  recvCv_.notify_one();
}

void Buffer::waitRecv() {
  // If the pair is in synchronous mode, the current thread is
  // responsible for doing reads.
  // Since a single pair potentially serves multiple buffers, a
  // read may be intended for another buffer.
  if (pair_->sync_) {
    // We can assume a single pair is never used by more than one
    // thread, so there is no need to acquire the mutex here.
    while (recvCompletions_ == 0) {
      pair_->recv();
    }
    recvCompletions_--;
  } else {
    // The device thread will signal completion. If the completion
    // hasn't arrived yet, wait until it does.
    std::unique_lock<std::mutex> lock(m_);
    while (recvCompletions_ == 0) {
      recvCv_.wait(lock);
    }
    recvCompletions_--;
  }
}

void Buffer::handleSendCompletion() {
  std::lock_guard<std::mutex> lock(m_);
  sendCompletions_++;
  sendCv_.notify_one();
}

void Buffer::waitSend() {
  if (pair_->sync_) {
    // The send operation must flush all data to the underlying socket
    // and then call handleSendCompletion. Therefore, the number of
    // send completions must always be positive when calling waitSend.
    GLOO_ENFORCE_GE(1, sendCompletions_);
    sendCompletions_--;
  } else {
    // The device thread will signal completion. If the completion
    // hasn't arrived yet, wait until it does.
    std::unique_lock<std::mutex> lock(m_);
    while (sendCompletions_ == 0) {
      sendCv_.wait(lock);
    }
    sendCompletions_--;
  }
}

void Buffer::send(size_t offset, size_t length) {
  Op op;

  memset(&op, 0, sizeof(op));

  op.preamble_.opcode_ = 0;
  op.preamble_.slot_ = slot_;
  op.preamble_.offset_ = offset;
  op.preamble_.length_ = length;
  op.buf_ = this;

  // Pass to pair
  pair_->send(op);
}

} // namespace tcp
} // namespace transport
} // namespace gloo
