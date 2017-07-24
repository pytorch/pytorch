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
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <iostream>

#include "gloo/common/error.h"
#include "gloo/common/logging.h"

namespace gloo {
namespace transport {
namespace tcp {

Buffer::Buffer(Pair* pair, int slot, void* ptr, size_t size)
    : ::gloo::transport::Buffer(slot, ptr, size),
      pair_(pair),
      recvCompletions_(0),
      sendCompletions_(0),
      sendPending_(0),
      ex_(nullptr) {}

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
  if (pair_->isSync()) {
    // We can assume a single pair is never used by more than one
    // thread, so there is no need to acquire the mutex here.
    while (recvCompletions_ == 0) {
      pair_->recv();
    }
    recvCompletions_--;
  } else {
    // The device thread will signal completion. If the completion
    // hasn't arrived yet, wait until it does or read times out.
    auto timeout = pair_->getTimeout();
    auto pred = [&]{
      checkErrorState();
      return recvCompletions_ > 0;
    };
    std::unique_lock<std::mutex> lock(m_);
    if (timeout == kNoTimeout) {
      // No timeout set. Wait for read to complete.
      recvCv_.wait(lock, pred);
    } else {
      auto done = recvCv_.wait_for(lock, timeout, pred);
      if (!done) {
        // Release the mutex before calling into the pair to avoid deadlock.
        // Calling signalIoFailureExternal() will throw, so no need to
        // reacquire.
        lock.unlock();
        pair_->signalIoFailureExternal(
            GLOO_ERROR_MSG("Read timeout ", pair_->peer().str()));
        GLOO_ENFORCE(false, "Unexpected code path");
      }
    }
    recvCompletions_--;
  }
}

void Buffer::handleSendCompletion() {
  std::lock_guard<std::mutex> lock(m_);
  sendCompletions_++;
  sendPending_--;
  sendCv_.notify_one();
}

void Buffer::waitSend() {
  if (pair_->isSync()) {
    // The send operation must flush all data to the underlying socket
    // and then call handleSendCompletion. Therefore, the number of
    // send completions must always be positive when calling waitSend.
    GLOO_ENFORCE_GE(1, sendCompletions_);
    sendCompletions_--;
  } else {
    // The device thread will signal completion. If the completion
    // hasn't arrived yet, wait until it does or write times out.
    auto timeout = pair_->getTimeout();
    auto pred = [&]{
      checkErrorState();
      return sendCompletions_ > 0;
    };
    std::unique_lock<std::mutex> lock(m_);
    if (sendCompletions_ == 0) {
      GLOO_ENFORCE_GT(sendPending_, 0, "No send to wait for");
      if (timeout == kNoTimeout) {
        // No timeout set. Wait for write to complete.
        sendCv_.wait(lock, pred);
      } else {
        auto done = sendCv_.wait_for(lock, timeout, pred);
        if (!done) {
          // Release the mutex before calling into the pair to avoid
          // deadlock. Calling signalIoFailureExternal() will throw,
          // so no need to reacquire.
          lock.unlock();
          pair_->signalIoFailureExternal(
              GLOO_ERROR_MSG("Write timeout ", pair_->peer().str()));
          GLOO_ENFORCE(false, "Unexpected code path");
        }
      }
    }
    sendCompletions_--;
  }
}

void Buffer::send(size_t offset, size_t length, size_t roffset) {
  Op op;

  // Can't assert on roffset, since we don't know the size of
  // the remote buffer. Refactor of initialization code needed
  // to support this.
  GLOO_ENFORCE_LE(offset + length, size_);

  if (debug_) {
    std::cout << "[" << getpid() << ": " << syscall(__NR_gettid) << "] ";
    std::cout << "send " << length << " bytes";
    std::cout << " to " << pair_->peer().str();
    std::cout << std::endl;
  }

  memset(&op, 0, sizeof(op));

  op.preamble_.opcode_ = 0;
  op.preamble_.slot_ = slot_;
  op.preamble_.offset_ = offset;
  op.preamble_.length_ = length;
  op.preamble_.roffset_ = roffset;
  op.buf_ = this;

  // Increment number of sends in flight
  sendPending_++;

  // Pass to pair
  pair_->send(op);
}

void Buffer::signalError(const std::exception_ptr& ex) {
  std::lock_guard<std::mutex> lock(m_);
  ex_ = ex;
  recvCv_.notify_all();
  sendCv_.notify_all();
}

void Buffer::checkErrorState() {
  if (ex_ != nullptr) {
    std::rethrow_exception(ex_);
  }
}

} // namespace tcp
} // namespace transport
} // namespace gloo
