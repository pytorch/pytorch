/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/ibverbs/buffer.h"

#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

#include "gloo/common/error.h"
#include "gloo/common/logging.h"

namespace gloo {
namespace transport {
namespace ibverbs {

Buffer::Buffer(Pair* pair, int slot, void* ptr, size_t size)
    : ::gloo::transport::Buffer(slot, ptr, size),
      pair_(pair),
      recvCompletions_(0),
      sendCompletions_(0),
      sendPending_(0),
      ex_(nullptr) {
  mr_ = ibv_reg_mr(
      pair_->dev_->pd_,
      ptr_,
      size_,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

  // Provide hint if the error is EFAULT and nv_peer_mem is not loaded
  if (mr_ == nullptr && errno == EFAULT) {
    if (!pair->dev_->hasNvPeerMem_) {
      GLOO_ENFORCE(
        mr_ != nullptr,
        "ibv_reg_mr: ",
        strerror(errno),
        " (kernel module 'nv_peer_mem' not loaded;"
        " did you specify a pointer to GPU memory?)");
    }
  }

  GLOO_ENFORCE(mr_ != nullptr, "ibv_reg_mr: ", strerror(errno));
}

Buffer::~Buffer() {
  ibv_dereg_mr(mr_);
}

// Wait for a receive operation to finish.
void Buffer::waitRecv() {
  // If the pair is in synchronous mode, the current thread is
  // responsible for polling for work completions.
  // Since a single pair potentially serves multiple buffers, a
  // completion may be intended for another buffer.
  auto timeout = pair_->getTimeout();
  if (pair_->sync_) {
    auto start = std::chrono::steady_clock::now();
    // We can assume a single pair is never used by more than one
    // thread, so there is no need to acquire the mutex here.
    while (recvCompletions_ == 0) {
      pair_->pollCompletions();
      if (timeout != kNoTimeout &&
          (std::chrono::steady_clock::now() - start) >= timeout) {
        pair_->signalIoFailure(
          GLOO_ERROR_MSG("Read timeout ", pair_->peer().str()));
        GLOO_ENFORCE(false, "Unexpected code path");
      }
    }
    recvCompletions_--;
  } else {
    // The device thread will signal completion. If the completion
    // hasn't arrived yet, wait until it does.
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
        // Calling signalIoFailure() will throw, so no need to
        // reacquire.
        lock.unlock();
        pair_->signalIoFailure(
          GLOO_ERROR_MSG("Read timeout ", pair_->peer().str()));
        GLOO_ENFORCE(false, "Unexpected code path");
      }
    }
    recvCompletions_--;
  }
}

// Wait for the previous send operation to finish.
void Buffer::waitSend() {
  // If the pair is in synchronous mode, the current thread is
  // responsible for polling for work completions.
  auto timeout = pair_->getTimeout();
  if (pair_->sync_) {
    // We can assume a single pair is never used by more than one
    // thread, so there is no need to acquire the mutex here.
    if (sendCompletions_ == 0) {
      GLOO_ENFORCE_GT(sendPending_, 0, "No send to wait for");
      auto start = std::chrono::steady_clock::now();
      // We can assume a single pair is never used by more than one
      // thread, so there is no need to acquire the mutex here.
      while (sendCompletions_ == 0) {
        pair_->pollCompletions();
        if (timeout != kNoTimeout &&
            (std::chrono::steady_clock::now() - start) >= timeout) {
          pair_->signalIoFailure(
            GLOO_ERROR_MSG("Send timeout ", pair_->peer().str()));
          GLOO_ENFORCE(false, "Unexpected code path");
        }
      }
    }
    sendCompletions_--;
  } else {
    // The device thread will signal completion. If the completion
    // hasn't arrived yet, wait until it does.
    std::unique_lock<std::mutex> lock(m_);
    checkErrorState();
    if (sendCompletions_ == 0) {
      GLOO_ENFORCE_GT(sendPending_, 0, "No send to wait for");
      auto pred = [&]{
        checkErrorState();
        return sendCompletions_ > 0;
      };
      if (timeout == kNoTimeout) {
        // No timeout set. Wait for read to complete.
        sendCv_.wait(lock, pred);
      } else {
        auto done = sendCv_.wait_for(lock, timeout, pred);
        if (!done) {
          // Release the mutex before calling into the pair to avoid deadlock.
          // Calling signalIoFailure() will throw, so no need to
          // reacquire.
          lock.unlock();
          pair_->signalIoFailure(
            GLOO_ERROR_MSG("Send timeout ", pair_->peer().str()));
          GLOO_ENFORCE(false, "Unexpected code path");
        }
      }
    }
    sendCompletions_--;
  }
}

void Buffer::send(size_t offset, size_t length, size_t roffset) {
  int rv;

  // Can't assert on roffset, since we don't know the size of
  // the remote buffer. Refactor of initialization code needed
  // to support this.
  GLOO_ENFORCE_LE(offset + length, size_);

  {
    std::unique_lock<std::mutex> lock(m_);
    checkErrorState();
  }

  if (debug_) {
    std::cout << "[" << getpid() << "] ";
    std::cout << "send " << length << " bytes";
    std::cout << std::endl;
  }

  // Increment number of sends in flight
  sendPending_++;

  pair_->send(this, offset, length, roffset);
}

void Buffer::handleCompletion(struct ibv_wc* wc) {
  if (wc->opcode & IBV_WC_RECV) {
    if (debug_) {
      std::cout << "[" << getpid() << "] ";
      std::cout << "recv " << wc->byte_len << " bytes";
      std::cout << std::endl;
    }
    std::unique_lock<std::mutex> lock(m_);
    recvCompletions_++;
    recvCv_.notify_one();
  } else if (wc->opcode == IBV_WC_RDMA_WRITE) {
    if (debug_) {
      std::cout << "[" << getpid() << "] ";
      std::cout << "send complete";
      std::cout << std::endl;
    }
    std::unique_lock<std::mutex> lock(m_);
    sendCompletions_++;
    sendPending_--;
    sendCv_.notify_one();
  } else {
    GLOO_ENFORCE(false, "Unexpected completion (opcode: ", wc->opcode, ")");
  }
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

} // namespace ibverbs
} // namespace transport
} // namespace gloo
