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
  if (pair_->sync_) {
    // We can assume a single pair is never used by more than one
    // thread, so there is no need to acquire the mutex here.
    while (recvCompletions_ == 0) {
      pair_->pollCompletions();
    }
    recvCompletions_--;
  } else {
    // The device thread will signal completion. If the completion
    // hasn't arrived yet, wait until it does.
    std::unique_lock<std::mutex> lock(m_);
    checkErrorState();
    while (recvCompletions_ == 0) {
      recvCv_.wait(lock);
      checkErrorState();
    }
    recvCompletions_--;
  }
}

// Wait for the previous send operation to finish.
void Buffer::waitSend() {
  // If the pair is in synchronous mode, the current thread is
  // responsible for polling for work completions.
  if (pair_->sync_) {
    // We can assume a single pair is never used by more than one
    // thread, so there is no need to acquire the mutex here.
    if (sendCompletions_ == 0) {
      GLOO_ENFORCE_GT(sendPending_, 0, "No send to wait for");
      while (sendCompletions_ == 0) {
        pair_->pollCompletions();
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
      while (sendCompletions_ == 0) {
        sendCv_.wait(lock);
        checkErrorState();
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

  struct ibv_sge list;
  list.addr = (uint64_t)ptr_ + offset;
  list.length = length;
  list.lkey = mr_->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = slot_;
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = slot_;

  const struct ibv_mr* peer = pair_->getMemoryRegion(slot_);
  GLOO_ENFORCE_NE(peer, (const struct ibv_mr*)nullptr);
  wr.wr.rdma.remote_addr = (uint64_t)peer->addr + roffset;
  wr.wr.rdma.rkey = peer->rkey;

  struct ibv_send_wr* bad_wr;
  rv = ibv_post_send(pair_->qp_, &wr, &bad_wr);
  if (rv != 0) {
    pair_->signalIoFailure(GLOO_ERROR_MSG("ibv_post_send: ", rv));
  }

  // Increment number of sends in flight
  sendPending_++;
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
