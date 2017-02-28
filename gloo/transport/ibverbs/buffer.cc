/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/ibverbs/buffer.h"

#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

#include "gloo/common/logging.h"

namespace gloo {
namespace transport {
namespace ibverbs {

Buffer::Buffer(Pair* pair, int slot, void* ptr, size_t size)
    : ::gloo::transport::Buffer(slot, ptr, size),
      pair_(pair),
      recvCompletions_(0),
      sendCompletions_(0) {
  mr_ = ibv_reg_mr(
      pair_->dev_->pd_,
      ptr_,
      size_,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
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
    while (recvCompletions_ == 0) {
      recvCv_.wait(lock);
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
    while (sendCompletions_ == 0) {
      pair_->pollCompletions();
    }
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
  int rv;

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
  wr.wr.rdma.remote_addr = (uint64_t)peer->addr;
  wr.wr.rdma.rkey = peer->rkey;

  struct ibv_send_wr* bad_wr;
  rv = ibv_post_send(pair_->qp_, &wr, &bad_wr);
  GLOO_ENFORCE_NE(rv, -1);
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
    sendCv_.notify_one();
  } else {
    GLOO_ENFORCE(false, "Unexpected completion (opcode: ", wc->opcode, ")");
  }
}

} // namespace ibverbs
} // namespace transport
} // namespace gloo
