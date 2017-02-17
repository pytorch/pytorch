/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/ibverbs/pair.h"
#include "gloo/transport/ibverbs/buffer.h"

#include <stdlib.h>
#include <string.h>

#include "gloo/common/logging.h"

namespace gloo {
namespace transport {
namespace ibverbs {

Pair::Pair(const std::shared_ptr<Device>& dev)
    : dev_(dev), peer_memory_regions_ready_(0) {
  int rv;

  memset(peer_memory_regions_.data(), 0, sizeof(peer_memory_regions_));
  memset(completion_handlers_.data(), 0, sizeof(completion_handlers_));

  // Create queue pair
  {
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(struct ibv_qp_init_attr));
    attr.send_cq = dev_->cq_;
    attr.recv_cq = dev_->cq_;
    attr.cap.max_send_wr = Pair::MAX_BUFFERS;
    attr.cap.max_recv_wr = Pair::MAX_BUFFERS;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.qp_type = IBV_QPT_RC;
    qp_ = ibv_create_qp(dev->pd_, &attr);
    GLOO_ENFORCE(qp_);
  }

  // Init queue pair
  {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = dev_->attr_.port;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    rv = ibv_modify_qp(
        qp_,
        &attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    GLOO_ENFORCE_NE(rv, -1);
  }

  // Populate local address.
  // The Packet Sequence Number field (PSN) is random which makes that
  // the remote end of this pair needs to have the contents of the
  // full address struct in order to connect, and vice versa.
  {
    struct ibv_port_attr attr;
    memset(&attr, 0, sizeof(struct ibv_port_attr));
    rv = ibv_query_port(dev_->context_, dev_->attr_.port, &attr);
    GLOO_ENFORCE_NE(rv, -1);
    rv = ibv_query_gid(
        dev_->context_,
        dev_->attr_.port,
        dev_->attr_.index,
        &self_.addr_.ibv_gid);
    GLOO_ENFORCE_NE(rv, -1);
    self_.addr_.lid = attr.lid;
    self_.addr_.qpn = qp_->qp_num;
    self_.addr_.psn = rand() & 0xffffff;
  }

  // Post receive requests for the remote memory regions.
  // The remote side of this pair will call the 'recv' function to
  // register a receive buffer. The memory region will be registered
  // and the identifier sent to this side of the pair.
  for (int i = 0; i < MAX_BUFFERS; ++i) {
    receiveMemoryRegion();
  }
}

Pair::~Pair() {
  int rv;

  rv = ibv_destroy_qp(qp_);
  GLOO_ENFORCE_NE(rv, -1);
}

const Address& Pair::address() const {
  return self_;
}

void Pair::connect(const std::vector<char>& bytes) {
  struct ibv_qp_attr attr;
  int rv;

  peer_ = Address(bytes);

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_1024;
  attr.dest_qp_num = peer_.addr_.qpn;
  attr.rq_psn = peer_.addr_.psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 20;
  attr.ah_attr.port_num = dev_->attr_.port;
  attr.ah_attr.is_global = 1;
  memcpy(&attr.ah_attr.grh.dgid, &peer_.addr_.ibv_gid, 16);
  attr.ah_attr.grh.hop_limit = 1;
  attr.ah_attr.grh.sgid_index = dev_->attr_.index;

  // Move to Ready To Receive (RTR) state
  rv = ibv_modify_qp(
      qp_,
      &attr,
      IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
          IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  GLOO_ENFORCE_NE(rv, -1);

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = self_.addr_.psn;
  attr.ah_attr.is_global = 1;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7; /* infinite */
  attr.max_rd_atomic = 1;

  // Move to Ready To Send (RTS) state
  rv = ibv_modify_qp(
      qp_,
      &attr,
      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  GLOO_ENFORCE_NE(rv, -1);
}

void Pair::setSync(bool /* sync */, bool /* busyPoll */) {
  GLOO_ENFORCE(false, "setSync not implemented");
}

void Pair::receiveMemoryRegion() {
  struct ibv_mr* mr = new struct ibv_mr;
  struct ibv_mr* init;
  int rv;

  // Keep list of ibv_mr's that the other side of this pair can write
  // into. They are written in a FIFO order so the handler can always
  // pop off the first entry upon receiving a write.
  tmp_memory_regions_.push_back(mr);

  // Map the memory region struct itself so the other side of this
  // pair can write into it.
  init = ibv_reg_mr(dev_->pd_, mr, sizeof(*mr), IBV_ACCESS_LOCAL_WRITE);
  mapped_recv_regions_.push_back(init);

  struct ibv_sge list;
  list.addr = (uint64_t)mr;
  list.length = sizeof(*mr);
  list.lkey = init->lkey;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)((Handler*)this);
  wr.sg_list = &list;
  wr.num_sge = 1;

  // The work request is serialized and sent to the driver so it
  // doesn't need to live beyond the ibv_post_recv call (and be kept
  // on the stack).
  struct ibv_recv_wr* bad_wr;
  rv = ibv_post_recv(qp_, &wr, &bad_wr);
  GLOO_ENFORCE_NE(rv, -1);
}

void Pair::sendMemoryRegion(Handler* h, struct ibv_mr* mr, int slot) {
  struct ibv_mr* init;
  int rv;

  // Register completion handler for this memory region.
  completion_handlers_[slot] = h;

  // First post receive work request to avoid racing with
  // a send to this region from the other side of this pair.
  postReceive();

  // Map the memory region struct itself so it can be sent to
  // the other side of this pair.
  init =
      ibv_reg_mr(dev_->pd_, mr, sizeof(struct ibv_mr), IBV_ACCESS_LOCAL_WRITE);
  mapped_send_regions_.push_back(init);

  struct ibv_sge list;
  list.addr = (uint64_t)mr;
  list.length = sizeof(struct ibv_mr);
  list.lkey = init->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)((Handler*)this);
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = slot;

  // The work request is serialized and sent to the driver so it
  // doesn't need to be valid after the ibv_post_send call.
  struct ibv_send_wr* bad_wr;
  rv = ibv_post_send(qp_, &wr, &bad_wr);
  GLOO_ENFORCE_NE(rv, -1);
}

const struct ibv_mr* Pair::getMemoryRegion(int slot) {
  if (peer_memory_regions_ready_.load() & (1 << slot)) {
    return peer_memory_regions_[slot];
  }

  std::unique_lock<std::mutex> lock(m_);
  while (peer_memory_regions_[slot] == nullptr) {
    cv_.wait(lock);
  }
  peer_memory_regions_ready_ &= (1 << slot);
  return peer_memory_regions_[slot];
}

void Pair::postReceive() {
  int rv;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)((Handler*)this);
  struct ibv_recv_wr* bad_wr;
  rv = ibv_post_recv(qp_, &wr, &bad_wr);
  GLOO_ENFORCE_NE(rv, -1);
}

std::unique_ptr<::gloo::transport::Buffer>
Pair::createSendBuffer(int slot, void* ptr, size_t size) {
  auto buffer = new Buffer(this, slot, ptr, size);
  return std::unique_ptr<::gloo::transport::Buffer>(buffer);
}

std::unique_ptr<::gloo::transport::Buffer>
Pair::createRecvBuffer(int slot, void* ptr, size_t size) {
  auto buffer = new Buffer(this, slot, ptr, size);
  sendMemoryRegion(buffer, buffer->mr_, buffer->slot_);
  return std::unique_ptr<::gloo::transport::Buffer>(buffer);
}

void Pair::handleCompletion(struct ibv_wc* wc) {
  if (wc->opcode & IBV_WC_RECV) {
    int slot = wc->imm_data & MASK_BUFFER_SLOT;
    if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      // Regular RDMA write.
      // Post new receive work request to backfill for this
      // completed work request.
      postReceive();
      // Forward to appropriate buffer completion handler.
      completion_handlers_[slot]->handleCompletion(wc);
    } else {
      // Memory region write.
      // The first receive work request has taken a write
      // containing the ibv_mr of the other side of the peer.
      struct ibv_mr* mr;
      int rv;

      // The buffer trying to write to this slot might be waiting for
      // the other side of this pair to send its memory region.
      // Lock access, and notify anybody waiting afterwards.
      {
        std::lock_guard<std::mutex> lock(m_);

        // Move ibv_mr from memory region 'inbox' to final slot.
        mr = tmp_memory_regions_.front();
        tmp_memory_regions_.pop_front();
        peer_memory_regions_[slot] = mr;

        // Deregister mapping for this ibv_mr.
        mr = mapped_recv_regions_.front();
        mapped_recv_regions_.pop_front();
        rv = ibv_dereg_mr(mr);
        GLOO_ENFORCE_NE(rv, -1);
      }

      cv_.notify_all();
    }
  } else if (wc->opcode == IBV_WC_SEND) {
    // Nop
  } else {
    GLOO_ENFORCE(false, "Unexpected completion with opcode: ", wc->opcode);
  }
}

} // namespace ibverbs
} // namespace transport
} // namespace gloo
