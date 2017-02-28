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

#include "gloo/common/common.h"
#include "gloo/common/logging.h"

namespace gloo {
namespace transport {
namespace ibverbs {

Pair::Pair(const std::shared_ptr<Device>& dev)
    : dev_(dev),
      peerMemoryRegionsReady_(0) {
  int rv;

  memset(peerMemoryRegions_.data(), 0, sizeof(peerMemoryRegions_));
  memset(sendCompletionHandlers_.data(), 0, sizeof(sendCompletionHandlers_));
  memset(recvCompletionHandlers_.data(), 0, sizeof(recvCompletionHandlers_));

  // Create completion queue
  {
    // Have to register this completion queue with the device's
    // completion channel to support asynchronous completion handling.
    // Pairs use asynchronous completion handling by default so
    // we call ibv_req_notify_cq(3) to request the first notification.
    cq_ = ibv_create_cq(
      dev_->context_,
      kCompletionQueueCapacity,
      this,
      dev_->comp_channel_,
      0);
    GLOO_ENFORCE(cq_);

    // Arm notification mechanism for completion queue.
    rv = ibv_req_notify_cq(cq_, kNotifyOnAnyCompletion);
    GLOO_ENFORCE_NE(rv, -1);
  }

  // Create queue pair
  {
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(struct ibv_qp_init_attr));
    attr.send_cq = cq_;
    attr.recv_cq = cq_;
    attr.cap.max_send_wr = Pair::kMaxBuffers;
    attr.cap.max_recv_wr = Pair::kMaxBuffers;
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
  for (int i = 0; i < kMaxBuffers; ++i) {
    receiveMemoryRegion();
  }
}

Pair::~Pair() {
  int rv;

  rv = ibv_destroy_qp(qp_);
  GLOO_ENFORCE_NE(rv, -1);

  rv = ibv_destroy_cq(cq_);
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
  auto mr = make_unique<MemoryRegion>(dev_->pd_);
  struct ibv_sge list = mr->sge();
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.sg_list = &list;
  wr.num_sge = 1;

  // The work request is serialized and sent to the driver so it
  // doesn't need to be valid after the ibv_post_recv call.
  struct ibv_recv_wr* bad_wr;
  int rv = ibv_post_recv(qp_, &wr, &bad_wr);
  GLOO_ENFORCE_NE(rv, -1);

  // Keep memory region around so that the other side of this pair can
  // write into it. They are written in a FIFO order so the handler
  // can always pop off the first entry upon handling the completion.
  mappedRecvRegions_.push_back(std::move(mr));
}

void Pair::sendMemoryRegion(struct ibv_mr* src, int slot) {
  auto mr = make_unique<MemoryRegion>(dev_->pd_, src);
  struct ibv_sge list = mr->sge();
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = slot;

  // First post receive work request to avoid racing with
  // a send to this region from the other side of this pair.
  postReceive();

  // The work request is serialized and sent to the driver so it
  // doesn't need to be valid after the ibv_post_send call.
  struct ibv_send_wr* bad_wr;
  int rv = ibv_post_send(qp_, &wr, &bad_wr);
  GLOO_ENFORCE_NE(rv, -1);

  // Keep memory region around until this send operation completes.
  // They are posted in a FIFO order so the handler can always pop off
  // the first entry upon handling the completion.
  mappedSendRegions_.push_back(std::move(mr));
}

const struct ibv_mr* Pair::getMemoryRegion(int slot) {
  if (peerMemoryRegionsReady_.load() & (1 << slot)) {
    return &peerMemoryRegions_[slot];
  }

  std::unique_lock<std::mutex> lock(m_);
  while (peerMemoryRegions_[slot].addr == nullptr) {
    cv_.wait(lock);
  }
  peerMemoryRegionsReady_ &= (1 << slot);
  return &peerMemoryRegions_[slot];
}

void Pair::postReceive() {
  int rv;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_recv_wr* bad_wr;
  rv = ibv_post_recv(qp_, &wr, &bad_wr);
  GLOO_ENFORCE_NE(rv, -1);
}

std::unique_ptr<::gloo::transport::Buffer>
Pair::createSendBuffer(int slot, void* ptr, size_t size) {
  auto buffer = new Buffer(this, slot, ptr, size);
  sendCompletionHandlers_[slot] = buffer;
  return std::unique_ptr<::gloo::transport::Buffer>(buffer);
}

std::unique_ptr<::gloo::transport::Buffer>
Pair::createRecvBuffer(int slot, void* ptr, size_t size) {
  auto buffer = new Buffer(this, slot, ptr, size);
  recvCompletionHandlers_[slot] = buffer;
  sendMemoryRegion(buffer->mr_, buffer->slot_);
  return std::unique_ptr<::gloo::transport::Buffer>(buffer);
}

// handleCompletions is called by the device thread when it
// received an event for this pair's completion queue on its
// completion channel.
void Pair::handleCompletions() {
  std::array<struct ibv_wc, kCompletionQueueCapacity> wc;
  int rv;

  ibv_ack_cq_events(cq_, 1);

  // Arm notification mechanism for completion queue.
  rv = ibv_req_notify_cq(cq_, kNotifyOnAnyCompletion);
  GLOO_ENFORCE_NE(rv, -1);

  // Invoke handler for every work completion.
  auto nwc = ibv_poll_cq(cq_, wc.size(), wc.data());

  GLOO_ENFORCE_GE(nwc, 0);
  for (int i = 0; i < nwc; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      continue;
    }

    handleCompletion(&wc[i]);
  }
}

void Pair::handleCompletion(struct ibv_wc* wc) {
  if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
    // Incoming RDMA write completed. Post new receive work request to
    // backfill for this completed work request.
    postReceive();

    // Slot is encoded in immediate data on receive work completion.
    // It is set in the Buffer::send function.
    // Bits outside of kBufferSlotMask are currently unused.
    int slot = wc->imm_data & kBufferSlotMask;
    GLOO_ENFORCE_EQ(wc->imm_data & ~kBufferSlotMask, 0);
    GLOO_ENFORCE(recvCompletionHandlers_[slot] != nullptr);
    recvCompletionHandlers_[slot]->handleCompletion(wc);
  } else if (wc->opcode == IBV_WC_RDMA_WRITE) {
    // Outbound RDMA write completed.
    // Slot is encoded in wr_id fields on send work request. Unlike
    // the receive work completions, the immediate data field on send
    // work requests are not pass to the respective work completion.
    int slot = wc->wr_id & kBufferSlotMask;
    GLOO_ENFORCE_EQ(wc->wr_id & ~kBufferSlotMask, 0);
    GLOO_ENFORCE(sendCompletionHandlers_[slot] != nullptr);
    sendCompletionHandlers_[slot]->handleCompletion(wc);
  } else if (wc->opcode == IBV_WC_RECV) {
    // Memory region recv completed.
    //
    // Only used by the remote side of the pair to pass ibv_mr's.
    // They are written to in FIFO order, so we can pick up
    // and use the first MemoryRegion instance in the list of
    // mapped receive regions.
    //
    // The buffer trying to write to this slot might be waiting for
    // the other side of this pair to send its memory region.
    // Lock access, and notify anybody waiting afterwards.
    //
    {
      std::lock_guard<std::mutex> lock(m_);
      GLOO_ENFORCE_GT(mappedRecvRegions_.size(), 0);
      auto mr = std::move(mappedRecvRegions_.front());
      mappedRecvRegions_.pop_front();

      // Slot is encoded in immediate data on receive work completion.
      // It is set in the Pair::sendMemoryRegion function.
      // Bits outside of kBufferSlotMask are currently unused.
      int slot = wc->imm_data & kBufferSlotMask;
      GLOO_ENFORCE_EQ(wc->imm_data & ~kBufferSlotMask, 0);

      // Move ibv_mr from memory region 'inbox' to final slot.
      peerMemoryRegions_[slot] = mr->mr();
    }

    // Notify any buffer waiting for the details of its remote peer.
    cv_.notify_all();
  } else if (wc->opcode == IBV_WC_SEND) {
    // Memory region send completed.
    mappedSendRegions_.pop_front();
  } else {
    GLOO_ENFORCE(false, "Unexpected completion with opcode: ", wc->opcode);
  }
}

} // namespace ibverbs
} // namespace transport
} // namespace gloo
