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
#include <list>
#include <memory>
#include <mutex>
#include <vector>

#include "gloo/transport/ibverbs/address.h"
#include "gloo/transport/ibverbs/device.h"
#include "gloo/transport/pair.h"

namespace gloo {
namespace transport {
namespace ibverbs {

// Forward declaration
class Buffer;

class Pair : public ::gloo::transport::Pair, public Handler {
  static const int MASK_BUFFER_SLOT = 0x7;
  static const int MAX_BUFFERS = MASK_BUFFER_SLOT + 1;

 public:
  explicit Pair(const std::shared_ptr<Device>& dev);
  virtual ~Pair();

  Pair(const Pair& that) = delete;

  Pair& operator=(const Pair& that) = delete;

  virtual const Address& address() const override;

  virtual void connect(const std::vector<char>& bytes) override;

  virtual void setSync(bool enable) override;

  virtual std::unique_ptr<::gloo::transport::Buffer>
  createSendBuffer(int slot, void* ptr, size_t size) override;

  virtual std::unique_ptr<::gloo::transport::Buffer>
  createRecvBuffer(int slot, void* ptr, size_t size) override;

  virtual void handleCompletion(struct ibv_wc* wc) override;

 protected:
  std::shared_ptr<Device> dev_;

  Address self_;
  Address peer_;

  struct ibv_cq* write_cq_;
  struct ibv_qp* qp_;

  std::mutex m_;
  std::condition_variable cv_;

  // For the remote peer to write their ibv_mr's into.
  // After writing, this pair's handler pops off the
  // first element and places it in peer_memory_regions_
  // according to the slot specified in the immediate
  // part of the write.
  std::list<struct ibv_mr*> tmp_memory_regions_;

  // For us to copy the remote peer's ibv_mr into.
  // Use an array instead of container so that the Buffer
  // class can use it without holding a lock.
  std::array<struct ibv_mr*, MAX_BUFFERS> peer_memory_regions_;
  std::atomic<uint64_t> peer_memory_regions_ready_;

  // Keep track of ibv_mr's for peer_memory_regions,
  // so they can be dereg'd when done.
  std::list<struct ibv_mr*> mapped_recv_regions_;
  std::list<struct ibv_mr*> mapped_send_regions_;

  // When we receive an RDMA write we need to dispatch
  // the completion to the right handler (buffer).
  std::array<Handler*, MAX_BUFFERS> completion_handlers_;

  void receiveMemoryRegion();
  void sendMemoryRegion(Handler* h, struct ibv_mr* mr, int slot);
  const struct ibv_mr* getMemoryRegion(int slot);

  void postReceive();

  friend class Buffer;
};

} // namespace ibverbs
} // namespace transport
} // namespace gloo
