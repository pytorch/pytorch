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
#include "gloo/transport/ibverbs/memory_region.h"
#include "gloo/transport/pair.h"

namespace gloo {
namespace transport {
namespace ibverbs {

// Forward declaration
class Buffer;

class Pair : public ::gloo::transport::Pair {
  static constexpr int kBufferSlotMask = 0x7;
  static constexpr int kMaxBuffers = kBufferSlotMask + 1;

  // Use 3x the maximum number of buffers as the capacity
  // for entries in this pair's completion queue.
  //
  // There are a maximum of:
  //   - MAX_BUFFERS posted receive work requests to receive memory
  //     regions from the other side of the pair.
  //   - MAX_BUFFERS posted send work requests to send memory
  //     regions to the other side of the pair.
  //   - MAX_BUFFERS posted receive work requests for RDMA writes
  //     from the other side of the pair. These requests are posted
  //     at the same time a buffer's local memory region is sent to
  //     the other side of the pair.
  static constexpr auto kCompletionQueueCapacity = 3 * kMaxBuffers;

  // The ibv_req_notify(3) function takes an argument called
  // 'solicited_only' which makes it only trigger a notification for
  // work requests that are flagged as solicited. Every completion
  // should trigger a notification, so always pass 0.
  static constexpr auto kNotifyOnAnyCompletion = 0;

 public:
  explicit Pair(const std::shared_ptr<Device>& dev);
  virtual ~Pair();

  Pair(const Pair& that) = delete;

  Pair& operator=(const Pair& that) = delete;

  virtual const Address& address() const override;

  virtual void connect(const std::vector<char>& bytes) override;

  virtual void setSync(bool enable, bool busyPoll) override;

  virtual std::unique_ptr<::gloo::transport::Buffer>
  createSendBuffer(int slot, void* ptr, size_t size) override;

  virtual std::unique_ptr<::gloo::transport::Buffer>
  createRecvBuffer(int slot, void* ptr, size_t size) override;

  void handleCompletionEvent();

  void pollCompletions();

  void handleCompletion(struct ibv_wc* wc);

 protected:
  std::shared_ptr<Device> dev_;

  // Whether or not this pair is running in sync mode.
  std::atomic<bool> sync_;

  // Whether or not this pair is busy polling in sync mode.
  std::atomic<bool> busyPoll_;

  // Number of completion events handled by this pair's completion
  // queue (also see ibv_get_cq_event(3)). This many events need to be
  // acknowledged prior to destructing the completion queue.
  // Otherwise, destruction will hang (see ibv_get_cq_event(3)).
  int completionEventsHandled_;

  Address self_;
  Address peer_;

  struct ibv_cq* cq_;
  struct ibv_qp* qp_;

  std::mutex m_;
  std::condition_variable cv_;

  // For us to copy the remote peer's ibv_mr into.
  // Use an array instead of container so that the Buffer
  // class can use it without holding a lock.
  std::array<struct ibv_mr, kMaxBuffers> peerMemoryRegions_;
  std::atomic<uint64_t> peerMemoryRegionsReady_;

  // These lists store memory regions that the remote side of the pair
  // can send to and that the local side of the pair can send from.
  //
  // After receiving a memory region from the remote side of the pair,
  // the memory region is popped off the front of this list and is
  // destructed, after storing the remote details in
  // peerMemoryRegions_.
  //
  // When registering a receive buffer, the local ibv_mr is sent
  // to the remote side of the pair, and the corresponding MemoryRegion
  // instance is kept around in the mappedSendRegions_ list until
  // the send operation complete.
  //
  std::list<std::unique_ptr<MemoryRegion> > mappedSendRegions_;
  std::list<std::unique_ptr<MemoryRegion> > mappedRecvRegions_;

  // Completions on behalf of buffers need to be forwarded to those buffers.
  std::array<Buffer*, kMaxBuffers> sendCompletionHandlers_;
  std::array<Buffer*, kMaxBuffers> recvCompletionHandlers_;

  void receiveMemoryRegion();
  void sendMemoryRegion(struct ibv_mr* mr, int slot);
  const struct ibv_mr* getMemoryRegion(int slot);

  void postReceive();

  friend class Buffer;
};

} // namespace ibverbs
} // namespace transport
} // namespace gloo
