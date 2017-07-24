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
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
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
  static constexpr int kMaxBuffers = 8;

  // Use 3x the maximum number of buffers as the capacity
  // for entries in this pair's completion queue.
  //
  // For receive completions, there are a maximum of:
  //   - MAX_BUFFERS posted receive work requests to receive memory
  //     regions from the other side of the pair (for send buffers).
  //   - MAX_BUFFERS posted receive work requests for RDMA writes
  //     from the other side of the pair.
  //
  // For send completions, there are a maximum of:
  //   - MAX_BUFFERS posted send work requests to send memory
  //     regions to the other side of the pair (for receive buffers).
  //   - MAX_BUFFERS posted send work requests for RDMA writes
  //     to the other side of the pair.
  //
  // While this sums up to 4x kMaxBuffers work requests, send work
  // requests can only be posted after receiving the corresponding
  // memory region from the other side of the pair. This leads to a
  // maximum of of 3x kMaxBuffers posted work requests at any given
  // time. However, since the majority can be made up by either
  // receive work requests or send work requests, we keep the capacity
  // at 4x kMaxBuffers and allocate half to each type.
  //
  static constexpr auto kRecvCompletionQueueCapacity = 2 * kMaxBuffers;
  static constexpr auto kSendCompletionQueueCapacity = 2 * kMaxBuffers;
  static constexpr auto kCompletionQueueCapacity =
    kRecvCompletionQueueCapacity + kSendCompletionQueueCapacity;

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

  void send(Buffer* buf, size_t offset, size_t length, size_t roffset);

 protected:
  std::shared_ptr<Device> dev_;

  // Whether or not this pair is running in sync mode.
  std::atomic<bool> sync_;

  // Whether or not this pair is busy polling in sync mode.
  std::atomic<bool> busyPoll_;

  const std::chrono::milliseconds timeout_;

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
  std::map<int, struct ibv_mr> peerMemoryRegions_;

  // These fields store memory regions that the remote side of the pair
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
  std::map<int, std::unique_ptr<MemoryRegion> > mappedSendRegions_;
  std::list<std::unique_ptr<MemoryRegion> > mappedRecvRegions_;

  // Completions on behalf of buffers need to be forwarded to those buffers.
  std::map<int, Buffer*> sendCompletionHandlers_;
  std::map<int, Buffer*> recvCompletionHandlers_;

  void receiveMemoryRegion();
  void sendMemoryRegion(struct ibv_mr* mr, int slot);
  const struct ibv_mr* getMemoryRegion(int slot);

  void postReceive();

  std::chrono::milliseconds getTimeout() const {
    return timeout_;
  }

  const Address& peer() const {
    return peer_;
  }

 private:
  std::exception_ptr ex_;

  // Used to signal IO exceptions from one thread and propagate onto others.
  void signalIoFailure(const std::string& msg);
  void checkErrorState();

  friend class Buffer;
};

} // namespace ibverbs
} // namespace transport
} // namespace gloo
