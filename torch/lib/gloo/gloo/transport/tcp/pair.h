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
#include <chrono>
#include <condition_variable>
#include <exception>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <sys/uio.h>

#include "gloo/transport/pair.h"
#include "gloo/transport/tcp/address.h"
#include "gloo/transport/tcp/device.h"

namespace gloo {
namespace transport {
namespace tcp {

// Forward declaration
class Buffer;

struct Op {
  struct {
    size_t opcode_;
    size_t slot_;
    size_t offset_;
    size_t length_;
    size_t roffset_;
  } preamble_;

  // Used internally
  Buffer* buf_;
  size_t nread_;
  size_t nwritten_;
};

class Pair : public ::gloo::transport::Pair {
  enum state {
    INITIALIZING = 1,
    LISTENING = 2,
    CONNECTING = 3,
    CONNECTED = 4,
    CLOSED = 5,
  };

 public:
  explicit Pair(const std::shared_ptr<Device>& dev);
  virtual ~Pair();

  Pair(const Pair& that) = delete;

  Pair& operator=(const Pair& that) = delete;

  virtual const Address& address() const override;

  virtual void connect(const std::vector<char>& bytes) override;

  virtual void setSync(bool sync, bool busyPoll) override;

  virtual std::unique_ptr<::gloo::transport::Buffer>
  createSendBuffer(int slot, void* ptr, size_t size) override;

  virtual std::unique_ptr<::gloo::transport::Buffer>
  createRecvBuffer(int slot, void* ptr, size_t size) override;

  void handleEvents(int events);

 protected:
  std::shared_ptr<Device> dev_;
  state state_;
  std::atomic<bool> sync_;
  const std::chrono::milliseconds timeout_;
  // When set, instructs pair to use busy-polling on receive.
  // Can only be used with sync receive mode.
  bool busyPoll_;
  int fd_;
  int sendBufferSize_;

  Address self_;
  Address peer_;

  std::mutex m_;
  std::condition_variable cv_;
  std::map<int, Buffer*> buffers_;

  void listen();
  void connect(const Address& peer);

  Buffer* getBuffer(int slot);
  void registerBuffer(Buffer* buf);
  void unregisterBuffer(Buffer* buf);

  void send(Op& op);
  void recv();

  const Address& peer() const {
    return peer_;
  }

  bool isSync() const {
    return sync_;
  }

  std::chrono::milliseconds getTimeout() const {
    return timeout_;
  }

  void signalIoFailureExternal(const std::string& msg);

  friend class Buffer;

 private:
  Op rx_;
  Op tx_;

  std::exception_ptr ex_;

  bool write(Op& op);
  bool read(Op& op);

  void handleListening();
  void handleConnecting();
  void handleConnected();

  void changeState(state nextState);
  void waitUntilConnected(std::unique_lock<std::mutex>& lock, bool useTimeout);
  void verifyConnected();

  // Used to signal IO exceptions from one thread and propagate onto others.
  void signalIoFailure(const std::string& msg);
  void checkErrorState();
};

} // namespace tcp
} // namespace transport
} // namespace gloo
