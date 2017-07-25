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
#include <memory>
#include <mutex>
#include <thread>

#include <sys/socket.h>

#include "gloo/transport/device.h"

namespace gloo {
namespace transport {
namespace tcp {

struct attr {
  attr() {}
  /* implicit */ attr(const char* ptr) : hostname(ptr) {}

  std::string hostname;

  // The address family defaults to AF_UNSPEC such that getaddrinfo(3)
  // will try to find either IPv4 or IPv6 addresses.
  int ai_family = AF_UNSPEC;
  int ai_socktype;
  int ai_protocol;
  struct sockaddr_storage ai_addr;
  int ai_addrlen;
};

std::shared_ptr<::gloo::transport::Device> CreateDevice(
    const struct attr&);

// Forward declarations
class Pair;
class Buffer;

class Device : public ::gloo::transport::Device,
               public std::enable_shared_from_this<Device> {
 public:
  explicit Device(const struct attr& attr);
  virtual ~Device();

  virtual std::string str() const override;

  virtual const std::string& getPCIBusID() const override;

  virtual int getInterfaceSpeed() const override;

  virtual void setTimeout(const std::chrono::milliseconds& timeout) override;

  virtual std::unique_ptr<::gloo::transport::Pair> createPair()
      override;

  virtual std::chrono::milliseconds getTimeout() const override;

 protected:
  void loop();

  void registerDescriptor(int fd, int events, Pair* p);
  void unregisterDescriptor(int fd);

  const struct attr attr_;
  std::atomic<bool> done_;
  std::unique_ptr<std::thread> loop_;

  friend class Pair;
  friend class Buffer;

 private:
  static constexpr auto capacity_ = 64;

  int fd_;
  std::atomic<std::chrono::milliseconds> timeout_;
  std::string interfaceName_;
  int interfaceSpeedMbps_;
  std::string pciBusID_;

  std::mutex m_;
  std::condition_variable cv_;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
