// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/exception.h>

namespace c10d {
namespace detail {

class SocketOptions {
 public:
  SocketOptions& prefer_ipv6(bool value) noexcept {
    prefer_ipv6_ = value;

    return *this;
  }

  bool prefer_ipv6() const noexcept {
    return prefer_ipv6_;
  }

  SocketOptions& connect_timeout(std::chrono::seconds value) noexcept {
    connect_timeout_ = value;

    return *this;
  }

  std::chrono::seconds connect_timeout() const noexcept {
    return connect_timeout_;
  }

 private:
  bool prefer_ipv6_ = true;
  std::chrono::seconds connect_timeout_{30};
};

class SocketImpl;

class Socket {
 public:
  // This function initializes the underlying socket library and must be called
  // before any other socket function.
  static void initialize();

  static Socket listen(std::uint16_t port, const SocketOptions& opts = {});

  static Socket listenFromFd(int fd, std::uint16_t expected_port);

  static Socket connect(
      const std::string& host,
      std::uint16_t port,
      const SocketOptions& opts = {});

  Socket() noexcept = default;

  Socket(const Socket& other) = delete;

  Socket& operator=(const Socket& other) = delete;

  Socket(Socket&& other) noexcept;

  Socket& operator=(Socket&& other) noexcept;

  ~Socket();

  Socket accept() const;

  int handle() const noexcept;

  std::uint16_t port() const;

  bool waitForInput(std::chrono::milliseconds timeout);

 private:
  explicit Socket(std::unique_ptr<SocketImpl>&& impl) noexcept;

  std::unique_ptr<SocketImpl> impl_;
};

} // namespace detail

} // namespace c10d
