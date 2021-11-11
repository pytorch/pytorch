// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <c10/macros/Macros.h>

namespace c10d {
namespace detail {

class SocketImpl;

class Socket {
 public:
  static void initialize();

  static Socket listen(std::uint16_t port, bool prefer_ipv6 = true);

  static Socket connect(const std::string& host, std::uint16_t port, bool prefer_ipv6 = true);

  Socket() noexcept = default;

  Socket(const Socket& other) = delete;

  Socket& operator=(const Socket& other) = delete;

  Socket(Socket&& other) noexcept;

  Socket& operator=(Socket&& other) noexcept;

  ~Socket();

  Socket accept() const;

  int handle() const noexcept;

  std::uint16_t port() const;

 private:
  explicit Socket(std::unique_ptr<SocketImpl>&& impl) noexcept;

  std::unique_ptr<SocketImpl> impl_;
};

} // namespace detail

class TORCH_API SocketError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;

  SocketError(const SocketError&) = default;

  SocketError& operator=(const SocketError&) = default;

  SocketError(SocketError&&) = default;

  SocketError& operator=(SocketError&&) = default;

  ~SocketError() override;
};

} // namespace c10d
