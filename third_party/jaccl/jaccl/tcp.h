// Copyright © 2025 Apple Inc.

#pragma once

#include <sys/socket.h>
#include <functional>
#include <string>

namespace jaccl {

struct address_t {
  sockaddr_storage addr;
  socklen_t len;

  const sockaddr* get() const {
    return (struct sockaddr*)&addr;
  }
};

/**
 * Parse a sockaddr from an ip and port provided as strings.
 */
address_t parse_address(const std::string& ip, const std::string& port);

/**
 * Parse a sockaddr provided as an <ip>:<port> string.
 */
address_t parse_address(const std::string& ip_port);

/**
 * Small wrapper over a TCP socket to simplify initiating connections.
 */
class TCPSocket {
 public:
  TCPSocket(const char* tag);
  TCPSocket(const TCPSocket&) = delete;
  TCPSocket& operator=(const TCPSocket&) = delete;
  TCPSocket(TCPSocket&& s);
  TCPSocket& operator=(TCPSocket&&);
  ~TCPSocket();

  void listen(const char* tag, const address_t& addr);
  TCPSocket accept(const char* tag);

  void send(const char* tag, const void* data, size_t len);
  void recv(const char* tag, void* data, size_t len);

  int detach();

  operator int() const {
    return sock_;
  }

  static TCPSocket connect(
      const char* tag,
      const address_t& addr,
      int num_retries = 1,
      int wait = 0,
      std::function<void(int, int)> cb = nullptr);

 private:
  TCPSocket(int sock);

  int sock_;
};

} // namespace jaccl
