// Copyright © 2025 Apple Inc.

#include <netdb.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <thread>

#include "jaccl/tcp.h"

namespace jaccl {

/**
 * Parse a sockaddr from an ip and port provided as strings.
 */
address_t parse_address(const std::string& ip, const std::string& port) {
  struct addrinfo hints, *res;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int status = getaddrinfo(ip.c_str(), port.c_str(), &hints, &res);
  if (status != 0) {
    std::ostringstream msg;
    msg << "Can't parse address " << ip << ":" << port;
    throw std::runtime_error(msg.str());
  }

  address_t result;
  memcpy(&result.addr, res->ai_addr, res->ai_addrlen);
  result.len = res->ai_addrlen;
  freeaddrinfo(res);

  return result;
}

/**
 * Parse a sockaddr provided as an <ip>:<port> string.
 */
address_t parse_address(const std::string& ip_port) {
  auto colon = ip_port.find(":");
  if (colon == std::string::npos) {
    std::ostringstream msg;
    msg << "Can't parse address " << ip_port;
    throw std::runtime_error(msg.str());
  }
  std::string ip(ip_port.begin(), ip_port.begin() + colon);
  std::string port(ip_port.begin() + colon + 1, ip_port.end());

  return parse_address(ip, port);
}

TCPSocket::TCPSocket(const char* tag) {
  sock_ = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_ < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't create socket (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }
}

TCPSocket::TCPSocket(TCPSocket&& s) {
  sock_ = s.sock_;
  s.sock_ = -1;
}

TCPSocket& TCPSocket::operator=(TCPSocket&& s) {
  if (this != &s) {
    sock_ = s.sock_;
    s.sock_ = -1;
  }
  return *this;
}

TCPSocket::TCPSocket(int s) : sock_(s) {}

TCPSocket::~TCPSocket() {
  if (sock_ > 0) {
    shutdown(sock_, 2);
    close(sock_);
  }
}

int TCPSocket::detach() {
  int s = sock_;
  sock_ = -1;
  return s;
}

void TCPSocket::listen(const char* tag, const address_t& addr) {
  int success;

  // Make sure we can launch immediately after shutdown by setting the
  // reuseaddr option so that we don't get address already in use errors
  int enable = 1;
  success = setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
  if (success < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't enable reuseaddr (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }
  success = setsockopt(sock_, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(int));
  if (success < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't enable reuseport (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }

  // Bind the socket to the address and port
  success = bind(sock_, addr.get(), addr.len);
  if (success < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't bind socket (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }

  // Prepare waiting for connections
  success = ::listen(sock_, 0);
  if (success < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't listen (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }
}

TCPSocket TCPSocket::accept(const char* tag) {
  int peer = ::accept(sock_, nullptr, nullptr);
  if (peer < 0) {
    std::ostringstream msg;
    msg << tag << " Accept failed (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }

  return TCPSocket(peer);
}

void TCPSocket::send(const char* tag, const void* data, size_t len) {
  while (len > 0) {
    auto n = ::send(sock_, data, len, 0);
    if (n <= 0) {
      std::ostringstream msg;
      msg << tag << " Send failed with errno=" << errno;
      throw std::runtime_error(msg.str());
    }
    len -= n;
    data = static_cast<const char*>(data) + n;
  }
}

void TCPSocket::recv(const char* tag, void* data, size_t len) {
  while (len > 0) {
    auto n = ::recv(sock_, data, len, 0);
    if (n <= 0) {
      std::ostringstream msg;
      msg << tag << " Recv failed with errno=" << errno;
      throw std::runtime_error(msg.str());
    }
    len -= n;
    data = static_cast<char*>(data) + n;
  }
}

TCPSocket TCPSocket::connect(
    const char* tag,
    const address_t& addr,
    int num_retries,
    int wait,
    std::function<void(int, int)> cb) {
  int sock, success;

  // Attempt to connect `num_retries` times with exponential backoff.
  for (int attempt = 0; attempt < num_retries; attempt++) {
    // Create the socket
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      std::ostringstream msg;
      msg << tag << " Couldn't create socket to connect (error: " << errno
          << ")";
      throw std::runtime_error(msg.str());
    }

    success = ::connect(sock, addr.get(), addr.len);
    if (success == 0) {
      break;
    }

    if (cb != nullptr) {
      cb(attempt, wait);
    }
    if (wait > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(wait));
    }

    wait <<= 1;
  }

  if (success < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't connect (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }

  return TCPSocket(sock);
}

} // namespace jaccl
