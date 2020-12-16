#pragma once

#include <c10d/Utils.hpp>

namespace c10d {
namespace tcputil {

#define AF_SELECTED AF_UNSPEC
#define CONNECT_SOCKET_OFFSET 2

inline void closeSocket(int socket) { ::close(socket); }

inline int setSocketAddrReUse(int socket) {
  int optval = 1;
  return ::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int));
}

inline int poll(struct pollfd *fds, unsigned long nfds, int timeout) {
  return ::poll(fds, nfds, timeout);
}

inline void addPollfd(std::vector<struct pollfd> &fds, int socket,
                      short events) {
  fds.push_back({.fd = socket, .events = events});
}

inline void waitSocketConnected(
    int socket,
    struct ::addrinfo *nextAddr,
    std::chrono::milliseconds timeout,
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime) {
  SYSCHECK_ERR_RETURN_NEG1(::fcntl(socket, F_SETFL, O_NONBLOCK));

  int ret = ::connect(socket, nextAddr->ai_addr, nextAddr->ai_addrlen);

  if (ret != 0 && errno != EINPROGRESS) {
    throw std::system_error(errno, std::system_category());
  }

  struct ::pollfd pfd;
  pfd.fd = socket;
  pfd.events = POLLOUT;

  int64_t pollTimeout = -1;
  if (timeout != kNoTimeout) {
    // calculate remaining time and use that as timeout for poll()
    const auto elapsed = std::chrono::high_resolution_clock::now() - startTime;
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(timeout) -
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    pollTimeout = std::max(static_cast<int64_t>(0),
                           static_cast<int64_t>(remaining.count()));
  }
  int numReady = ::poll(&pfd, 1, pollTimeout);
  if (numReady < 0) {
    throw std::system_error(errno, std::system_category());
  } else if (numReady == 0) {
    errno = 0;
    throw std::runtime_error(kConnectTimeoutMsg);
  }

  socklen_t errLen = sizeof(errno);
  errno = 0;
  ::getsockopt(socket, SOL_SOCKET, SO_ERROR, &errno, &errLen);

  // `errno` is set when:
  //  1. `getsockopt` has failed
  //  2. there is awaiting error in the socket
  //  (the error is saved to the `errno` variable)
  if (errno != 0) {
    throw std::system_error(errno, std::system_category());
  }

  // Disable non-blocking mode
  int flags;
  SYSCHECK_ERR_RETURN_NEG1(flags = ::fcntl(socket, F_GETFL));
  SYSCHECK_ERR_RETURN_NEG1(::fcntl(socket, F_SETFL, flags & (~O_NONBLOCK)));
}

// Linux socket does not need init libs first
inline void socketInitialize() {}

inline struct ::pollfd getPollfd(int socket, short events) {
  struct ::pollfd res = {.fd = socket, .events = events};
  return res;
}

} // namespace tcputil
} // namespace c10d
