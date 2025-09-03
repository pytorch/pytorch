#pragma once

#include <torch/csrc/distributed/c10d/Utils.h>

namespace c10d::tcputil {

#define CONNECT_SOCKET_OFFSET 1

inline int poll(struct pollfd* fdArray, unsigned long fds, int timeout) {
  return WSAPoll(fdArray, fds, timeout);
}

inline void addPollfd(
    std::vector<struct pollfd>& fds,
    int socket,
    short events) {
  fds.push_back({(SOCKET)socket, events});
}

inline struct ::pollfd getPollfd(int socket, short events) {
  struct ::pollfd res = {(SOCKET)socket, events};
  return res;
}

} // namespace c10d::tcputil
