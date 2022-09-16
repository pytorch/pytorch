#pragma once

#include <c10d/Utils.hpp>

namespace c10d {
namespace tcputil {

#define CONNECT_SOCKET_OFFSET 1

inline int poll(struct pollfd *fdArray, unsigned long fds, int timeout) {
  return WSAPoll(fdArray, fds, timeout);
}

inline void addPollfd(std::vector<struct pollfd> &fds, int socket,
                      short events) {
  fds.push_back({(SOCKET)socket, events});
}

inline struct ::pollfd getPollfd(int socket, short events) {
  struct ::pollfd res = {(SOCKET)socket, events};
  return res;
}

} // namespace tcputil
} // namespace c10d
