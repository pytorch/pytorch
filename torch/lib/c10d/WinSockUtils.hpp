#pragma once

#include <c10d/Utils.hpp>

namespace c10d {
namespace tcputil {

#define AF_SELECTED AF_INET
#define CONNECT_SOCKET_OFFSET 1

inline void closeSocket(int socket) { ::closesocket(socket); }

inline int setSocketAddrReUse(int socket) {
  bool optval = false;
  return ::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, (char *)&optval,
                      sizeof(bool));
}

inline int poll(struct pollfd *fdArray, unsigned long fds, int timeout) {
  return WSAPoll(fdArray, fds, timeout);
}

inline void addPollfd(std::vector<struct pollfd> &fds, int socket,
                      short events) {
  fds.push_back({(SOCKET)socket, events});
}

inline void waitSocketConnected(
    int socket,
    struct ::addrinfo *nextAddr,
    std::chrono::milliseconds timeout,
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime) {
  unsigned long block_mode = 1;
  SYSCHECK_ERR_RETURN_NEG1(ioctlsocket(socket, FIONBIO, &block_mode));

  int ret;
  do {
    ret = connect(socket, nextAddr->ai_addr, nextAddr->ai_addrlen);
    if (ret == SOCKET_ERROR) {
      int err = WSAGetLastError();
      if (err == WSAEISCONN) {
        break;
      } else if (err == WSAEALREADY || err == WSAEWOULDBLOCK) {
        if (timeout != kNoTimeout) {
          const auto elapsed =
              std::chrono::high_resolution_clock::now() - startTime;
          if (elapsed > timeout) {
            errno = 0;
            throw std::runtime_error(kConnectTimeoutMsg);
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      throw std::system_error(err, std::system_category(),
                              "Socket connect failed");
    }
  } while (ret == SOCKET_ERROR);

  block_mode = 0;
  SYSCHECK_ERR_RETURN_NEG1(ioctlsocket(socket, FIONBIO, &block_mode));
}

// All processes (applications or DLLs) that call Winsock
// functions must initialize the use of the Windows Sockets
// DLL before making other Winsock function calls.
// This also makes certain that Winsock is supported on the system.
// Ref to
// https://docs.microsoft.com/en-us/windows/win32/winsock/initializing-winsock
inline void socketInitialize() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    WSADATA wsa_data;
    SYSCHECK_ERR_RETURN_NEG1(WSAStartup(MAKEWORD(2, 2), &wsa_data))
  });
}

inline struct ::pollfd getPollfd(int socket, short events) {
  struct ::pollfd res = {(SOCKET)socket, events};
  return res;
}

} // namespace tcputil
} // namespace c10d
