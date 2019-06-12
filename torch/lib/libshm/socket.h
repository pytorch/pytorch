#pragma once

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <poll.h>
#include <cstdio>
#include <string>
#include <sstream>
#include <iostream>
#include <cstring>

#include "err.h"
#include "alloc_info.h"

class Socket {
public:
  int socket_fd;

protected:
  Socket() {
    SYSCHECK(socket_fd = socket(AF_UNIX, SOCK_STREAM, 0));
  }
  Socket(const Socket& other) = delete;
  Socket(Socket&& other) noexcept : socket_fd(other.socket_fd) { other.socket_fd = -1; };
  explicit Socket(int fd) : socket_fd(fd) {}

  virtual ~Socket() {
    if (socket_fd != -1)
      close(socket_fd);
  }

  struct sockaddr_un prepare_address(const char *path) {
    struct sockaddr_un address;
    address.sun_family = AF_UNIX;
    strcpy(address.sun_path, path);
    return address;
  }

  size_t address_length(struct sockaddr_un address) {
    return strlen(address.sun_path) + sizeof(address.sun_family);
  }

  void recv(void *_buffer, size_t num_bytes) {
    char *buffer = (char*)_buffer;
    size_t bytes_received = 0;
    ssize_t step_received;
    struct pollfd pfd = {0};
    pfd.fd = socket_fd;
    pfd.events = POLLIN;
    while (bytes_received < num_bytes) {
      SYSCHECK(poll(&pfd, 1, 1000));
      if (pfd.revents & POLLIN) {
        SYSCHECK(step_received = ::read(socket_fd, buffer, num_bytes - bytes_received));
        if (step_received == 0)
          throw std::runtime_error("Other end has closed the connection");
        bytes_received += step_received;
        buffer += step_received;
      } else if (pfd.revents & (POLLERR | POLLHUP)) {
        throw std::runtime_error("An error occurred while waiting for the data");
      } else {
        throw std::runtime_error("Shared memory manager connection has timed out");
      }
    }
  }

  void send(const void *_buffer, size_t num_bytes) {
    const char *buffer = (const char*)_buffer;
    size_t bytes_sent = 0;
    ssize_t step_sent;
    while (bytes_sent < num_bytes) {
      SYSCHECK(step_sent = ::write(socket_fd, buffer, num_bytes));
      bytes_sent += step_sent;
      buffer += step_sent;
    }
  }


};

class ManagerSocket: public Socket {
public:
  explicit ManagerSocket(int fd): Socket(fd) {}

  AllocInfo receive() {
    AllocInfo info;
    recv(&info, sizeof(info));
    return info;
  }

  void confirm() {
    send("OK", 2);
  }

};


class ManagerServerSocket: public Socket {
public:
  explicit ManagerServerSocket(const std::string &path) {
    socket_path = path;
    try {
      struct sockaddr_un address = prepare_address(path.c_str());
      size_t len = address_length(address);
      SYSCHECK(bind(socket_fd, (struct sockaddr *)&address, len));
      SYSCHECK(listen(socket_fd, 10));
    } catch(std::exception &e) {
      close(socket_fd);
      throw;
    }
  }

  virtual ~ManagerServerSocket() {
    unlink(socket_path.c_str());
  }

  ManagerSocket accept() {
    int client_fd;
    struct sockaddr_un addr;
    socklen_t addr_len = sizeof(addr);
    SYSCHECK(client_fd = ::accept(socket_fd, (struct sockaddr *)&addr, &addr_len));
    return ManagerSocket(client_fd);
  }

  std::string socket_path;
};

class ClientSocket: public Socket {
public:
  explicit ClientSocket(const std::string &path) {
    try {
      struct sockaddr_un address = prepare_address(path.c_str());
      size_t len = address_length(address);
      SYSCHECK(connect(socket_fd, (struct sockaddr *)&address, len));
    } catch(std::exception &e) {
      close(socket_fd);
      throw;
    }
  }

  void register_allocation(AllocInfo &info) {
    char buffer[3] = {0, 0, 0};
    ssize_t bytes_read;
    send(&info, sizeof(info));
    recv(buffer, 2);
    if (strcmp(buffer, "OK") != 0)
      throw std::runtime_error("Shared memory manager didn't respond with an OK");
  }

  void register_deallocation(AllocInfo &info) {
    send(&info, sizeof(info));
  }

};
