#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <poll.h>
#include <cstdio>
#include <string>
#include <sstream>
#include <iostream>

#include "err.h"
#include "alloc_info.h"

class Socket {
public:
  int socket_fd;

protected:
  Socket() {
    SYSCHECK(socket_fd = socket(AF_UNIX, SOCK_SEQPACKET, 0));
  }
  Socket(const Socket& other) = delete;
  Socket(Socket&& other): socket_fd(other.socket_fd) { other.socket_fd = -1; };

  Socket(int fd): socket_fd(fd) {}

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


};

class ManagerSocket: public Socket {
public:
  ManagerSocket(int fd): Socket(fd) {}

  AllocInfo recieve() {
    AllocInfo info;
    SYSCHECK(::recv(socket_fd, &info, sizeof(info), 0));
    return info;
  }

  void confirm() {
    SYSCHECK(::send(socket_fd, "OK", 2, 0));
  }

};


class ManagerServerSocket: public Socket {
public:
  ManagerServerSocket(const std::string &path) {
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
    SYSCHECK(client_fd = ::accept(socket_fd, NULL, NULL));
    return ManagerSocket(client_fd);
  }

  std::string socket_path;
};

class ClientSocket: public Socket {
public:
  ClientSocket(const std::string &path) {
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
    char buffer[10];
    ssize_t bytes_read;
    SYSCHECK(::send(socket_fd, &info, sizeof(info), 0));
    struct pollfd pfd = {0};
    pfd.fd = socket_fd;
    pfd.events = POLLIN;
    SYSCHECK(poll(&pfd, 1, 1000));
    if (pfd.revents & POLLIN) {
        SYSCHECK(bytes_read = ::recv(socket_fd, buffer, sizeof(buffer)-1, 0));
        buffer[bytes_read] = 0;
        if (strcmp(buffer, "OK") != 0)
            throw std::exception();
    } else {
        // no data arrived before the timeout
        throw std::exception();
    }
  }

  void register_deallocation(AllocInfo &info) {
    ::send(socket_fd, &info, sizeof(info), 0);
  }

};
