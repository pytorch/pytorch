#pragma once

#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

#include <libshm/alloc_info.h>
#include <libshm/err.h>

class Socket {
 public:
  int socket_fd;
  Socket(const Socket& other) = delete;

 protected:
  Socket() {
    SYSCHECK_ERR_RETURN_NEG1(socket_fd = socket(AF_UNIX, SOCK_STREAM, 0));
  }
  Socket(Socket&& other) noexcept : socket_fd(other.socket_fd) {
    other.socket_fd = -1;
  };
  explicit Socket(int fd) : socket_fd(fd) {}

  virtual ~Socket() {
    if (socket_fd != -1)
      close(socket_fd);
  }

  struct sockaddr_un prepare_address(const char* path) {
    struct sockaddr_un address;
    address.sun_family = AF_UNIX;
    strcpy(address.sun_path, path);
    return address;
  }

  // Implemented based on https://man7.org/linux/man-pages/man7/unix.7.html
  size_t address_length(struct sockaddr_un address) {
    return offsetof(sockaddr_un, sun_path) + strlen(address.sun_path) + 1;
  }

  void recv(void* _buffer, size_t num_bytes) {
    char* buffer = (char*)_buffer;
    size_t bytes_received = 0;
    ssize_t step_received;
    struct pollfd pfd = {};
    pfd.fd = socket_fd;
    pfd.events = POLLIN;
    while (bytes_received < num_bytes) {
      SYSCHECK_ERR_RETURN_NEG1(poll(&pfd, 1, 1000));
      if (pfd.revents & POLLIN) {
        SYSCHECK_ERR_RETURN_NEG1(
            step_received =
                ::read(socket_fd, buffer, num_bytes - bytes_received));
        TORCH_CHECK(step_received != 0, "Other end has closed the connection");
        bytes_received += step_received;
        buffer += step_received;
      } else if (pfd.revents & (POLLERR | POLLHUP)) {
        TORCH_CHECK(false, "An error occurred while waiting for the data");
      } else {
        TORCH_CHECK(false, "Shared memory manager connection has timed out");
      }
    }
  }

  void send(const void* _buffer, size_t num_bytes) {
    const char* buffer = (const char*)_buffer;
    size_t bytes_sent = 0;
    ssize_t step_sent;
    while (bytes_sent < num_bytes) {
      SYSCHECK_ERR_RETURN_NEG1(
          step_sent = ::write(socket_fd, buffer, num_bytes));
      bytes_sent += step_sent;
      buffer += step_sent;
    }
  }
};

class ManagerSocket : public Socket {
 public:
  explicit ManagerSocket(int fd) : Socket(fd) {}

  AllocInfo receive() {
    AllocInfo info;
    recv(&info, sizeof(info));
    return info;
  }

  void confirm() {
    send("OK", 2);
  }
};

class ManagerServerSocket : public Socket {
 public:
  explicit ManagerServerSocket(const std::string& path) {
    socket_path = path;
    try {
      struct sockaddr_un address = prepare_address(path.c_str());
      size_t len = address_length(address);
      SYSCHECK_ERR_RETURN_NEG1(
          bind(socket_fd, (struct sockaddr*)&address, len));
      SYSCHECK_ERR_RETURN_NEG1(listen(socket_fd, 10));
    } catch (std::exception&) {
      SYSCHECK_ERR_RETURN_NEG1(close(socket_fd));
      throw;
    }
  }

  void remove() {
    struct stat file_stat;
    if (fstat(socket_fd, &file_stat) == 0)
      SYSCHECK_ERR_RETURN_NEG1(unlink(socket_path.c_str()));
  }

  ~ManagerServerSocket() override {
    unlink(socket_path.c_str());
  }

  ManagerSocket accept() {
    int client_fd;
    struct sockaddr_un addr;
    socklen_t addr_len = sizeof(addr);
    SYSCHECK_ERR_RETURN_NEG1(
        client_fd = ::accept(socket_fd, (struct sockaddr*)&addr, &addr_len));
    return ManagerSocket(client_fd);
  }

  std::string socket_path;
};

class ClientSocket : public Socket {
 public:
  explicit ClientSocket(const std::string& path) {
    try {
      struct sockaddr_un address = prepare_address(path.c_str());
      size_t len = address_length(address);
      SYSCHECK_ERR_RETURN_NEG1(
          connect(socket_fd, (struct sockaddr*)&address, len));
    } catch (std::exception&) {
      SYSCHECK_ERR_RETURN_NEG1(close(socket_fd));
      throw;
    }
  }

  void register_allocation(AllocInfo& info) {
    char buffer[3] = {0, 0, 0};
    send(&info, sizeof(info));
    recv(buffer, 2);
    TORCH_CHECK(
        strcmp(buffer, "OK") == 0,
        "Shared memory manager didn't respond with an OK");
  }

  void register_deallocation(AllocInfo& info) {
    send(&info, sizeof(info));
  }
};
