#include "ChannelUtils.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/poll.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <algorithm>

namespace thd {
namespace {

constexpr int LISTEN_QUEUE_SIZE = 64;

void setSocketNoDelay(int socket) {
  int flag = 1;
  socklen_t optlen = sizeof(flag);
  SYSCHECK(setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, optlen));
}

port_type getSocketPort(int fd) {
  port_type listen_port;
  struct sockaddr_storage addr_storage;
  socklen_t addr_len = sizeof(addr_storage);
  SYSCHECK(getsockname(fd, reinterpret_cast<struct sockaddr*>(&addr_storage), &addr_len));
  if (addr_storage.ss_family == AF_INET) {
    struct sockaddr_in *addr = reinterpret_cast<struct sockaddr_in*>(&addr_storage);
    listen_port = ntohs(addr->sin_port);
  } else if (addr_storage.ss_family == AF_INET6) { // AF_INET6
    struct sockaddr_in6 *addr = reinterpret_cast<struct sockaddr_in6*>(&addr_storage);
    listen_port = ntohs(addr->sin6_port);
  } else {
    throw std::runtime_error("unsupported protocol");
  }
  return listen_port;
}

} // anonymous namespace

std::pair<std::string, std::string> splitAddress(const std::string &addr) {
  std::string host, port;
  auto num_colons = std::count(addr.begin(), addr.end(), ':');
  if (num_colons > 1) {
    // IPv6
    auto end_pos = addr.find(']');
    if (addr[0] != '[' || end_pos == std::string::npos) {
      throw std::invalid_argument("IPv6 address in an incorrect format (maybe you forgot to add [ ])");
    }
    host = addr.substr(1, end_pos - 1);
    port = addr.substr(end_pos + 2);
  } else if (num_colons == 1) {
    // IPv4 or HOSTNAME:PORT
    auto sep_pos = addr.find(':');
    host = addr.substr(0, sep_pos);
    port = addr.substr(sep_pos + 1);
  } else {
    throw std::invalid_argument("expected an address in format IP:PORT or HOSTNAME:PORT");
  }
  if (addr == "" || port == "") {
    throw std::invalid_argument("expected an address in format IP:PORT");
  }
  return std::make_pair(host, port);
}

std::string sockaddrToString(struct sockaddr *addr) {
  char address[INET6_ADDRSTRLEN + 1];
  if (addr->sa_family == AF_INET) {
    struct sockaddr_in *s = reinterpret_cast<struct sockaddr_in*>(addr);
    SYSCHECK(::inet_ntop(AF_INET, &(s->sin_addr), address, INET_ADDRSTRLEN))
    address[INET_ADDRSTRLEN] = '\0';
  } else if (addr->sa_family == AF_INET6) {
    struct sockaddr_in6 *s = reinterpret_cast<struct sockaddr_in6*>(addr);
    SYSCHECK(::inet_ntop(AF_INET6, &(s->sin6_addr), address, INET6_ADDRSTRLEN))
    address[INET6_ADDRSTRLEN] = '\0';
  } else {
    throw std::runtime_error("unsupported protocol");
  }
  return address;
}

std::pair<int, port_type> listen(port_type port) {
  struct addrinfo hints, *res = NULL;

  std::memset(&hints, 0x00, sizeof(hints));
  hints.ai_flags = AI_PASSIVE | AI_ADDRCONFIG;
  hints.ai_family = AF_UNSPEC; // either IPv4 or IPv6
  hints.ai_socktype = SOCK_STREAM; // TCP

  // `getaddrinfo` will sort addresses according to RFC 3484 and can be tweeked
  // by editing `/etc/gai.conf`. so there is no need to manual sorting
  // or protocol preference.
  int err = ::getaddrinfo(nullptr, std::to_string(port).data(), &hints, &res);
  if (err != 0 || !res) {
    throw std::invalid_argument("cannot find host to listen on: " + std::string(gai_strerror(err)));
  }

  std::shared_ptr<struct addrinfo> addresses(res, [](struct addrinfo* p) {
    ::freeaddrinfo(p);
  });

  struct addrinfo *next_addr = addresses.get();
  int socket;
  while (true) {
    try {
      SYSCHECK(socket = ::socket(next_addr->ai_family, next_addr->ai_socktype, next_addr->ai_protocol))

      int optval = 1;
      SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int)))
      SYSCHECK(::bind(socket, next_addr->ai_addr, next_addr->ai_addrlen))
      SYSCHECK(::listen(socket, LISTEN_QUEUE_SIZE))
      break;
    } catch (const std::system_error& e) {
      ::close(socket);
      next_addr = next_addr->ai_next;

      // we have tried all addresses but could not start listening on any of them
      if (!next_addr) {
        throw;
      }
    }
  }

  // get listen port and address
  return {socket, getSocketPort(socket)};
}


int connect(const std::string& address, port_type port, bool wait, int timeout) {
  struct addrinfo hints, *res = NULL;

  std::memset(&hints, 0x00, sizeof(hints));
  hints.ai_flags = AI_NUMERICSERV; // specifies that port (service) is numeric
  hints.ai_family = AF_UNSPEC; // either IPv4 or IPv6
  hints.ai_socktype = SOCK_STREAM; // TCP

  // `getaddrinfo` will sort addresses according to RFC 3484 and can be tweeked
  // by editing `/etc/gai.conf`. so there is no need to manual sorting
  // or protcol preference.
  int err = ::getaddrinfo(address.data(), std::to_string(port).data(), &hints, &res);
  if (err != 0 || !res) {
    throw std::invalid_argument("host not found: " + std::string(gai_strerror(err)));
  }

  std::shared_ptr<struct addrinfo> addresses(res, [](struct addrinfo* p) {
    ::freeaddrinfo(p);
  });

  struct addrinfo *next_addr = addresses.get();
  int socket;
  // we'll loop over the addresses only if at least of them gave us ECONNREFUSED.
  // Maybe the host was up, but the server wasn't running.
  bool any_refused = false;
  while (true) {
    try {
      SYSCHECK(socket = ::socket(next_addr->ai_family, next_addr->ai_socktype, next_addr->ai_protocol))
      ResourceGuard socket_guard([socket]() { ::close(socket); });

      // We need to connect in non-blocking mode, so we can use a timeout
      SYSCHECK(fcntl(socket, F_SETFL, O_NONBLOCK));

      int ret = ::connect(socket, next_addr->ai_addr, next_addr->ai_addrlen);
      if (ret != 0 && errno != EINPROGRESS)
        throw std::system_error(errno, std::system_category());

      struct pollfd pfd;
      pfd.fd = socket;
      pfd.events = POLLOUT;

      int num_ready = poll(&pfd, 1, timeout);
      if (num_ready < 0) {
        throw std::system_error(errno, std::system_category());
      } else if (num_ready == 0) {
        errno = 0;
        throw std::runtime_error("connect() timed out");
      }

      socklen_t err_len = sizeof(errno);
      SYSCHECK(getsockopt(socket, SOL_SOCKET, SO_ERROR, &errno, &err_len));
      if (errno != 0) {
        throw std::system_error(errno, std::system_category());
      }

      // Disable non-blocking mode
      int flags;
      SYSCHECK(flags = fcntl(socket, F_GETFL));
      SYSCHECK(fcntl(socket, F_SETFL, flags & (~O_NONBLOCK)));
      socket_guard.release();
      break;
    } catch (std::exception& e) {
      // if `connect` fails, the state of the socket is unspecified.
      // we should close the socket and create a new one before attempting to reconnect.
      ::close(socket);
      if (errno == ECONNREFUSED) any_refused = true;

      // we need to move to next address because this was not available
      // to connect or to create socket
      next_addr = next_addr->ai_next;

      // we have tried all addresses but could not connect to any of them
      if (!next_addr) {
        if (!wait || !any_refused) throw;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        any_refused = false;
        next_addr = addresses.get();
      }
    }
  }

  setSocketNoDelay(socket);

  return socket;
}

std::tuple<int, std::string> accept(int listen_socket, int timeout) {
  // poll on listen socket, it allows to make timeout
  std::unique_ptr<struct pollfd[]> events(new struct pollfd[1]);
  events[0] = {.fd = listen_socket, .events = POLLIN};

  while (true) {
    int res = ::poll(events.get(), 1, timeout);
    if (res == 0) {
      throw std::runtime_error("waiting for processes to connect has timed out");
    } else if (res == -1) {
      if (errno == EINTR) {
        continue;
      }
      throw std::system_error(errno, std::system_category());
    } else {
      if (!(events[0].revents & POLLIN))
        throw std::system_error(ECONNABORTED, std::system_category());
      break;
    }
  }

  int socket;
  SYSCHECK(socket = ::accept(listen_socket, NULL, NULL))

  // Get address of the connecting process
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  SYSCHECK(::getpeername(socket, reinterpret_cast<struct sockaddr*>(&addr), &addr_len))

  setSocketNoDelay(socket);

  return std::make_tuple(socket, sockaddrToString(reinterpret_cast<struct sockaddr*>(&addr)));
}

} // namespace thd
