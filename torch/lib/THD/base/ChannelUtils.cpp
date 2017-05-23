#include "ChannelUtils.hpp"
#include "ChannelEnvVars.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/poll.h>
#include <unistd.h>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

namespace thd {
namespace {

constexpr int LISTEN_QUEUE_SIZE = 64;


void setSocketNoDelay(int socket) {
  int flag = 1;
  socklen_t optlen = sizeof(flag);
  SYSCHECK(setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, optlen));
}

} // anonymous namespace

const char* must_getenv(const char* env) {
  const char* value = std::getenv(env);
  if (value == nullptr) {
    throw std::logic_error(std::string("") + "failed to read the " + env +
        " environmental variable; maybe you forgot to set it properly?");
  }
  return value;
}

std::tuple<int, std::string, port_type> listen(port_type port) {
  struct addrinfo hints, *res = NULL;

  std::memset(&hints, 0x00, sizeof(hints));
  hints.ai_flags = AI_PASSIVE | AI_ADDRCONFIG;
  hints.ai_family = AF_UNSPEC; // either IPv4 or IPv6
  hints.ai_socktype = SOCK_STREAM; // TCP

  // `getaddrinfo` will sort addresses according to RFC 3484 and can be tweeked
  // by editing `/etc/gai.conf`. so there is no need to manual sorting
  // or protocol preference.
  int err = ::getaddrinfo(NULL, std::to_string(port).data(), &hints, &res);
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

      // we have tried all addresses but could not establish listening on any of them
      if (!next_addr) {
        throw e;
      }
    }
  }

  // get listen port and address
  char address[INET6_ADDRSTRLEN];
  port_type listen_port;
  if (next_addr->ai_family == AF_INET) {
    struct sockaddr_in *addr = reinterpret_cast<struct sockaddr_in*>(next_addr->ai_addr);
    SYSCHECK(::inet_ntop(next_addr->ai_family, &(addr->sin_addr), address, sizeof(address)));
    listen_port = ntohs(addr->sin_port);
  } else { // AF_INET6
    struct sockaddr_in6 *addr = reinterpret_cast<struct sockaddr_in6*>(next_addr->ai_addr);
    SYSCHECK(::inet_ntop(next_addr->ai_family, &(addr->sin6_addr), address, sizeof(address)));
    listen_port = ntohs(addr->sin6_port);
  }

  return std::make_tuple(socket, std::string(address), listen_port);
}


int connect(const std::string& address, port_type port, bool wait) {
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
      SYSCHECK(::connect(socket, next_addr->ai_addr, next_addr->ai_addrlen))
      break;
    } catch (const std::system_error& e) {
      // if `connect` fails, the state of the socket is unspecified.
      // we should close the socket and create a new one before attempting to reconnect.
      ::close(socket);
      if (errno == ECONNREFUSED) any_refused = true;

      // we need to move to next address because this was not available
      // to connect or to create socket
      next_addr = next_addr->ai_next;

      // we have tried all addresses but could not connect to any of them
      if (!next_addr) {
        if (!wait || !any_refused) throw e;
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

  int res;
  SYSCHECK(res = ::poll(events.get(), 1, timeout))
  if (res == 0) {
    throw std::runtime_error("waiting for processes to connect has timed out");
  } else {
    if (!(events[0].revents & POLLIN))
      throw std::system_error(ECONNABORTED, std::system_category());
  }

  int socket;
  SYSCHECK(socket = ::accept(listen_socket, NULL, NULL))

  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  char address[INET6_ADDRSTRLEN + 1];

  SYSCHECK(::getpeername(socket, reinterpret_cast<struct sockaddr*>(&addr), &addr_len))

  if (addr.ss_family == AF_INET) {
    struct sockaddr_in *s = reinterpret_cast<struct sockaddr_in*>(&addr);
    SYSCHECK(::inet_ntop(AF_INET, &(s->sin_addr), address, INET_ADDRSTRLEN))
    address[INET_ADDRSTRLEN] = '\0';
  } else {
    struct sockaddr_in6 *s = reinterpret_cast<struct sockaddr_in6*>(&addr);
    SYSCHECK(::inet_ntop(AF_INET6, &(s->sin6_addr), address, INET6_ADDRSTRLEN))
    address[INET6_ADDRSTRLEN] = '\0';
  }

  setSocketNoDelay(socket);

  return std::make_tuple(socket, std::string(address));
}

std::tuple<port_type, rank_type> load_master_env() {
  auto port = convertToPort(std::stoul(getenv(MASTER_PORT_ENV)));

  rank_type world_size = std::stoul(getenv(WORLD_SIZE_ENV));
  if (world_size == 0)
    throw std::domain_error(std::string(WORLD_SIZE_ENV) + " env variable cannot be 0");

  return std::make_tuple(port, world_size);
}


std::tuple<std::string, port_type> load_worker_env() {
  std::string full_address = std::string(getenv(MASTER_ADDR_ENV));
  auto found_pos = full_address.rfind(":");
  if (found_pos == std::string::npos)
    throw std::domain_error("invalid master address, usage: IP:PORT | HOSTNAME:PORT");

  std::string str_port = full_address.substr(found_pos + 1);
  auto port = convertToPort(std::stoul(str_port));
  return std::make_tuple(full_address.substr(0, found_pos), port);
}

rank_type load_rank_env() {
  return convertToRank(std::stol(getenv(RANK_ENV)));
}

rank_type load_world_size_env() {
  return convertToRank(std::stol(getenv(WORLD_SIZE_ENV)));
}

} // namespace thd
