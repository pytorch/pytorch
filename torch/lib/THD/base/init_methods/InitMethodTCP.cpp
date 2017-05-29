#include "InitMethod.hpp"
#include "InitMethodUtils.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <set>
#include <thread>
#include <cstring>
#include <random>
#include <sstream>
#include <iterator>

#define UID_LENGTH 60
#define MAX_MSG_LENGTH 4000

namespace thd {
namespace init {
namespace {

std::string getRandomString()
{
  static constexpr char charset[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";
  constexpr std::size_t max_index = (sizeof(charset) - 1);

  int fd;
  unsigned int seed;
  SYSCHECK(fd = open("/dev/urandom", O_RDONLY));
  SYSCHECK(read(fd, &seed, sizeof(seed)));
  SYSCHECK(::close(fd));
  std::mt19937 prng {seed};

  std::string str(UID_LENGTH, 0);
  for (std::size_t i = 0; i < UID_LENGTH; ++i) {
    str[i] = charset[prng() % max_index];
  }
  return str;
}

struct MulticastMessage {
  std::string uid;
  std::string group_name;
  std::vector<std::string> addresses;
  port_type port;

  MulticastMessage(std::string group_name, port_type port)
    : uid(getRandomString())
    , group_name(group_name)
    , addresses(getInterfaceAddresses())
    , port(port) {}

  MulticastMessage(std::string msg) {
    std::istringstream ss {msg};
    ss >> uid >> group_name >> port;
    addresses = {std::istream_iterator<std::string>(ss),
                 std::istream_iterator<std::string>()};
  }

  std::string pack() {
    std::ostringstream ss;
    ss << uid << ' ' << group_name << ' ' << port << ' ';
    for (const auto& address : addresses) {
      ss << address << ' ';
    }
    return ss.str();
  }
};

bool isMulticastAddress(struct sockaddr* address) {
  if (address->sa_family == AF_INET) {
    struct sockaddr_in *address_ipv4 = reinterpret_cast<struct sockaddr_in*>(address);
    uint32_t host_addr = ntohl(address_ipv4->sin_addr.s_addr);
    return (host_addr & 0xF0000000) == 0xE0000000;
  } else if (address->sa_family == AF_INET6) {
    struct sockaddr_in6 *address_ipv6 = reinterpret_cast<struct sockaddr_in6*>(address);
    auto& addr_bytes = address_ipv6->sin6_addr.s6_addr;
    // NOTE: address is in network byte order
    return addr_bytes[0] == 0xff;
  } else {
    throw std::invalid_argument("unsupported address family");
  }
}

int bindMulticastSocket(struct sockaddr* address, struct sockaddr_storage *sock_addr, int timeout_sec = 1, int ttl = 1) {
  struct timeval timeout = {.tv_sec = timeout_sec, .tv_usec = 0};

  int socket, optval;
  SYSCHECK(socket = ::socket(address->sa_family, SOCK_DGRAM, 0));
  optval = 1; SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int)));
  optval = 1; SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(int)));

  if (address->sa_family == AF_INET) {
    struct sockaddr_in *sock_addr_ipv4 = reinterpret_cast<struct sockaddr_in*>(sock_addr);
    struct sockaddr_in *address_ipv4 = reinterpret_cast<struct sockaddr_in*>(address);
    std::memset(sock_addr_ipv4, 0, sizeof(*sock_addr_ipv4));
    sock_addr_ipv4->sin_family = address->sa_family;
    sock_addr_ipv4->sin_addr.s_addr = INADDR_ANY;
    sock_addr_ipv4->sin_port = address_ipv4->sin_port;

    SYSCHECK(::bind(socket, reinterpret_cast<struct sockaddr*>(sock_addr_ipv4), sizeof(*sock_addr_ipv4)));
    SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)));

    struct ip_mreq mreq;
    mreq.imr_multiaddr = address_ipv4->sin_addr;
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    SYSCHECK(::setsockopt(socket, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)));
    SYSCHECK(::setsockopt(socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)));

    sock_addr_ipv4->sin_addr = address_ipv4->sin_addr;
  } else if (address->sa_family == AF_INET6) {
    struct sockaddr_in6 *sock_addr_ipv6 = reinterpret_cast<struct sockaddr_in6*>(sock_addr);
    struct sockaddr_in6 *address_ipv6 = reinterpret_cast<struct sockaddr_in6*>(address);
    std::memset(sock_addr_ipv6, 0, sizeof(*sock_addr_ipv6));
    sock_addr_ipv6->sin6_family = address->sa_family;
    sock_addr_ipv6->sin6_addr = in6addr_any;
    sock_addr_ipv6->sin6_port = address_ipv6->sin6_port;

    SYSCHECK(::bind(socket, reinterpret_cast<struct sockaddr*>(sock_addr_ipv6), sizeof(*sock_addr_ipv6)));
    SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)));

    struct ipv6_mreq mreq;
    mreq.ipv6mr_multiaddr = address_ipv6->sin6_addr;
    mreq.ipv6mr_interface = 0;
    SYSCHECK(::setsockopt(socket, IPPROTO_IPV6, IPV6_MULTICAST_HOPS, &ttl, sizeof(ttl)));
    SYSCHECK(::setsockopt(socket, IPPROTO_IPV6, IPV6_JOIN_GROUP, &mreq, sizeof(mreq)));

    sock_addr_ipv6->sin6_addr = address_ipv6->sin6_addr;
  }

  return socket;
}

InitMethod::Config initTCPMaster(struct sockaddr* addr) {
  // TODO
  throw std::runtime_error("non-multicast tcp initialization not supported");
}

InitMethod::Config initTCPMulticast(std::string group_name, rank_type world_size, struct sockaddr* addr) {
  InitMethod::Config config;
  struct sockaddr_storage sock_addr;
  int socket = bindMulticastSocket(addr, &sock_addr);
  // NOTE: Multicast membership is dropped on close
  ResourceGuard socket_guard([socket]() { ::close(socket); });

  int listen_socket;
  port_type listen_port;
  std::tie(listen_socket, listen_port) = listen();
  MulticastMessage msg {group_name, listen_port};

  std::string packed_msg = msg.pack();
  std::set<std::string> processes;
  processes.insert(packed_msg);

  char recv_message[MAX_MSG_LENGTH];
  if (packed_msg.length() + 1 > MAX_MSG_LENGTH) {
    throw std::logic_error("message too long for multicast init");
  }

  auto broadcast = [socket, &sock_addr, &packed_msg]() {
    SYSCHECK(::sendto(socket, packed_msg.c_str(), packed_msg.size() + 1, 0,
                reinterpret_cast<struct sockaddr*>(&sock_addr),
                sock_addr.ss_family == AF_INET
                    ? sizeof(struct sockaddr_in)
                    : sizeof(struct sockaddr_in6)));
  };

  broadcast();

  // Wait for messages from all processes
  while (processes.size() < world_size) {
    try {
      SYSCHECK(::recv(socket, recv_message, sizeof(recv_message), 0));
      std::string recv_message_str(recv_message);

      // We should ignore messages comming from different group
      auto recv_msg = MulticastMessage(recv_message_str);
      if (recv_msg.group_name != group_name) {
        continue;
      }

      processes.insert(recv_message_str); // set will automatically deduplicate messages
    } catch (const std::system_error& e) {
      // Check if this was really a timeout from `recvfrom` or a different error.
      if (errno != EAGAIN && errno != EWOULDBLOCK)
        throw;
    }

    broadcast();
  }

  // Just to decrease the probability of packet loss deadlocking the system
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  broadcast();

  auto master_msg = MulticastMessage(*processes.begin());
  std::size_t rank = 0;
  for (auto it = processes.begin(); it != processes.end(); ++it, ++rank) {
    auto packed_recv_msg = *it;
    auto recv_msg = MulticastMessage(packed_recv_msg);

    if (packed_msg == packed_recv_msg) {
      config.rank = rank;
      if (config.rank == 0) {
        config.master = {
          .world_size = world_size,
          .listen_socket = listen_socket,
          .listen_port = master_msg.port,
        };

        config.public_address = discoverWorkers(listen_socket, world_size);
      } else {
        std::string master_address;
        std::tie(master_address, config.public_address) = discoverMaster(master_msg.addresses, master_msg.port);
        config.worker = {
          .address = master_address,
          .port = master_msg.port,
        };
      }
      break;
    }
  }

  return config;
}


} // anonymous namespace

InitMethod::Config initTCP(std::string argument, rank_type world_size, std::string group_name) {
  // Parse arguments
  std::string address, str_port;
  std::tie(address, str_port) = splitAddress(argument);

  // Resolve addr and select init method
  struct addrinfo hints = {0};
  hints.ai_family = AF_UNSPEC;
  struct addrinfo *res;
  if (getaddrinfo(address.c_str(), str_port.c_str(), &hints, &res)) {
    throw std::invalid_argument("invalid init address");
  }
  ResourceGuard res_guard([res]() { ::freeaddrinfo(res); });

  for (struct addrinfo *head = res; head != NULL; head = head->ai_next) {
    if (head->ai_family != AF_INET && head->ai_family != AF_INET6) continue;
    try {
      if (isMulticastAddress(head->ai_addr)) {
        return initTCPMulticast(group_name, world_size, head->ai_addr);
      } else {
        return initTCPMaster(head->ai_addr);
      }
    } catch (std::exception &e) {
      if (!head->ai_next) throw;
    }
  }
  throw std::runtime_error("failed to initialize THD using given address");
}

} // namespace init
} // namespace thd
