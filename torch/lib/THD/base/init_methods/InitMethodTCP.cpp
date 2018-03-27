#include "InitMethod.hpp"
#include "InitMethodUtils.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
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

constexpr size_t num_rand_bytes = 32;
constexpr size_t max_msg_length = 4000;

namespace thd {
namespace init {
namespace {

std::string getRandomString()
{
  static constexpr char charset[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";
  int fd;
  uint8_t rand_bytes[num_rand_bytes];
  ssize_t bytes_read;
  SYSCHECK(fd = open("/dev/urandom", O_RDONLY));
  SYSCHECK(bytes_read = read(fd, &rand_bytes, sizeof(rand_bytes)));
  if (bytes_read != sizeof(rand_bytes))
    throw std::runtime_error("failed to read from /dev/urandom");
  SYSCHECK(::close(fd));

  std::string str;
  str.reserve(num_rand_bytes);
  for (uint8_t *byte = rand_bytes; byte != rand_bytes + num_rand_bytes; ++byte) {
    str.push_back(charset[(*byte) % (sizeof(charset) - 1)]);
  }
  return str;
}

struct MulticastMessage {
  std::string uid;
  std::string group_name;
  std::vector<std::string> addresses;
  port_type port;
  int rank;

  MulticastMessage(std::string group_name, port_type port, int rank)
    : uid(getRandomString())
    , group_name(group_name)
    , addresses(getInterfaceAddresses())
    , port(port)
    , rank(rank) {}

  MulticastMessage(std::string msg) {
    std::istringstream ss {msg};
    ss >> uid >> group_name >> port >> rank;
    addresses = {std::istream_iterator<std::string>(ss),
                 std::istream_iterator<std::string>()};
  }

  std::string pack() {
    std::ostringstream ss;
    ss << uid << ' ' << group_name << ' ' << port << ' ' << rank;
    for (const auto& address : addresses) {
      ss << ' ' << address;
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

int bindMulticastSocket(struct sockaddr* address, struct sockaddr_storage *sock_addr,
                        int timeout_sec = 1, int ttl = 1) {
  struct timeval timeout = {.tv_sec = timeout_sec, .tv_usec = 0};

  int socket, optval;
  SYSCHECK(socket = ::socket(address->sa_family, SOCK_DGRAM, 0));
  optval = 1; SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int)));

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

// messages
std::vector<MulticastMessage> getMessages(struct sockaddr* addr, rank_type world_size,
                                          std::string group_name, std::string packed_msg) {
  struct sockaddr_storage sock_addr;
  int socket = bindMulticastSocket(addr, &sock_addr);
  // NOTE: Multicast membership is dropped on close
  ResourceGuard socket_guard([socket]() { ::close(socket); });

  std::set<std::string> msgs = {packed_msg};

  char recv_message[max_msg_length];
  if (packed_msg.length() + 1 > max_msg_length) {
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
  while (msgs.size() < world_size) {
    try {
      SYSCHECK(::recv(socket, recv_message, sizeof(recv_message), 0));
      std::string recv_message_str(recv_message);

      if (recv_message_str == packed_msg) continue; // ignore multicast loopback

      // We should ignore messages coming from different group
      auto recv_msg = MulticastMessage(recv_message_str);
      if (recv_msg.group_name != group_name) {
        continue;
      }

      msgs.insert(recv_message_str); // set will automatically deduplicate messages
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

  std::vector<MulticastMessage> unpacked_msgs;
  for (auto& msg : msgs) {
    unpacked_msgs.emplace_back(msg);
  }

  return unpacked_msgs;
}


InitMethod::Config initTCPMaster(std::string address, std::string str_port,
                                 rank_type world_size, int assigned_rank) {
  InitMethod::Config config;
  if (assigned_rank == -1) {
    throw std::invalid_argument("tcp:// method with non-multicast addresses "
                                "requires manual rank assignment");
  }

  config.rank = convertToRank(assigned_rank);
  config.world_size = world_size;
  auto port = convertToPort(std::stoul(str_port));
  if (config.rank == 0) {
    config.master.listen_port = port;
    std::tie(config.master.listen_socket, std::ignore) = listen(port);
    config.public_address = discoverWorkers(config.master.listen_socket, world_size);
  } else {
    config.worker.master_addr = address;
    config.worker.master_port = port;
    std::tie(std::ignore, config.public_address) = discoverMaster({address}, port);
  }

  return config;
}

InitMethod::Config initTCPMulticast(std::string group_name, rank_type world_size,
                                    int assigned_rank, struct sockaddr* addr) {
  InitMethod::Config config;

  int listen_socket;
  port_type listen_port;
  std::tie(listen_socket, listen_port) = listen();
  ResourceGuard listen_socket_guard([listen_socket]() { ::close(listen_socket); });

  MulticastMessage msg {group_name, listen_port, assigned_rank};
  std::string packed_msg = msg.pack();

  std::vector<MulticastMessage> msgs = getMessages(addr, world_size, group_name,
                                                   packed_msg);

  std::vector<MulticastMessage*> sorted_msgs(msgs.size());

  // Pre-fill sorted_msgs with processes that had their ranks assigned manually
  for (auto& msg : msgs) {
    if (msg.rank >= 0) {
      if (sorted_msgs[msg.rank] != nullptr)
        throw std::logic_error("more than one node have assigned same rank");
      sorted_msgs[msg.rank] = &msg;
    }
  }

  // NOTE: msgs are already sorted lexicographically, so we can greedily
  // insert them into free slots
  std::size_t free_pos = 0;
  for (auto& msg : msgs) {
    if (msg.rank >= 0) continue; // These were sorted in the previous loop
    while (sorted_msgs[free_pos] != nullptr) free_pos++;
    sorted_msgs[free_pos] = &msg;
  }

  auto& master_msg = *sorted_msgs[0];
  for (std::size_t rank = 0; rank < sorted_msgs.size(); ++rank) {
    if (packed_msg == sorted_msgs[rank]->pack()) {
      config.rank = rank;
      config.world_size = world_size;
      if (config.rank == 0) {
        listen_socket_guard.release();
        config.master = {
          .listen_socket = listen_socket,
          .listen_port = master_msg.port,
        };

        config.public_address = discoverWorkers(listen_socket, world_size);
      } else {
        std::string master_address;
        std::tie(master_address, config.public_address) =
          discoverMaster(master_msg.addresses, master_msg.port);
        config.worker = {
          .master_addr = master_address,
          .master_port = master_msg.port,
        };
      }
      break;
    }
  }

  return config;
}


} // anonymous namespace

InitMethod::Config initTCP(std::string argument, rank_type world_size,
                           std::string group_name, int rank) {
  // Parse arguments
  std::string address, str_port;
  std::tie(address, str_port) = splitAddress(argument);

  // Resolve addr and select init method
  struct addrinfo hints = {0};
  hints.ai_family = AF_UNSPEC;
  struct addrinfo *res;
  if (::getaddrinfo(address.c_str(), str_port.c_str(), &hints, &res)) {
    throw std::invalid_argument("invalid init address");
  }
  ResourceGuard res_guard([res]() { ::freeaddrinfo(res); });

  for (struct addrinfo *head = res; head != NULL; head = head->ai_next) {
    if (head->ai_family != AF_INET && head->ai_family != AF_INET6) continue;
    try {
      if (isMulticastAddress(head->ai_addr)) {
        return initTCPMulticast(group_name, world_size, rank, head->ai_addr);
      } else {
        return initTCPMaster(address, str_port, world_size, rank);
      }
    } catch (std::exception &e) {
      if (!head->ai_next) throw;
    }
  }
  throw std::runtime_error("failed to initialize THD using given address");
}

} // namespace init
} // namespace thd
