#include "InitMethodMulticast.hpp"
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

#define UID_LENGTH 60
#define MAX_MSG_LENGTH 4000

namespace thd {
namespace {

std::string getRandomString()
{
  int fd;
  unsigned int seed;
  SYSCHECK(fd = open("/dev/urandom", O_RDONLY));
  SYSCHECK(read(fd, &seed, sizeof(seed)));
  SYSCHECK(::close(fd));
  std::srand(seed);

  auto randchar = []() -> char {
    const char charset[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[std::rand() % max_index];
  };

  std::string str(UID_LENGTH, 0);
  std::generate_n(str.begin(), UID_LENGTH, randchar);
  return str;
}

struct MulticastMessage {
  std::string uid;
  std::string group_name;
  std::vector<std::string> addresses;
  port_type port;

  MulticastMessage(std::string group_name, port_type port)
    : MulticastMessage(getRandomString(), group_name, port) {}

  MulticastMessage(std::string msg) {
    std::array<std::string, 3> arr;
    std::size_t prev_pos = 0;
    for (std::size_t i = 0; i < 3; ++i) {
      auto next_sep_pos = msg.find_first_of(';', prev_pos);
      arr[i] = msg.substr(prev_pos, next_sep_pos - prev_pos);
      prev_pos = next_sep_pos + 1;
    }

    auto sep_pos = msg.rfind('#');
    if (sep_pos == std::string::npos)
      throw std::runtime_error("corrupted multicast message");

    std::vector<std::string> addresses;
    while (true) {
      auto next_sep_pos = msg.find(';', sep_pos + 1);
      if (next_sep_pos == std::string::npos) break;
      addresses.emplace_back(msg.substr(sep_pos + 1, next_sep_pos - sep_pos - 1));
      sep_pos = next_sep_pos;
    }


    uid = arr[0];
    group_name = arr[1];
    this->addresses = addresses;
    port = convertToPort(std::stoul(arr[2]));
  }

  std::string pack() {
    std::string packed_msg = uid + ";" + group_name + ";" + std::to_string(port) + ";#";
    for (const auto& address : addresses) {
      packed_msg += address + ";";
    }
    return packed_msg;
  }

private:
  MulticastMessage(std::string uid, std::string group_name, port_type port)
    : uid(getRandomString())
    , group_name(group_name)
    , addresses(getInterfaceAddresses())
    , port(port) {}
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
    return addr_bytes[0] == 0xff && addr_bytes[1] == 0x01;
  } else {
    throw std::invalid_argument("unsupported address family");
  }
}

int bindMulticastSocket(struct sockaddr* address, port_type port, struct sockaddr_storage *sock_addr, int timeout_sec = 1, int ttl = 1) {
  struct timeval timeout = {.tv_sec = timeout_sec, .tv_usec = 0};

  int socket, optval;
  SYSCHECK(socket = ::socket(address->sa_family, SOCK_DGRAM, 0));
  optval = 1; SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int)));
  optval = 1; SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(int)));

  if (address->sa_family == AF_INET) {
    struct sockaddr_in *sock_addr_ipv4 = reinterpret_cast<struct sockaddr_in*>(sock_addr);
    std::memset(sock_addr_ipv4, 0, sizeof(*sock_addr_ipv4));
    sock_addr_ipv4->sin_family = address->sa_family;
    sock_addr_ipv4->sin_addr.s_addr = INADDR_ANY;
    sock_addr_ipv4->sin_port = htons(port);

    SYSCHECK(::bind(socket, reinterpret_cast<struct sockaddr*>(sock_addr_ipv4), sizeof(*sock_addr_ipv4)));
    SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)));

    struct ip_mreq mreq;
    struct sockaddr_in* address_ipv4 = reinterpret_cast<struct sockaddr_in*>(address);
    mreq.imr_multiaddr = address_ipv4->sin_addr;
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    SYSCHECK(::setsockopt(socket, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)));
    SYSCHECK(::setsockopt(socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)));

    sock_addr_ipv4->sin_addr = address_ipv4->sin_addr;
  } else if (address->sa_family == AF_INET6) {
    struct sockaddr_in6 *addr_ipv6 = reinterpret_cast<struct sockaddr_in6*>(sock_addr);
    std::memset(addr_ipv6, 0, sizeof(*addr_ipv6));
    addr_ipv6->sin6_family = address->sa_family;
    addr_ipv6->sin6_addr = in6addr_any;
    addr_ipv6->sin6_port = htons(port);

    SYSCHECK(::bind(socket, reinterpret_cast<struct sockaddr*>(addr_ipv6), sizeof(*addr_ipv6)));
    SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)));

    struct ipv6_mreq mreq;
    struct sockaddr_in6* address_ipv6 = reinterpret_cast<struct sockaddr_in6*>(address);
    mreq.ipv6mr_multiaddr = address_ipv6->sin6_addr;
    mreq.ipv6mr_interface = 0;
    SYSCHECK(::setsockopt(socket, IPPROTO_IPV6, IPV6_MULTICAST_HOPS, &ttl, sizeof(ttl)));
    SYSCHECK(::setsockopt(socket, IPPROTO_IPV6, IPV6_JOIN_GROUP, &mreq, sizeof(mreq)));

    addr_ipv6->sin6_addr = address_ipv6->sin6_addr;
  }

  return socket;
}

} // anonymous namespace

InitMethodMulticast::InitMethodMulticast(std::string address, port_type port,
                                         rank_type world_size, std::string group_name)
 : _address(address)
 , _port(port)
 , _world_size(world_size)
 , _group_name(group_name)
{}

InitMethodMulticast::~InitMethodMulticast() {}

InitMethod::Config InitMethodMulticast::getConfig() {
  struct addrinfo hints = {0};
  hints.ai_family = AF_UNSPEC;
  struct addrinfo *res;
  if (getaddrinfo(_address.c_str(), std::to_string(_port).c_str(), &hints, &res)) {
    throw std::invalid_argument("invalid init address");
  }
  ResourceGuard<struct addrinfo*> res_guard([res]() { ::freeaddrinfo(res); });

  for (struct addrinfo *head = res; head != NULL; head = head->ai_next) {
    if (head->ai_family != AF_INET && head->ai_family != AF_INET6) continue;
    try {
      if (isMulticastAddress(head->ai_addr)) {
        return getMulticastConfig(head->ai_addr);
      } else {
        return getMasterConfig(head->ai_addr);
      }
    } catch (std::exception &e) {
      if (head->ai_next) continue;
      throw;
    }
  }
  throw std::runtime_error("failed to initialize THD using given address");
}

InitMethod::Config InitMethodMulticast::getMasterConfig(struct sockaddr* addr) {
  // TODO
  throw std::runtime_error("non-multicast tcp initialization not supported");
}

InitMethod::Config InitMethodMulticast::getMulticastConfig(struct sockaddr* addr) {
  InitMethod::Config config;
  struct sockaddr_storage sock_addr;
  int socket = bindMulticastSocket(addr, _port, &sock_addr);
  /* Multicast membership is dropped on close */
  ResourceGuard<int> socket_guard([socket]() { ::close(socket); });

  int listen_socket;
  port_type listen_port;
  std::tie(listen_socket, listen_port) = listen();
  MulticastMessage msg {_group_name, listen_port};

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
                sock_addr.ss_family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6)));
  };

  broadcast();

  while (processes.size() < _world_size) {
    try {
      SYSCHECK(::recv(socket, recv_message, sizeof(recv_message), 0));
      std::string recv_message_str(recv_message);

      /* We should ignore messages comming from different group */
      auto recv_msg = MulticastMessage(recv_message_str);
      if (recv_msg.group_name != _group_name) {
        continue;
      }

      processes.insert(recv_message_str); // set will automatically deduplicate messages
    } catch (const std::system_error& e) {
      /* Check if this was really a timeout from `recvfrom` or a different error. */
      if (errno != EAGAIN && errno != EWOULDBLOCK)
        throw;
    }

    broadcast();
  }

  // Just to make decrease the probability of packet loss deadlocking the system
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
          .world_size = _world_size,
          .listen_socket = listen_socket,
          .listen_port = master_msg.port,
        };

        config.public_address = discoverWorkers(listen_socket, _world_size);
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

} // namespace thd
