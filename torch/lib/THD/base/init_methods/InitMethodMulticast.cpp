#include "InitMethodMulticast.hpp"
#include "InitMethodUtils.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <set>
#include <thread>
#include <cstring>

#define UID_LENGTH 100
#define MAX_MSG_LENGTH 1000

namespace thd {
namespace {

struct MulticastMessage {
  std::string uid;
  std::string group_name;
  std::vector<std::string> addresses;
  port_type port;

  static std::string pack(const MulticastMessage& msg) {
    std::string packed_msg = msg.uid + ";" + msg.group_name + ";" + std::to_string(msg.port) + ";#";
    for (const auto& address : msg.addresses) {
      packed_msg += address + ";";
    }

    return packed_msg;
  }

  static MulticastMessage unpack(std::string msg) {
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

    return {
      .uid = arr[0],
      .group_name = arr[1],
      .addresses = addresses,
      .port = convertToPort(std::stoul(arr[2])),
    };
  }
};

std::string getRandomString()
{
  std::srand(std::time(0));
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

static int getAddressIPVersion(const std::string& address) {
  char buf[16];
  if (inet_pton(AF_INET, address.c_str(), buf)) {
    return AF_INET;
  } else if (inet_pton(AF_INET6, address.c_str(), buf)) {
    return AF_INET6;
  } else {
    throw std::invalid_argument("invalid ip address");
  }
  return -1;
}

std::tuple<int, struct sockaddr> connectUDP(const std::string& address, port_type port, int timeout_sec = 10, int ttl = 1) {
  struct timeval timeout = {.tv_sec = timeout_sec, .tv_usec = 0};

  struct sockaddr addr;

  int version = getAddressIPVersion(address);

  int socket, optval;
  SYSCHECK(socket = ::socket(version, SOCK_DGRAM, 0))
  optval = 1; SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int)))
  optval = 1; SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(int)))

  if (version == AF_INET) {
    struct sockaddr_in addr_ipv4;
    std::memset(&addr_ipv4, 0x00, sizeof(addr_ipv4));
    addr_ipv4.sin_family = version;
    addr_ipv4.sin_addr.s_addr = INADDR_ANY;
    addr_ipv4.sin_port = htons(port);

    SYSCHECK(::bind(socket, reinterpret_cast<struct sockaddr*>(&addr_ipv4), sizeof(addr_ipv4)))
    SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)))

    struct ip_mreq mreq;
    SYSCHECK(::inet_pton(AF_INET, address.c_str(), &mreq.imr_multiaddr));
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    SYSCHECK(::setsockopt(socket, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)))
    SYSCHECK(::setsockopt(socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)))

    SYSCHECK(::inet_pton(AF_INET, address.c_str(), &addr_ipv4.sin_addr))
    addr = *reinterpret_cast<struct sockaddr*>(&addr_ipv4);
  } else if (version == AF_INET6) {
    struct sockaddr_in6 addr_ipv6;
    std::memset(&addr_ipv6, 0x00, sizeof(addr_ipv6));
    addr_ipv6.sin6_family = version;
    addr_ipv6.sin6_addr = in6addr_any;
    addr_ipv6.sin6_port = htons(port);

    SYSCHECK(::bind(socket, reinterpret_cast<struct sockaddr*>(&addr_ipv6), sizeof(addr_ipv6)))
    SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)))

    struct ipv6_mreq mreq;
    SYSCHECK(::inet_pton(AF_INET6, address.c_str(), &mreq.ipv6mr_multiaddr))
    mreq.ipv6mr_interface = 0;
    SYSCHECK(::setsockopt(socket, IPPROTO_IPV6, IPV6_MULTICAST_HOPS, &ttl, sizeof(ttl)))
    SYSCHECK(::setsockopt(socket, IPPROTO_IPV6, IPV6_JOIN_GROUP, &mreq, sizeof(mreq)))

    SYSCHECK(::inet_pton(AF_INET6, address.c_str(), &addr_ipv6.sin6_addr))
    addr = *reinterpret_cast<struct sockaddr*>(&addr_ipv6);
  }

  /* Reduce the chance that we use socket before joining the group */
  std::this_thread::sleep_for(std::chrono::seconds(1));

  return std::make_tuple(socket, addr);
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
  InitMethod::Config config;

  int socket;
  struct sockaddr send_addr;
  std::tie(socket, send_addr) = connectUDP(_address, _port);

  int listen_socket;
  MulticastMessage msg;
  msg.uid = getRandomString();
  msg.group_name = _group_name;
  msg.addresses = getInterfaceAddresses();
  std::tie(listen_socket, msg.port) = listen();

  std::string send_message = MulticastMessage::pack(msg);
  std::set<std::string> processes;
  processes.insert(send_message);

  // TODO: check if send_message is not to long

  char recv_message[MAX_MSG_LENGTH];
  SYSCHECK(::sendto(socket, send_message.c_str(), send_message.size(), 0, &send_addr, sizeof(send_addr)))

  while (processes.size() < _world_size) {
    try {
      SYSCHECK(::recvfrom(socket, recv_message, sizeof(recv_message), 0, nullptr, nullptr))
      std::string recv_message_str(recv_message);

      /* We should ignore messages comming from different group */
      auto recv_msg = MulticastMessage::unpack(recv_message_str);
      if (recv_msg.group_name != _group_name)
        continue;

      processes.insert(recv_message_str); // set will automatically delete duplicates
    } catch (const std::system_error& e) {
      /* Check if this was really a timeout from `recvfrom` or some different error. */
      if (errno != EAGAIN && errno != EWOULDBLOCK)
        throw e;
    }

    /* Even timeout has occured we want to resend message (just to be sure). */
    SYSCHECK(::sendto(socket, send_message.c_str(), send_message.size(), 0, &send_addr, sizeof(send_addr)))
  }

  /*
   * To ensure that all other processes have received our message,
   * send it again few times.
   */
  for (std::size_t i = 0; i < 10; ++i) {
    SYSCHECK(::sendto(socket, send_message.c_str(), send_message.size(), 0, &send_addr, sizeof(send_addr)))
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  MulticastMessage master_msg;
  std::size_t rank = 0;
  for (auto it = processes.begin(); it != processes.end(); ++it, ++rank) {
    MulticastMessage process_msg = MulticastMessage::unpack(*it);
    if (rank == 0) { // remember master message
      master_msg = process_msg;
    }

    if (msg.uid == process_msg.uid) {
      config.rank = rank;
      if (config.rank == 0) {
        config.master = {
          .world_size = _world_size,
          .listen_socket = listen_socket,
          .listen_port = master_msg.port,
        };

        discoverWorkers(listen_socket, _world_size);
      } else {
        std::string master_address = discoverMaster(master_msg.addresses, master_msg.port);
        config.worker = {
          .address = master_address,
          .port = master_msg.port,
        };
      }
      break;
    }
  }

  /* On close multicast membership is dropped, so no need to do it manually */
  ::close(socket);

  return config;
}

} // namespace thd
