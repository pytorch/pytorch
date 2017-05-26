#include "InitMethodMulticast.hpp"

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

namespace thd {
namespace {

struct MulticastMessage {
  std::string uid;
  std::string group_name;
  std::string address;
  port_type port;

  static std::string pack(const MulticastMessage& msg) {
    return msg.uid + ";" + msg.group_name + ";" + msg.address + ";" + std::to_string(msg.port);
  }

  static MulticastMessage unpack(std::string msg) {
    std::array<std::string, 4> arr;
    std::size_t prev_pos = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      auto pos = msg.find_first_of(';', prev_pos + 1);
      arr[i] = msg.substr(prev_pos, pos);
      prev_pos = pos;
    }

    return {
      .uid = arr[0],
      .group_name = arr[1],
      .address = arr[2],
      .port = convertToPort(std::stoul(arr[3])),
    };
  }
};

struct attr {
  int socket;
  struct sockaddr ai_addr;
  socklen_t ai_addrlen;
};

std::string getRandomString()
{
  auto randchar = []() -> char {
    const char charset[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
  };

  std::string str(UID_LENGTH, 0);
  std::generate_n(str.begin(), UID_LENGTH, randchar);
  return str;
}

std::tuple<int, struct sockaddr, socklen_t> connectUDP(const std::string& address, port_type port, int timeout_sec = 10, int ttl = 1) {
  struct addrinfo hints, *res = NULL;
  struct timeval timeout = {.tv_sec = timeout_sec, .tv_usec = 0};

  std::memset(&hints, 0x00, sizeof(hints));
  hints.ai_flags = AI_NUMERICSERV; // specifies that port (service) is numeric
  hints.ai_family = AF_UNSPEC; // either IPv4 or IPv6
  hints.ai_socktype = SOCK_DGRAM; // UDP
  hints.ai_protocol = IPPROTO_UDP;

  int err = ::getaddrinfo(address.data(), std::to_string(port).data(), &hints, &res);
  if (err != 0 || !res) {
    throw std::invalid_argument("host not found: " + std::string(::gai_strerror(err)));
  }

  std::shared_ptr<struct addrinfo> addresses(res, [](struct addrinfo* p) {
    ::freeaddrinfo(p);
  });

  struct addrinfo *next_addr = addresses.get();
  int socket;
  while (true) {
    try {
      int optval = 1;
      SYSCHECK(socket = ::socket(next_addr->ai_family, next_addr->ai_socktype, next_addr->ai_protocol))
      SYSCHECK(::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int)))
      SYSCHECK(::bind(socket, next_addr->ai_addr, next_addr->ai_addrlen))
      SYSCHECK(::setsockopt (socket, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)))
      if (next_addr->ai_family == AF_INET) {
        struct ip_mreq mreq;
        SYSCHECK(::inet_pton(AF_INET, address.data(), &mreq.imr_multiaddr));
        SYSCHECK(::setsockopt(socket, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)))
        SYSCHECK(::setsockopt(socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)))
      } else if (next_addr->ai_family == AF_INET6) {
        struct ipv6_mreq mreq;
        SYSCHECK(::inet_pton(AF_INET6, address.data(), &mreq.ipv6mr_multiaddr))
        SYSCHECK(::setsockopt(socket, IPPROTO_IPV6, IPV6_MULTICAST_HOPS, &ttl, sizeof(ttl)))
        SYSCHECK(::setsockopt(socket, IPPROTO_IPV6, IPV6_JOIN_GROUP, &mreq, sizeof(mreq)))
      } else {
        throw std::system_error(EAFNOSUPPORT, std::generic_category());
      }


      break;
    } catch (const std::system_error& e) {
      ::close(socket);
      next_addr = next_addr->ai_next;

      // we have tried all addresses but could not connect to any of them
      if (!next_addr) {
        throw e;
      }

      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  /* Reduce the chance that we use socket before joining the group */
  std::this_thread::sleep_for(std::chrono::seconds(1));

  struct sockaddr ai_addr;
  memcpy(&ai_addr, next_addr->ai_addr, next_addr->ai_addrlen);
  return std::make_tuple(socket, ai_addr, next_addr->ai_addrlen);
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
  struct sockaddr addr;
  socklen_t addr_len;
  std::tie(socket, addr, addr_len) = connectUDP(_address, _port);

  int listen_socket;
  MulticastMessage msg;
  msg.uid = getRandomString();
  msg.group_name = _group_name;
  std::tie(listen_socket, msg.address, msg.port) = listen();

  std::string send_message = MulticastMessage::pack(msg);
  std::set<std::string> processes;
  processes.insert(send_message);

  char recv_message[UID_LENGTH + 100];
  SYSCHECK(::sendto(socket, send_message.c_str(), send_message.size(), 0, &addr, addr_len))

  while (processes.size() < _world_size) {
    try {
      SYSCHECK(::recvfrom(socket, recv_message, sizeof(recv_message), 0, &addr, &addr_len))
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
    SYSCHECK(::sendto(socket, send_message.c_str(), send_message.size(), 0, &addr, addr_len))
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
      } else {
        config.worker = {
          .address = master_msg.address,
          .listen_port = master_msg.port,
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
