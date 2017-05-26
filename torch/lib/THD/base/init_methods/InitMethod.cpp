#include "InitMethod.hpp"
#include "InitMethodEnv.hpp"
#include "InitMethodFile.hpp"
#include "InitMethodMulticast.hpp"

#include <sys/types.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <ifaddrs.h>

namespace thd {

InitMethod::Config getInitConfig(std::string argument, int world_size,
                                 std::string group_name) {
  if (argument.find("tcp") == 0) { // multicast tcp
    argument.erase(0, 6); // chop: "tcp://"

    auto found_pos = argument.rfind(":");
    if (found_pos == std::string::npos)
      throw std::invalid_argument("invalid multicast address, usage: IP:PORT | HOSTNAME:PORT");

    rank_type r_world_size;
    try {
      r_world_size = convertToRank(world_size);
    } catch(std::exception& e) {
      throw std::invalid_argument("invalid world_size");
    }

    return InitMethodMulticast(
      argument.substr(0, found_pos), // address
      convertToPort(std::stoul(argument.substr(found_pos + 1))), // port
      r_world_size, // world size
      group_name
    ).getConfig();
  } else if (argument.find("file") == 0) { // shared folder
    argument.erase(0, 7); // chop: "file://"

    rank_type r_world_size;
    try {
      r_world_size = convertToRank(world_size);
    } catch(std::exception& e) {
      throw std::invalid_argument("invalid world_size");
    }

    return InitMethodFile(
      argument, // file path
      r_world_size // world size
    ).getConfig();
  } else if (argument == "env://") {
    return InitMethodEnv().getConfig();
  }

  throw std::invalid_argument("unsupported initialization method");
}

std::vector<std::string> getInterfaceAddresses() {
  struct ifaddrs *ifa;
  SYSCHECK(getifaddrs(&ifa));
  std::shared_ptr<struct ifaddrs> ifaddr_list(ifa, [](struct ifaddrs* ptr) {
      freeifaddrs(ptr);
  });

  std::vector<std::string> addresses;

  while (ifa != NULL) {
    struct sockaddr *addr = ifa->ifa_addr;
    if (addr) {
      bool is_loopback = ifa->ifa_flags & IFF_LOOPBACK;
      bool is_ip = addr->sa_family == AF_INET || addr->sa_family == AF_INET6;
      if (is_ip && !is_loopback) {
        addresses.push_back(sockaddrToString(addr));
      }
    }
    ifa = ifa->ifa_next;
  }

  return addresses;
}

} // namespace thd
