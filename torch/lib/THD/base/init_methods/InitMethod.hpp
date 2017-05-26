#pragma once

#include <string>
#include <stdexcept>
#include <tuple>

#include "../ChannelUtils.hpp"

namespace thd {

struct InitMethod {
  struct Config {
    struct MasterConfig {
      rank_type world_size;
      int listen_socket;
      port_type listen_port;
    };

    struct WorkerConfig {
      std::string address;
      port_type listen_port;
    };

    rank_type rank;
    MasterConfig master;
    WorkerConfig worker;
  };

  virtual Config getConfig() = 0;
};

InitMethod::Config getInitConfig(std::string argument, int world_size = -1,
                                 std::string group_name = "");

std::vector<std::string> getInterfaceAddresses();

} // namespace thd
