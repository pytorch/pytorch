#pragma once

#include "../ChannelUtils.hpp"

#include <string>
#include <stdexcept>
#include <tuple>

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
      port_type port;
    };

    rank_type rank;
    std::string public_address;
    MasterConfig master;
    WorkerConfig worker;
  };

  virtual Config getConfig() = 0;
};

InitMethod::Config getInitConfig(std::string argument, int world_size = -1,
                                 std::string group_name = "", int rank = -1);

} // namespace thd
