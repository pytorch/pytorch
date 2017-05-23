#pragma once

#include "InitMethod.hpp"

namespace thd {

struct InitMethodMulticast : InitMethod {
  InitMethodMulticast(std::string address, port_type port, rank_type world_size,
                      std::string group_name);
  virtual ~InitMethodMulticast();

  InitMethod::Config getConfig() override;

private:
  std::string _address;
  port_type _port;
  rank_type _world_size;
  std::string _group_name;
};

} // namespace thd
