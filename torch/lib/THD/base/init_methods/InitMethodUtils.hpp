#pragma once

#include "../ChannelUtils.hpp"

#include <string>
#include <vector>

namespace thd {

std::vector<std::string> getInterfaceAddresses();

std::string discoverWorkers(int listen_socket, rank_type world_size);
// pair of master_address, my_address
std::pair<std::string, std::string> discoverMaster(std::vector<std::string> addresses, port_type port);

} // namespace thd
