#pragma once

#include "../ChannelUtils.hpp"

#include <string>
#include <vector>

namespace thd {

void discoverWorkers(int listen_socket, rank_type world_size);
std::string discoverMaster(std::vector<std::string> addresses, port_type port);

} // namespace thd
