#pragma once

#include <THD/base/ChannelUtils.hpp>

#include <string>
#include <vector>

namespace thd {

std::vector<std::string> getInterfaceAddresses();

std::string discoverWorkers(int listen_socket, rank_type world_size);

// pair of master_address, my_address
std::pair<std::string, std::string> discoverMaster(
    std::vector<std::string> addresses,
    port_type port);

// Helper that gets the rank based on the input order
rank_type getRank(
    const std::vector<int>& ranks,
    int assigned_rank,
    size_t order);
} // namespace thd
