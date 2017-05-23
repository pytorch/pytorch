#include "DataChannel.hpp"
#ifdef WITH_GLOO
#include "data_channels/DataChannelGloo.hpp"
#endif // WITH_GLOO
#ifdef WITH_MPI
#include "data_channels/DataChannelMPI.hpp"
#endif // WITH_MPI
#include "data_channels/DataChannelTCP.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace thd {

DataChannel* DataChannel::newChannel(THDChannelType type, std::string init_method,
                                     int world_size, std::string group_name) {
  InitMethod::Config config = getInitConfig(init_method, world_size, group_name);
  switch (type) {
    case THDChannelTCP:
      return new DataChannelTCP(config);

    case THDChannelMPI:
#ifdef WITH_MPI
      return new DataChannelMPI();
#endif // WITH_MPI
      throw std::runtime_error(
        "the MPI backend is not available; "
        "try to recompile the THD package with MPI support"
      );

    case THDChannelGloo:
#ifdef WITH_GLOO
      return new DataChannelGloo(config);
#endif // WITH_GLOO
      throw std::runtime_error(
        "the Gloo backend is not available; "
        "try to recompile the THD package with Gloo support"
      );

    default:
      throw std::runtime_error("unsupported data channel type");
  }
}


DataChannel::Group::Group()
{}


DataChannel::Group::Group(std::vector<rank_type> ranks, rank_type max_rank)
{
  if (ranks.size() == 0)
    throw std::logic_error("cannot create empty group");

  sort(ranks.begin(), ranks.end());
  if (ranks.back() > max_rank) {
    throw std::out_of_range(
      "array of ranks contains invalid rank, "
      "all ranks should be in range: [0, " + std::to_string(max_rank) + "]"
    );
  }

  _new2old.reserve(ranks.size());
  for (std::size_t i = 0; i < ranks.size(); ++i) {
    _new2old.push_back(ranks[i]);
    _old2new.insert({ranks[i], i});
  }
}


DataChannel::Group::~Group()
{}


auto DataChannel::Group::size() const -> rank_type {
  return static_cast<rank_type>(_new2old.size());
}


auto DataChannel::Group::mustGetGroupRank(rank_type global_rank) const -> rank_type {
  rank_type group_rank;
  bool exists;
  std::tie(group_rank, exists) = getGroupRank(global_rank);

  if (!exists) {
    throw std::logic_error(
      "rank(" + std::to_string(global_rank) + ") is not member of group"
    );
  }

  return group_rank;
}


auto DataChannel::Group::getGroupRank(rank_type global_rank) const -> std::pair<rank_type, bool> {
  auto global_rank_it = _old2new.find(global_rank); // O(1) operation
  if (global_rank_it != _old2new.end())
    return std::make_pair(global_rank_it->second, true);

  return std::make_pair(0, false);
}


auto DataChannel::Group::mustGetGlobalRank(rank_type group_rank) const -> rank_type {
  rank_type global_rank;
  bool exists;
  std::tie(global_rank, exists) = getGlobalRank(group_rank);

  if (!exists) {
    throw std::logic_error(
      "group rank is invalid, rank should be in "
      "range: [0, " + std::to_string(_new2old.size() - 1) + "]"
    );
  }

  return global_rank;
}


auto DataChannel::Group::getGlobalRank(rank_type group_rank) const -> std::pair<rank_type, bool> {
  if (group_rank >= _new2old.size())
    return std::make_pair(0, false);

  return std::make_pair(_new2old[group_rank], true);
}

} // namespace thd
