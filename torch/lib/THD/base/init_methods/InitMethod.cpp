#include "InitMethod.hpp"

namespace thd {
namespace init {

InitMethod::Config initTCP(std::string argument, rank_type world_size, std::string group_name);
InitMethod::Config initFile(std::string argument, rank_type world_size, std::string group_name);
InitMethod::Config initEnv(int world_size);

}

InitMethod::Config getInitConfig(std::string argument, int world_size,
                                 std::string group_name) {
  if (argument.find("env://") == 0) {
    return init::initEnv(world_size);
  } else {
    rank_type r_world_size;
    try {
      r_world_size = convertToRank(world_size);
    } catch(std::exception& e) {
      throw std::invalid_argument("invalid world_size");
    }

    group_name.append("#"); // To make sure it's not empty

    if (argument.find("tcp://") == 0) {
      argument.erase(0, 6); // chop "tcp://"
      return init::initTCP(argument, r_world_size, group_name);
    } else if (argument.find("file://") == 0) {
      argument.erase(0, 7); // chop "file://"
      return init::initFile(argument, r_world_size, group_name);
    }
  }

  throw std::invalid_argument("unsupported initialization method");
}

} // namespace thd
