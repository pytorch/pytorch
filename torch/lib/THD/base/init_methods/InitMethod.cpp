#include "InitMethod.hpp"
#include "InitMethodEnv.hpp"
#include "InitMethodFile.hpp"
#include "InitMethodMulticast.hpp"

namespace thd {

InitMethod::Config getInitConfig(std::string argument, int world_size,
                                 std::string group_name) {
  if (argument.find("tcp") == 0) { // multicast tcp
    argument.erase(0, 6); // chop: "tcp://"

    std::string host, port;
    std::tie(host, port) = splitAddress(argument);

    rank_type r_world_size;
    try {
      r_world_size = convertToRank(world_size);
    } catch(std::exception& e) {
      throw std::invalid_argument("invalid world_size value");
    }

    return InitMethodMulticast(
      host,
      convertToPort(std::stoul(port)), // port
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
      r_world_size, // world size
      group_name
    ).getConfig();
  } else if (argument == "env://") {
    return InitMethodEnv().getConfig();
  }

  throw std::invalid_argument("unsupported initialization method");
}

} // namespace thd
