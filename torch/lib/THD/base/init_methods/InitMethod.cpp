#include "InitMethod.hpp"
#include "InitMethodEnv.hpp"
#include "InitMethodFile.hpp"
#include "InitMethodMulticast.hpp"

namespace thd {

InitMethod::Config getInitConfig(std::string argument, int world_size,
                                 std::string group_name) {
  if (argument.find("tcp") == 0) { // multicast tcp
    argument.erase(0, 6); // chop: "tcp://"

    auto found_pos = argument.rfind(":");
    if (found_pos == std::string::npos)
      throw std::invalid_argument("invalid multicast address, usage: IP:PORT | HOSTNAME:PORT");

    return InitMethodMulticast(
      argument.substr(0, found_pos), // address
      convertToPort(std::stoul(argument.substr(found_pos + 1))), // port
      convertToRank(world_size), // world size
      group_name
    ).getConfig();
  } else if (argument.find("file") == 0) { // shared folder
    argument.erase(0, 7); // chop: "file://"
    return InitMethodFile(
      argument, // file path
      convertToRank(world_size) // world size
    ).getConfig();
  } else if (argument == "env://") {
    return InitMethodEnv().getConfig();
  }

  throw std::invalid_argument("unsupported initialization method");
}

} // namespace thd
