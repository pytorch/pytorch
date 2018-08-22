#include "InitMethod.hpp"

namespace thd {

InitMethod::InitMethod() {
}

InitMethod::~InitMethod() {
}

InitMethod::Config getInitConfig(
    std::string argument,
    int world_size,
    std::string group_name,
    int rank) {
  // Find init method that shares prefix with the specified argument
  const auto& keys = InitMethodRegistry()->Keys();
  const auto& it = std::find_if(
      keys.begin(),
      keys.end(),
      [&argument] (const std::string& key) -> bool {
        return argument.find(key) == 0;
      });
  if (it == keys.end()) {
    std::stringstream ss;
    ss << "unknown init method: " << argument;
    throw std::logic_error(ss.str());
  }

  auto initMethod = InitMethodRegistry()->Create(*it);
  auto config = initMethod->init(argument, world_size, group_name, rank);
  config.validate();
  return config;
}

AT_DEFINE_REGISTRY(InitMethodRegistry, InitMethod);

} // namespace thd
