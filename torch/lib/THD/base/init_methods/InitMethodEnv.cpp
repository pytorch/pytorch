#include "InitMethodEnv.hpp"

namespace thd {

InitMethodEnv::InitMethodEnv() {}
InitMethodEnv::~InitMethodEnv() {}

InitMethod::Config InitMethodEnv::getConfig() {
  InitMethod::Config config;
  config.rank = load_rank_env();
  if (config.rank == 0) {
    config.master.world_size = load_world_size_env();
    std::tie(config.master.listen_port, config.master.world_size) = load_master_env();
    std::tie(config.master.listen_socket, std::ignore, std::ignore) = listen(config.master.listen_port);
  } else {
    std::tie(config.worker.address, config.worker.listen_port) = load_worker_env();
  }
  return config;
}

} // namespace thd
