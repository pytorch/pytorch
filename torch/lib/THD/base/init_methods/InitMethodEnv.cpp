#include "InitMethod.hpp"
#include "InitMethodUtils.hpp"

namespace thd {
namespace init {

namespace {

constexpr char RANK_ENV[] = "RANK";
constexpr char WORLD_SIZE_ENV[] = "WORLD_SIZE";
constexpr char MASTER_PORT_ENV[] = "MASTER_PORT";
constexpr char MASTER_ADDR_ENV[] = "MASTER_ADDR";

const char* must_getenv(const char* env) {
  const char* value = std::getenv(env);
  if (value == nullptr) {
    throw std::logic_error(std::string("") + "failed to read the " + env +
        " environmental variable; maybe you forgot to set it?");
  }
  return value;
}

std::tuple<port_type, rank_type> load_master_env() {
  auto port = convertToPort(std::stoul(must_getenv(MASTER_PORT_ENV)));

  rank_type world_size = std::stoul(must_getenv(WORLD_SIZE_ENV));
  if (world_size == 0)
    throw std::domain_error(std::string(WORLD_SIZE_ENV) + " env variable cannot be 0");

  return std::make_tuple(port, world_size);
}


std::tuple<std::string, port_type> load_worker_env() {
  std::string str_port = must_getenv(MASTER_PORT_ENV);
  auto port = convertToPort(std::stoul(str_port));
  return std::make_tuple(must_getenv(MASTER_ADDR_ENV), port);
}

rank_type load_rank_env() {
  return convertToRank(std::stol(must_getenv(RANK_ENV)));
}

} // anonymous namespace

InitMethod::Config initEnv(int world_size) {
  InitMethod::Config config;
  config.rank = load_rank_env();
  if (config.rank == 0) {
    const char *env_world_size_str = std::getenv(WORLD_SIZE_ENV);
    int env_world_size = world_size;
    if (env_world_size_str != nullptr)
      env_world_size = convertToRank(std::stol(must_getenv(WORLD_SIZE_ENV)));
    if (env_world_size != world_size)
      throw std::runtime_error("world size specified both as an environmental variable"
        "and to the initializer");
    config.master.world_size = env_world_size;

    std::tie(config.master.listen_port, config.master.world_size) = load_master_env();
    std::tie(config.master.listen_socket, std::ignore) = listen(config.master.listen_port);
    config.public_address = discoverWorkers(config.master.listen_socket, config.master.world_size);
  } else {
    std::tie(config.worker.address, config.worker.port) = load_worker_env();
    std::tie(std::ignore, config.public_address) = discoverMaster({config.worker.address}, config.worker.port);
  }
  return config;
}

} // namespace init
} // namespace thd
