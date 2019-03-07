#include <THD/base/init_methods/InitMethod.hpp>
#include <THD/base/init_methods/InitMethodUtils.hpp>

namespace thd {
namespace init {

namespace {

constexpr char RANK_ENV[] = "RANK";
constexpr char WORLD_SIZE_ENV[] = "WORLD_SIZE";
constexpr char MASTER_PORT_ENV[] = "MASTER_PORT";
constexpr char MASTER_ADDR_ENV[] = "MASTER_ADDR";

const char* mustGetEnv(const char* env) {
  const char* value = std::getenv(env);
  if (value == nullptr) {
    throw std::logic_error(
        std::string("") + "failed to read the " + env +
        " environmental variable; maybe you forgot to set it?");
  }
  return value;
}

std::tuple<std::string, port_type> loadWorkerEnv() {
  std::string str_port = mustGetEnv(MASTER_PORT_ENV);
  auto port = convertToPort(std::stoul(str_port));
  return std::make_tuple(mustGetEnv(MASTER_ADDR_ENV), port);
}

rank_type maybeLoadEnv(
    const char* env_name,
    int value,
    std::string parameter_name) {
  const char* env_value_str = std::getenv(env_name);
  int env_value = value;
  if (env_value_str != nullptr)
    env_value = std::stol(env_value_str);
  if (value != -1 && env_value != value)
    throw std::runtime_error(
        parameter_name +
        " specified both as an "
        "environmental variable and to the initializer");
  if (env_value == -1)
    throw std::runtime_error(
        parameter_name +
        " is not set but it is required for "
        "env:// init method");

  return convertToRank(env_value);
}

} // anonymous namespace

InitMethod::Config initEnv(
    std::string argument, /* unused */
    int world_size_r,
    std::string group_name,
    int rank) {
  InitMethod::Config config;

  config.rank = maybeLoadEnv(RANK_ENV, rank, "rank");
  config.world_size = maybeLoadEnv(WORLD_SIZE_ENV, world_size_r, "world_size");

  if (group_name != "") {
    throw std::runtime_error(
        "group_name is not supported in env:// init method");
  }

  if (config.rank == 0) {
    config.master.listen_port =
        convertToPort(std::stoul(mustGetEnv(MASTER_PORT_ENV)));
    std::tie(config.master.listen_socket, std::ignore) =
        listen(config.master.listen_port);
    config.public_address =
        discoverWorkers(config.master.listen_socket, config.world_size);
  } else {
    std::tie(config.worker.master_addr, config.worker.master_port) =
        loadWorkerEnv();
    std::tie(std::ignore, config.public_address) =
        discoverMaster({config.worker.master_addr}, config.worker.master_port);
  }
  return config;
}

} // namespace init
} // namespace thd
