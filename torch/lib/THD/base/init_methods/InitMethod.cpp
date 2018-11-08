#include "InitMethod.hpp"

#ifdef THD_INIT_EXTENSION_H
#define INCF(F) INCF_(F)
#define INCF_(F) #F
#include INCF(THD_INIT_EXTENSION_H)
#endif

namespace thd {
namespace init {

InitMethod::Config initTCP(
    std::string argument,
    int world_size_r,
    std::string group_name,
    int rank);
InitMethod::Config initFile(
    std::string argument,
    int world_size_r,
    std::string group_name,
    int rank);
InitMethod::Config initEnv(
    std::string argument,
    int world_size_r,
    std::string group_name,
    int rank);

InitMethodFuncMap initMethods(
    {{"env://", ::thd::init::initEnv},
     {"file://", ::thd::init::initFile},
     {"tcp://", ::thd::init::initTCP}

#ifdef THD_INIT_EXTENSION_H
     ,
     /**
      * Additional method pairs can be defined in THD_INIT_EXTENSION_H header
      * to extend the init methods
      */
     THD_INIT_EXTENSION_METHODS
#endif

    });

} // namespace init

InitMethod::Config getInitConfig(
    std::string argument,
    int world_size,
    std::string group_name,
    int rank) {
  InitMethod::Config config;

  for (auto& methodPair : init::initMethods) {
    auto initMethodPrefix = methodPair.first;
    auto initMethodFunc = methodPair.second;
    if (argument.find(initMethodPrefix) == 0) {
      config = initMethodFunc(argument, world_size, group_name, rank);
    }
  }
  config.validate();
  return config;
}

} // namespace thd
