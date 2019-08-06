#pragma once

#include <torch/csrc/distributed/rpc/RpcAgent.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>

namespace torch {
namespace distributed {
namespace rpc {

class MapNameStore : public NameStore {
 public:
  MapNameStore(std::unordered_map<std::string, std::string> nameMap)
      : nameMap_(std::move(nameMap)) {}

  ~MapNameStore() override {}

  const std::string resolve(std::string name) {
    return nameMap_[name];
  }

 private:
  std::unordered_map<std::string, std::string> nameMap_;
};

}
}
}
