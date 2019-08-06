#pragma once

#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>

namespace torch {
namespace distributed {
namespace rpc {

class Serializer {
 public:
  Serializer(const std::vector<IValue>& values) : values_(values) {}
  int64_t writeNext(std::ostream& os, uint64_t size);

 private:
  const std::vector<IValue>& values_;
};

}
}
}
