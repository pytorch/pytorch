#pragma once

#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>

namespace torch {
namespace distributed {
namespace rpc {

class Deserializer {
 public:
  std::vector<at::IValue> readNext(std::istream& is, int64_t size);
};

}
}
}
