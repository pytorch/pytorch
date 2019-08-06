#pragma once

#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

class Response final {
 public:
  Response(int64_t code,
           std::vector<at::IValue> values)
      : code_(code), values_(std::move(values)) {}

  ~Response() {}

  int64_t code();
  const std::vector<at::IValue> values();
  std::vector<at::IValue> toIValues();
  static Response fromIValues(std::vector<at::IValue> values);

 private:
  const int64_t code_;
  const std::vector<at::IValue> values_;
};

}
}
}
