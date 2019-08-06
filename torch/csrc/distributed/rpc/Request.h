#pragma once

#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

static std::shared_ptr<Operator> matchOperator(
    at::Symbol symbol, std::string str_schema);

class Request final {
 public:
  Request(std::shared_ptr<Operator> op,
          const std::vector<at::IValue> args)
      : args_(std::move(args)), op_(op) {}

  ~Request() {}

  std::shared_ptr<Operator> op();
  const std::vector<at::IValue>& args();
  std::vector<at::IValue> toIValues();
  static Request fromIValues(std::vector<at::IValue> values);

 private:
  const std::vector<at::IValue> args_;
  std::shared_ptr<Operator> op_;
};

}
}
}
