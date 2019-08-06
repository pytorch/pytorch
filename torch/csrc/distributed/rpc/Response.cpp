#include <torch/csrc/distributed/rpc/Response.h>

namespace torch {
namespace distributed {
namespace rpc {

// Response
int64_t Response::code() {
  return code_;
}

const std::vector<at::IValue> Response::values() {
  return values_;
}

std::vector<at::IValue> Response::toIValues() {
  std::vector<at::IValue> values = values_;
  values.push_back(code_);
  return std::move(values);
}

Response Response::fromIValues(std::vector<at::IValue> values) {
  auto code = values.back().toInt();
  values.pop_back();
  return Response(code, std::move(values));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
