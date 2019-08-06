#include <torch/csrc/distributed/rpc/Request.h>

namespace torch {
namespace distributed {
namespace rpc {

std::shared_ptr<Operator> matchOperator(
    at::Symbol symbol, std::string str_schema) {
  for (auto op: torch::jit::getAllOperatorsFor(symbol)) {
    if (toString(op->schema()).compare(str_schema) == 0) {
      return op;
    }
  }
  throw std::runtime_error("Cannot find matching operator");
}

std::shared_ptr<Operator> Request::op() {
  return op_;
}

const std::vector<at::IValue>& Request::args() {
  return args_;
}

std::vector<at::IValue> Request::toIValues() {
  std::vector<at::IValue> values = args_;
  values.emplace_back(toString(op_->schema()));
  return std::move(values);
}

Request Request::fromIValues(std::vector<at::IValue> values) {
  auto str_schema = values.back().toStringRef();
  values.pop_back();

  auto str_symbol = str_schema.substr(0, str_schema.find("("));
  auto symbol = at::Symbol::fromQualString(str_symbol);
  auto op = matchOperator(symbol, str_schema);

  return Request(op, std::move(values));
}

} // namespace rpc
}
}
