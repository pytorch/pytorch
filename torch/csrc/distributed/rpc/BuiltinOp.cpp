#include <torch/csrc/distributed/rpc/BuiltinOp.h>

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

std::shared_ptr<Operator> BuiltinOp::op() {
  return op_;
}

std::vector<at::IValue>& BuiltinOp::stack() {
  return stack_;
}

Message BuiltinOp::toMessage() {
  std::vector<torch::Tensor> tensor_table;
  Pickler pickler(&tensor_table);

  pickler.start();
  pickler.startTuple();
  for (auto& value: stack_) {
    pickler.addIValue(std::move(value));
  }
  pickler.addIValue(toString(op_->schema()));
  pickler.endTuple();
  pickler.finish();

  return Message(pickler.stack(),
                 std::move(tensor_table),
                 MessageType::BUILTIN_OP);
}

BuiltinOp BuiltinOp::fromMessage(Message message) {
  auto data = static_cast<void*>(message.meta().data());
  auto size = message.meta().size();
  Unpickler unpickler(data, size, &message.tensors(), nullptr);

  std::vector<IValue> values = unpickler.parse_ivalue_list();

  auto str_schema = values.back().toStringRef();
  values.pop_back();

  auto str_symbol = str_schema.substr(0, str_schema.find("("));
  auto symbol = at::Symbol::fromQualString(str_symbol);
  auto op = matchOperator(symbol, str_schema);

  return BuiltinOp(op, std::move(values));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
