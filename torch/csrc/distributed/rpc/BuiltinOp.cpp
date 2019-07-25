#include <torch/csrc/distributed/rpc/BuiltinOp.h>

namespace torch {
namespace distributed {
namespace rpc {

BuiltinOp::BuiltinOp(
    std::shared_ptr<Operator> op, std::vector<at::IValue>&& args)
    : op_(std::move(op)), stack_(args) {}

BuiltinOp::~BuiltinOp() = default;

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
    pickler.addIValue(value);
  }
  pickler.addIValue(toString(op_->schema()));
  pickler.endTuple();
  pickler.finish();

  auto meta = pickler.stack();
  return Message(std::move(meta),
                 std::move(tensor_table),
                 MessageType::BUILTIN_OP);
}

BuiltinOp BuiltinOp::fromMessage(const Message& message) {
  auto meta = static_cast<const void*>(message.meta().data());
  auto meta_size = message.meta().size();
  Unpickler unpickler(meta, meta_size, &message.tensors(), nullptr);

  std::vector<IValue> values = unpickler.parse_ivalue_list();

  TORCH_CHECK(values.size() >= 1, "Message of a BuiltinOp must at least "
      "contain one IValue as the operator schema.");

  const std::string& str_schema = values.back().toStringRef();
  // extract symbol from the schema
  auto str_symbol = str_schema.substr(0, str_schema.find('('));
  auto symbol = at::Symbol::fromQualString(str_symbol);
  auto op = matchOperator(symbol, str_schema);
  // remove str_schema from values
  values.pop_back();

  return BuiltinOp(op, std::move(values));
}

std::shared_ptr<Operator> BuiltinOp::matchOperator(
    at::Symbol& symbol, const std::string& str_schema) {
  for (auto op: torch::jit::getAllOperatorsFor(symbol)) {
    if (toString(op->schema()).compare(str_schema) == 0) {
      return op;
    }
  }
  AT_ERROR("Cannot find matching operator for schema ", str_schema);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
