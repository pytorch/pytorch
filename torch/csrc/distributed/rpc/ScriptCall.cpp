#include <torch/csrc/distributed/rpc/ScriptCall.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

using torch::jit::Pickler;
using torch::jit::Unpickler;

} // namespace

const std::string ScriptCall::BUILTIN_OP_NAMESPACE_("torch.ops.aten.");
const std::string ScriptCall::ATEN_PREFIX_("aten::");

ScriptCall::ScriptCall(
    std::shared_ptr<Operator> op, std::vector<at::IValue>&& args)
    : op_(std::move(op)), stack_(args) {}

std::shared_ptr<Operator> ScriptCall::op() const {
  return *op_;
}

const std::vector<at::IValue>& ScriptCall::stack() const {
  return stack_;
}

Message ScriptCall::toMessage() {
  std::vector<torch::Tensor> tensor_table;
  Pickler pickler(&tensor_table);

  pickler.protocol();
  pickler.startTuple();
  for (auto& value: stack_) {
    pickler.pushIValue(value);
  }
  if (op_) {
    // builtin ops

    // TODO: replace this with a real overload_name when FunctionSchema supports
    // that.
    pickler.pushIValue(toString((*op_)->schema()));
    // insert qualified name
    auto opName = (*op_)->schema().name();
    TORCH_CHECK(opName.find("::") == opName.rfind("::") &&
                opName.rfind(ATEN_PREFIX_) == 0,
                "Unexpected operator name ", opName);
    // aten::add -> torch.ops.aten.add
    opName.replace(0, ATEN_PREFIX_.length(), BUILTIN_OP_NAMESPACE_);
    pickler.pushIValue(opName);
  }
  pickler.endTuple();
  pickler.stop();

  auto payload = pickler.stack();
  return Message(std::move(payload),
                 std::move(tensor_table),
                 MessageType::SCRIPT_CALL);
}

ScriptCall ScriptCall::fromMessage(const Message& message) {
  auto payload = static_cast<const void*>(message.payload().data());
  auto payload_size = message.payload().size();
  Unpickler unpickler(payload, payload_size, &message.tensors(), nullptr);

  std::vector<IValue> values = unpickler.parse_ivalue_list();

  TORCH_CHECK(values.size() >= 1, "Message of a ScriptCall must at least "
      "contain one IValue as the operator schema.");

  const std::string& qualifiedName = values.back().toStringRef();
  if (qualifiedName.rfind(BUILTIN_OP_NAMESPACE_) == 0) {
    values.pop_back();

    const std::string& str_schema = values.back().toStringRef();
    // extract symbol from the schema
    auto str_symbol = str_schema.substr(0, str_schema.find('('));
    auto symbol = at::Symbol::fromQualString(str_symbol);
    auto op = matchOperator(symbol, str_schema);
    // remove str_schema from values
    values.pop_back();

    return ScriptCall(op, std::move(values));
  } else {
    AT_ERROR("Unrecognized qualified name ", qualifiedName);
  }
}

std::shared_ptr<Operator> ScriptCall::matchOperator(
    at::Symbol& symbol, const std::string& str_schema) {
  // TODO: This is a temporary solution. We should pass enough information to
  // allow deterministically matched to one operator.
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
