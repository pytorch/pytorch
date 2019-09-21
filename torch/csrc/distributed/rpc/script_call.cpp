#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/jit/pickle.h>

namespace torch {
namespace distributed {
namespace rpc {

const std::string ScriptCall::BUILTIN_OP_NAMESPACE_("torch.ops.aten.");
const std::string ScriptCall::ATEN_PREFIX_("aten::");

ScriptCall::ScriptCall(
    std::shared_ptr<Operator> op,
    std::vector<at::IValue>&& args)
    : op_(std::move(op)), stack_(args) {}

std::shared_ptr<Operator> ScriptCall::op() const {
  return *op_;
}

const std::vector<at::IValue>& ScriptCall::stack() const {
  return stack_;
}

std::vector<at::IValue>& ScriptCall::stackRef() {
  return stack_;
}

void ScriptCall::toIValues(std::vector<at::IValue>& ivalues) const {
  for (auto& value : stack_) {
    ivalues.push_back(value);
  }

  if (op_) {
    // TODO: replace this with a real overload_name when FunctionSchema supports
    // that.
    ivalues.emplace_back(toString((*op_)->schema()));
    // insert qualified name
    auto opName = (*op_)->schema().name();
    TORCH_CHECK(
        opName.find("::") == opName.rfind("::") &&
            opName.rfind(ATEN_PREFIX_) == 0,
        "Unexpected operator name ",
        opName);
    // aten::add -> torch.ops.aten.add
    opName.replace(0, ATEN_PREFIX_.length(), BUILTIN_OP_NAMESPACE_);
    ivalues.emplace_back(std::move(opName));
  }
}

std::shared_ptr<Operator> ScriptCall::fromIValues(
    std::vector<at::IValue>& ivalues) {
  const std::string& qualifiedName = ivalues.back().toStringRef();

  if (qualifiedName.rfind(BUILTIN_OP_NAMESPACE_) == 0) {
    ivalues.pop_back();
    const std::string& str_schema = ivalues.back().toStringRef();
    auto op = matchOperator(str_schema);

    ivalues.pop_back();
    // remove str_schema from ivalues
    return op;
  } else {
    AT_ERROR("Unrecognized qualified name ", qualifiedName);
  }
}

Message ScriptCall::toMessage() {
  std::vector<IValue> ivalues;
  toIValues(ivalues);

  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  return Message(
      std::move(payload), std::move(tensor_table), MessageType::SCRIPT_CALL);
}

std::unique_ptr<ScriptCall> ScriptCall::fromMessage(const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();
  auto value =
      jit::unpickle(payload, payload_size, nullptr, &message.tensors());

  auto values = value.toTuple()->elements();
  auto op = fromIValues(values);
  return c10::guts::make_unique<ScriptCall>(op, std::move(values));
}

std::shared_ptr<Operator> ScriptCall::matchOperator(
    const std::string& str_schema) {
  // TODO: This is a temporary solution. We should pass enough information to
  // allow deterministically matched to one operator.

  // extract symbol from the schema
  auto schema = torch::jit::parseSchema(str_schema);
  auto symbol = at::Symbol::fromQualString(schema.name());

  for (auto op : torch::jit::getAllOperatorsFor(symbol)) {
    if (toString(op->schema()).compare(str_schema) == 0) {
      return op;
    }
  }

  AT_ERROR("Cannot find matching operator for schema ", str_schema);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
