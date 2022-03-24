#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>

C10_DEFINE_bool(ltc_enable_dynamic_shapes, false, "Whether dynamic shape is enabled");

namespace torch {
namespace lazy {

hash_t OperandHashes(const OpList& operands, const hash_t& seed, bool bakeInSizes) {
  hash_t hash = seed;
  for (auto& operand : operands) {
    if (!operand) {
      hash = HashCombine(hash, static_cast<uint64_t>(kNullOpt));
      continue;
    }
    auto operand_hash = bakeInSizes ? operand.hash_with_sizes() : operand.hash_without_sizes();
    hash = HashCombine(hash, operand_hash);
  }
  return hash;
}

void Node::AddOperand(NodePtr node, size_t index) {
  CHECK_LT(index, node->num_outputs());
  operands_.push_back(std::move(node));
  operands_as_outputs_.emplace_back(operands_.back().get(), index);
}

size_t Output::Hasher::operator()(const Output& output) const {
  return StdHashCombine(
      reinterpret_cast<std::ptrdiff_t>(output.node), output.index);
}

hash_t Output::hash() const {
  return HashCombine(node->hash(), Hash(index));
}

std::string Output::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ", index=" << index;
  return ss.str();
}

hash_t Value::hash() const {
  return HashCombine(node->hash(), Hash(index));
}

hash_t Value::hash_with_sizes() const {
  return HashCombine(node->hash_with_sizes(), Hash(index));
}

hash_t Value::hash_without_sizes() const {
  return HashCombine(node->hash_without_sizes(), Hash(index));
}

OpKind OpKind::Get(const std::string& name) {
  return OpKind(c10::Symbol::fromQualString(name));
}

hash_t OpKind::hash() const {
  return StringHash(op.toQualString());
}

bool Node::enableDynamicShape() {
  static bool enabled = std::getenv("LTC_ENABLE_DYNAMIC_SHAPES") != nullptr;
  return enabled || FLAGS_ltc_enable_dynamic_shapes;
}

Node::Node(OpKind op, OpList operands, size_t num_outputs, hash_t node_hash, std::function<hash_t(bool)> dag_hash_fn)
    : op_(op),
      num_outputs_(num_outputs),
      node_hash_(node_hash),
      dag_hash_without_sizes_(dag_hash_fn(false)),
      dag_hash_with_sizes_(dag_hash_fn(true)),
      metadata_(GetMetaDataIfDebugging()) {
  for (auto& operand : operands) {
    // Ideally, optional operands should be filtered by the leaf node classes,
    // but it's just much easier to do it here.
    // TODO(alanwaketan): Find a way to move the below logic to the leaf node
    // classes.
    if (!operand) {
      continue;
    }
    AddOperand(operand.node, operand.index);
  }
}

Node::Node(OpKind op, size_t num_outputs, hash_t node_hash, std::function<hash_t(bool)> dag_hash_fn)
    : Node(op, OpList{}, num_outputs, node_hash, dag_hash_fn) {}

Node::Node(OpKind op, OpList operands, size_t num_outputs, std::function<hash_t(bool)> node_hash_fn)
    : op_(op),
      num_outputs_(num_outputs),
      node_hash_(node_hash_fn(!enableDynamicShape())),
      dag_hash_without_sizes_(node_hash_fn(false)),
      dag_hash_with_sizes_(node_hash_fn(true)),
      metadata_(GetMetaDataIfDebugging())  {
  for (auto& operand : operands) {
    // Ideally, optional operands should be filtered by the leaf node classes,
    // but it's just much easier to do it here.
    // TODO(alanwaketan): Find a way to move the below logic to the leaf node
    // classes.
    if (!operand) {
      continue;
    }
    AddOperand(operand.node, operand.index);
  }
}

Node::Node(OpKind op, size_t num_outputs, std::function<hash_t(bool)> node_hash_fn)
    : Node(op, OpList{}, num_outputs, node_hash_fn) {}

Node::~Node() = default;

std::string Node::ToString() const {
  std::stringstream ss;
  ss << op();
  if (num_outputs() > 1) {
    ss << ", num_outputs=" << num_outputs();
  }
  if (!metadata_.scope.empty()) {
    ss << ", scope=" << metadata_.scope;
  }
  EmitShortFrameInfo(ss, metadata_.frame_info);
  return ss.str();
}

} // namespace lazy
} // namespace torch
