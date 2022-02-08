#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>

C10_DEFINE_bool(ltc_enable_dynamic_shapes, false, "Whether dynamic shape is enabled");

namespace torch {
namespace lazy {

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

Node::Node(OpKind op, size_t num_outputs, hash_t node_hash, std::function<hash_t(bool)> dag_hash_fn)
    : op_(op),
      num_outputs_(num_outputs),
      node_hash_(node_hash),
      dag_hash_without_sizes_(dag_hash_fn(false)),
      dag_hash_with_sizes_(dag_hash_fn(true)),
      metadata_(GetMetaDataIfDebugging()) {}

Node::Node(OpKind op, size_t num_outputs, std::function<hash_t(bool)> node_hash_fn)
    : op_(op),
      num_outputs_(num_outputs),
      node_hash_(node_hash_fn(!enableDynamicShape())),
      dag_hash_without_sizes_(node_hash_fn(false)),
      dag_hash_with_sizes_(node_hash_fn(true)),
      metadata_(GetMetaDataIfDebugging()) {}

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
