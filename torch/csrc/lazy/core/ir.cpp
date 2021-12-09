#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>

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

OpKind OpKind::Get(const std::string& name) {
  return OpKind(c10::Symbol::fromQualString(name));
}

hash_t OpKind::hash() const {
  return StringHash(op.toQualString());
}

Node::Node(OpKind op, size_t num_outputs, hash_t node_hash, hash_t dag_hash)
    : op_(op),
      num_outputs_(num_outputs),
      node_hash_(node_hash),
      dag_hash_(dag_hash),
      metadata_(GetMetaDataIfDebugging()) {}

Node::Node(OpKind op, size_t num_outputs, hash_t node_hash)
    : op_(op),
      num_outputs_(num_outputs),
      node_hash_(node_hash),
      dag_hash_(node_hash),
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
