#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>

// Enables caching on for dynamic shapes (aka disable hash on shapes)
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

bool Node::enableDynamicShape() {
  static bool enabled = std::getenv("LTC_ENABLE_DYNAMIC_SHAPES") != nullptr;
  return enabled || FLAGS_ltc_enable_dynamic_shapes;
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


Node::Node(OpKind op, OpList operands, std::vector<Shape>&& shapes,
           size_t num_outputs, hash_t hash_seed)
    : Node(op, num_outputs,
           // TODO(WHC) this is inefficient (having to compute node_hash twice
           // since I can't call hash() yet) so probably move dag_hash
           // initialization to a separate function?
           /* node_hash */ HashCombine(op.hash(), hash_seed),
           /* dag_hash */
           [&](bool bakeInSizes) { return OperandHashes(operands, HashCombine(op.hash(), hash_seed), bakeInSizes); }) {
  // Move shapes into node
  shapes_.insert(
    shapes_.end(),
    std::make_move_iterator(shapes.begin()),
    std::make_move_iterator(shapes.end()));

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

Node::Node(OpKind op, OpList operands, size_t num_outputs, hash_t hash_seed)
  : Node(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {}

Node::Node(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed)
    : Node(op, num_outputs, [&](bool bakeInSizes) -> hash_t { return GetOpHash(op, shape, hash_seed, bakeInSizes); }) {
  shapes_.push_back(std::move(shape));
}

Node::~Node() = default;

hash_t Node::GetOpHash(OpKind op, const Shape& shape, hash_t hash_seed, bool bakeInSizes) {
  hash_t h = HashCombine(op.hash(), shape.hash(bakeInSizes));
  return HashCombine(h, hash_seed);
}

// Retrieves the full shape of the IR Node.
c10::ArrayRef<Shape> Node::shapes() const { return shapes_; }

// Retrieves the shape of the output at a given index.
const Shape& Node::shape(size_t output_index) const {
  return shapes_.at(output_index);
}

const std::vector<Output>& Node::operands() const {
  return operands_as_outputs_;
}
const Output& Node::operand(size_t i) const {
  return operands_as_outputs_.at(i);
}

std::string Node::ToString() const {
  std::stringstream ss;
  ss << shapes() << " " << op();
  if (num_outputs() > 1) {
    ss << ", num_outputs=" << num_outputs();
  }
  if (!metadata().scope.empty()) {
    ss << ", scope=" << metadata().scope;
  }
  EmitShortFrameInfo(ss, metadata().frame_info);
  return ss.str();
}

void Node::AddOperand(NodePtr node, size_t index) {
  CHECK_LT(index, node->num_outputs());
  operands_.push_back(std::move(node));
  operands_as_outputs_.emplace_back(operands_.back().get(), index);
}


} // namespace lazy
} // namespace torch
