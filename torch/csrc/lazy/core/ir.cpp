#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>

// Enables caching on for dynamic shapes (aka disable hash on shapes)
// NOLINTNEXTLINE(misc-use-internal-linkage)
// clang-format off
C10_DEFINE_bool(
    ltc_enable_dynamic_shapes,
    false,
    "Whether dynamic shape is enabled")

namespace torch::lazy {
static const torch::lazy::Output kNullOutput = torch::lazy::Output();

size_t Output::Hasher::operator()(const Output& output) const {
  return StdHashCombine(
      reinterpret_cast<std::ptrdiff_t>(output.node), output.index);
}

hash_t Output::hash() const {
  return HashCombine(node->hash(), Hash(index));
}

hash_t Output::shapeHash() const {
  return HashCombine(node->shapeHash(), Hash(index));
}

std::string Output::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ", index=" << index;
  return ss.str();
}

bool Output::operator==(const Value& rhs) const {
  // Either side could be kNullValue which has node as nullptr
  return (!node == !rhs.node) &&
      (!node || (node->hash() == rhs.node->hash() && index == rhs.index));
}

hash_t Value::hash() const {
  return HashCombine(node->hash(), Hash(index));
}

hash_t Value::shapeHash() const {
  return HashCombine(node->shapeHash(), Hash(index));
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

Node::Node(OpKind op, size_t num_outputs)
    : op_(op), num_outputs_(num_outputs), metadata_(GetMetaDataIfDebugging()) {}

Node::Node(
    OpKind op,
    OpList operands,
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    std::vector<Shape>&& shapes,
    size_t num_outputs)
    : Node(op, num_outputs) {
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

Node::Node(OpKind op, OpList operands, size_t num_outputs)
    : Node(op, operands, std::vector<Shape>{}, num_outputs) {}

Node::Node(OpKind op, Shape shape, size_t num_outputs) : Node(op, num_outputs) {
  shapes_.push_back(std::move(shape));
}

// Retrieves the full shape of the IR Node.
c10::ArrayRef<Shape> Node::shapes() const {
  return shapes_;
}

// Retrieves the shape of the output at a given index.
const Shape& Node::shape(size_t output_index) const {
  return shapes_.at(output_index);
}

// Add the shape computed by the shape_fn

void Node::addComputedShape(const std::function<Shape()>& shape_fn) {
  shapes_.push_back(computeShape(shape_fn));
}

using ShapeCache = Cache<hash_t, Shape, HashReducer>;

// Compute the shape using the provided shape_fn.
Shape Node::computeShape(const std::function<Shape()>& shape_fn) {
  static ShapeCache* cache = new ShapeCache(FLAGS_torch_lazy_shape_cache_size);

  auto hash = shapeHash();
  auto shape = cache->Get(hash);
  if (shape == nullptr) {
    shape = cache->Add(hash, std::make_shared<Shape>(shape_fn()));
  }
  return *shape;
}

const std::vector<Output>& Node::operands() const {
  return operands_as_outputs_;
}

const Output& Node::operand(size_t i) const {
  return operands_as_outputs_.at(i);
}

const Output& Node::nullable_operand(size_t i) const {
  // We use kNullOutput instead of kNullValue here to avoid implicit casting,
  // which would prevent this method from returning a reference.
  return i < operands_as_outputs_.size() ? operand(i) : kNullOutput;
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

void Node::AddOperand(const NodePtr& node, size_t index) {
  TORCH_CHECK_LT(index, node->num_outputs());
  operands_.push_back(node);
  operands_as_outputs_.emplace_back(operands_.back().get(), index);
}

} // namespace torch::lazy
