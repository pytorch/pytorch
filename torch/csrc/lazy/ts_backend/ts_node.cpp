#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include <torch/csrc/lazy/ts_backend/config.h>
#include <torch/csrc/lazy/core/cache.h>

namespace torch {
namespace lazy {

const Shape& GetShapeFromTsOutput(const Output& output) {
  if (auto* tsnode = dynamic_cast<const TsNode*>(output.node)) {
    return tsnode->shape(output.index);
  }
  throw std::runtime_error("Expected TsNode but could not dynamic cast");
}

const Shape& GetShapeFromTsValue(const Value& value) {
  if (auto* tsnode = dynamic_cast<const TsNode*>(value.node.get())) {
    return tsnode->shape(value.index);
  }
  throw std::runtime_error("Expected TsNode but could not dynamic cast");
}

void TsNodeSetShapeDeferred(
    NodePtr node, const std::function<Shape()>& shape_fn) {
  if (auto tsnode = std::dynamic_pointer_cast<TsNode>(node)) {
    tsnode->SetShapeDeferred(shape_fn);
    return;
  }
  throw std::runtime_error("Expected TsNode but could not dynamic cast");
}

hash_t OperandHashes(const OpList& operands, const hash_t& seed) {
  hash_t hash = seed;
  for (auto& operand : operands) {
    if (!operand) {
      hash = HashCombine(hash, static_cast<uint64_t>(kNullOpt));
      continue;
    }
    hash = HashCombine(hash, operand.hash());
  }
  return hash;
}

TsNode::TsNode(OpKind op, OpList operands, std::vector<Shape>&& shapes,
               size_t num_outputs, hash_t hash_seed)
    : Node(op, num_outputs,
           // TODO(WHC) this is inefficient (having to compute node_hash twice
           // since I can't call hash() yet) so probably move dag_hash
           // initialization to a separate function?
           /* node_hash */ HashCombine(op.hash(), hash_seed),
           /* dag_hash */
           OperandHashes(operands, HashCombine(op.hash(), hash_seed))),
      shapes_(shapes) {
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

TsNode::TsNode(OpKind op, OpList operands,
               const std::function<Shape()>& shape_fn,
               size_t num_outputs, hash_t hash_seed)
    : TsNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {
  shapes_.push_back(GetOpShape(shape_fn));
}

TsNode::TsNode(OpKind op, OpList operands, size_t num_outputs,
               hash_t hash_seed)
    : TsNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {}

void TsNode::SetShapeDeferred(
    const std::function<Shape()>& shape_fn) {
  shapes_.push_back(GetOpShape(shape_fn));
}

TsNode::TsNode(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed)
    : Node(op, num_outputs, GetOpHash(op, shape, hash_seed))
{
  shapes_.push_back(std::move(shape));
}

const Shape& TsNode::shape(size_t output_index) const {
  return shapes_.at(output_index);
}

using ShapeCache = Cache<hash_t, Shape, HashReducer>;

ShapeCache* GetShapeCache() {
  static ShapeCache* cache = new ShapeCache(FLAGS_torch_lazy_ts_shape_cache_size);
  return cache;
}

Shape TsNode::GetOpShape(
    const std::function<Shape()>& shape_fn) const {
  ShapeCache* shape_cache = GetShapeCache();
  auto shape = shape_cache->Get(hash());
  if (shape == nullptr) {
    shape = shape_cache->Add(hash(),
                             std::make_shared<Shape>(shape_fn()));
  }
  return *shape;
}

std::string TsNode::ToString() const {
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

hash_t TsNode::GetOpHash(OpKind op, const Shape& shape, hash_t hash_seed) {
  hash_t h = HashCombine(op.hash(), shape.hash());
  return HashCombine(h, hash_seed);
}

void TsNode::AddOperand(NodePtr node, size_t index) {
  CHECK_LT(index, node->num_outputs());
  operands_.push_back(std::move(node));
  operands_as_outputs_.emplace_back(operands_.back().get(), index);
}

TSOpVector TsNode::Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                         TSLoweringContext* loctx) const {
  // TODO(whc) beginning to invert the design here.  Move to provide a Lower()
  // method on each node, starting with codegen.  Once we delete most
  // non-codegen ops, make this pure-virtual and put Lower() on the remaining
  // non-codegen ops.  For now, returning empty list here triggers fallback to
  // old lowering path.
  return {};
}

}  // namespace lazy
}  // namespace torch
