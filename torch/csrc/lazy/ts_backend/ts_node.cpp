#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include <torch/csrc/lazy/ts_backend/config.h>

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

/**
 * Constructors used by legacy IR
 *  - to avoid having to modify each legacy IR class to compute hashes,
 *    just provide special constructors for them which compute the hashes the old way
 */
TsNode::TsNode(OpKind op, OpList operands, std::vector<Shape>&& shapes,
               size_t num_outputs, hash_t hash_seed)
    : TsNode(op, operands, std::move(shapes), num_outputs,
           /* node_hash */ HashCombine(op.hash(), hash_seed),
           /* dag_hash */
           OperandHashes(operands, HashCombine(op.hash(), hash_seed))) {}

TsNode::TsNode(OpKind op, OpList operands,
               const std::function<Shape()>& shape_fn,
               size_t num_outputs, hash_t hash_seed)
    : TsNode(op, operands, std::vector<Shape>{}, num_outputs, 
           /* node_hash */ HashCombine(op.hash(), hash_seed),
           /* dag_hash */
           OperandHashes(operands, HashCombine(op.hash(), hash_seed))) {
  shapes_ = GetOpShape(shape_fn);
}

TsNode::TsNode(OpKind op, OpList operands, size_t num_outputs,
               hash_t hash_seed)
    : TsNode(op, operands, std::vector<Shape>{}, num_outputs,
           /* node_hash */ HashCombine(op.hash(), hash_seed),
           /* dag_hash */
           OperandHashes(operands, HashCombine(op.hash(), hash_seed))) {}

void TsNode::SetShapeDeferred(
    const std::function<Shape()>& shape_fn) {
  shapes_ = GetOpShape(shape_fn);
}

TsNode::TsNode(OpKind op, Shape shape, size_t num_outputs)
    : Node(op, num_outputs, HashCombine(op.hash(), shape.hash())) {
  shapes_.push_back(std::move(shape));
}

/**
 * Constructors used by codegen IR
 *  - node_ and dag_hash are explicitly computed in the codegen code, so shape-caching
 *    can be done outside Node class based on hash
 */
TsNode::TsNode(OpKind op, OpList operands, std::vector<Shape>&& shapes,
               size_t num_outputs, hash_t node_hash, hash_t dag_hash)
    : Node(op, num_outputs,
           node_hash,
           dag_hash),
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

// TODO(whc) is this ctor even used anywhere? are there any codegenned leaf nodes?
TsNode::TsNode(OpKind op, Shape shape, size_t num_outputs, hash_t node_hash)
    : Node(op, num_outputs, HashCombine(op.hash(), HashCombine(shape.hash(), node_hash)))
{
  shapes_.push_back(std::move(shape));
}

const Shape& TsNode::shape(size_t output_index) const {
  return shapes_.at(output_index);
}


ShapeCache* GetShapeCache() {
  static ShapeCache* cache = new ShapeCache(FLAGS_torch_lazy_ts_shape_cache_size);
  return cache;
}

// Used by legacy IR which still returns a single 'Shape' rather than vector<Shape>
// so this helper wraps the shape in a vector for consistency with newer codegenned IR shapes 
std::vector<Shape> TsNode::GetOpShape(
    const std::function<Shape()>& shape_fn) const {
  ShapeCache* shape_cache = GetShapeCache();
  auto shape = shape_cache->Get(hash());
  if (shape == nullptr) {
    shape = shape_cache->Add(hash(),
                             std::make_shared<std::vector<Shape>>(std::initializer_list<Shape>{shape_fn()}));
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
