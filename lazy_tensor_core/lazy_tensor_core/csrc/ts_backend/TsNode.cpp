#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

#include "lazy_tensors/computation_client/sys_util.h"
#include "third_party/computation_client/cache.h"

namespace torch_lazy_tensors {
namespace ir {
using torch::lazy::Output;
using torch::lazy::OpKind;

const lazy_tensors::Shape& GetShapeFromTsOutput(const torch::lazy::Output& output) {
  if (auto* tsnode = dynamic_cast<const TsNode*>(output.node)) {
    return tsnode->shape(output.index);
  }
  throw std::runtime_error("Expected TsNode but could not dynamic cast");
}

const lazy_tensors::Shape& GetShapeFromTsValue(const torch::lazy::Value& value) {
  if (auto* tsnode = dynamic_cast<const TsNode*>(value.node.get())) {
    return tsnode->shape(value.index);
  }
  throw std::runtime_error("Expected TsNode but could not dynamic cast");
}

void TsNodeSetShapeDeferred(
    NodePtr node, const std::function<lazy_tensors::Shape()>& shape_fn) {
  if (auto tsnode = std::dynamic_pointer_cast<TsNode>(node)) {
    tsnode->SetShapeDeferred(shape_fn);
    return;
  }
  throw std::runtime_error("Expected TsNode but could not dynamic cast");
}

torch::lazy::hash_t OperandHashes(const OpList& operands,
                                  const torch::lazy::hash_t& seed) {
  torch::lazy::hash_t hash = seed;
  for (auto& operand : operands) {
    if (!operand) {
      hash = torch::lazy::HashCombine(
          hash, static_cast<uint64_t>(torch::lazy::kNullOpt));
      continue;
    }
    hash = torch::lazy::HashCombine(hash, operand.hash());
  }
  return hash;
}

TsNode::TsNode(OpKind op, OpList operands, std::vector<lazy_tensors::Shape>&& shapes,
               size_t num_outputs, torch::lazy::hash_t hash_seed)
    : Node(op, num_outputs,
           // TODO(WHC) this is inefficient (having to compute node_hash twice
           // since I can't call hash() yet) so probably move dag_hash
           // initialization to a separate function?
           /* node_hash */ torch::lazy::HashCombine(op.hash(), hash_seed),
           /* dag_hash */
           OperandHashes(operands,
                         torch::lazy::HashCombine(op.hash(), hash_seed))),
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
               const std::function<lazy_tensors::Shape()>& shape_fn,
               size_t num_outputs, torch::lazy::hash_t hash_seed)
    : TsNode(op, operands, std::vector<lazy_tensors::Shape>{}, num_outputs, hash_seed) {
  auto shape = GetOpShape(shape_fn);
  CHECK(!shape.IsTuple());
  shapes_.push_back(std::move(shape));
}

TsNode::TsNode(OpKind op, OpList operands, size_t num_outputs,
               torch::lazy::hash_t hash_seed)
    : TsNode(op, operands, std::vector<lazy_tensors::Shape>{}, num_outputs, hash_seed) {}

void TsNode::SetShapeDeferred(
    const std::function<lazy_tensors::Shape()>& shape_fn) {
  auto shape = GetOpShape(shape_fn);
  CHECK(!shape.IsTuple());
  shapes_.push_back(std::move(shape));
}

TsNode::TsNode(OpKind op, lazy_tensors::Shape shape, size_t num_outputs,
               torch::lazy::hash_t hash_seed)
    : Node(op, num_outputs, GetOpHash(op, shape, hash_seed))
{
  CHECK(!shape.IsTuple());
  shapes_.push_back(std::move(shape));
}

const lazy_tensors::Shape& TsNode::shape(size_t output_index) const {
  return shapes_.at(output_index);
}

using ShapeCache =
    lazy_tensors::util::Cache<torch::lazy::hash_t, lazy_tensors::Shape,
                              torch::lazy::HashReducer>;

ShapeCache* GetShapeCache() {
  static int64_t shape_cache_size =
      lazy_tensors::sys_util::GetEnvInt("LTC_IR_SHAPE_CACHE_SIZE", 4096);
  static ShapeCache* cache = new ShapeCache(shape_cache_size);
  return cache;
}

lazy_tensors::Shape TsNode::GetOpShape(
    const std::function<lazy_tensors::Shape()>& shape_fn) const {
  ShapeCache* shape_cache = GetShapeCache();
  auto shape = shape_cache->Get(hash());
  if (shape == nullptr) {
    shape = shape_cache->Add(hash(),
                             std::make_shared<lazy_tensors::Shape>(shape_fn()));
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

torch::lazy::hash_t TsNode::GetOpHash(OpKind op,
                                      const lazy_tensors::Shape& shape,
                                      torch::lazy::hash_t hash_seed) {
  torch::lazy::hash_t h =
      torch::lazy::HashCombine(op.hash(), torch::lazy::Hash(shape.ToString()));
  return torch::lazy::HashCombine(h, hash_seed);
}

void TsNode::AddOperand(NodePtr node, size_t index) {
  CHECK_LT(index, node->num_outputs());
  operands_.push_back(std::move(node));
  operands_as_outputs_.push_back(Output(operands_.back().get(), index));
}

TSOpVector TsNode::Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                         ts_backend::TSLoweringContext* loctx) const {
  // TODO(whc) beginning to invert the design here.  Move to provide a Lower()
  // method on each node, starting with codegen.  Once we delete most
  // non-codegen ops, make this pure-virtual and put Lower() on the remaining
  // non-codegen ops.  For now, returning empty list here triggers fallback to
  // old lowering path.
  return {};
}

}  // namespace ir
}  // namespace torch_lazy_tensors
