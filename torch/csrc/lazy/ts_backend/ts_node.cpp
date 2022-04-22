#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include <torch/csrc/lazy/core/debug_util.h>

namespace {
  std::string GetFirstUserFrameInPythonIfEnabled() {
    static const auto LTC_ENABLE_SOURCE_INFO = std::getenv("LTC_ENABLE_SOURCE_INFO");
    if (!LTC_ENABLE_SOURCE_INFO) {
      return {};
    }

    return torch::lazy::GetFirstUserFrameInPython();
  }
}

namespace torch {
namespace lazy {


hash_t OperandHashes(const OpList& operands, const hash_t& seed, bool bakeInSizes) {
  hash_t hash = seed;
  for (auto& operand : operands) {
    if (!operand) {
      hash = HashCombine(hash, static_cast<uint64_t>(kNullOpt));
      continue;
    }
    auto operand_hash = operand.hash();
    hash = HashCombine(hash, operand_hash);
  }
  return hash;
}

hash_t GetOpHash(OpKind op, const Shape& shape, hash_t hash_seed, bool bakeInSizes) {
  hash_t h = HashCombine(op.hash(), shape.hash(bakeInSizes));
  return HashCombine(h, hash_seed);
}

TsNode::TsNode(OpKind op, OpList operands, std::vector<Shape>&& shapes, size_t num_outputs, hash_t hash_seed)
    : Node(op, operands, std::move(shapes), num_outputs) {
  hash_seed = HashCombine(op.hash(), hash_seed);
  shape_hash_ = OperandHashes(operands, hash_seed, true);
  dag_hash_ = (enableDynamicShape() ? OperandHashes(operands, hash_seed, false) : shape_hash_);
}


TsNode::TsNode(OpKind op, OpList operands, const std::function<Shape()>& shape_fn,
               size_t num_outputs, hash_t hash_seed)
    : TsNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {
  addComputedShape(shape_fn);
}

TsNode::TsNode(OpKind op, OpList operands, size_t num_outputs, hash_t hash_seed)
    : TsNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {}

TsNode::TsNode(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed)
    : Node(op, num_outputs),
      shape_hash_(GetOpHash(op, shape, hash_seed, true)),
      dag_hash_(enableDynamicShape() ? GetOpHash(op, shape, hash_seed, false) : shape_hash_) {
  shapes_.push_back(std::move(shape));
}

hash_t TsNode::hash() const { return dag_hash_; }

hash_t TsNode::shapeHash() const { return shape_hash_; }

TSOpVector TsNode::Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                         TSLoweringContext* loctx) const {
  // TODO(whc) beginning to invert the design here.  Move to provide a Lower()
  // method on each node, starting with codegen.  Once we delete most
  // non-codegen ops, make this pure-virtual and put Lower() on the remaining
  // non-codegen ops.  For now, returning empty list here triggers fallback to
  // old lowering path.
  return {};
}

TensorList::TensorList(OpList values)
  : TsNode(/*op=*/tensor_list_opkind,
           /*operands=*/values,
           /*shapes=*/std::vector<Shape>(),
         /*num_outputs=*/1,
         /*hash_seed=*/OperandHashes(values, /*seed=*/kHashSeed, enableDynamicShape())) {}

TSOpVector TensorList::Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                             TSLoweringContext* loctx) const {

  std::vector<torch::jit::Value*> tensor_list;
  CHECK(!operands().empty());
  for (const torch::lazy::Output& operand : operands()) {
    tensor_list.emplace_back(loctx->GetOutputOp(operand));
  }
  auto graph = function->graph();
  auto listnode = graph->insertNode(graph->createList(tensor_list[0]->type(), tensor_list));
  return {listnode->output()};
}

}  // namespace lazy
}  // namespace torch
