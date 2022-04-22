#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include <torch/csrc/lazy/ts_backend/config.h>
#include <torch/csrc/lazy/core/cache.h>
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

TsNode::TsNode(OpKind op, OpList operands,
               const std::function<Shape()>& shape_fn,
               size_t num_outputs, hash_t hash_seed)
    : TsNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {
  shapes_.push_back(GetOpShape(shape_fn));
}

void TsNode::SetShapeDeferred(
    const std::function<Shape()>& shape_fn) {
  shapes_.push_back(GetOpShape(shape_fn));
}

using ShapeCache = Cache<hash_t, Shape, HashReducer>;

ShapeCache* GetShapeCache() {
  static ShapeCache* cache = new ShapeCache(FLAGS_torch_lazy_ts_shape_cache_size);
  return cache;
}

Shape TsNode::GetOpShape(
    const std::function<Shape()>& shape_fn) const {
  auto hash = hash_with_sizes();
  ShapeCache* shape_cache = GetShapeCache();
  auto shape = shape_cache->Get(hash);
  if (shape == nullptr) {
    shape = shape_cache->Add(hash,
                             std::make_shared<Shape>(shape_fn()));
  }
  return *shape;
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
