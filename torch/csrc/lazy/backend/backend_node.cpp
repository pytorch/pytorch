#include <torch/csrc/lazy/backend/backend_node.h>
#include <torch/csrc/lazy/backend/config.h>
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

BackendNode::BackendNode(OpKind op, OpList operands, std::vector<Shape>&& shapes,
               size_t num_outputs, hash_t hash_seed)
    : Node(op, num_outputs,
           // TODO(WHC) this is inefficient (having to compute node_hash twice
           // since I can't call hash() yet) so probably move dag_hash
           // initialization to a separate function?
           /* node_hash */ HashCombine(op.hash(), hash_seed),
           /* dag_hash */
           [&](bool bakeInSizes) { return OperandHashes(operands, HashCombine(op.hash(), hash_seed), bakeInSizes); }),
      shapes_(shapes),
      python_stacktrace_(GetFirstUserFrameInPythonIfEnabled()) {
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

BackendNode::BackendNode(OpKind op, OpList operands,
               const std::function<Shape()>& shape_fn,
               size_t num_outputs, hash_t hash_seed)
    : BackendNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {
  shapes_.push_back(GetOpShape(shape_fn));
}

BackendNode::BackendNode(OpKind op, OpList operands, size_t num_outputs,
               hash_t hash_seed)
    : BackendNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {}

void BackendNode::SetShapeDeferred(
    const std::function<Shape()>& shape_fn) {
  shapes_.push_back(GetOpShape(shape_fn));
}

BackendNode::BackendNode(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed)
    : Node(op, num_outputs, [&](bool bakeInSizes) -> hash_t { return GetOpHash(op, shape, hash_seed, bakeInSizes); }),
    python_stacktrace_(GetFirstUserFrameInPythonIfEnabled())
{
  shapes_.push_back(std::move(shape));
}

const Shape& BackendNode::shape(size_t output_index) const {
  return shapes_.at(output_index);
}

using ShapeCache = Cache<hash_t, Shape, HashReducer>;

ShapeCache* GetShapeCache() {
  static ShapeCache* cache = new ShapeCache(FLAGS_torch_lazy_tensors_shape_cache_size);
  return cache;
}

Shape BackendNode::GetOpShape(
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

std::string BackendNode::ToString() const {
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

hash_t BackendNode::GetOpHash(OpKind op, const Shape& shape, hash_t hash_seed, bool bakeInSizes) {
  hash_t h = HashCombine(op.hash(), shape.hash(bakeInSizes));
  return HashCombine(h, hash_seed);
}

void BackendNode::AddOperand(NodePtr node, size_t index) {
  CHECK_LT(index, node->num_outputs());
  operands_.push_back(std::move(node));
  operands_as_outputs_.emplace_back(operands_.back().get(), index);
}


TensorList::TensorList(OpList values)
  : BackendNode(
      /*op=*/tensor_list_opkind,
      /*operands=*/values,
      /*shapes=*/std::vector<Shape>(),
      /*num_outputs=*/1,
      /*hash_seed=*/OperandHashes(values, /*seed=*/kHashSeed, enableDynamicShape())) {}

const Shape& TensorList::shape(size_t output_index) const {
  TORCH_CHECK(false, "NotImplementedError");
}

}  // namespace lazy
}  // namespace torch
