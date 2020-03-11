#include <torch/csrc/jit/codegen/cuda/partition.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Check all outputs are:
//   1. TensorType
//   2. on the same device;
// TODO: update this when codegen can output scalar
static c10::optional<c10::Device> getDevice(const Value* const value) {
  if (!value->type()->isSubtypeOf(TensorType::get())) {
    // not tensor type, return false as the op is not outputing scalar.
    return c10::nullopt;
  }
  return value->type()->expect<TensorType>()->device();
}

static c10::optional<c10::Device> getDevice(const Node* const node) {
  auto outputs = node->outputs();
  if (outputs.size() == 0) {
    return c10::nullopt;
  }
  return getDevice(outputs[0]);
}

static bool isFusibleDevice(const Node* node, const c10::Device device) {
  for (auto value : node->outputs()) {
    auto output_device = getDevice(value);
    if (!output_device.has_value() || output_device.value() != device) {
      return false;
    }
  }
  return true;
}

// TODO: we need to check input type when we handle `to()`
static bool isFusibleDevice(const Node* node) {
  auto device = getDevice(node);
  if (!device.has_value()) {
    return false;
  }
  // Technically we don't need to check device for node->outputs()[0] again, meh
  return isFusibleDevice(node, device.value());
}

// TODO: fusible_ops should be a registry unordered_map<Node,Expr>
static OperatorSet fusible_ops = {
    // "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
    // "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
    // "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
    // "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
    // "aten::div(Tensor self, Tensor other) -> Tensor",
    // "aten::div(Tensor self, Scalar other) -> Tensor",
    // "aten::mul(Tensor self, Tensor other) -> Tensor",
    // "aten::mul(Tensor self, Scalar other) -> Tensor"
};

inline bool isFusibleNode(const Node* const node)  {
  // TODO: update code base so we can use `node->is_MemberOf(fusible_ops)`
  return ((node->isMemberOf(fusible_ops)) || node->kind() == prim::CudaFusionGroup);
}

} // namespace

bool isFusibleCudaFusionGroup(const Node* const node) {
  if (isFusibleNode(node)) {
    return isFusibleDevice(node);
  }
  return false;
}

bool isFusibleCudaFusionGroup(
    const Node* const fusion,
    const Node* const node) {
  if (isFusibleNode(node)) {
    auto device = getDevice(fusion);

    return (device.has_value() && isFusibleDevice(node, device.value()));
  }
  return false;
}

}}}}
