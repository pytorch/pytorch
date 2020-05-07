#include <torch/csrc/jit/codegen/cuda/partition.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>

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

static bool isFusableDevice(const Node* node, const c10::Device device) {
  for (auto value : node->outputs()) {
    auto output_device = getDevice(value);
    if (!output_device.has_value() || output_device.value() != device) {
      return false;
    }
  }
  return true;
}

// TODO: we need to check input type when we handle `to()`
static bool isFusableDevice(const Node* node) {
  auto device = getDevice(node);
  if (!device.has_value()) {
    return false;
  }
  // Technically we don't need to check device for node->outputs()[0] again, meh
  return isFusableDevice(node, device.value());
}

inline bool isFusableNode(const Node* const node) {
  // checks if node is compatible with parser:
  // 1. if we have a parsing rule; or 2. if the node is already a fusion group.
  return (isNodeParsible(node) || node->kind() == prim::CudaFusionGroup);
}

// TODO: how would symbolic shape from profiling executor play with this?
static bool compatible_broadcast_shape(
    const c10::VaryingShape<int64_t>& e,
    const c10::VaryingShape<int64_t>& a) {
  if (e.isComplete() && a.isComplete()) {
    auto e_size = e.concrete_sizes().value();
    auto a_size = a.concrete_sizes().value();
    for (size_t i = 0; i < e_size.size(); i++) {
      if (e_size[i] != a_size[i]) {
        return false;
      }
    }
    return true;
  }
  return false;
}

} // namespace

bool isFusableCudaFusionGroup(const Node* const node) {
  if (isFusableNode(node)) {
    return isFusableDevice(node);
  }
  return false;
}

bool isFusableCudaFusionGroup(
    const Node* const fusion,
    const Node* const node) {
  if (isFusableNode(node)) {
    auto device = getDevice(fusion);

    auto tensor_type = fusion->outputs()[0]->type()->cast<TensorType>();
    if (tensor_type) {
      for (auto output : node->outputs()) {
        // We only check shape of tensor output
        auto output_type = output->type()->cast<TensorType>();
        if (output_type) {
          bool output_tensor = false;
          for (auto use : output->uses()) {
            if (use.user != fusion) {
              output_tensor = true;
              break;
            }
          }
          // if the output is not used by outside, there's no need to check its
          // shape
          if (output_tensor &&
              !compatible_broadcast_shape(
                  tensor_type->sizes(), output_type->sizes())) {
            return false;
          }
        }
      }
      return (device.has_value() && isFusableDevice(node, device.value()));
    }
  }
  return false;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
