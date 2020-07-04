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
  for (auto output : outputs) {
    auto device = getDevice(output);
    if (device.has_value()) {
      return device;
    }
  }
  return c10::nullopt;
}

static bool isFusableDevice(const Node* node, const c10::Device device) {
  for (auto value : node->outputs()) {
    auto output_device = getDevice(value);
    if (output_device.has_value() && output_device.value() != device) {
      return false;
    }
  }
  return true;
}

// TODO: we need to check input type when we handle `to()`
static bool isFusableDevice(const Node* node) {
  auto device = getDevice(node);
  if (!device.has_value()) {
    return true;
  }
  return device->is_cuda();
}

inline bool isFusableNode(const Node* const node) {
  // checks if node is compatible with parser:
  // 1. if we have a parsing rule; or 2. if the node is already a fusion group.
  return (isNodeParsible(node) || node->kind() == prim::CudaFusionGroup);
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
  if (isFusableCudaFusionGroup(node)) {
    // TODO: ensure legit fusion.
    // issue 0: currently codegen doesn't support broadcasting, except in the
    //          form of stride 0.
    // We WAR by explicitly extend all tensor to be the broadcasted size. This
    // however requires that we have identical tensor shape for all output
    // tensors.
    // We previously have a check, where for any `node` that we try to fuse, all
    // `auto output : node->outputs(); auto use : output->uses();` has to meet
    // one of the three conditions:
    //   a. use.user == fusion;
    //   b. node->outputs() sizes are compatible with `fusion` outputs;
    //   c. isFusableNode(use.user) && use.user->outputs() sizes are compatible
    //      with `fusion` outputs;
    //
    // However, given the instance of legacy executor, it is not guaranteed the
    // necessary shape information is available to do the check. Hence we are
    // omitting it for now and we'll wait until proper support from profiling is
    // implemented to justify another look at this.
    // And the broadcasting Hack won't be applicable after reduction is
    // supported in codegen. So it's going to be a more complicated story.
    //
    // For now, a voilating fusion would result in no codegen kernel (fallback
    // execution with interpreter and non-optimized graph is used instead)

    // ensure if the node has a designated device, it's on the same device with
    // fusion.
    // TODO: is there a danger of us fusing operations that's supposed to be on
    //       separate GPUs? And is that necessarily bad?
    auto device = getDevice(fusion);
    return (!device.has_value() || isFusableDevice(node, device.value()));
  }
  return false;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
