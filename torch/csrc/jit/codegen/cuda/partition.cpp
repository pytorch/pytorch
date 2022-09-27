#include <torch/csrc/jit/codegen/cuda/partition.h>

#include <ATen/core/jit_type.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

const c10::DeviceIndex INVALID_INDEX = -2;

namespace {

bool hasNonElementWiseOperation(const Node* node) {
  if (node->kind() == prim::CudaFusionGroup) {
    for (auto n : node->g(attr::Subgraph)->nodes()) {
      if (hasNonElementWiseOperation(n)) {
        return true;
      }
    }
  } else {
    // prim::Constant is not parsible, but it is also not nonElementWise
    if (node->kind() != prim::Constant && !isElementWiseNode(node)) {
      return true;
    }
  }
  return false;
}

// Check all outputs are:
//   1. TensorType
//   2. on the same device;
// TODO: update this when codegen can output scalar
static c10::optional<c10::Device> getDevice(const Value* value) {
  if (!value->type()->isSubtypeOf(*TensorType::get())) {
    // not tensor type, return false as the op is not outputing scalar.
    return c10::nullopt;
  }
  auto tensor_type = value->type()->expectRef<TensorType>();
  // special case for scalar tensor: return c10::nullopt instead of cpu device.
  // this allows us to fuse scalar cpu tensor with cuda tensor, while avoid
  // merging ops with pure scalar cpu tensors.
  if (is_cpu_scalar(tensor_type)) {
    return c10::nullopt;
  }
  return tensor_type.device();
}

static bool hasBfloat(const Node* node) {
  auto has_bfloat = [](const Value* value) {
    if (!value->type()->isSubtypeOf(*TensorType::get())) {
      return false;
    }
    auto opt_scalar_type = value->type()->expectRef<TensorType>().scalarType();
    if (opt_scalar_type.has_value() &&
        opt_scalar_type.value() == at::ScalarType::BFloat16) {
      return true;
    }
    return false;
  };

  if (std::any_of(node->inputs().begin(), node->inputs().end(), has_bfloat) ||
      std::any_of(node->outputs().begin(), node->outputs().end(), has_bfloat)) {
    return true;
  }
  return false;
}

static c10::optional<c10::Device> getDevice(const Node* node) {
  c10::optional<c10::Device> ret = c10::nullopt;
  auto merge_devices = [&ret](const c10::optional<c10::Device>& device) {
    if (device.has_value()) {
      if (ret.has_value()) {
        if (ret.value() != device.value()) {
          // invalidate device to reflect conflicts
          ret->set_index(INVALID_INDEX);
          // return false to indicate early termination
          return false;
        } else {
          // same device, do nothing
          return true;
        }
      } else {
        // initialize return device
        ret = device.value();
        return true;
      }
    }
    // no device information, do nothing
    return true;
  };
  for (auto val : node->inputs()) {
    if (!merge_devices(getDevice(val))) {
      return ret;
    }
  }
  for (auto val : node->outputs()) {
    if (!merge_devices(getDevice(val))) {
      return ret;
    }
  }
  return ret;
}

static bool isDeviceCompatible(const Node* node, const c10::Device& device) {
  // only fuses cuda device
  if (!device.is_cuda()) {
    GRAPH_UPDATE("rejecting node (non-cuda device): ", *node);
    return false;
  }
  const auto major = at::cuda::getDeviceProperties(device.index())->major;
  // disable non-elementwise fusion on pre-volta devices
  if (major < 7 && hasNonElementWiseOperation(node)) {
    GRAPH_UPDATE(
        "rejecting node (non element-wise op not supported on SM < 7X): ",
        *node);
    return false;
  }
  // disable bfloat fusion on pre-ampere devices
  if (major < 8 && hasBfloat(node)) {
    GRAPH_UPDATE("rejecting node (bfloat not supported on SM < 8X): ", *node);
    return false;
  }
  return true;
}

static bool isFusibleDevice(const Node* node, const c10::Device& device) {
  TORCH_INTERNAL_ASSERT(
      device.index() != INVALID_INDEX, "fusible device needs to be validate");
  auto opt_device = getDevice(node);
  // we can be more relaxed here as we known that this function tries to merge
  // node into an existing `device`
  if (opt_device.has_value() &&
      (opt_device->index() == INVALID_INDEX || opt_device != device)) {
    GRAPH_UPDATE(
        "rejecting node from fusion (outputs device not matching fusion): ",
        *node);
    return false;
  }
  if (!isDeviceCompatible(node, device)) {
    return false;
  }
  return true;
}

// TODO: we need to check input type when we handle `to()`
static bool isFusibleDevice(const Node* node) {
  auto device = getDevice(node);
  // be conservative and only fuse cuda operations, this avoids us initializing
  // operations that produces cpu scalar outputs
  if (!device.has_value() || device->index() == INVALID_INDEX) {
    return false;
  }

  if (!isDeviceCompatible(node, device.value())) {
    return false;
  }
  return true;
}

bool compatibleType(const torch::jit::Value* val) {
  if (auto tensor_type = val->type()->cast<c10::TensorType>()) {
    if (tensor_type->scalarType().has_value()) {
      if (aten_to_data_type(tensor_type->scalarType().value()) ==
          DataType::Null) {
        return false;
      }
      if (!isOptionEnabled(EnableOption::Complex)) {
        // Complex is disabled by default until its support is completely added
        // TODO: remove this logic
        if (isComplexType(
                aten_to_data_type(tensor_type->scalarType().value()))) {
          return false;
        }
      }
    }
    // magic number 8 here since our kernel argument only supports rank <= 8
    if (tensor_type->dim().has_value() && (tensor_type->dim().value() > 8)) {
      return false;
    }
  }
  return true;
}

bool checkInputTensorTypes(const Node* node) {
  for (const auto i : c10::irange(node->inputs().size())) {
    const auto& val = node->inputs()[i];
    if (!compatibleType(val)) {
      // special case on aten::_batch_norm_impl_index_backward, the 11th output
      // is going to be discarded, so no need to check data type there.
      if (node->kind() ==
              c10::Symbol::fromQualString(
                  "aten::_batch_norm_impl_index_backward") &&
          i == 11) {
        continue;
      }
      return false;
    }
  }
  return true;
}

bool checkOutputTensorTypes(const Node* node) {
  for (const auto i : c10::irange(node->outputs().size())) {
    const auto& val = node->outputs()[i];
    if (!compatibleType(val)) {
      // special case on aten::_batch_norm_impl_index, the 4th output
      // is going to be discarded, so no need to check data type there.
      if (node->kind() ==
              c10::Symbol::fromQualString("aten::_batch_norm_impl_index") &&
          i == 3) {
        continue;
      }
      return false;
    }
  }
  return true;
}

inline bool isFusibleNode(const Node* node) {
  // Check if already part of a fusion group
  if (node->kind() == prim::CudaFusionGroup)
    return true;
  // Check we have a parsing rule
  if (!isNodeParsible(node)) {
    // ignoring profile nodes & constant nodes to avoid noise from debugging
    if (node->kind() != prim::Constant &&
        node->kind() != prim::profile_ivalue && node->kind() != prim::profile &&
        node->kind() != prim::Param) {
      GRAPH_UPDATE("rejecting node from fusion (node not parsible): ", *node);
    }
    return false;
  }
  // Check if we have a tensor type it's one we support
  if (!checkInputTensorTypes(node)) {
    GRAPH_UPDATE(
        "rejecting node from fusion (input scalar type not supported): ",
        *node);
    return false;
  }
  if (!checkOutputTensorTypes(node)) {
    GRAPH_UPDATE(
        "rejecting node from fusion (output scalar type not supported): ",
        *node);
    return false;
  }
  return true;
}

} // namespace

bool isFusibleCudaFusionGroup(const Node* node) {
  FUSER_PERF_SCOPE("isFusibleCudaFusionGroup");

  if (isFusibleNode(node)) {
    auto ret = isFusibleDevice(node);
    return ret;
  }
  return false;
}

bool isFusibleCudaFusionGroup(const Node* fusion, const Node* node) {
  FUSER_PERF_SCOPE("isFusibleCudaFusionGroup");
  bool fused = false;
  // TODO: lift the restriction of not fusing producer containing reduction when
  //       we have proper scheduling.
  if (isFusibleNode(node)) {
    // ensure if the node has a designated device, it's on the same device with
    // fusion.
    // TODO: is there a danger of us fusing operations that's supposed to be on
    //       separate GPUs? And is that necessarily bad?
    auto device = getDevice(fusion);
    fused = (!device.has_value() || isFusibleDevice(node, device.value()));
  }
  return fused;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
