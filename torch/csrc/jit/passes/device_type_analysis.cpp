#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>
#include <c10/core/Device.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/device_type_analysis.h>
#include <torch/library.h>
#include <cstddef>
#include <memory>
#include <utility>

namespace torch {
namespace jit {

namespace {

using Tensor = at::Tensor;
using Device = at::Device;

using PropRule = std::function<bool(Node*)>;
/*
A Propagation Rule takes the Node, and
applies the relevant properties to the Tensor outputs
of the Node (based on the rule itself)

Returns: Bool indicating if anything was changed
*/

bool setDeviceType(Value* value, c10::optional<Device> device) {
  auto tensor_type = value->type()->cast<TensorType>();
  TORCH_INTERNAL_ASSERT(tensor_type, "Expecting a tensor type");
  bool changed = tensor_type->device() != device;
  if (changed) {
    value->setType(tensor_type->withDevice(device));
  }
  return changed;
}

bool setReturnsToDevice(Node* n, c10::optional<Device> device) {
  bool changed = false;
  for (Value* out : n->outputs()) {
    auto tensor_type = out->type()->cast<TensorType>();
    if (!tensor_type) {
      continue;
    }
    changed |= setDeviceType(out, device);
  }
  return changed;
}

PropRule setReturnstoDeviceRule(DeviceType deviceType) {
  Device device = Device(deviceType);
  return [=](Node* n) { return setReturnsToDevice(n, device); };
}

bool returnFirstArgDeviceRule(Node* n) {
  // Custom Rule for when multiple args can have mismatched device types
  auto tensor_type = n->inputs()[0]->type()->cast<TensorType>();
  TORCH_INTERNAL_ASSERT(tensor_type, "Expecting a tensor type");
  return setReturnsToDevice(n, tensor_type->device());
}

bool returnSecondArgDeviceRule(Node* n) {
  // Custom Rule for when multiple args can have mismatched device types
  auto tensor_type = n->inputs()[1]->type()->cast<TensorType>();
  TORCH_INTERNAL_ASSERT(tensor_type, "Expecting a tensor type");
  return setReturnsToDevice(n, tensor_type->device());
}

bool propWithNoDevice(Node* n) {
  // Figure out what the common device to propagate is
  // Types of tensors must match, except CPU zerodim, which any
  // other type can overwrite

  c10::optional<Device> device;
  bool seen_any_device = false;
  bool only_seen_cpu_zerodim = false;

  for (Value* inp : n->inputs()) {
    auto tensor_type = inp->type()->cast<TensorType>();
    if (!tensor_type) {
      continue;
    }

    bool is_zerodim = tensor_type->symbolic_sizes().rank().value_or(-1) == 0;
    bool is_cpu = tensor_type->device() && tensor_type->device()->is_cpu();
    bool is_cpu_zerodim = is_zerodim && is_cpu;

    if (seen_any_device) {
      if (device != tensor_type->device() && !is_cpu_zerodim) {
        if (only_seen_cpu_zerodim) {
          device = tensor_type->device();
          only_seen_cpu_zerodim = false;
        } else {
          // Bail on the type not match case
          return setReturnsToDevice(n, c10::nullopt);
        }
      }
    } else {
      seen_any_device = true;
      only_seen_cpu_zerodim = is_cpu_zerodim;
      device = tensor_type->device();
    }
  }
  return setReturnsToDevice(n, device);
}

bool defaultDeviceProp(Node* n) {
  // Detecting of the op has a device object argument
  // as there is implicit string conversion to device
  auto schema = n->maybeSchema();
  if (!schema) {
    return false;
  }
  auto arguments = schema->arguments();
  auto input_vec = n->inputs();
  for (int i = 0; i < arguments.size(); i++) {
    Argument& argument = arguments[i];
    if (DeviceObjType::get()->isSubtypeOf(argument.type())) {
      // Optional args are filled in by torchscript with default val
      auto input_val = toIValue(n->inputs().at(i));
      if (!input_val.has_value()) {
        // Can't propagate if there is a dynamic device type
        return false;
      }
      if (input_val->isNone()) {
        continue;
      }
      if (!input_val->isDevice()) {
        // Bail on union types
        return false;
      }
      TORCH_INTERNAL_ASSERT(input_val->isDevice())
      Device device = input_val->toDevice();
      return setReturnsToDevice(n, device);
    }
  }
  return propWithNoDevice(n);
}

struct DeviceTypePropagationPass {
  explicit DeviceTypePropagationPass(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {
    buildRuleRegistry();
  }

  // returns true if at least one node has its scalar type set on a tensor node
  bool run() {
    processBlock(graph_->block());
    return changed_;
  }

 private:
  void processBlock(Block* block) {
    GRAPH_DEBUG("processBlock");
    for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
      processNode(*it);
    }
  }

  void processNode(Node* n) {
    GRAPH_DEBUG("processNode");
    switch (n->kind()) {
      case prim::If:
        return processIf(n);
      case prim::Loop:
      case prim::CallMethod:
      case prim::CallFunction:
        return; // Not handled for now
      default:
        break;
    }

    bool has_tensor_output =
        std::any_of(n->outputs().begin(), n->outputs().end(), [](Value* v) {
          return (bool)v->type()->cast<TensorType>();
        });

    if (!has_tensor_output) {
      // if output contains no tensor, nothing to propagate
      return;
    }

    switch (n->kind()) {
      case prim::Constant:
        // This is already been propagated by something else in freezing
      case prim::ListConstruct:
      case prim::ListUnpack:
        return; // Not handled for now
      default:
        if (n->kind().is_aten()) {
          return processAtenOps(n);
        } else {
          return; // Not handled for now
        }
    }
  }

  bool mergeAndApplyTensorProps(
      const at::ArrayRef<Value*>& src1,
      const at::ArrayRef<Value*>& src2,
      const at::ArrayRef<Value*>& dst) {
    bool changed = false;
    TORCH_INTERNAL_ASSERT(src1.size() == src2.size());
    TORCH_INTERNAL_ASSERT(src1.size() == dst.size());

    for (int i = 0; i < dst.size(); i++) {
      auto src1_type = src1[i]->type()->cast<TensorType>();
      auto src2_type = src2[i]->type()->cast<TensorType>();
      if (!src1_type || !src2_type) {
        continue;
      }

      if (!src1_type->device() || !src2_type->device() ||
          !(src1_type->device().value() == src2_type->device().value())) {
        changed |= setDeviceType(dst[i], c10::nullopt);
      } else {
        changed |= setDeviceType(dst[i], src1_type->device());
      }
    }
    return changed;
  }

  /*
  bool processLoop(Node* n) {
    GRAPH_DEBUG("processLoop");
  }
  */

  void processIf(Node* node) {
    GRAPH_DEBUG("processIf");
    auto blocks = node->blocks();
    auto true_block = blocks.at(0);
    auto false_block = blocks.at(1);

    processBlock(true_block);
    processBlock(false_block);

    mergeAndApplyTensorProps(
        true_block->outputs(), false_block->outputs(), node->outputs());
  }
  // for efficiency
  void processAtenOps(Node* n) {
    GRAPH_DEBUG("processAtenOps");
    GRAPH_DEBUG("case = ", n->kind(), " ", *n);
    // Custom Rule Matching
    auto op = n->maybeOperator();
    if (!op) {
      return;
    }
    auto prop_fn = device_prop_registry_->find(*op);
    if (prop_fn) {
      PropRule rule = *prop_fn;
      changed_ |= rule(n);
      return;
    }
    changed_ |= defaultDeviceProp(n);
  }

  void buildRuleRegistry() {
    // building a registry for all of the custom Device Type rules
    if (device_prop_registry_)
      return;

    static OperatorMap<PropRule> temp_registry{
        {"aten::cpu(Tensor self) -> Tensor",
         setReturnstoDeviceRule(DeviceType::CPU)},
        {"aten::cuda(Tensor self) -> Tensor",
         setReturnstoDeviceRule(DeviceType::CUDA)},
        {"aten::to_mkldnn(Tensor self, ScalarType? dtype) -> Tensor",
         setReturnstoDeviceRule(DeviceType::MKLDNN)},
        {"aten::reshape_as(Tensor self, Tensor other) -> Tensor",
         returnFirstArgDeviceRule},
        {"aten::view_as(Tensor self, Tensor other) -> Tensor",
         returnFirstArgDeviceRule},
        {"aten::expand_as(Tensor self, Tensor other) -> Tensor",
         returnFirstArgDeviceRule},
        {"aten::type_as(Tensor self, Tensor other) -> Tensor",
         returnSecondArgDeviceRule},
    };
    device_prop_registry_ =
        std::make_unique<OperatorMap<PropRule>>(std::move(temp_registry));
  }

  static std::unique_ptr<OperatorMap<PropRule>> device_prop_registry_;
  std::shared_ptr<Graph> graph_;
  bool changed_ = false;
};

std::unique_ptr<OperatorMap<PropRule>>
    DeviceTypePropagationPass::device_prop_registry_ = nullptr;

} // anonymous namespace

// This analysis propagates input device types (if any) throughout the
// graph.
bool DeviceTypePropagation(std::shared_ptr<Graph>& graph) {
  auto tp = std::make_unique<DeviceTypePropagationPass>((graph));
  bool changed = tp->run();
  if (changed) {
    GRAPH_DUMP("After TensorPropertyPropagation pass:", graph);
  }
  return changed;
}

} // namespace jit
} // namespace torch
