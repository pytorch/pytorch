#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>
#include <c10/core/Device.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/device_type_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <memory>
#include <optional>
#include <utility>

namespace torch::jit {

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

bool setDeviceType(Value* value, std::optional<Device> device) {
  auto tensor_type = value->type()->expect<TensorType>();
  bool changed = tensor_type->device() != device;
  if (changed) {
    value->setType(tensor_type->withDevice(device));
  }
  return changed;
}

bool setReturnsToDevice(Node* n, std::optional<Device> device) {
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

bool isZerodimCPUTensor(const std::shared_ptr<TensorType>& tensor_type) {
  // CPU devices on zerodim tensors are the only device that can be
  // overwritten by another device. Therefore, to be conservative
  // assume that it is not a zerodim cpu tensor if something is not known.
  bool is_zerodim = tensor_type->symbolic_sizes().rank().value_or(-1) == 0;
  bool is_cpu = tensor_type->device() && tensor_type->device()->is_cpu();
  return is_zerodim && is_cpu;
}

bool propWithNoDevice(Node* n) {
  // Propagate if we can verify that all input devices match,
  // except CPU zerodim, which any other type can overwrite
  size_t input_num = 0;

  for (; input_num < n->inputs().size(); input_num++) {
    if (n->inputs()[input_num]->type()->cast<TensorType>()) {
      break;
    }
  }
  if (input_num == n->inputs().size()) {
    // No tensor found
    return setReturnsToDevice(n, std::nullopt);
  }

  auto tensor_type = n->inputs()[input_num]->type()->expect<TensorType>();
  bool only_seen_cpu_zerodim = isZerodimCPUTensor(tensor_type);
  std::optional<Device> device = tensor_type->device();

  // Now see if all inputs have a consistent device type
  for (input_num++; input_num < n->inputs().size(); input_num++) {
    auto tensor_type = n->inputs()[input_num]->type()->cast<TensorType>();
    if (!tensor_type || isZerodimCPUTensor(tensor_type)) {
      continue;
    }

    if (device != tensor_type->device()) {
      if (only_seen_cpu_zerodim) {
        device = tensor_type->device();
        only_seen_cpu_zerodim = false;
      } else {
        // Bail on the type not match case
        return setReturnsToDevice(n, std::nullopt);
      }
    }
  }
  return setReturnsToDevice(n, device);
}

bool defaultDeviceProp(Node* n) {
  // Detecting if the op has a device object argument
  // as there is implicit string conversion to device
  auto schema = n->maybeSchema();
  if (!schema) {
    return false;
  }
  auto arguments = schema->arguments();
  for (size_t i = 0; i < arguments.size(); i++) {
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

struct DeviceTypePropagationPass : public PropertyPropBase {
  explicit DeviceTypePropagationPass(std::shared_ptr<Graph> graph)
      : PropertyPropBase(std::move(graph)) {
    buildRuleRegistry();
  }

  // returns true if at least one node has its scalar type set on a tensor node
  bool run() {
    propagateBlock(graph_->block(), false);
    return changed_;
  }

 private:
  void propagateNode(Node* n, bool _ = false) override {
    GRAPH_DEBUG("processNode");
    switch (n->kind()) {
      case prim::If:
        return processIf(n);
      case prim::Loop:
        return processLoop(n);
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
        // This is already been propagated by something else
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

} // namespace torch::jit
