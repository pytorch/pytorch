#include <torch/nativert/graph/GraphUtils.h>

#include <c10/core/Device.h>

#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

bool areAllIOTensorsAttributesOnCpu(const Node& node) {
  const auto& tensorValuesMeta = node.owningGraph()->tensorValuesMeta();

  // Check inputs
  for (auto& input : node.inputs()) {
    if (input.value->type() == Type::Kind::Tensor) {
      if (auto it = tensorValuesMeta.find(std::string{input.value->name()});
          it != tensorValuesMeta.end()) {
        const auto& device = it->second.device();
        if (!device.is_cpu()) {
          return false;
        }
      }
    } else if (input.value->type() == Type::Kind::TensorList) {
      for (const auto& el : input.value->getListElements()) {
        if (auto it = tensorValuesMeta.find(std::string{el->name()});
            it != tensorValuesMeta.end()) {
          const auto& device = it->second.device();
          if (!device.is_cpu()) {
            return false;
          }
        }
      }
    } else {
      // other input types doesn't affect if the node is on CPU or not
    }
  }

  // Check outputs
  for (auto& output : node.outputs()) {
    if (!output) {
      // When a node's output is a Constant, its Value* is nullptr
      // TODO: this is breaking the invariant of all nodes outputs are non-null
      // in the graph. We should fix this.
      continue;
    }
    if (output->type() == Type::Kind::Tensor) {
      if (auto it = tensorValuesMeta.find(std::string{output->name()});
          it != tensorValuesMeta.end()) {
        const auto& device = it->second.device();
        if (!device.is_cpu()) {
          return false;
        }
      }
    } else if (output->type() == Type::Kind::TensorList) {
      for (const auto& el : output->getListElements()) {
        if (auto it = tensorValuesMeta.find(std::string{el->name()});
            it != tensorValuesMeta.end()) {
          const auto& device = it->second.device();
          if (!device.is_cpu()) {
            return false;
          }
        }
      }
    } else {
      // other output types doesn't affect if the node is on CPU or not
    }
  }

  // Check attributes
  for (auto& attribute : node.attributes()) {
    if (std::holds_alternative<c10::Device>(attribute.value)) {
      auto device = std::get<c10::Device>(attribute.value);
      if (!device.is_cpu()) {
        return false;
      }
    }
  }
  return true;
}

} // namespace torch::nativert
