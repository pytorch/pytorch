#ifndef MetalNeuronType_h
#define MetalNeuronType_h

#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace metal {

enum class NeuronType {
  None,
  Clamp,
  Relu,
  Sigmoid,
  HardSigmoid,
  Tanh,
};

static inline NeuronType neuronType(
    c10::optional<c10::Scalar> output_min,
    c10::optional<c10::Scalar> output_max) {
  float inf_max = std::numeric_limits<float>::infinity();
  float inf_min = -std::numeric_limits<float>::infinity();
  float output_max_ =
      output_max.has_value() ? output_max.value().toFloat() : inf_max;
  float output_min_ =
      output_min.has_value() ? output_min.value().toFloat() : inf_min;
  if (output_max_ == inf_max && output_min_ == 0) {
    return NeuronType::Relu;
  } else if (output_max_ < inf_max && output_min_ > inf_min) {
    return NeuronType::Clamp;
  } else {
    return NeuronType::None;
  }
}

static inline MPSCNNNeuron* neuronType(NeuronType type) {
  if (type == NeuronType::Relu) {
    return [MPSCNNNeuronOp relu];
  } else if (type == NeuronType::Sigmoid) {
    return [MPSCNNNeuronOp sigmoid];
  } else if (type == NeuronType::Tanh) {
    return [MPSCNNNeuronOp tanh];
  } else if (type == NeuronType::HardSigmoid) {
    if (@available(iOS 11.0, *)) {
      return [MPSCNNNeuronOp hardSigmoid];
    } else {
      return nil;
    }
  } else {
    return nil;
  }
}

}
}
}

#endif /* MetalNeuronType_h */
