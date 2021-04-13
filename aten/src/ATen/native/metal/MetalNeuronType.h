#ifndef MetalNeuronType_h
#define MetalNeuronType_h

#import <ATen/native/metal/MetalPrepackOpContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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

static inline NeuronType neuronType(const Conv2dOpContext& context) {
  float inf_max = std::numeric_limits<float>::infinity();
  float inf_min = -std::numeric_limits<float>::infinity();
  float output_max = context.output_max.has_value()
      ? context.output_max.value().toFloat()
      : inf_max;
  float output_min = context.output_min.has_value()
      ? context.output_min.value().toFloat()
      : inf_min;
  if (output_max == inf_max && output_min == 0) {
    return NeuronType::Relu;
  } else if (output_max < inf_max && output_min > inf_min) {
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
  } else if (type == NeuronType::HardSigmoid){
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
