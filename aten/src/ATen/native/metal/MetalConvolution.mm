#import <ATen/native/metal/MetalConvolution.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNOps.h>

namespace at {
namespace native {
namespace metal {

Conv2DParams::Conv2DParams(
    c10::IntArrayRef inputSizes,
    c10::IntArrayRef weightSizes,
    c10::IntArrayRef padding,
    c10::IntArrayRef stride,
    c10::IntArrayRef dilation,
    int64_t groups)
    : N(inputSizes[0]),
      C(inputSizes[1]),
      H(inputSizes[2]),
      W(inputSizes[3]),
      OC(weightSizes[0]),
      IC(weightSizes[1]),
      KH(weightSizes[2]),
      KW(weightSizes[3]),
      SY(stride[0]),
      SX(stride[1]),
      PY(padding[0]),
      PX(padding[1]),
      DY(dilation[0]),
      DX(dilation[1]),
      G(groups) {
  OH = std::floor((H + 2 * PY - DY * (KH - 1) - 1) / SY + 1);
  OW = std::floor((W + 2 * PX - DX * (KW - 1) - 1) / SX + 1);
};

std::vector<int64_t> Conv2DParams::output_sizes() const {
  return {N, OC, OH, OW};
}

bool Conv2DParams::isDepthwise() const {
  // Currently, only channel multipler of 1 is supported
  // i.e. inputFeatureChannels == outputFeatureChannels
  return G > 1 && IC == 1 && OC == G && OC == C;
}

NeuronType neuronType(const Conv2dOpContext& context) {
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

} // namespace metal
} // namespace native
} // namespace at
