#import <ATen/native/metal/MetalPrepackOpContext.h>

#include <torch/script.h>

namespace at {
namespace native {
namespace metal {

enum class NeuronType {
  None,
  Clamp,
  Relu,
  Sigmoid,
  Tanh,
};

struct Conv2DParams final {
  Conv2DParams() = delete;
  Conv2DParams(
      c10::IntArrayRef inputSizes,
      c10::IntArrayRef weightSizes,
      c10::IntArrayRef padding,
      c10::IntArrayRef stride,
      c10::IntArrayRef dilation,
      int64_t groups);

  std::vector<int64_t> output_sizes() const;
  bool isDepthwise() const;

  int64_t N; // batch size
  int64_t C; // channels
  int64_t H; // input height
  int64_t W; // input width
  int64_t OC; // output channels
  int64_t IC; // input channels
  int64_t KH; // kernel height
  int64_t KW; // kernel width
  int64_t SY; // stride y (height)
  int64_t SX; // stride x (width)
  int64_t PY; // padding y (height)
  int64_t PX; // padding x (width)
  int64_t DY; // dilation y (height)
  int64_t DX; // dilation x (width)
  int64_t G; // groups
  int64_t OW; // output width
  int64_t OH; // output height
};

NeuronType neuronType(const Conv2dOpContext& context);

} // namespace metal
} // namespace native
} // namespace at
