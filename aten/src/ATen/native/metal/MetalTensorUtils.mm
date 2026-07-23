#import <ATen/native/metal/MetalTensorUtils.h>

namespace at {
namespace native {
namespace metal {

uint32_t batchSize(const Tensor& tensor) {
  const IntArrayRef sizes = tensor.sizes();
  const uint32_t dims = tensor.dim();
  if (dims < 4) {
    return 1;
  }
  return sizes[dims - 4];
}

uint32_t channelsSize(const Tensor& tensor) {
  const IntArrayRef sizes = tensor.sizes();
  const uint32_t dims = tensor.dim();
  if (dims < 3) {
    return 1;
  }
  return sizes[dims - 3];
}

uint32_t heightSize(const Tensor& tensor) {
  const IntArrayRef sizes = tensor.sizes();
  const uint32_t dims = tensor.dim();
  if (dims < 2) {
    return 1;
  }
  return sizes[dims - 2];
}

uint32_t widthSize(const Tensor& tensor) {
  const IntArrayRef sizes = tensor.sizes();
  const uint32_t dims = tensor.dim();
  if (dims < 1) {
    return 1;
  }
  return sizes[dims - 1];
}

}
}
}
