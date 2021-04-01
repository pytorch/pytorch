#ifndef MetalCopy_h
#define MetalCopy_h

#include <ATen/Tensor.h>

namespace at {
namespace native {
namespace metal {

// TODO: Remove the header once we are able to call it through dispatcher
Tensor t(const Tensor& input);

} // namespace metal
} // namespace native
} // namespace at

#endif
