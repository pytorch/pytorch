#include <ATen/Tensor.h>

namespace at {
namespace native {
namespace metal {

uint32_t batchSize(const Tensor& tensor);
uint32_t channelsSize(const Tensor& tensor);
uint32_t heightSize(const Tensor& tensor);
uint32_t widthSize(const Tensor& tensor);

} // namespace metal
} // namespace native
} // namespace at
