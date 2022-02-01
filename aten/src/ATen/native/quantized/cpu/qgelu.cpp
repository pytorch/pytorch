#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <c10/util/irange.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qgelu_stub);

Tensor gelu_quantized_cpu(const Tensor& qx, bool approximation) {
  Tensor qy;
  // no need to fix this. should be handled in #61439
  TORCH_INTERNAL_ASSERT(!approximation, "quantization not supported");
  qgelu_stub(qx.device().type(), qx, qy);
  return qy;
}
}}  // namespace at::native
