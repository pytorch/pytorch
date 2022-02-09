#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

namespace at {

namespace native {

Tensor ffn_cpu(const Tensor& input, const Tensor& w1, const Tensor& b1, const Tensor& w2, const Tensor& b2, bool use_gelu, bool add_norm){
  TORCH_CHECK(add_norm == false, "TODO add_norm to be supported in FFN");
  TORCH_CHECK(use_gelu == false, "TODO gelu to be supported in FFN");
  Tensor res = at::baddbmm(b1, input, w1);
  if (!use_gelu) {
    res = at::relu(res);
  }
  res = at::baddbmm(b2, res, w2);
  return res;
}
} // namespace native
} // namespace at
