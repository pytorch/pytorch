#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/mkldnn/Matmul.h>
#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

void mkldnn_matmul(
    const Tensor &mat1,
    const Tensor &mat2,
    Tensor &result,
    float beta,
    float alpha) {
  TORCH_CHECK(false, "mkldnn_matmul: ATen not compiled with MKLDNN support");
}
} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at {
namespace native {

void mkldnn_matmul(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &result,
    float beta,
    float alpha) {
  TORCH_CHECK((mat1.dim() == 2 && mat2.dim() == 2) || (mat1.dim() == 3 && mat2.dim() == 3),
    "mkldnn_matmul:  expect mat1 to be 2-D or 3-D tensor");
  TORCH_CHECK(mat1.scalar_type() == at::kBFloat16 &&
   mat2.scalar_type() == at::kBFloat16 &&
   result.scalar_type() == at::kBFloat16, "mkldnn_matmul:  only enabled for bf16 path");
  TORCH_CHECK(mkldnn_bf16_device_check(),
    "mkldnn_matmul: mkldnn_matmul bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  ideep::attr_t op_attr;
  // "addmm", "addbmm" "baddbmm" in pytorch allow bias to be 2-D or 3-D tensor
  // but mkldnn matmul primitive only support bias be 1-D tensors
  // to address their differences, we use mkldnn post ops to perform a fused "add" after matrix multiplication is over
  if (beta != 0.0f) op_attr = ideep::attr_t::fuse_sum();
  // If alpha = 0, dose not need actually do gemm computation
  if (alpha == 0)
    return;

  auto is_mkldnn_optimized_format = [&](const Tensor& t) {
    if (t.is_contiguous()) return true;
    const auto sizes = t.sizes();
    const auto strides = t.strides();
    if (t.dim() == 2){
      return strides[0] == 1 && strides[1] == sizes[0];
    } else {
      // dim = 3
      return strides[0] == sizes[1] * sizes[2] && strides[1] == 1 && strides[2] == sizes[1];
    }
  };

  // Mkldnn only optimized for contiguous or transposed (transpose last 2 dim if 3-D tensor) format now
  // Will remove this "contiguous" after mkldnn have fully supported
  Tensor mat1_ = is_mkldnn_optimized_format(mat1) ? mat1 : mat1.contiguous();
  Tensor mat2_ = is_mkldnn_optimized_format(mat2) ? mat2 : mat2.contiguous();
  Tensor mat1_reshaped = mat1_;
  Tensor mat2_reshaped = mat2_;
  if (result.dim() == 2 && mat1.dim() == 3 && mat2.dim() == 3){
    // addbmm(batch1*batch2) [b,n,m] * [b,m,p] = [n,p] can be treated as:
    // [n, b*m] * [b*m, p] = [n, p]
    // For batch1: reorder from [b, n, m] to [n, b, m], reshape to [n, b*m]
    // For batch2: reshape from [b, m, p] to [b*m, p]
    auto mat1_size = mat1.sizes();
    auto mat2_size = mat2.sizes();
    mat1_ = mat1_size[0] > 1 ? mat1_.transpose(0, 1) : mat1_;
    mat1_reshaped = mat1_.reshape({mat1_size[1], mat1_size[0] * mat1_size[2]});
    mat2_reshaped = mat2_.reshape({mat2_size[0] * mat2_size[1], mat2_size[2]});
 }

  // mkldnn_matmul only proceed CPU tensor
  const ideep::tensor x = itensor_view_from_dense(mat1_reshaped);
  const ideep::tensor w = itensor_view_from_dense(mat2_reshaped);
  ideep::tensor y = itensor_view_from_dense(result);
  ideep::matmul_forward::compute(x, w, y, alpha, beta,
      ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), op_attr);
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
