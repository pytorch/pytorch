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
    Tensor &result,
    float beta,
    float alpha) {
  TORCH_CHECK(mat1.scalar_type() == at::kBFloat16, "mkldnn_matmul:  only enabled for bf16 path");
  TORCH_CHECK(mat2.scalar_type() == at::kBFloat16, "mkldnn_matmul:  only enabled for bf16 path");
  TORCH_CHECK(result.scalar_type() == at::kBFloat16, "mkldnn_matmul:  only enabled for bf16 path");
  TORCH_CHECK(mkldnn_bf16_device_check(),
    "mkldnn_matmul: mkldnn_matmul bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");

  // In aten, expect beta * result = 0 if beta = 0
  if (beta == 0.0f)
    result.zero_();
  // If alpha = 0, dose not need actually do gemm computation
  if (alpha == 0)
    return;
  at::Tensor mat1_reshaped = mat1;
  at::Tensor mat2_reshaped = mat2;
 if (result.dim() == 2 && mat1.dim() == 3 && mat2.dim() == 3){
    // addbmm(batch1*batch2) [b,n,m] * [b,m,p] = [n,p] can be treated as:
    // [n, b*m] * [b*m, p] = [n, p]
    // For batch1: reorder from [b, n, m] to [n, b, m], reshape to [n, b*m]
    // For batch2: reshape from [b, m, p] to [b*m, p]
    auto mat1_size = mat1.sizes();
    auto mat2_size = mat2.sizes();
    auto mat1_ = mat1_size[0] > 1 ? mat1.transpose(0, 1) : mat1;
    mat1_reshaped = mat1_.reshape({mat1_size[1], mat1_size[0] * mat1_size[2]});
    mat2_reshaped = mat2.reshape({mat2_size[0] * mat2_size[1], mat2_size[2]});
 }

  // mkldnn_matmul only proceed CPU tensor
  const ideep::tensor x = itensor_view_from_dense(mat1_reshaped);
  const ideep::tensor w = itensor_view_from_dense(mat2_reshaped);
  ideep::tensor y = itensor_view_from_dense(result);
  ideep::matmul_forward::compute(x, w, y, alpha, beta,
      ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), ideep::attr_t::fuse_sum());
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
