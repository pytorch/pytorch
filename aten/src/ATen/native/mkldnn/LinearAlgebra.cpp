#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()
namespace at {
namespace native {

at::Tensor mkldnn_mm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(false, "mkldnn_mm: ATen not compiled with MKLDNN support");
}

at::Tensor& mkldnn_mm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& result) {
  TORCH_CHECK(false, "mkldnn_mm_out: ATen not compiled with MKLDNN support");
}

at::Tensor& mkldnn_bmm_out(const at::Tensor& mat1, const at::Tensor& mat2, at::Tensor& result) {
  TORCH_CHECK(false, "mkldnn_bmm_out: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_addmm_out(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Tensor &result) {
  TORCH_CHECK(false, "mkldnn_addmm_out: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_addbmm_out(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Tensor &result) {
  TORCH_CHECK(false, "mkldnn_addbmm_out: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_baddbmm_(Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  TORCH_CHECK(false, "mkldnn_baddbmm_: ATen not compiled with MKLDNN support");
}
} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/Resize.h>
namespace at {
namespace native {

void matmul_common(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &bias,
    Tensor &result,
    Scalar beta=1,
    Scalar alpha=1,
    const ideep::attr_t& attr = ideep::attr_t()) {
  TORCH_CHECK(mat1.scalar_type() == at::kBFloat16, "mkldnn_matmul:  only enabled for bf16 path");
  TORCH_CHECK(mat2.scalar_type() == at::kBFloat16, "mkldnn_matmul:  only enabled for bf16 path");
  TORCH_CHECK(result.scalar_type() == at::kBFloat16, "mkldnn_matmul:  only enabled for bf16 path");
  TORCH_CHECK(mkldnn_bf16_device_check(),
    "mkldnn_matmul: mkldnn_matmul bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");

  auto batch_items_are_expanded = [&](const Tensor& t) {
    const auto strides = t.strides();
    return 0 == std::accumulate(strides.begin(), strides.end(), 1, std::multiplies<int64_t>());
  };

  float dst_coeff = alpha.to<float>();
  float sum_coeff = beta.to<float>();

  // In OneDnn, 0 * nan = nan
  // In aten, expect beta * result = 0 if beta = 0
  if (sum_coeff == 0.0f)
    result.zero_();
  auto mat1_c = batch_items_are_expanded(mat1) ? mat1.contiguous(): mat1;
  auto mat2_c = batch_items_are_expanded(mat2) ? mat2.contiguous(): mat2;
  const ideep::tensor x = itensor_from_tensor(mat1_c);
  const ideep::tensor w = itensor_from_tensor(mat2_c);
  ideep::tensor y = itensor_from_tensor(result);
  if (bias.defined()) {
    TORCH_CHECK(bias.scalar_type() == at::kBFloat16, "mkldnn_matmul:  only enabled for bf16 path");
    const ideep::tensor b = itensor_from_tensor(bias);
    // DNNL only supports bias in 1xN dims
    // For bias = 1 x N case, directly use matmul primitive
    // For other case, use a fused sum to add bias to gemm result
    if (dst_coeff == 1.0f  && sum_coeff == 1.0f && bias.size(0) == 1 && bias.dim() == 2) {
      ideep::matmul_forward::compute(x, w, b, y);
      setStrided(result, y.get_dims(), y.get_strides(), result.storage_offset());
      return;
    }
    // avoid tensor copy if beta = 0
    if (sum_coeff != 0.0f && !result.is_same(bias))
      result.copy_(bias);
  }

  ideep::matmul_forward::compute(x, w, y, dst_coeff, sum_coeff,
      ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), attr);
  setStrided(result, y.get_dims(), y.get_strides(), result.storage_offset());
}

at::Tensor mkldnn_mm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  Tensor result = at::empty({self.sizes()[0], mat2.sizes()[1]}, self.options());
  return mkldnn_mm_out(self, mat2, result);
}

at::Tensor& mkldnn_mm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& result) {
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  matmul_common(self, mat2, Tensor(), result);
  return result;
}

at::Tensor& mkldnn_bmm_out(const at::Tensor& mat1, const at::Tensor& mat2, at::Tensor& result) {
  TORCH_CHECK(mat1.dim() == 3, "self must be a 3-D tensor");
  TORCH_CHECK(mat2.dim() == 3, "mat2 must be a 3-D tensor");
  auto batch1_size = mat1.sizes();
  auto batch2_size = mat2.sizes();
  TORCH_CHECK(batch1_size[0] == batch2_size[0] && batch1_size[2] == batch2_size[1]);
  result.resize_({batch1_size[0], batch1_size[1], batch2_size[2]});
  matmul_common(mat1, mat2, Tensor(), result);
  return result;
}

Tensor& mkldnn_addmm_out(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Tensor &result) {
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  native::resize_(result, {mat1.sizes()[0], mat2.sizes()[1]});
  matmul_common(mat1, mat2, self, result, beta, alpha, ideep::attr_t::fuse_sum());
  return result;
}

Tensor& mkldnn_addbmm_out(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Tensor &result) {
  TORCH_CHECK(mat1.dim() == 3, "mat1 must be a 3-D tensor");
  TORCH_CHECK(mat2.dim() == 3, "mat2 must be a 3-D tensor");
  auto batch1_size = mat1.sizes();
  auto batch2_size = mat2.sizes();
  TORCH_CHECK(batch1_size[0] == batch2_size[0] && batch1_size[2] == batch2_size[1]);
  // addbmm(batch1*batch2) [b,n,m] * [b,m,p] = [n,p] can be treated as:
  // [n, b*m] * [b*m, p] = [n, p]
  // For batch1: reorder from [b, n, m] to [n, b, m], reshape to [n, b*m]
  // For batch2: reshape from [b, m, p] to [b*m, p]
  auto mat1_ = batch1_size[0] > 1 ? mat1.transpose(0, 1) : mat1;
  auto batch1 = mat1_.reshape({batch1_size[1], batch1_size[0] * batch1_size[2]});
  auto batch2 = mat2.reshape({batch2_size[0] * batch2_size[1], batch2_size[2]});
  return mkldnn_addmm_out(self, batch1, batch2, beta, alpha, result);
}

Tensor& mkldnn_baddbmm_(Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  TORCH_CHECK(mat1.dim() == 3, "mat1 must be a 3-D tensor");
  TORCH_CHECK(mat2.dim() == 3, "mat2 must be a 3-D tensor");
  auto result_size = self.sizes();
  auto batch1_size = mat1.sizes();
  auto batch2_size = mat2.sizes();
  TORCH_CHECK(batch1_size[0] == batch2_size[0] && batch1_size[2] == batch2_size[1]);
  matmul_common(mat1, mat2, self, self, beta, alpha, ideep::attr_t::fuse_sum());
  return self;
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
