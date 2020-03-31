#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_bmm(
    const Tensor& self,
    const Tensor& mat2) {
  TORCH_CHECK(false, "mkldnn_bmm: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_bmm_out(
    Tensor &result, 
    const Tensor& batch1, 
    const Tensor& batch2) {
  TORCH_CHECK(false, "mkldnn_bmm_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_mm(
    const Tensor& self,
    const Tensor& mat2) {
  TORCH_CHECK(false, "mkldnn_mm: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_mm_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat2) {
  TORCH_CHECK(false, "mkldnn_mm_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_baddbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(false, "mkldnn_baddbmm: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_baddbmm_out(
    Tensor &result, 
    const Tensor& self, 
    const Tensor& batch1, 
    const Tensor& batch2, 
    Scalar beta, 
    Scalar alpha) {
  TORCH_CHECK(false, "mkldnn_baddbmm_out: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_baddbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(false, "mkldnn_baddbmm_: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_addmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(false, "mkldnn_addmm: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_addmm_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {      
  TORCH_CHECK(false, "mkldnn_addmm_out: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_addmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(false, "mkldnn_addmm_: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_addbmm(
    const Tensor &self,
    const Tensor &batch1,
    const Tensor &batch2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(false, "mkldnn_addbmm: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_addbmm_out(
    Tensor& result,
    const Tensor &self,
    const Tensor &batch1,
    const Tensor &batch2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(false, "mkldnn_addbmm_out: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_addbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(false, "mkldnn_addbmm_: ATen not compiled with MKLDNN support");
}
} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

void matmul_common(
    const ideep::tensor &x,
    const ideep::tensor &w,
    const ideep::tensor &bias, 
    ideep::tensor &y,
    Scalar beta=1,
    Scalar alpha=1, 
    const ideep::attr_t& attr = ideep::attr_t()) {
  float dst_coeff = alpha.to<float>();
  float sum_coeff = beta.to<float>();
  if (!bias.is_empty()) { 
    // DNNL only supports bias in 1xN dims
    // use bias for sum can save tensor memory copy 
    if (dst_coeff == 1.0f  && sum_coeff == 1.0f && bias.get_dim(0) == 1) {
      ideep::matmul_forward::compute(x, w, bias, y);
      return;
    }
    ideep::direct_copy::compute(bias, y);
  }

  ideep::matmul_forward::compute(x, w, y, dst_coeff, sum_coeff,
      ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), attr);
}

Tensor mkldnn_bmm(
    const Tensor& self, 
    const Tensor& mat2) {
  auto self_size = self.sizes();
  std::vector<int64_t> result_size(self_size.begin(), self_size.end()-1);
  result_size.push_back(mat2.size(-1));
  Tensor result = empty_mkldnn(result_size, self.options());
  return mkldnn_bmm_out(result, self, mat2);
}

Tensor& mkldnn_bmm_out(
    Tensor &result, 
    const Tensor& batch1, 
    const Tensor& batch2) {
  const ideep::tensor x = itensor_from_mkldnn(batch1);
  const ideep::tensor w = itensor_from_mkldnn(batch2);
  ideep::tensor& y = itensor_from_mkldnn(result);
  matmul_common(x, w, ideep::tensor(), y);
  return result;
}

Tensor mkldnn_mm(
    const Tensor& self,
    const Tensor& mat2) {
  return at::native::mkldnn_bmm(self, mat2);
}

Tensor& mkldnn_mm_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat2) {
  return at::native::mkldnn_bmm_out(result, self, mat2);
}

Tensor& mkldnn_baddbmm_out(
    Tensor &result, 
    const Tensor& self, 
    const Tensor& batch1, 
    const Tensor& batch2, 
    Scalar beta, 
    Scalar alpha) {
  const ideep::tensor x = itensor_from_mkldnn(batch1);
  const ideep::tensor w = itensor_from_mkldnn(batch2);
  ideep::tensor bias;
  if (self.numel() != 0) {
    bias = itensor_from_mkldnn(self);
    if (bias.ndims() < x.ndims()) {
      auto bias_dims = bias.get_dims();
      bias_dims.insert(bias_dims.begin(), 1);
      bias.reshape(bias_dims);
    }
  }
  ideep::tensor& y = itensor_from_mkldnn(result);
  auto attr_ = ideep::attr_t::fuse_sum();
  matmul_common(x, w, bias, y, beta, alpha, attr_);
  return result;
}

Tensor mkldnn_baddbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  Tensor result = empty_mkldnn(self.sizes(), self.options());
  return at::native::mkldnn_baddbmm_out(result, self, batch1, batch2, beta, alpha);
}

Tensor& mkldnn_baddbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  Tensor result = at::empty({0}, self.options());
  return mkldnn_baddbmm_out(self, result, batch1, batch2, beta, alpha);
}

Tensor& mkldnn_addmm_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {      
  return at::native::mkldnn_baddbmm_out(result, self, mat1, mat2, beta, alpha);
}

Tensor mkldnn_addmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  return at::native::mkldnn_baddbmm(self, batch1, batch2, beta, alpha);
}

Tensor& mkldnn_addmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  return at::native::mkldnn_baddbmm_(self, batch1, batch2, beta, alpha);
}

Tensor& mkldnn_addbmm_out(
    Tensor& result,
    const Tensor &self,
    const Tensor &batch1,
    const Tensor &batch2,
    Scalar beta,
    Scalar alpha) {
  // addbmm(batch1*batch2) [b,n,m] * [b,m,p] = [n,p] can be treated as:
  // [n, b*m] * [b*m, p] = [n, p]
  // For batch1: reorder from [b, n, m] to [n, b, m], reshape to [n, b*m]
  // For batch2: reshape from [b, m, p] to [b*m, p]
  const ideep::tensor x = itensor_from_mkldnn(batch1);
  ideep::tensor w = itensor_from_mkldnn(batch2);

  auto x_ = x;
  if (x.get_dim(0) > 1) {
    x_ = x.transpose(0, 1);
  }
  ideep::dims x_dims = {x.get_dim(1), x.get_dim(0) * x.get_dim(2)};
  x_ = x_.reshape(x_dims);
  ideep::dims w_dims = {w.get_dim(0) * w.get_dim(1), w.get_dim(2)};
  auto w_ = w.reshape(w_dims);
  ideep::tensor& y = itensor_from_mkldnn(result);
  auto attr_ = ideep::attr_t::fuse_sum();
 
  ideep::tensor bias; 
  if (self.numel() != 0) {
    bias = itensor_from_mkldnn(self);
    if (bias.ndims() < x_.ndims()) {
      auto bias_dims = bias.get_dims();
      bias_dims.insert(bias_dims.begin(), 1);
      bias.reshape(bias_dims);
    }
  }
  matmul_common(x_, w_, bias, y, beta, alpha, attr_);
  return result;
}

Tensor mkldnn_addbmm(
    const Tensor &self,
    const Tensor &batch1,
    const Tensor &batch2,
    Scalar beta,
    Scalar alpha) {
  Tensor result = empty_mkldnn(self.sizes(), self.options());
  return mkldnn_addbmm_out(result, self, batch1, batch2, beta, alpha);
}

Tensor& mkldnn_addbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  Tensor result = at::empty({0}, self.options());
  return mkldnn_addbmm_out(self, result, batch1, batch2, beta, alpha);
}

Tensor mkldnn_maybe_multiply(const Tensor& grad_output, Scalar alpha=1) {
  if (alpha.toFloat() == 1) {
    return grad_output;
  }
  const ideep::tensor grady = itensor_from_mkldnn(grad_output);
  ideep::tensor gradx;
  ideep::eltwise_forward::compute(grady, gradx, 
      ideep::algorithm::eltwise_linear, 
      ideep::prop_kind::forward_inference, alpha.toFloat());
  return new_with_itensor_mkldnn(std::move(gradx), grad_output.options());
}

std::tuple<Tensor, Tensor> mkldnn_matmul_backward(const Tensor& grad_output, const Tensor& mat1,  const Tensor& mat2, Scalar alpha=1) {
  auto grady = itensor_from_mkldnn(grad_output);
  auto ndim = mat1.dim();
  auto m1 = itensor_from_mkldnn(mat1).transpose_(ndim-2, ndim-1);
  auto m2 = itensor_from_mkldnn(mat2).transpose_(ndim-2, ndim-1);
  ideep::tensor gradx, gradw;
  ideep::matmul_forward::compute(grady, m2, gradx, alpha.toFloat());
  ideep::matmul_forward::compute(m1, grady, gradw, alpha.toFloat());
  return std::tuple<Tensor, Tensor>{
      new_with_itensor_mkldnn(std::move(gradx), grad_output.options()), 
      new_with_itensor_mkldnn(std::move(gradw), grad_output.options())};
}

std::tuple<Tensor, Tensor> mkldnn_mm_backward(const Tensor& grad_output, const Tensor& self, const Tensor& mat2) {
  return mkldnn_matmul_backward(grad_output, self, mat2);
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_addmm_backward(const Tensor& grad_output, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  Tensor grad1 = mkldnn_maybe_multiply(grad_output, beta);
  Tensor grad2, grad3;
  std::tie(grad2, grad3) = mkldnn_matmul_backward(grad_output, mat1, mat2, alpha);
  return std::tuple<Tensor, Tensor, Tensor>{grad1, grad2, grad3}; 
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_addbmm_backward(const Tensor& grad_output, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  Tensor grad1 = mkldnn_maybe_multiply(grad_output, beta);
  
  // Workaround: MkldnnTensor not support unsqueeze and expand operations now
  auto grad_dense = at::native::mkldnn_to_dense(grad_output, grad_output.scalar_type());
  grad_dense = grad_dense.unsqueeze(0).expand({batch1.size(0), batch1.size(1), batch2.size(2)});
  auto grad_expand = at::native::dense_to_mkldnn(grad_dense, grad_output.scalar_type());
  Tensor grad2, grad3;
  std::tie(grad2, grad3) = mkldnn_matmul_backward(grad_expand, batch1, batch2, alpha);
  return std::tuple<Tensor, Tensor, Tensor>{grad1, grad2, grad3}; 
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_baddbmm_backward(const Tensor& grad_output, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  Tensor grad1 = mkldnn_maybe_multiply(grad_output, beta);
  Tensor grad2, grad3;
  std::tie(grad2, grad3) = mkldnn_matmul_backward(grad_output, batch1, batch2, alpha);
  return std::tuple<Tensor, Tensor, Tensor>{grad1, grad2, grad3}; 
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
