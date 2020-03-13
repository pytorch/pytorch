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
  const ideep::tensor w = itensor_from_tensor(batch2);
  ideep::tensor& y = itensor_from_mkldnn(result);
  ideep::matmul_forward::compute(x, w, y);
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

Tensor mkldnn_baddbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
    Tensor result = empty_mkldnn(self.sizes(), self.options());
    return at::native::mkldnn_baddbmm_out(result, self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm_common(
    Tensor &result,
    const ideep::tensor &bias, 
    const ideep::tensor &x,
    const ideep::tensor &w,
    Scalar beta,
    Scalar alpha) {
    
    ideep::tensor& y = itensor_from_mkldnn(result);
    float dst_coeff = alpha.to<float>();
    float sum_coeff = beta.to<float>();
    // DNNL only supports bias in 1xN dims
    // use bias for sum can save tensor memory copy 
    if (dst_coeff == 1.0f  && sum_coeff == 1.0f && bias.get_dim(0) == 1) {
      ideep::matmul_forward::compute(x, w, bias, y);
      return result;
    }

    ideep::direct_copy::compute(bias, y);
    auto attr_ = ideep::attr_t::fuse_sum();
    ideep::matmul_forward::compute(x, w, y, dst_coeff, sum_coeff,
        ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), attr_);
    return result;
}

Tensor& mkldnn_baddbmm_out(
    Tensor &result, 
    const Tensor& self, 
    const Tensor& batch1, 
    const Tensor& batch2, 
    Scalar beta, 
    Scalar alpha) {
    const ideep::tensor x = itensor_from_mkldnn(batch1);
    const ideep::tensor w = itensor_from_tensor(batch2);
    ideep::tensor bias = itensor_from_tensor(self);
    if (bias.get_dims().size() < x.get_dims().size()) {
      auto bias_dims = bias.get_dims();
      bias_dims.insert(bias_dims.begin(), 1);
      bias.reshape(bias_dims);
    }
    return baddbmm_common(result, bias, x, w, beta, alpha);
}

Tensor& mkldnn_baddbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
    const ideep::tensor x = itensor_from_mkldnn(batch1);
    const ideep::tensor w = itensor_from_tensor(batch2);
    ideep::tensor y = itensor_from_mkldnn(self);
    float dst_coeff = alpha.to<float>();
    float sum_coeff = beta.to<float>();
    auto attr_ = ideep::attr_t::fuse_sum();
    ideep::matmul_forward::compute(x, w, y, dst_coeff, sum_coeff,
        ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), attr_);
    return self;
}

// mkldnn_addmm will go to DNNL matmul jit path
Tensor mkldnn_addmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
    return at::native::mkldnn_baddbmm(self, batch1, batch2, beta, alpha);
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

Tensor& mkldnn_addmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  return at::native::mkldnn_baddbmm_(self, batch1, batch2, beta, alpha);
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
  ideep::tensor w = itensor_from_tensor(batch2);

  auto x_ = x;
  if (x.get_dim(0) > 1) {
    auto x_desc = ideep::tensor::desc(x.get_dims(), x.get_data_type(), ideep::tag::bac);
    x_ = x.reorder_if_differ_in(x_desc);
  }
  ideep::dims x_dims = {x.get_dim(1), x.get_dim(0) * x.get_dim(2)};
  x_ = x_.reshape(x_dims);

  ideep::dims w_dims = {w.get_dim(0) * w.get_dim(1), w.get_dim(2)};
  auto w_ = w.reshape(w_dims);
   
  ideep::tensor bias = itensor_from_tensor(self);
  if (bias.get_dims().size() < x_.get_dims().size()) {
    auto bias_dims = bias.get_dims();
    bias_dims.insert(bias_dims.begin(), 1);
    bias.reshape(bias_dims);
  }
  return baddbmm_common(result, bias, x_, w_, beta, alpha);
}

Tensor& mkldnn_addbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  const ideep::tensor x = itensor_from_mkldnn(batch1);
  ideep::tensor w = itensor_from_tensor(batch2);

  auto x_ = x;
  if (x.get_dim(0) > 1) {
    auto x_desc = ideep::tensor::desc(x.get_dims(), x.get_data_type(), ideep::tag::bac);
    x_ = x.reorder_if_differ_in(x_desc);
  }
  ideep::dims x_dims = {x.get_dim(1), x.get_dim(0) * x.get_dim(2)};
  x_ = x_.reshape(x_dims);

  ideep::dims w_dims = {w.get_dim(0) * w.get_dim(1), w.get_dim(2)};
  auto w_ = w.reshape(w_dims);

  ideep::tensor y = itensor_from_mkldnn(self);
  float dst_coeff = alpha.to<float>();
  float sum_coeff = beta.to<float>();
  auto attr_ = ideep::attr_t::fuse_sum();
  ideep::matmul_forward::compute(x_, w_, y, dst_coeff, sum_coeff,
      ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), attr_);
  return self;
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
