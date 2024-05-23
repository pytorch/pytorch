#include <ATen/TensorMeta.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmm_native.h>
#endif

namespace at::native {

void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  TORCH_CHECK(options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");
  TORCH_CHECK(options.device() == out.device(),
      "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
  }
}

Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions& options){
  if(strides.empty()){
    return at::empty(sizes, options);
  }else{
    return at::empty_strided(sizes, strides, options);
  }
}

void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {
  TORCH_CHECK(options.dtype() == self.dtype(),
      "Bad in-place call: ",
      "input tensor dtype ", self.dtype(), " and output tensor dtype ", options.dtype(), " should match");
  TORCH_CHECK(options.device() == self.device(),
      "Bad in-place call: ",
      "input tensor device ", self.device(), " and output tensor device ", options.device(), " should match");
  TORCH_CHECK(sizes == self.sizes(),
      "Bad in-place call: ",
      "input tensor size ", self.sizes(), " and output tensor size ", sizes, " should match");
}


// result = beta * self + alpha * (mat1 * mat2)
// Tensor& addmm_out_impl(
TORCH_IMPL_FUNC(addmm_out_xpu)(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const c10::Scalar& beta,
    const c10::Scalar& alpha,
    const at::Tensor& result) {
  Tensor self_ = self.sizes().size() == 0 ? self.view({1}) : self;
  // corner cases
  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
      return ;
  }

  if (mat1.numel() == 0){
    if(beta.to<float>() == 0.f){
      result.zero_();
    }
    // at::mul_out(
    //   result,
    //   self_.expand(result.sizes()),
    //   at::native::scalar_tensor(
    //     beta,
    //     self.scalar_type(),
    //     c10::nullopt,
    //     at::kCPU,
    //     c10::nullopt
    //   )
    // );
    result.mul_(beta);
  }

  std::vector<int64_t> result_shape = {mat1.size(0), mat2.size(1)};

  TORCH_CHECK(
      are_expandable(self_.sizes(), result_shape),
      "addmm_out input must be expanable to:",
      result_shape,
      " but got:",
      self_.sizes());

  // complex/double case
  if (mat1.is_complex() || mat1.scalar_type() == ScalarType::Double) {
    AT_ERROR(
        "Double and complex datatype matmul is not supported in oneDNN");
  }

  // proxy output
  Tensor result_ = result;

  // general case
  Tensor bias = Tensor();
  at::native::onednn::Attr attr;
  float beta_ = beta.to<float>();
  if (beta_ == 0.f) {
    if (alpha.to<float>() != 1.f) {
      attr.append_post_eltwise(
          1.f, alpha.to<float>(), 0.f, attr.kind_with_linear);
    }
  } else {
    if (alpha.to<float>() == 1.f && beta_ == 1.f) {
      bias = self_;
    } else {
      Tensor binary = self_.dim() == 1 ? self_.unsqueeze(0) : self_;
      // Tensor binary = self.expand_as(result);
      // For post-binary-add, onednn needs binary scale=1.f
      // Thus we need the following transformation
      // alpha * matmul(mat1, mat2) + beta * binary
      // beta * (alpha/beta * matmul(src, wei) + binary)
      float alpha_ = alpha.to<float>() / beta_;
      if (alpha_ != 1.f)
        attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
      attr.append_post_binary(attr.kind_with_binary_add, binary);
      if (beta_ != 1.f)
        attr.append_post_eltwise(1.f, beta_, 0.f, attr.kind_with_linear);
    }
  }
  at::native::onednn::matmul(result_, mat1, mat2, bias, true, attr);
  if (!result.is_same(result_))
    result.copy_(result_);
}

// Tensor& addmm_out(
//   const Tensor& self,
//   const Tensor& mat1,
//   const Tensor& mat2,
//   const Scalar& beta,
//   const Scalar& alpha,
//   at::Tensor& result
// ){
//   // set_output_raw_strided
//   checkBackend("addmm_out", {result, self, mat1, mat2}, Backend::XPU);
//   TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "self and mat2 must have the same dtype, but got ", self.scalar_type(), " and ", mat2.scalar_type());
//   TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "mat1 and mat2 must have the same dtype, but got ", mat1.scalar_type(), " and ", mat2.scalar_type());
//   TORCH_CHECK(
//       mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
//   TORCH_CHECK(
//       mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
//   TORCH_CHECK(
//       mat1.sizes()[1] == mat2.sizes()[0],
//       "mat1 and mat2 shapes cannot be multiplied (",
//       mat1.sizes()[0],
//       "x",
//       mat1.sizes()[1],
//       " and ",
//       mat2.sizes()[0],
//       "x",
//       mat2.sizes()[1],
//       ")");
//   IntArrayRef res_size = {mat1.sizes()[0], mat2.sizes()[1]};
//   IntArrayRef res_strides = {};
//   resize_out(result, res_size, res_strides, mat1.options());

//   return at::native::xpu::addmm_out_impl(self, mat1, mat2, beta, alpha, result);
// }

// Tensor addmm(
//   const Tensor& self,
//   const Tensor& mat1,
//   const Tensor& mat2,
//   const Scalar& beta,
//   const Scalar& alpha
// ){
//   // set_output_raw_strided
//   checkBackend("addmm_out", {self, mat1, mat2}, Backend::XPU);
//   TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "self and mat2 must have the same dtype, but got ", self.scalar_type(), " and ", mat2.scalar_type());
//   TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "mat1 and mat2 must have the same dtype, but got ", mat1.scalar_type(), " and ", mat2.scalar_type());
//   TORCH_CHECK(
//       mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
//   TORCH_CHECK(
//       mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
//   TORCH_CHECK(
//       mat1.sizes()[1] == mat2.sizes()[0],
//       "mat1 and mat2 shapes cannot be multiplied (",
//       mat1.sizes()[0],
//       "x",
//       mat1.sizes()[1],
//       " and ",
//       mat2.sizes()[0],
//       "x",
//       mat2.sizes()[1],
//       ")");
//   IntArrayRef res_size = {mat1.sizes()[0], mat2.sizes()[1]};
//   IntArrayRef res_strides = {};
//   Tensor out = create_out(res_size, res_strides, mat1.options());
//   return at::native::xpu::addmm_out_impl(self, mat1, mat2, beta, alpha, out);
// }

// Tensor& addmm_(
//   Tensor& self,
//   const Tensor& mat1,
//   const Tensor& mat2,
//   const Scalar& beta,
//   const Scalar& alpha
// ){
//   checkBackend("addmm_out", {self, mat1, mat2}, Backend::XPU);
//   TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "self and mat2 must have the same dtype, but got ", self.scalar_type(), " and ", mat2.scalar_type());
//   TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "mat1 and mat2 must have the same dtype, but got ", mat1.scalar_type(), " and ", mat2.scalar_type());
//   TORCH_CHECK(
//       mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
//   TORCH_CHECK(
//       mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
//   TORCH_CHECK(
//       mat1.sizes()[1] == mat2.sizes()[0],
//       "mat1 and mat2 shapes cannot be multiplied (",
//       mat1.sizes()[0],
//       "x",
//       mat1.sizes()[1],
//       " and ",
//       mat2.sizes()[0],
//       "x",
//       mat2.sizes()[1],
//       ")");
//   IntArrayRef res_size = {mat1.sizes()[0], mat2.sizes()[1]};
//   IntArrayRef res_strides = {};
//   check_inplace(self, res_size, mat1.options());

//   return at::native::xpu::addmm_out_impl(self, mat1, mat2, beta, alpha, self);
// }

// Tensor& _addmm_activation_out(
//     const Tensor& self,
//     const Tensor& mat1,
//     const Tensor& mat2,
//     const Scalar& beta,
//     const Scalar& alpha,
//     bool use_gelu,
//     at::Tensor& result) {
//   addmm_out_impl(self, mat1, mat2, beta, alpha, result);
//   if (use_gelu) {
//     at::gelu_(result);
//   } else {
//     at::relu_(result);
//   }
//   return result;
// }

// Tensor& mm_out(const Tensor& self, const Tensor& mat2, Tensor& result) {
//   checkBackend("mm_out", {result, self, mat2}, Backend::XPU);
//   TORCH_CHECK(self.dim() == 2, "self must be a matrix");
//   TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
//   TORCH_CHECK(
//       self.sizes()[1] == mat2.sizes()[0],
//       "mat1 and mat2 shapes cannot be multiplied (",
//       self.sizes()[0],
//       "x",
//       self.sizes()[1],
//       " and ",
//       mat2.sizes()[0],
//       "x",
//       mat2.sizes()[1],
//       ")");
//   TORCH_CHECK(
//         mat2.dtype() == result.dtype(),
//         "mm(): expected out tensor to have dtype ",
//         mat2.dtype(),
//         " but got ",
//         result.dtype());

//   result.resize_({self.size(0), mat2.size(1)});
//   if (self.numel() == 0 || mat2.numel() == 0) {
//     if (result.numel() > 0)
//       result.zero_();
//     return result;
//   }

//   if (self.is_complex() || self.scalar_type() == ScalarType::Double) {
//     AT_ERROR(
//         "Double and complex datatype matmul is not supported in oneDNN");
//   }

//   onednn::matmul(result, self, mat2, Tensor(), true, onednn::Attr());
//   return result;
// }

// Tensor mm(const Tensor& self, const Tensor& mat2) {
//   checkBackend("mm", {self, mat2}, Backend::XPU);
//   auto result = at::empty({0}, self.options());
//   xpu::mm_out(self, mat2, result);
//   return result;
// }

// Tensor mv(const Tensor& self, const Tensor& vec) {
//   Tensor result = at::empty({self.size(0)}, self.options());
//   return at::addmv_(result, self, vec, 0, 1);
// }


// result = beta * input + alpha * (batch1 @ batch2)
// Tensor& baddbmm_out(
//     const Tensor& input,
//     const Tensor& batch1,
//     const Tensor& batch2,
//     const Scalar& beta,
//     const Scalar& alpha,
//     Tensor& result) {
//   checkBackend("baddbmm_out", {input, batch1, batch2, result}, Backend::XPU);
//   TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
//   TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

//   std::vector<int64_t> result_shape = {
//       batch1.size(0), batch1.size(1), batch2.size(2)};
//   result.resize_(result_shape);
//   if (result.numel() == 0){
//     return result;
//   } else if (batch1.size(2) == 0){
//     if (beta.to<c10::complex<double>>() == 0.0){
//       return result.zero_();
//     }else{
//       at::mul_out(result, input, beta);
//       return result;
//     }
//   }

//   TORCH_CHECK(
//       are_expandable(input.sizes(), result_shape),
//       "baddbmm_out input must be expanable to:",
//       result_shape,
//       " but got:",
//       input.sizes());

//   // complex and double case
//   if (batch1.is_complex() || batch2.scalar_type() == ScalarType::Double) {
//     AT_ERROR(
//         "Double and complex datatype matmul is not supported in oneDNN");
//   }

//   // general case
//   onednn::Attr attr;
//   float beta_ = beta.to<float>();
//   Tensor binary;
//   if (beta_ == 0.f) {
//     if (alpha.to<float>() != 1.f) {
//       attr.append_post_eltwise(
//           1.f, alpha.to<float>(), 0.f, attr.kind_with_linear);
//     }
//   } else {
//     if(input.sizes().size()==0)
//       binary = input.view({1});
//     binary = input.dim() < 3 ? input.unsqueeze(0) : input;
//     binary = binary.dim() < 3 ? binary.unsqueeze_(0) : binary;
//     float alpha_ = alpha.to<float>() / beta_;
//     if (alpha_ != 1.f)
//       attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
//     attr.append_post_binary(attr.kind_with_binary_add, binary);
//     if (beta_ != 1.f)
//       attr.append_post_eltwise(1.f, beta_, 0.f, attr.kind_with_linear);
//   }
//   onednn::matmul(result, batch1, batch2, at::Tensor(), true, attr);
//   return result;
// }

// Tensor& baddbmm_(
//     Tensor& self,
//     const Tensor& batch1,
//     const Tensor& batch2,
//     const Scalar& beta,
//     const Scalar& alpha) {
//   TORCH_CHECK(self.dtype() == batch1.dtype(), "Input dtypes must be the same, got: input ", self.dtype(), ", batch1: ", batch1.dtype(), ", batch2: ", batch2.dtype());
//   std::vector<int64_t> result_shape = {batch1.size(0), batch1.size(1), batch2.size(2)};
//   check_inplace(self, result_shape, batch1.options());
//   return at::native::xpu::baddbmm_out(
//       self, batch1, batch2, beta, alpha, self);
// }

// Tensor baddbmm(
//     const Tensor& input,
//     const Tensor& batch1,
//     const Tensor& batch2,
//     const Scalar& beta,
//     const Scalar& alpha) {
//   Tensor r = at::empty({0}, input.options());
//   TORCH_CHECK(input.dtype() == batch1.dtype(), "Input dtypes must be the same, got: input ", input.dtype(), ", batch1: ", batch1.dtype(), ", batch2: ", batch2.dtype());
//   r = at::native::xpu::baddbmm_out(input, batch1, batch2, beta, alpha, r);
//   return r;
// }

// Tensor& addbmm_out(
//     const Tensor& self,
//     const Tensor& batch1,
//     const Tensor& batch2,
//     const Scalar& beta,
//     const Scalar& alpha,
//     Tensor& out) {
//   checkBackend("addbmm_out", {out, self, batch1, batch2}, Backend::XPU);
//   TORCH_CHECK(
//       batch1.dim() == 3 && batch2.dim() == 3,
//       "Batch tensors should be 3D, got dimensions ",
//       batch1.dim(),
//       " and ",
//       batch2.dim());

//   out.resize_({batch1.size(1), batch2.size(2)});
//   if (alpha.to<float>() == 0.f || batch1.numel() == 0 || batch2.numel() == 0) {
//     out.resize_({batch1.size(1), batch2.size(2)});
//     if (out.numel() == 0)
//       return out;

//     if (self.defined() && beta.to<float>() != 0.f) {
//       out = at::mul_out(
//           out, self, at::native::wrapped_scalar_tensor(at::Scalar(beta)));
//     } else {
//       out.zero_();
//     }
//     return out;
//   }

//   Tensor b1;
//   if (batch1.size(0) > 1) {
//     b1 = batch1.transpose(0, 1).contiguous().view({batch1.size(1), -1});
//   } else {
//     b1 = batch1.contiguous().view({batch1.size(1), -1});
//   }
//   auto b2 = batch2.contiguous().view({-1, batch2.size(2)});
//   at::native::xpu::addmm_out_impl(self, b1, b2, beta, alpha, out);

//   return out;
// }

// Tensor& addbmm_(
//     Tensor& self,
//     const Tensor& batch1,
//     const Tensor& batch2,
//     const Scalar& beta,
//     const Scalar& alpha) {
//   at::native::xpu::addbmm_out(self, batch1, batch2, beta, alpha, self);
//   return self;
// }

// Tensor addbmm(
//     const Tensor& self,
//     const Tensor& batch1,
//     const Tensor& batch2,
//     const Scalar& beta,
//     const Scalar& alpha) {
//   Tensor out = at::empty({0}, self.options());
//   at::native::xpu::addbmm_out(self, batch1, batch2, beta, alpha, out);
//   return out;
// }

// Tensor& bmm_out(const Tensor& self, const Tensor& batch2, Tensor& result) {
//   checkBackend("bmm_out", {result, self, batch2}, Backend::XPU);
//   TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
//   TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

//   result.resize_({self.size(0), self.size(1), batch2.size(2)});
//   if (self.numel() == 0 || batch2.numel() == 0) {
//     if (result.numel() > 0)
//       result.zero_();
//     return result;
//   }

//   if (self.is_complex() || self.scalar_type() == ScalarType::Double) {
//     AT_ERROR(
//         "Double and complex datatype matmul is not supported in oneDNN");
//   }
//   onednn::matmul(result, self, batch2, at::Tensor(), true, onednn::Attr());
//   return result;
// }

// Tensor bmm(const Tensor& self, const Tensor& batch2) {
//   auto result = at::empty({0}, self.options());
//   at::native::xpu::bmm_out(self, batch2, result);
//   return result;
// }

// Tensor& addmv_out(
//     const Tensor& self,
//     const Tensor& mat,
//     const Tensor& vec,
//     const Scalar& beta,
//     const Scalar& alpha,
//     Tensor& out) {
//   Tensor self_v;
//   TORCH_CHECK(
//       (mat.dim() == 2 && vec.dim() == 1 && self.dim() <= 1),
//       "vector + matrix @ vector expected, got ",
//       self.dim(),
//       ", ",
//       mat.dim(),
//       ", ",
//       vec.dim());
//   if (self.dim() == 1 && self.size(0) != 1) {
//     TORCH_CHECK(
//         (mat.size(1) == vec.size(0) && mat.size(0) == self.size(0)),
//         "size mismatch, get ",
//         self.size(0),
//         ", ",
//         mat.size(0),
//         "x",
//         mat.size(1),
//         ",",
//         vec.size(0));
//     self_v = self.view({self.size(0), 1});
//   } else {
//     TORCH_CHECK(
//         (mat.size(1) == vec.size(0)),
//         "size mismatch, get ",
//         mat.size(0),
//         "x",
//         mat.size(1),
//         ",",
//         vec.size(0));
//     self_v = self;
//   }

//   Tensor vec_v = vec.view({vec.size(0), 1});
//   at::native::xpu::addmm_out_impl(self_v, mat, vec_v, beta, alpha, out);

//   return out;
// }

// Tensor addmv(const Tensor& self, const Tensor& mat, const Tensor& vec, const Scalar& beta, const Scalar& alpha){
//   Tensor out = at::empty({mat.size(0)}, mat.options());
//   return at::native::xpu::addmv_out(self, mat, vec, beta, alpha, out);
// }

// Tensor& tensordot_out(
//     const Tensor& input1,
//     const Tensor& input2,
//     IntArrayRef dims1,
//     IntArrayRef dims2,
//     Tensor& result) {
//   Tensor result_tmp = at::tensordot(input1, input2, dims1, dims2);
//   auto result_dtype = result_tmp.scalar_type();
//   auto output_tensor_dtype = result.scalar_type();
//   auto output_device = result.device();
//   auto input1_device = input1.device();
//   auto input2_device = input2.device();
//   // check if the input & output tensors are on the same device.
//   TORCH_CHECK(
//       (output_device == input1_device) && (input1_device == input2_device),
//       "tensordot: Expected the output and input tensors to be on the "
//       "same device, but got the output tensor on ",
//       output_device,
//       ", input tensor a on ",
//       input1_device,
//       ", and input tensor b on ",
//       input2_device);
//   // check if the computed result has the same dtype as the out tensor
//   // (because tensordot does not support type promotion)
//   TORCH_CHECK(
//       result_dtype == output_tensor_dtype,
//       "tensordot",
//       ": Expected the output tensor to have dtype ",
//       result_dtype,
//       ", but got an output tensor with dtype ",
//       output_tensor_dtype);
//   at::native::resize_output(result, result_tmp.sizes());
//   result.copy_(result_tmp);
//   return result;
// }

// TORCH_LIBRARY_IMPL(aten, XPU, m){
  // m.impl("addmm", TORCH_FN(addmm));
  // m.impl("addmm_", TORCH_FN(addmm_));
  // m.impl("addmm.out", TORCH_FN(addmm_out));
  // m.impl("_addmm_activation.out", TORCH_FN(_addmm_activation_out));
  // m.impl("mm.out", TORCH_FN(mm_out));
  // m.impl("mm", TORCH_FN(mm));
  // m.impl("baddbmm.out", TORCH_FN(baddbmm_out));
  // m.impl("baddbmm_", TORCH_FN(baddbmm_));
  // m.impl("baddbmm", TORCH_FN(baddbmm));
  // m.impl("addbmm.out", TORCH_FN(addbmm_out));
  // m.impl("addbmm_", TORCH_FN(addbmm_));
  // m.impl("addbmm", TORCH_FN(addbmm));
  // m.impl("bmm.out", TORCH_FN(bmm_out));
  // m.impl("bmm", TORCH_FN(bmm));
  // m.impl("addmv.out", TORCH_FN(addmv_out));
  // m.impl("addmv", TORCH_FN(addmv));
  // m.impl("tensordot.out", TORCH_FN(tensordot_out));
// }

} // namespace at::native::xpu
