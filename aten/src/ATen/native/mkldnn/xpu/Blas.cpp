#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <torch/library.h>
#ifndef AT_PER_OPERATOR_HEADERS

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mm_native.h>
#endif

namespace at::native {
namespace xpu {

// result = beta * self + alpha * (mat1 * mat2)
Tensor& addmm_out(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  checkBackend("addmm_out", {result, self, mat1, mat2}, Backend::XPU);
  TORCH_CHECK(
      mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(
      mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");
  TORCH_CHECK(
      mat1.dtype() == mat2.dtype(),
      "expected mat1 and mat2 to have the same dtype, but got: ",
      mat1.dtype(),
      " != ",
      mat2.dtype())
  // complex case
  TORCH_CHECK(
      !mat1.is_complex(), "Complex datatype matmul is not supported in oneDNN");

  std::vector<int64_t> result_shape = {mat1.size(0), mat2.size(1)};
  result.resize_(result_shape);

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  if (mat1.numel() == 0) {
    if (beta.to<float>() == 0.f) {
      return result.zero_();
    }
    return at::mul_out(
        result,
        self.expand(result.sizes()),
        at::native::scalar_tensor(
            beta, self.scalar_type(), std::nullopt, at::kCPU, std::nullopt));
  }

  TORCH_CHECK(
      are_expandable(self.sizes(), result_shape),
      "addmm_out input must be expanable to:",
      result_shape,
      " but got:",
      self.sizes());

  // general case
  Tensor bias = Tensor();
  onednn::Attr attr;
  float beta_ = beta.to<float>();
  float alpha_ = beta_ == 0.f ? alpha.to<float>() : alpha.to<float>() / beta_;
  if (beta_ == 0.f) {
    attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
  } else if (alpha_ == 1.f && beta_ == 1.f && !result.is_same(self)) {
    // if result and self are the same tensor, we use post op sum.
    bias = self;
  } else {
    Tensor binary = self.dim() < 1 ? self.unsqueeze(0) : self;
    binary = binary.dim() == 1 ? binary.unsqueeze(0) : binary;
    bool inplace = binary.is_same(result);
    if (inplace) {
      attr.append_post_eltwise(
          1.f, alpha.to<float>(), 0.f, attr.kind_with_linear);
      attr.append_post_sum(beta_);
    } else {
      if (at::native::onednn::is_broadcast(binary)) {
        at::native::onednn::undo_broadcast(binary);
      }
      // in test_addmv_rowmajor_colmajor_incx_incy_lda, binary is a tensor with
      // shape (5, 1) but stride(2, 2)
      binary = at::native::onednn::is_onednn_matmul_strides(binary)
          ? binary
          : binary.contiguous();
      // Tensor binary = self.expand_as(result);
      // For post-binary-add, onednn needs binary scale=1.f
      // Thus we need the following transformation
      // alpha * matmul(mat1, mat2) + beta * binary
      // beta * (alpha/beta * matmul(src, wei) + binary)
      attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
      attr.append_post_binary<true>(attr.kind_with_binary_add, binary);
      attr.append_post_eltwise(1.f, beta_, 0.f, attr.kind_with_linear);
    }
  }
  onednn::matmul(result, mat1, mat2, bias, true, attr);
  return result;
}

Tensor& _addmm_activation_out(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    bool use_gelu,
    at::Tensor& result) {
  addmm_out(self, mat1, mat2, beta, alpha, result);
  if (use_gelu) {
    at::gelu_(result);
  } else {
    at::relu_(result);
  }
  return result;
}

Tensor& mm_out(const Tensor& self, const Tensor& mat2, Tensor& result) {
  checkBackend("mm_out", {result, self, mat2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      self.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      self.sizes()[0],
      "x",
      self.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");
  TORCH_CHECK(
      self.dtype() == mat2.dtype(),
      "expected mat1 and mat2 to have the same dtype, but got: ",
      self.dtype(),
      " != ",
      mat2.dtype())

  result.resize_({self.size(0), mat2.size(1)});
  if (self.numel() == 0 || mat2.numel() == 0) {
    if (result.numel() > 0)
      result.zero_();
    return result;
  }

  TORCH_CHECK(
      !self.is_complex(), "Complex datatype matmul is not supported in oneDNN");

  onednn::matmul(result, self, mat2, Tensor(), true, onednn::Attr());
  return result;
}

// result = beta * input + alpha * (batch1 @ batch2)
Tensor& baddbmm_out(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  checkBackend("baddbmm_out", {input, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  std::vector<int64_t> result_shape = {
      batch1.size(0), batch1.size(1), batch2.size(2)};
  result.resize_(result_shape);
  if (result.numel() == 0) {
    return result;
  } else if (batch1.size(2) == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      return result.zero_();
    } else {
      at::mul_out(result, input, beta);
      return result;
    }
  }

  TORCH_CHECK(
      are_expandable(input.sizes(), result_shape),
      "baddbmm_out input must be expanable to:",
      result_shape,
      " but got:",
      input.sizes());

  // complex case
  TORCH_CHECK(
      !batch1.is_complex(),
      "Complex datatype matmul is not supported in oneDNN");

  // general case
  onednn::Attr attr;
  float beta_ = beta.to<float>();
  float alpha_ = beta_ == 0.f ? alpha.to<float>() : alpha.to<float>() / beta_;
  Tensor binary;
  if (beta_ == 0.f) {
    attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
  } else {
    Tensor binary = input.dim() < 1 ? input.unsqueeze(0) : input;
    binary = binary.dim() < 3 ? binary.unsqueeze(0) : binary;
    // If input is a 1d tensor need be broadcasted, we need unsqueeze twice.
    binary = binary.dim() < 3 ? binary.unsqueeze_(0) : binary;
    bool inplace = binary.is_same(result);
    if (inplace) {
      attr.append_post_eltwise(
          1.f, alpha.to<float>(), 0.f, attr.kind_with_linear);
      attr.append_post_sum(beta_);
    } else {
      if (at::native::onednn::is_broadcast(binary)) {
        at::native::onednn::undo_broadcast(binary);
      }
      binary = at::native::onednn::is_onednn_matmul_strides(binary)
          ? binary
          : binary.contiguous();
      attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
      attr.append_post_binary<true>(attr.kind_with_binary_add, binary);
      attr.append_post_eltwise(1.f, beta_, 0.f, attr.kind_with_linear);
    }
  }
  onednn::matmul(result, batch1, batch2, at::Tensor(), true, attr);
  return result;
}

Tensor& bmm_out(const Tensor& self, const Tensor& batch2, Tensor& result) {
  checkBackend("bmm_out", {result, self, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  result.resize_({self.size(0), self.size(1), batch2.size(2)});
  if (self.numel() == 0 || batch2.numel() == 0) {
    if (result.numel() > 0)
      result.zero_();
    return result;
  }

  TORCH_CHECK(
      !self.is_complex(), "Complex datatype matmul is not supported in oneDNN");
  onednn::matmul(result, self, batch2, at::Tensor(), true, onednn::Attr());
  return result;
}

Tensor bmm(const Tensor& self, const Tensor& batch2) {
  auto result = at::empty({0}, self.options());
  at::native::xpu::bmm_out(self, batch2, result);
  return result;
}

Tensor& addmv_out(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  Tensor self_v;
  TORCH_CHECK(
      (mat.dim() == 2 && vec.dim() == 1 && self.dim() <= 1),
      "vector + matrix @ vector expected, got ",
      self.dim(),
      ", ",
      mat.dim(),
      ", ",
      vec.dim());
  if (self.dim() == 1 && self.size(0) != 1) {
    TORCH_CHECK(
        (mat.size(1) == vec.size(0) && mat.size(0) == self.size(0)),
        "size mismatch, get ",
        self.size(0),
        ", ",
        mat.size(0),
        "x",
        mat.size(1),
        ",",
        vec.size(0));
    self_v = self.view({self.size(0), 1});
  } else {
    TORCH_CHECK(
        (mat.size(1) == vec.size(0)),
        "size mismatch, get ",
        mat.size(0),
        "x",
        mat.size(1),
        ",",
        vec.size(0));
    self_v = self;
  }

  Tensor vec_v = vec.view({vec.size(0), 1});
  at::native::xpu::addmm_out(self_v, mat, vec_v, beta, alpha, out);
  out.resize_({mat.size(0)});
  return out;
}

Tensor& tensordot_out(
    const Tensor& input1,
    const Tensor& input2,
    IntArrayRef dims1,
    IntArrayRef dims2,
    Tensor& result) {
  Tensor result_tmp = at::tensordot(input1, input2, dims1, dims2);
  auto result_dtype = result_tmp.scalar_type();
  auto output_tensor_dtype = result.scalar_type();
  auto output_device = result.device();
  auto input1_device = input1.device();
  auto input2_device = input2.device();
  // check if the input & output tensors are on the same device.
  TORCH_CHECK(
      (output_device == input1_device) && (input1_device == input2_device),
      "tensordot: Expected the output and input tensors to be on the "
      "same device, but got the output tensor on ",
      output_device,
      ", input tensor a on ",
      input1_device,
      ", and input tensor b on ",
      input2_device);
  // check if the computed result has the same dtype as the out tensor
  // (because tensordot does not support type promotion)
  TORCH_CHECK(
      result_dtype == output_tensor_dtype,
      "tensordot",
      ": Expected the output tensor to have dtype ",
      result_dtype,
      ", but got an output tensor with dtype ",
      output_tensor_dtype);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("tensordot.out", TORCH_FN(tensordot_out));
}
} // namespace xpu

TORCH_IMPL_FUNC(addmm_out_xpu)
(const Tensor& self,
 const Tensor& mat1,
 const Tensor& mat2,
 const Scalar& beta,
 const Scalar& alpha,
 const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  xpu::addmm_out(self, mat1, mat2, beta, alpha, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(mm_out_xpu)
(const Tensor& self, const Tensor& mat2, const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  xpu::mm_out(self, mat2, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(bmm_out_xpu)
(const Tensor& self, const Tensor& batch2, const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  xpu::bmm_out(self, batch2, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(addmm_activation_out_xpu)
(const Tensor& self,
 const Tensor& mat1,
 const Tensor& mat2,
 const Scalar& beta,
 const Scalar& alpha,
 bool use_gelu,
 const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  xpu::_addmm_activation_out(
      self, mat1, mat2, beta, alpha, use_gelu, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(baddbmm_out_xpu)
(const Tensor& self,
 const Tensor& batch1,
 const Tensor& batch2,
 const Scalar& beta,
 const Scalar& alpha,
 const Tensor& result) {
  xpu::baddbmm_out(
      self,
      batch1,
      batch2,
      beta,
      alpha,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(addmv_out_xpu)
(const Tensor& self,
 const Tensor& mat,
 const Tensor& vec,
 const Scalar& beta,
 const Scalar& alpha,
 const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  xpu::addmv_out(self, mat, vec, beta, alpha, const_cast<Tensor&>(result));
}

Tensor _weight_int4pack_mm_xpu(
    const Tensor& A,
    const Tensor& B,
    int64_t qGroupSize,
    const Tensor& qScale,
    const Tensor& qZeros) {
  auto M = A.size(0); // M
  auto N = B.size(0); // N1=LCM(N, K)
  TORCH_CHECK(
      A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
      __func__,
      " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

  TORCH_CHECK(B.dtype() == kInt, __func__, " : expect B to be int32 tensor.");
  TORCH_CHECK(
      qZeros.dtype() == kChar,
      __func__,
      " : expect qZeros to be int8 tensor currently.");
  TORCH_CHECK(B.dim() == 2, __func__, " : expect B to 2d tensor.");

  TORCH_CHECK(
      qGroupSize > 1 && qGroupSize % 32 == 0,
      __func__,
      " : expect qGroupSize to be multiple of 32 and greater than 1, got ",
      qGroupSize);

  TORCH_CHECK(
      qScale.dim() == 2 && qScale.size(1) == N,
      __func__,
      ": expect qScale to be 2d tensor with sizes [:, ",
      N,
      "]");
  TORCH_CHECK(
      qZeros.dim() == 2 && qZeros.size(1) == N,
      __func__,
      ": expect qZeros to be 2d tensor with sizes [:, ",
      N,
      "]");

  auto C = at::empty({M, N}, A.options());

  // qscale:[K/qGroupSize, N]
  // qzp:[K/qGroupSize, N]
  at::native::onednn::woq_matmul_int4(C, A, B, qScale, qZeros, qGroupSize);

  return C;
}
} // namespace at::native
