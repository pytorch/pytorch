#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/ceil_div.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/xpu/Blas.h>
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
namespace {
/*
 * Scaling Type Determination:
 * ---------------------------
 * Conditions and corresponding Scaling Types:
 *
 * - If scale tensor is `Float8_e8m0fnu` or `Float8_e4m3fn`:
 *   - Returns BlockWise (with additional size checks).
 *
 * - Else if scale.numel() == 1:
 *   - Returns TensorWise.
 *
 * - Else if scale.dim() == 2 && scale.size(0) == outer_dim && scale.size(1) ==
 * 1:
 *   - Returns RowWise.
 *
 * - Otherwise:
 *   - Returns Error.
 */
using at::native::onednn::ScalingType;

bool is_tensorwise_scaling(const at::Tensor& t, const at::Tensor& scale) {
  return at::isFloat8Type(t.scalar_type()) &&
      scale.scalar_type() == at::kFloat && scale.numel() == 1;
}

bool is_rowwise_scaling(const at::Tensor& t, const at::Tensor& scale) {
  return (
      at::isFloat8Type(t.scalar_type()) && scale.scalar_type() == at::kFloat &&
      scale.dim() == 2 && scale.size(0) == t.size(0) && scale.size(1) == 1 &&
      scale.is_contiguous());
}

bool is_desired_scaling(
    const at::Tensor& t,
    const at::Tensor& scale,
    ScalingType desired_scaling) {
  switch (desired_scaling) {
    case ScalingType::TensorWise:
      return is_tensorwise_scaling(t, scale);
    case ScalingType::RowWise:
      return is_rowwise_scaling(t, scale);
    default:
      TORCH_CHECK(
          false,
          "Unsupported scaling type. t.type()=",
          t.toString(),
          ", scale.type()=",
          scale.toString());
      return false;
  }
}

std::pair<ScalingType, ScalingType> get_joint_scaling(
    std::initializer_list<std::pair<ScalingType, ScalingType>> options,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b) {
  for (auto [lhs, rhs] : options) {
    if (is_desired_scaling(a, scale_a, lhs) &&
        is_desired_scaling(b.t(), scale_b.t(), rhs)) {
      return {lhs, rhs};
    }
  }
  TORCH_CHECK(
      false,
      "Invalid scaling configuration.\n"
      "- For TensorWise scaling, a and b should be float8, scales should be float and singletons.\n"
      "- For RowWise scaling, a and b should be float8, scales should be float, scale_a should be (",
      a.size(0),
      ", 1) and scale_b should be (1, ",
      b.size(1),
      "), and both should be contiguous.\n"
      "Got a.dtype()=",
      a.scalar_type(),
      ", scale_a.dtype()=",
      scale_a.scalar_type(),
      ", scale_a.size()=",
      scale_a.sizes(),
      ", scale_a.stride()=",
      scale_a.strides(),
      ", ",
      "b.dtype()=",
      b.scalar_type(),
      ", scale_b.dtype()=",
      scale_b.scalar_type(),
      ", scale_b.size()=",
      scale_b.sizes(),
      " and scale_b.stride()=",
      scale_b.strides());
}

} // namespace

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
  if (self.is_complex()) {
    at::native::addmm_complex_out_xpu(self, mat1, mat2, beta, alpha, result);

    return result;
  }

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

  if (self.is_complex()) {
    at::native::mm_complex_out_xpu(self, mat2, result);

    return result;
  }

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
  if (input.is_complex()) {
    at::native::baddbmm_complex_out_xpu(
        input, batch1, batch2, beta, alpha, result);

    return result;
  }

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

  // complex case
  if (self.is_complex()) {
    at::native::bmm_complex_out_xpu(self, batch2, result);

    return result;
  }

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

Tensor& _scaled_mm_out_xpu(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& scale_a,
    const Tensor& scale_b,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& scale_result,
    std::optional<c10::ScalarType> out_dtype,
    bool use_fast_accum,
    Tensor& out) {
  // Note: fast_accum is not supported in XPU for now.
  TORCH_CHECK(!use_fast_accum, "fast_accum is not supported in XPU for now.");

  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");

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

  // Check what type of scaling we are doing based on inputs. This list is
  // sorted by decreasing priority.
  auto [scaling_choice_a, scaling_choice_b] = get_joint_scaling(
      {
          std::make_pair(ScalingType::TensorWise, ScalingType::TensorWise),
          std::make_pair(ScalingType::RowWise, ScalingType::RowWise),
      },
      mat1,
      mat2,
      scale_a,
      scale_b);
  TORCH_CHECK(
      !scale_result ||
          (scale_result->numel() == 1 && scale_result->scalar_type() == kFloat),
      "scale_result must be a float scalar");
  TORCH_CHECK(
      !bias || bias->numel() == mat2.sizes()[1],
      "Bias must be size ",
      mat2.sizes()[1],
      " but got ",
      bias->numel());
  TORCH_CHECK(
      mat1.sizes()[1] % 16 == 0,
      "Expected trailing dimension of mat1 to be divisible by 16 ",
      "but got mat1 shape: (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      ").");
  TORCH_CHECK(
      mat2.sizes()[0] % 16 == 0 && mat2.sizes()[1] % 16 == 0,
      "mat2 shape (",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ") must be divisible by 16");
  // Check types
  TORCH_CHECK(
      !out_dtype || *out_dtype == out.scalar_type(),
      "out_dtype must match output matrix type");
  TORCH_CHECK(
      at::isFloat8Type(mat1.scalar_type()),
      "Expected mat1 to be Float8 matrix got ",
      mat1.scalar_type());
  TORCH_CHECK(
      at::isFloat8Type(mat2.scalar_type()),
      "Expected mat2 to be Float8 matrix got ",
      mat2.scalar_type());
  // TODO: oneDNN Currently only supports e4m3 with group scales on BMG. Not
  // support 2D scales, only 1D. Needs to add more checks there.

  if (bias) {
    TORCH_CHECK(
        out.scalar_type() != kFloat,
        "Bias is not supported when out_dtype is set to Float32");

    TORCH_CHECK(
        bias->scalar_type() == ScalarType::BFloat16 ||
            bias->scalar_type() == ScalarType::Half,
        "Bias must be BFloat16 or Half, but got ",
        bias->scalar_type());

    TORCH_CHECK(
        (out.scalar_type() != kFloat &&
         out.scalar_type() != ScalarType::BFloat16) ||
            bias->scalar_type() == ScalarType::BFloat16,
        "Bias must be BFloat16 to compute ",
        out.scalar_type(),
        " output, but got ",
        bias->scalar_type());

    TORCH_CHECK(
        out.scalar_type() != ScalarType::Half ||
            bias->scalar_type() == ScalarType::Half,
        "Bias must be Float16 to compute ",
        out.scalar_type(),
        " output, but got ",
        bias->scalar_type());
  }

  {
    auto bias_ = bias.value_or(Tensor());
    auto scale_result_ = scale_result.value_or(Tensor());

    // NOLINTNEXTLINE(*c-array*)
    TensorArg targs[]{
        {out, "out", 0},
        {mat1, "mat1", 1},
        {mat2, "mat2", 2},
        {bias_, "bias", 3},
        {scale_a, "scale_a", 4},
        {scale_b, "scale_b", 5},
        {scale_result_, "scale_result", 6}};
    checkAllSameGPU(__func__, targs);
  }

  // Validation checks have passed lets resize the output to actual size
  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

  // If any of M, K, N is 0 - return early (the tensorwise/rowwise float8 gemm
  // kernels do not support this case).
  if (mat1_sizes[0] == 0 || mat1_sizes[1] == 0 || mat2_sizes[1] == 0) {
    // `out` was created with `at::empty`. In the case where we are multiplying
    // MxK by KxN and K is the zero dim, we need to initialize here to properly
    // return a tensor of zeros.
    if (mat1_sizes[1] == 0) {
      out.zero_();
    }

    return out;
  }

  // TODO: We need further checks on inputs and fail early.
  // TODO: Handle scale_result when provided.
  onednn::scaled_matmul(
      mat1,
      mat2,
      out,
      scale_a,
      scale_b,
      scaling_choice_a,
      scaling_choice_b,
      bias,
      scale_result,
      use_fast_accum);

  return out;
}

Tensor _scaled_mm_xpu(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_a,
    const Tensor& scale_b,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& scale_result,
    std::optional<c10::ScalarType> out_dtype,
    bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));
  return _scaled_mm_out_xpu(
      mat_a,
      mat_b,
      scale_a,
      scale_b,
      bias,
      scale_result,
      out_dtype,
      use_fast_accum,
      out);
}

Tensor& _int_mm_out_xpu(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& result) {
  TORCH_CHECK(
      self.dim() == 2,
      "Expected self to be of dimension 2 but got ",
      self.dim());
  TORCH_CHECK(
      mat2.dim() == 2,
      "Expected mat2 to be of dimension 2 but got ",
      mat2.dim());
  TORCH_CHECK(
      self.size(1) == mat2.size(0),
      "self.size(1) needs to match mat2.size(0) but got ",
      self.size(1),
      " and ",
      mat2.size(0));

  TORCH_CHECK(
      self.dtype() == at::kChar,
      "Expected self dtype to be of type int8 but got ",
      self.dtype());
  TORCH_CHECK(
      mat2.dtype() == at::kChar,
      "Expected mat2 dtype to be of type int8 but got ",
      mat2.dtype());
  TORCH_CHECK(
      result.dtype() == at::kInt,
      "Expected result dtype to be of type kInt but got ",
      result.dtype());
  TORCH_CHECK(
      result.size(0) == self.size(0),
      "Expected result.size(0) to be ",
      self.size(0),
      " but got ",
      result.size(0));
  TORCH_CHECK(
      result.size(1) == mat2.size(1),
      "Expected result.size(1) to be ",
      mat2.size(1),
      " but got ",
      result.size(1));

  TORCH_CHECK(
      result.dim() == 2,
      "Expected result to be of dimension 2 but got ",
      result.dim());

  TORCH_CHECK(result.is_contiguous(), "Expected result to be contiguous.");

  if (result.numel() == 0 || self.size(1) == 0) {
    return result.zero_();
  }

  Tensor bias = at::Tensor();
  Tensor mat2_scales = at::ones({1}, mat2.options().dtype(at::kFloat));
  Tensor mat2_zero_points = at::Tensor();
  auto post_op_args = torch::List<std::optional<at::Scalar>>();

  at::native::onednn::quantized_matmul(
      self.contiguous(),
      1.0,
      0,
      mat2.contiguous(),
      mat2_scales,
      mat2_zero_points,
      bias,
      result,
      1.0,
      0,
      result.scalar_type(),
      /*other*/ std::nullopt,
      /*other scale*/ 1.0,
      /*other zp*/ 0,
      /*binary post op*/ "none",
      /*binary alpha*/ 1.0,
      /*post_op_name*/ "none",
      post_op_args,
      /*post_op_algorithm*/ "none",
      /*m2_trans*/ true);
  return result;
}

Tensor _int_mm_xpu(const Tensor& self, const Tensor& mat2) {
  Tensor result =
      at::empty({self.size(0), mat2.size(1)}, self.options().dtype(at::kInt));
  return _int_mm_out_xpu(self, mat2, result);
}

Tensor _weight_int8pack_mm_xpu(
    const Tensor& A,
    const Tensor& B,
    const Tensor& scales) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  TORCH_CHECK(
      A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
      " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");
  TORCH_CHECK(
      A.stride(1) == 1, " : A must be contiguous on the last dimension.");
  TORCH_CHECK(B.dtype() == kChar, " : expect B to be int8 tensor.");
  TORCH_CHECK(B.is_contiguous(), " : expect B to be contiguous.");
  TORCH_CHECK(B.size(1) == K, " : expect B.size(1) == ", K);

  TORCH_CHECK(
      scales.dim() == 1 && scales.size(0) == N,
      " : expect scales to be 1d tensor with size ",
      N);

  auto C = at::empty({M, N}, A.options());

  // --- Launch kernel ---
  Tensor bias = at::Tensor();
  Tensor mat2_zero_points = at::Tensor();
  Tensor non_const_scales = scales;
  auto post_op_args = torch::List<std::optional<at::Scalar>>();

  at::native::onednn::quantized_matmul(
      A.contiguous(),
      1.0,
      0,
      B,
      non_const_scales,
      mat2_zero_points,
      bias,
      C,
      1.0,
      0,
      C.scalar_type(),
      /*other*/ std::nullopt,
      /*other scale*/ 1.0,
      /*other zp*/ 0,
      /*binary post op*/ "none",
      /*binary alpha*/ 1.0,
      /*post_op_name*/ "none",
      post_op_args,
      /*post_op_algorithm*/ "none",
      /*m2_trans*/ false);

  return C;
}
} // namespace at::native
