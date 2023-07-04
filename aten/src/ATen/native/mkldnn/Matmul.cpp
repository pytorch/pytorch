#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/native/mkldnn/Matmul.h>
#include <ATen/OpMathType.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

void mkldnn_matmul(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &result,
    float beta,
    float alpha) {
  TORCH_CHECK(false, "mkldnn_matmul: ATen not compiled with MKLDNN support");
}

bool use_mkldnn_bf16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt){
  return false;
}

bool use_mkldnn_fp16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt){
  return false;
}

bool mkldnn_bf16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::BFloat16 *a, int64_t lda,
    const c10::BFloat16 *b, int64_t ldb,
    float beta,
    c10::BFloat16 *c, int64_t ldc) {
  return false;
}

bool mkldnn_fp16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::Half *a, int64_t lda,
    const c10::Half *b, int64_t ldb,
    float beta,
    c10::Half *c, int64_t ldc) {
  return false;
}

bool use_mkldnn_lower_precision_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result) {
    return false;
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at {
namespace native {

static bool use_mkldnn_bf16_matmul() {
  return (
      at::globalContext().userEnabledMkldnn() &&
      mkldnn_bf16_device_check());
}

static bool use_mkldnn_fp16_matmul() {
  return (
      at::globalContext().userEnabledMkldnn() &&
      mkldnn_fp16_device_check());
}


template<typename scalar_t>
inline typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, bool>
mkldnn_lowerp_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const scalar_t *a_data, int64_t lda,
    const scalar_t *b_data, int64_t ldb,
    float beta,
    scalar_t *c_data, int64_t ldc) {
  if (!(std::is_same_v<scalar_t, c10::BFloat16> ? use_mkldnn_bf16_matmul()
                                                : use_mkldnn_fp16_matmul()) ||
      (m * n * k <= 16 * 16 * 16) || (alpha == 0.0f)) {
    return false;
  }

  ideep::attr_t op_attr;
  // Use mkldnn post ops to perform the add.
  if (beta != 0.0f) {
    op_attr = ideep::attr_t::fuse_sum();
  }

  // NOTE: View as c-contiguous to avoid extra reordering in mkldnn
  // Use identity: C = AB <=> C^T = B^T A^T
  ideep::tensor::dims a_strides{{lda, 1}}, b_strides{{ldb, 1}}, c_strides{{ldc, 1}};
  if (transa != TransposeType::NoTranspose) {
    std::swap(a_strides[0], a_strides[1]);
  }
  if (transb != TransposeType::NoTranspose) {
    std::swap(b_strides[0], b_strides[1]);
  }

  auto idtype = ideep::tensor::data_type::bf16;
  if constexpr (!std::is_same_v<scalar_t, c10::BFloat16>) {
    idtype = ideep::tensor::data_type::f16;
  }

  ideep::tensor a({
      /*sizes=*/{k, m},
      idtype,
      /*strides=*/a_strides},
    const_cast<scalar_t*>(a_data));
  ideep::tensor b({
      /*sizes=*/{n, k},
      idtype,
      /*strides=*/b_strides},
    const_cast<scalar_t*>(b_data));
  ideep::tensor c({
      /*sizes=*/{n, m},
      idtype,
      /*strides=*/c_strides},
    c_data);

  ideep::matmul_forward::compute(
      b, a, c, alpha, beta,
      ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), op_attr);

  if (c.get_data_handle() != c_data){
    // ideep will query onednn expect format of output
    // if given output format is not expected, ideep will re-init an output buffer
    // under this case, we need copy the re-inited buffer back to given buffer
    ideep::tensor real_output({
        /*sizes=*/{n, m},
        idtype,
        /*strides=*/c_strides},
      c_data);
    c.reorder_to(real_output);
  }

  return true;
}

bool mkldnn_bf16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::BFloat16 *a, int64_t lda,
    const c10::BFloat16 *b, int64_t ldb,
    float beta,
    c10::BFloat16 *c, int64_t ldc) {
  return mkldnn_lowerp_gemm<c10::BFloat16>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

bool mkldnn_fp16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::Half *a, int64_t lda,
    const c10::Half *b, int64_t ldb,
    float beta,
    c10::Half *c, int64_t ldc) {
  return mkldnn_lowerp_gemm<c10::Half>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


void mkldnn_matmul(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &result,
    float beta,
    float alpha) {
  TORCH_CHECK((mat1.dim() == 2 && mat2.dim() == 2) || // aten::addmm
              (mat1.dim() == 3 && mat2.dim() == 3) || // aten::bmm, aten::baddbmm
              (mat1.dim() == 2 && mat2.dim() == 1) || // aten::mv
              (mat1.dim() == 1 && mat2.dim() == 1),  // aten::dot
              "mkldnn_matmul:  unsupported dims for mat and mat2");

#if defined(__aarch64__)
  // oneDNN fast-maths mode (enabled by setting the environment variable ONEDNN_DEFAULT_FPMATH_MODE=BF16) will dispatch
  // fp32 inputs to bf16 kernels where HW permits. So, both fp32 and bf16 inputs are permitted.
  TORCH_CHECK((mat1.scalar_type() == mat2.scalar_type()) && (mat1.scalar_type() == result.scalar_type()) &&
              ((mat1.scalar_type() == at::kFloat) || (mat1.scalar_type() == at::kBFloat16)),
              "mkldnn_matmul:  only enabled for fp32 and bf16 path");
  // device needs to support bf16 if the inputs are of bf16 type
  if (mat1.scalar_type() == at::kBFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check_arm(),
                "mkldnn_matmul: mkldnn_matmul bf16 path needs a cpu with bf16 support");
  }
#else
  TORCH_CHECK((mat1.scalar_type() == at::kBFloat16 || mat1.scalar_type() == at::kHalf) &&
                 mat2.scalar_type() == mat1.scalar_type() &&
                 result.scalar_type() == mat1.scalar_type(), "mkldnn_matmul:  only enabled for bf16 and fp16 path");
  if (mat1.scalar_type() == at::kBFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
    "mkldnn_matmul: mkldnn_matmul bf16 path needs the cpu support avx512bw, avx512vl and avx512dq, or AWS Graviton3");
  } else {
    TORCH_CHECK(mkldnn_fp16_device_check(),
    "mkldnn_matmul: mkldnn_matmul fp16 path needs the cpu support avx512_fp16");
  }
#endif

  auto mat1_unsqueezed = mat1.dim() == 1 ? mat1.unsqueeze(0) : mat1;
  auto mat2_unsqueezed = mat2.dim() == 1 ? mat2.unsqueeze(1) : mat2;
  auto result_unsqueezed = result.dim() == 1 ? result.unsqueeze(1) : result;

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
  Tensor mat1_ = is_mkldnn_optimized_format(mat1_unsqueezed) ? mat1_unsqueezed : mat1_unsqueezed.contiguous();
  Tensor mat2_ = is_mkldnn_optimized_format(mat2_unsqueezed) ? mat2_unsqueezed : mat2_unsqueezed.contiguous();
  // Make sure mat1 and mat2 have default contiguous strides if they are contiguous tensors for better performance.
  auto mat1_sizes = mat1_.sizes();
  auto mat1_default_contiguous_strides = c10::contiguous_strides(mat1_sizes);
  if (mat1_.is_contiguous() &&
      mat1_.strides() != c10::IntArrayRef(mat1_default_contiguous_strides)) {
     mat1_ = mat1_.as_strided(mat1_sizes, mat1_default_contiguous_strides);
  }
  auto mat2_sizes = mat2_.sizes();
  auto mat2_default_contiguous_strides = c10::contiguous_strides(mat2_sizes);
  if (mat2_.is_contiguous() &&
      mat2_.strides() != c10::IntArrayRef(mat2_default_contiguous_strides)) {
    mat2_ = mat2_.as_strided(mat2_sizes, mat2_default_contiguous_strides);
  }

  // mkldnn_matmul only proceed CPU tensor
  const ideep::tensor x = itensor_view_from_dense(mat1_);
  const ideep::tensor w = itensor_view_from_dense(mat2_);
  ideep::tensor y = itensor_view_from_dense(result_unsqueezed);
  ideep::matmul_forward::compute(x, w, y, alpha, beta,
      ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), op_attr);
  if (y.get_data_handle() != result.data_ptr()){
    // ideep will query onednn expect format of output
    // if given output format is not expected, ideep will re-init an output buffer
    // under this case, we need copy the re-inited buffer back to given buffer
    ideep::tensor public_y = itensor_view_from_dense(result);
    y.reorder_to(public_y);
  }

  if (mat1.dim() == 1 && mat2.dim() == 1){
    // aten::dot
    result.squeeze_();
  }

}

inline bool checksize(const Tensor& mat1, const Tensor& mat2){
  // if dim = 2, mat1's size = (m * n), mat2's size = (n * k)
  // else if dim = 3, mat1's size = (b * m * n), mat2's size = (b * n * k)
  // else called from aten::mv, mat1.size = (m * n), mat2.size = (n)
  // only m * n * b * k(if exist) are large enough we can get benefit from mkldnn optimized gemm kernel
  static const int64_t mkldnn_gemm_min_size = 16 * 16 * 16;
  if (mat1.dim() == 1 && mat2.dim() == 1) {
    // aten::dot
    return mat1.size(0) > mkldnn_gemm_min_size;
  } else if (mat1.dim() == 2 && mat2.dim() == 1) {
    // aten::mv
    return mat1.size(0) * mat1.size(1) > mkldnn_gemm_min_size;
  } else if (mat2.dim() == 2 && mat2.dim() == 2) {
    // aten::addmm
    return mat1.size(0) * mat1.size(1) * mat2.size(1) > mkldnn_gemm_min_size;
  } else {
    // aten::bmm, aten::baddbmm
    return mat1.size(0) * mat1.size(1) * mat1.size(2) * mat2.size(2) > mkldnn_gemm_min_size;
  }
}

bool use_mkldnn_bf16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result) {
#if defined(__aarch64__)
  if (mkldnn_bf16_device_check_arm()) {
     //onednn fastmath mode can leverage bf16 HW even for the fp32 input, e.g. Arm Neoverse V1
     //so, don't restrict the mkldnn_matmul only for bf16 inputs, allow it for float as well
     return (
        use_mkldnn_bf16_matmul() &&
        (mat1.scalar_type() == mat2.scalar_type()) && (!result.defined() || (mat1.scalar_type() == result.scalar_type())) &&
        ((mat1.scalar_type() == kFloat) || (mat1.scalar_type() == kBFloat16)) &&
        mat1.numel() != 0 &&
        mat2.numel() != 0 &&
        checksize(mat1, mat2));
  } else
#endif
  {
     return (
        use_mkldnn_bf16_matmul() &&
        mat1.scalar_type() == kBFloat16 &&
        mat2.scalar_type() == kBFloat16 &&
        (!result.defined() || result.scalar_type() == kBFloat16) &&
        mat1.numel() != 0 &&
        mat2.numel() != 0 &&
        checksize(mat1, mat2));
  }
}

bool use_mkldnn_fp16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result) {

    return (
      use_mkldnn_fp16_matmul() &&
      mat1.scalar_type() == kHalf &&
      mat2.scalar_type() == kHalf &&
      (!result.defined() || result.scalar_type() == kHalf) &&
      mat1.numel() != 0 &&
      mat2.numel() != 0 &&
      checksize(mat1, mat2));
}

bool use_mkldnn_lower_precision_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result) {
    return (use_mkldnn_bf16_matmul(mat1, mat2, result) || use_mkldnn_fp16_matmul(mat1, mat2, result));
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED
