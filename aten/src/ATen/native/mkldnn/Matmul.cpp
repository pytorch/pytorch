#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/native/mkldnn/Matmul.h>

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
bool mkldnn_bf32_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc){
      return false;
    }

bool use_mkldnn_bf32_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result) {
    return false;
}

bool use_mkldnn_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result) {
    return false;
}

void mkldnn_matmul_i8i8i32(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &result) {
  TORCH_INTERNAL_ASSERT(false, __func__, ": ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at {
namespace native {

static bool use_mkldnn_bf16_matmul() {
  return at::globalContext().userEnabledMkldnn() && mkldnn_bf16_device_check();
}

static bool use_mkldnn_fp16_matmul() {
  return at::globalContext().userEnabledMkldnn() && mkldnn_fp16_device_check();
}

static bool use_mkldnn_bf32_matmul() {
  return use_mkldnn_bf16_matmul() && at::globalContext().float32MatmulPrecision() == at::Float32MatmulPrecision::MEDIUM;
}


template<typename scalar_t>
inline typename std::enable_if_t<
    std::is_same_v<scalar_t, float> ||
    std::is_same_v<scalar_t, c10::Half> ||
    std::is_same_v<scalar_t, c10::BFloat16>,
    bool>
mkldnn_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const scalar_t *a_data, int64_t lda,
    const scalar_t *b_data, int64_t ldb,
    float beta,
    scalar_t *c_data, int64_t ldc) {
  bool bf16_usable = std::is_same_v<scalar_t, c10::BFloat16> && use_mkldnn_bf16_matmul();
  bool fp16_usable = std::is_same_v<scalar_t, c10::Half> && use_mkldnn_fp16_matmul();
  bool bf32_usable = std::is_same_v<scalar_t, float> && use_mkldnn_bf32_matmul();
  if ( !(bf16_usable || fp16_usable || bf32_usable) ||
      (m * n * k <= 16 * 16 * 16) || (alpha == 0.0f)) {
    return false;
  }

  ideep::attr_t op_attr;
  // Use mkldnn post ops to perform the add.
  if (beta != 0.0f) {
    op_attr = ideep::attr_t::fuse_sum();
  }
  if (bf32_usable) op_attr.set_fpmath_mode(dnnl_fpmath_mode_bf16); // bf32 path

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
  if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    idtype = ideep::tensor::data_type::f16;
  }
  if constexpr (std::is_same_v<scalar_t, float>) {
    idtype = ideep::tensor::data_type::f32;
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
  return mkldnn_gemm<c10::BFloat16>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

bool mkldnn_fp16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::Half *a, int64_t lda,
    const c10::Half *b, int64_t ldb,
    float beta,
    c10::Half *c, int64_t ldc) {
  return mkldnn_gemm<c10::Half>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

bool mkldnn_bf32_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc){
      return mkldnn_gemm<float>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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
  TORCH_CHECK(
      (mat1.scalar_type() == at::kBFloat16 ||
       mat1.scalar_type() == at::kHalf ||
       mat1.scalar_type() == at::kFloat) &&
          mat2.scalar_type() == mat1.scalar_type() &&
          result.scalar_type() == mat1.scalar_type(),
      "mkldnn_matmul:  only enabled for bf16 and fp16 path");
  if (mat1.scalar_type() == at::kBFloat16 || mat1.scalar_type() == at::kFloat) {
    TORCH_CHECK(
        mkldnn_bf16_device_check(),
        "mkldnn_matmul: mkldnn_matmul bf16 path needs the cpu support avx_ne_convert or avx512bw, avx512vl and avx512dq, or AWS Graviton3");
  } else {
    TORCH_INTERNAL_ASSERT(mat1.scalar_type() == at::kHalf);
    TORCH_CHECK(
        mkldnn_fp16_device_check(),
        "mkldnn_matmul: mkldnn_matmul fp16 path needs the cpu support avx_ne_convert or avx512_fp16");
  }
#endif

  auto mat1_unsqueezed = mat1.dim() == 1 ? mat1.unsqueeze(0) : mat1;
  auto mat2_unsqueezed = mat2.dim() == 1 ? mat2.unsqueeze(1) : mat2;
  auto result_unsqueezed = result.dim() == 1 ? result.unsqueeze(1) : result;
  bool bf32_usable = mat1.scalar_type() == at::kFloat && use_mkldnn_bf32_matmul();

  ideep::attr_t op_attr;
  // "addmm", "addbmm" "baddbmm" in pytorch allow bias to be 2-D or 3-D tensor
  // but mkldnn matmul primitive only support bias be 1-D tensors
  // to address their differences, we use mkldnn post ops to perform a fused "add" after matrix multiplication is over
  if (beta != 0.0f) op_attr = ideep::attr_t::fuse_sum();
  if (bf32_usable) op_attr.set_fpmath_mode(dnnl_fpmath_mode_bf16); // bf32 path
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
  mat1_ = may_convert_to_default_contiguous_strides(mat1_);
  mat2_ = may_convert_to_default_contiguous_strides(mat2_);

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

bool use_mkldnn_bf32_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result) {

    return (
      use_mkldnn_bf32_matmul() &&
      mat1.scalar_type() == kFloat &&
      mat2.scalar_type() == kFloat &&
      (!result.defined() || result.scalar_type() == kFloat) &&
      mat1.numel() != 0 &&
      mat2.numel() != 0 &&
      checksize(mat1, mat2));
}

bool use_mkldnn_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result) {
  return (use_mkldnn_bf16_matmul(mat1, mat2, result) || use_mkldnn_fp16_matmul(mat1, mat2, result) || use_mkldnn_bf32_matmul(mat1, mat2, result));
}

static void _mkldnn_matmul_i8i8i32_with_primitive(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &result) {
  // Create ideep tensors for oneDNN computation
  auto src = ideep::tensor(
      {mat1.sizes().vec(),
       ideep::tensor::data_type::s8,
       mat1.strides().vec()},
      mat1.data_ptr());
  auto wei = ideep::tensor(
      {mat2.sizes().vec(),
       ideep::tensor::data_type::s8,
       mat2.strides().vec()},
      mat2.data_ptr());
  auto dst = ideep::tensor(
      {result.sizes().vec(),
       ideep::tensor::data_type::s32,
       result.strides().vec()},
      result.data_ptr());
  // Create primitive desc
  auto engine = ideep::engine::cpu_engine();
  ideep::attr_t op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto src_desc = src.get_desc();
  auto wei_desc = wei.get_desc();
  auto dst_desc = dst.get_desc();
  auto prim_desc = dnnl::matmul::primitive_desc(
      engine, src_desc, wei_desc, dst_desc, op_attr);
  // Reorder mat2 if needed
  auto expected_weight = wei.reorder_if_differ_in(prim_desc.weights_desc());
  // Prepare args for primitive
  ideep::tensor scratchpad(prim_desc.scratchpad_desc());
  ideep::exec_args args;
  args.insert({DNNL_ARG_SRC, src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weight});
  args.insert({DNNL_ARG_DST, dst});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  // Create primitve and execute
  auto primitive = dnnl::matmul(prim_desc);
  primitive.execute(ideep::stream::default_stream(), args);
}

static void _mkldnn_gemm_i8i8i32_with_blas(
  const Tensor& self,
  const Tensor& mat2,
  const Tensor& result) {
    const int m = result.size(0);
    const int n = result.size(1);
    const int k = self.size(1);

    const char transa = self.strides()[1] == 1 ? 'N' : 'T';
    const char transb = mat2.strides()[1] == 1 ? 'N' : 'T';
    const char offsetc = 'F';

    const int lda = transa == 'T' ? self.stride(1) : self.stride(0);
    const int ldb = transb == 'T' ? mat2.stride(1) : mat2.stride(0);
    const int ldc = n;

    const float alpha = 1;
    const float beta = 0;

    int8_t ao = 0;
    int8_t bo = 0;
    int32_t co = 0;

    dnnl::gemm_s8s8s32(
        transa,
        transb,
        offsetc,
        m,
        n,
        k,
        alpha,
        (int8_t*)self.data_ptr(),
        lda,
        ao,
        (int8_t*)mat2.data_ptr(),
        ldb,
        bo,
        beta,
        (int32_t*)result.data_ptr(),
        ldc,
        &co);
  }

void mkldnn_matmul_i8i8i32(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &result) {
  // x:s8 * w:s8 -> y:s32
  // both inputs should be 2d
  // In most cases, using DNNL blas API is faster but it requires a/b contiguous along one dimentsion
  bool a_is_contigous = (mat1.stride(0) == 1 || mat1.stride(1) == 1);
  bool b_is_contigous = (mat2.stride(0) == 1 || mat2.stride(1) == 1);

  if (a_is_contigous && b_is_contigous) {
    _mkldnn_gemm_i8i8i32_with_blas(mat1, mat2, result);
  } else {
    _mkldnn_matmul_i8i8i32_with_primitive(mat1, mat2, result);
  }
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED
