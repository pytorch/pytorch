#include <ATen/native/cuda/CublasGroupedArgs.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

namespace {

// Scale size modes for the GPU kernel
enum ScaleSizeMode : int8_t {
  SCALE_NONE = 0,
  SCALE_VEC32 = 1,
  SCALE_VEC128 = 2,
  SCALE_BLK128x128 = 3,
};

// cuBLAS VEC32_UE8M0 scale tensor size (bytes, since e8m0 is 1 byte each).
// Mirrors getScaleTensorSize() from the cuBLAS samples.
__device__ __forceinline__ int64_t cublas_vec32_scale_size(int inner, int outer) {
  const int BLOCK_ROWS = 128; // S_BLOCK_INNER(4) * S_VSCALE(32)
  const int BLOCK_COLS = 128; // S_BLOCK_COLS(32) * S_BLOCK_ROWS(4)
  const int S_VSCALE = 32;
  int64_t s_rows = ((inner + BLOCK_ROWS - 1) / BLOCK_ROWS) * (BLOCK_ROWS / S_VSCALE);
  int64_t s_cols = ((outer + BLOCK_COLS - 1) / BLOCK_COLS) * BLOCK_COLS;
  return s_rows * s_cols;
}

// VEC128_32F scale tensor size in bytes. One float32 per 128-element block
// along the inner dimension.
__device__ __forceinline__ int64_t cublas_vec128_scale_size(int inner, int outer) {
  return static_cast<int64_t>(outer) * ((inner + 127) / 128) * 4;
}

// BLK128x128_32F scale tensor size in bytes. One float32 per 128x128 block
// with L4 padding on the inner dimension.
__device__ __forceinline__ int64_t cublas_blk128x128_scale_size(int inner, int outer) {
  int64_t L4 = ((inner + 127) / 128 + 3) / 4 * 4;
  return L4 * ((outer + 127) / 128) * 4;
}

__device__ __forceinline__ int64_t compute_scale_bytes(int inner, int outer, int8_t mode) {
  switch (mode) {
    case SCALE_VEC32: return cublas_vec32_scale_size(inner, outer);
    case SCALE_VEC128: return cublas_vec128_scale_size(inner, outer);
    case SCALE_BLK128x128: return cublas_blk128x128_scale_size(inner, outer);
    default: return 0;
  }
}

__global__ void populate_cublas_grouped_args_kernel(
    const int32_t* __restrict__ offs,
    int64_t base_A, int64_t base_B, int64_t base_D,
    int32_t cublas_m, int32_t cublas_n, int32_t cublas_k,
    bool m_is_delta, bool n_is_delta, bool k_is_delta,
    int32_t lda_val, int32_t ldb_val, int32_t ldd_val,
    int64_t a_offs_stride, int64_t a_idx_stride,
    int64_t b_offs_stride, int64_t b_idx_stride,
    int64_t d_offs_stride, int64_t d_idx_stride,
    int32_t* __restrict__ m_out, int32_t* __restrict__ n_out, int32_t* __restrict__ k_out,
    int32_t* __restrict__ lda_out, int32_t* __restrict__ ldb_out, int32_t* __restrict__ ldd_out,
    int64_t* __restrict__ APtr_out, int64_t* __restrict__ BPtr_out, int64_t* __restrict__ DPtr_out,
    int64_t* __restrict__ alphaPtr_out, int64_t* __restrict__ betaPtr_out,
    float* __restrict__ alpha_ptr, float* __restrict__ beta_ptr,
    int64_t base_scale_a, int64_t base_scale_b,
    int64_t scale_a_stride_bytes, int64_t scale_b_stride_bytes,
    int32_t scale_a_inner, int32_t scale_a_outer,
    int32_t scale_b_inner, int32_t scale_b_outer,
    int64_t* __restrict__ scalePtrA_out, int64_t* __restrict__ scalePtrB_out,
    int8_t scale_a_mode, int8_t scale_b_mode) {
  int i = threadIdx.x;

  if (i == 0) {
    *alpha_ptr = 1.0f;
    *beta_ptr = 0.0f;
  }

  int32_t delta = 0;
  int64_t group_start = 0;
  if (offs != nullptr) {
    int32_t end = offs[i];
    int32_t start_val = (i == 0) ? 0 : offs[i - 1];
    delta = end - start_val;
    group_start = static_cast<int64_t>(start_val);
  }

  m_out[i] = m_is_delta ? delta : cublas_m;
  n_out[i] = n_is_delta ? delta : cublas_n;
  k_out[i] = k_is_delta ? delta : cublas_k;

  lda_out[i] = lda_val;
  ldb_out[i] = ldb_val;
  ldd_out[i] = ldd_val;

  APtr_out[i] = base_A + group_start * a_offs_stride + i * a_idx_stride;
  BPtr_out[i] = base_B + group_start * b_offs_stride + i * b_idx_stride;
  DPtr_out[i] = base_D + group_start * d_offs_stride + i * d_idx_stride;

  alphaPtr_out[i] = reinterpret_cast<int64_t>(alpha_ptr);
  betaPtr_out[i] = reinterpret_cast<int64_t>(beta_ptr);

  if (scalePtrA_out != nullptr) {
    if (scale_a_stride_bytes != 0) {
      // Uniform stride (3D/3D or GroupWise)
      scalePtrA_out[i] = base_scale_a + i * scale_a_stride_bytes;
    } else {
      // Variable-size groups: prefix-sum over per-group scale sizes.
      // A 0 value in scale_a_inner or scale_a_outer means "use delta from offs".
      int64_t offset = 0;
      for (int j = 0; j < i; j++) {
        int32_t dim_j = (j == 0) ? offs[j] : offs[j] - offs[j - 1];
        int32_t inner = scale_a_inner ? scale_a_inner : dim_j;
        int32_t outer = scale_a_outer ? scale_a_outer : dim_j;
        offset += compute_scale_bytes(inner, outer, scale_a_mode);
      }
      scalePtrA_out[i] = base_scale_a + offset;
    }
  }
  if (scalePtrB_out != nullptr) {
    if (scale_b_stride_bytes != 0) {
      scalePtrB_out[i] = base_scale_b + i * scale_b_stride_bytes;
    } else {
      int64_t offset = 0;
      for (int j = 0; j < i; j++) {
        int32_t dim_j = (j == 0) ? offs[j] : offs[j] - offs[j - 1];
        int32_t inner = scale_b_inner ? scale_b_inner : dim_j;
        int32_t outer = scale_b_outer ? scale_b_outer : dim_j;
        offset += compute_scale_bytes(inner, outer, scale_b_mode);
      }
      scalePtrB_out[i] = base_scale_b + offset;
    }
  }
}

void launch_populate_cublas_grouped_args(
    int batchCount,
    const int32_t* offs,
    int64_t base_A, int64_t base_B, int64_t base_D,
    int32_t cublas_m, int32_t cublas_n, int32_t cublas_k,
    bool m_is_delta, bool n_is_delta, bool k_is_delta,
    int32_t lda_val, int32_t ldb_val, int32_t ldd_val,
    int64_t a_offs_stride, int64_t a_idx_stride,
    int64_t b_offs_stride, int64_t b_idx_stride,
    int64_t d_offs_stride, int64_t d_idx_stride,
    int32_t* m_out, int32_t* n_out, int32_t* k_out,
    int32_t* lda_out, int32_t* ldb_out, int32_t* ldd_out,
    int64_t* APtr_out, int64_t* BPtr_out, int64_t* DPtr_out,
    int64_t* alphaPtr_out, int64_t* betaPtr_out,
    float* alpha_ptr, float* beta_ptr,
    int64_t base_scale_a, int64_t base_scale_b,
    int64_t scale_a_stride_bytes, int64_t scale_b_stride_bytes,
    int32_t scale_a_inner, int32_t scale_a_outer,
    int32_t scale_b_inner, int32_t scale_b_outer,
    int64_t* scalePtrA_out, int64_t* scalePtrB_out,
    int8_t scale_a_mode, int8_t scale_b_mode,
    cudaStream_t stream) {
  TORCH_CHECK(batchCount > 0 && batchCount <= 1024,
      "batchCount must be in [1, 1024], got ", batchCount);
  populate_cublas_grouped_args_kernel<<<1, batchCount, 0, stream>>>(
      offs, base_A, base_B, base_D,
      cublas_m, cublas_n, cublas_k,
      m_is_delta, n_is_delta, k_is_delta,
      lda_val, ldb_val, ldd_val,
      a_offs_stride, a_idx_stride,
      b_offs_stride, b_idx_stride,
      d_offs_stride, d_idx_stride,
      m_out, n_out, k_out,
      lda_out, ldb_out, ldd_out,
      APtr_out, BPtr_out, DPtr_out,
      alphaPtr_out, betaPtr_out,
      alpha_ptr, beta_ptr,
      base_scale_a, base_scale_b,
      scale_a_stride_bytes, scale_b_stride_bytes,
      scale_a_inner, scale_a_outer,
      scale_b_inner, scale_b_outer,
      scalePtrA_out, scalePtrB_out,
      scale_a_mode, scale_b_mode);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020
cublasGroupedArgs::cublasGroupedArgs(
    const Tensor& mat1,
    const Tensor& mat2,
    const std::optional<Tensor>& offs,
    Tensor& c,
    const std::optional<Tensor>& scale_a,
    const std::optional<Tensor>& scale_b,
    const std::optional<Tensor>& scale_result,
    const std::optional<at::blas::ScalingType>& scaling_choice_a,
    const std::optional<at::blas::ScalingType>& scaling_choice_b) {
  const bool a_is_2d = mat1.dim() == 2;
  const bool b_is_2d = mat2.dim() == 2;
  if (a_is_2d || b_is_2d) {
    TORCH_CHECK(offs.has_value(), "Offsets tensor must be provided when at least one input is 2D");
  }

  A_dtype = mat2.scalar_type();
  B_dtype = mat1.scalar_type();
  result_dtype = c.scalar_type();
  const int64_t esz = mat1.element_size();
  const int64_t out_esz = c.element_size();

  if (offs.has_value()) {
    batchCount = offs.value().size(0);
  } else {
    batchCount = mat1.size(0);
  }

  // cuBLAS is column-major. To get a row-major result C = mat1 × mat2,
  // we use the identity C^T = mat2^T × mat1^T. So cuBLAS-A = mat2 and
  // cuBLAS-B = mat1. The transpose flags depend on inner-dim layout:
  //   row-major (stride(-1)==1): cuBLAS sees it as col-major "already
  //     transposed" → after the B^T×A^T flip, the op flag is 'n'
  //   col-major (stride(-2)==1): cuBLAS sees it naturally → after
  //     the flip, the op flag is 't'
  const bool mat2_row_major = mat2.stride(-1) == 1;
  const bool mat1_row_major = mat1.stride(-1) == 1;
  transa = mat2_row_major ? 'n' : 't';
  transb = mat1_row_major ? 'n' : 't';

  // User-space dimensions
  const int64_t user_M = mat1.size(-2);
  const int64_t user_N = mat2.size(-1);
  const int64_t user_K = mat1.size(-1);

  // In the cuBLAS B^T×A^T convention:
  //   cublas_m = user_N, cublas_n = user_M, cublas_k = user_K
  const int32_t cublas_m = static_cast<int32_t>(user_N);
  const int32_t cublas_n = static_cast<int32_t>(user_M);
  const int32_t cublas_k = static_cast<int32_t>(user_K);

  // Leading dimensions (constant across groups, from inner-dim strides)
  // cuBLAS-A = mat2, cuBLAS-B = mat1
  const int32_t lda_val = static_cast<int32_t>(transa == 't' ? mat2.stride(-1) : mat2.stride(-2));
  const int32_t ldb_val = static_cast<int32_t>(transb == 't' ? mat1.stride(-1) : mat1.stride(-2));
  const int32_t ldd_val = static_cast<int32_t>(c.stride(-2));

  if (scale_a && scale_b) {
    scale_mata_ptr = scale_b->data_ptr();
    scale_matb_ptr = scale_a->data_ptr();
    scale_mata_dtype = scale_b->scalar_type();
    scale_matb_dtype = scale_a->scalar_type();

    auto infer = [&](const Tensor& scale) -> at::blas::ScalingType {
      if (scale.scalar_type() == at::kFloat8_e8m0fnu)
        return at::blas::ScalingType::BlockWise1x32;
      if (scale.numel() == 1)
        return at::blas::ScalingType::TensorWise;
      return at::blas::ScalingType::GroupWise;
    };
    // mata corresponds to scale_b (cuBLAS-A = mat2)
    // scaling_choice_b is the user's B recipe → cuBLAS-A scale mode
    scale_mata_scaling_type = scaling_choice_b.value_or(infer(*scale_b));
    // matb corresponds to scale_a (cuBLAS-B = mat1)
    // scaling_choice_a is the user's A recipe → cuBLAS-B scale mode
    scale_matb_scaling_type = scaling_choice_a.value_or(infer(*scale_a));
  }
  if (scale_result) {
    scale_result_ptr = scale_result->data_ptr();
  }

  // GroupWise, BlockWise1x32, and VEC128/BLK128x128 scales need device-side
  // pointer arrays (one pointer per group) because cuBLAS expects the scale
  // pointer to be an array of device pointers for grouped GEMM.
  auto needs_ptr_array = [](at::blas::ScalingType st) {
    return st == at::blas::ScalingType::GroupWise
        || st == at::blas::ScalingType::BlockWise1x32
        || st == at::blas::ScalingType::BlockWise1x128
        || st == at::blas::ScalingType::BlockWise128x128;
  };
  const bool mata_needs_ptr = needs_ptr_array(scale_mata_scaling_type);
  const bool matb_needs_ptr = needs_ptr_array(scale_matb_scaling_type);

  // Determine per-case which dimensions are variable (delta-based)
  // and how pointer strides work
  bool m_is_delta = false, n_is_delta = false, k_is_delta = false;
  int64_t a_offs_stride = 0, a_idx_stride = 0;
  int64_t b_offs_stride = 0, b_idx_stride = 0;
  int64_t d_offs_stride = 0, d_idx_stride = 0;

  if (a_is_2d && b_is_2d) {
    // 2D x 2D: jagged K
    k_is_delta = true;
    a_offs_stride = mat2.stride(-2) * esz;
    b_offs_stride = mat1.stride(-1) * esz;
    d_idx_stride = c.stride(0) * out_esz;
    avgM = cublas_m;
    avgN = cublas_n;
    avgK = user_K / batchCount;
  } else if (a_is_2d && !b_is_2d) {
    // 2D x 3D: jagged M (user M varies, cublas n varies)
    n_is_delta = true;
    a_idx_stride = mat2.stride(0) * esz;
    b_offs_stride = mat1.stride(-2) * esz;
    d_offs_stride = c.stride(-2) * out_esz;
    avgM = cublas_m;
    avgN = user_M / batchCount;
    avgK = cublas_k;
  } else if (!a_is_2d && b_is_2d) {
    // 3D x 2D: jagged N (user N varies, cublas m varies)
    m_is_delta = true;
    a_offs_stride = mat2.stride(-1) * esz;
    b_idx_stride = mat1.stride(0) * esz;
    d_offs_stride = c.stride(-1) * out_esz;
    avgM = user_N / batchCount;
    avgN = cublas_n;
    avgK = cublas_k;
  } else {
    // 3D x 3D: all dimensions fixed
    a_idx_stride = mat2.stride(0) * esz;
    b_idx_stride = mat1.stride(0) * esz;
    d_idx_stride = c.stride(0) * out_esz;
    avgM = cublas_m;
    avgN = cublas_n;
    avgK = cublas_k;
  }

  // Single device allocation for all arrays:
  //   6 x int32[batchCount]  (m, n, k, lda, ldb, ldd)
  //   5 x int64[batchCount]  (A, B, D, alpha, beta ptrs)
  //   2 x float              (alpha, beta scalars)
  // + optionally up to 2 x int64[batchCount] for per-group scale pointer arrays
  //   (GroupWise + VEC128/BLK128x128 scales: embedded in the main buffer)
  //   (BlockWise1x32 scales: separate allocation required by cuBLAS)
  const bool mata_blockwise_1x32 = scale_mata_scaling_type == at::blas::ScalingType::BlockWise1x32;
  const bool matb_blockwise_1x32 = scale_matb_scaling_type == at::blas::ScalingType::BlockWise1x32;
  const int embedded_ptr_arrays =
      ((mata_needs_ptr && !mata_blockwise_1x32) ? 1 : 0) +
      ((matb_needs_ptr && !matb_blockwise_1x32) ? 1 : 0);
  const int64_t buf_bytes =
      static_cast<int64_t>(batchCount) * 6 * sizeof(int32_t) +
      static_cast<int64_t>(batchCount) * 5 * sizeof(int64_t) +
      2 * sizeof(float) +
      static_cast<int64_t>(embedded_ptr_arrays) * batchCount * sizeof(int64_t);
  buf = at::empty({buf_bytes}, mat1.options().dtype(at::kByte));

  // Typed pointer arithmetic (same pattern as GroupMM.cu).
  // reinterpret_cast only at type boundaries.
  mArray   = reinterpret_cast<int32_t*>(buf.data_ptr());
  nArray   = mArray + batchCount;
  kArray   = nArray + batchCount;
  ldaArray = kArray + batchCount;
  ldbArray = ldaArray + batchCount;
  lddArray = ldbArray + batchCount;

  APtrArray     = reinterpret_cast<int64_t*>(lddArray + batchCount);
  BPtrArray     = APtrArray + batchCount;
  DPtrArray     = BPtrArray + batchCount;
  alphaPtrArray = DPtrArray + batchCount;
  betaPtrArray  = alphaPtrArray + batchCount;

  float* alpha_scalar = reinterpret_cast<float*>(betaPtrArray + batchCount);
  float* beta_scalar  = alpha_scalar + 1;
  alphaScalar = alpha_scalar;
  betaScalar = beta_scalar;

  // Per-group scale pointer arrays: embedded for GroupWise/VEC128/BLK128x128,
  // separate for BlockWise1x32.
  // Place after the two float scalars (alpha_scalar, beta_scalar are 8 bytes
  // total, so the next int64_t is naturally aligned).
  int64_t* scaleAPtrArray = nullptr;
  int64_t* scaleBPtrArray = nullptr;
  int64_t* scale_ptr_base = reinterpret_cast<int64_t*>(beta_scalar + 1);
  if (mata_needs_ptr && !mata_blockwise_1x32) {
    scaleAPtrArray = scale_ptr_base;
    scale_ptr_base += batchCount;
  }
  if (matb_needs_ptr && !matb_blockwise_1x32) {
    scaleBPtrArray = scale_ptr_base;
  }
  // BlockWise1x32 scale pointer arrays need separate allocations
  const int blockwise_ptr_arrays = (mata_blockwise_1x32 ? 1 : 0) + (matb_blockwise_1x32 ? 1 : 0);
  if (blockwise_ptr_arrays > 0) {
    scale_ptr_buf = at::empty(
        {static_cast<int64_t>(blockwise_ptr_arrays) * batchCount * 8},
        mat1.options().dtype(at::kByte));
    char* sbase = static_cast<char*>(scale_ptr_buf.data_ptr());
    int64_t soffset = 0;
    if (mata_blockwise_1x32) {
      scaleAPtrArray = reinterpret_cast<int64_t*>(sbase + soffset);
      soffset += batchCount * 8;
    }
    if (matb_blockwise_1x32) {
      scaleBPtrArray = reinterpret_cast<int64_t*>(sbase + soffset);
    }
  }

  // Base addresses for scale data (cuBLAS-A = mat2 → scale_b, cuBLAS-B = mat1 → scale_a)
  const int64_t base_scale_a = scale_b ? reinterpret_cast<int64_t>(scale_b->data_ptr()) : 0;
  const int64_t base_scale_b = scale_a ? reinterpret_cast<int64_t>(scale_a->data_ptr()) : 0;

  // Byte stride between consecutive groups' scale data.
  // For GroupWise (1D float): stride(0)*elem_size = 1*4 = sizeof(float).
  // For BlockWise 3D/3D: stride(0)*elem_size = per_group_numel*elem_size.
  // For BlockWise with jagged dims: 0 signals variable-size mode.
  auto is_blockwise = [](at::blas::ScalingType st) {
    return st == at::blas::ScalingType::BlockWise1x32
        || st == at::blas::ScalingType::BlockWise1x128
        || st == at::blas::ScalingType::BlockWise128x128;
  };
  const bool blockwise_variable_a = is_blockwise(scale_mata_scaling_type) && (a_is_2d || b_is_2d);
  const bool blockwise_variable_b = is_blockwise(scale_matb_scaling_type) && (a_is_2d || b_is_2d);
  const int64_t scale_a_stride_bytes = blockwise_variable_a ? 0 :
      (scale_b && scale_b->dim() >= 2
          ? scale_b->stride(0) * scale_b->element_size()
          : (scale_b ? scale_b->element_size() : 0));
  const int64_t scale_b_stride_bytes = blockwise_variable_b ? 0 :
      (scale_a && scale_a->dim() >= 2
          ? scale_a->stride(0) * scale_a->element_size()
          : (scale_a ? scale_a->element_size() : 0));

  // For variable-size blockwise scales, pass inner/outer dims for per-group
  // scale size computation. A value of 0 means "substitute the jagged
  // dimension (delta from offsets) at runtime".
  // cuBLAS-A scale: size(inner=K, outer=cublas_m=user_N)
  // cuBLAS-B scale: size(inner=K, outer=cublas_n=user_M)
  int32_t scale_a_inner = 0, scale_a_outer = 0;
  int32_t scale_b_inner = 0, scale_b_outer = 0;
  if (blockwise_variable_a) {
    // K varies (2D/2D): inner=0, outer=cublas_m
    // cublas_m varies (3D/2D): inner=cublas_k, outer=0
    scale_a_inner = k_is_delta ? 0 : cublas_k;
    scale_a_outer = m_is_delta ? 0 : cublas_m;
  }
  if (blockwise_variable_b) {
    // K varies (2D/2D): inner=0, outer=cublas_n
    // cublas_n varies (2D/3D): inner=cublas_k, outer=0
    scale_b_inner = k_is_delta ? 0 : cublas_k;
    scale_b_outer = n_is_delta ? 0 : cublas_n;
  }

  const int64_t base_A = reinterpret_cast<int64_t>(mat2.data_ptr());
  const int64_t base_B = reinterpret_cast<int64_t>(mat1.data_ptr());
  const int64_t base_D = reinterpret_cast<int64_t>(c.data_ptr());

  const int32_t* offs_ptr = offs.has_value()
      ? static_cast<const int32_t*>(offs.value().data_ptr())
      : nullptr;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto to_scale_mode = [](at::blas::ScalingType st) -> int8_t {
    switch (st) {
      case at::blas::ScalingType::BlockWise1x32: return SCALE_VEC32;
      case at::blas::ScalingType::BlockWise1x128: return SCALE_VEC128;
      case at::blas::ScalingType::BlockWise128x128: return SCALE_BLK128x128;
      default: return SCALE_NONE;
    }
  };
  const int8_t scale_a_mode = to_scale_mode(scale_mata_scaling_type);
  const int8_t scale_b_mode = to_scale_mode(scale_matb_scaling_type);

  launch_populate_cublas_grouped_args(
        batchCount, offs_ptr,
        base_A, base_B, base_D,
        cublas_m, cublas_n, cublas_k,
        m_is_delta, n_is_delta, k_is_delta,
        lda_val, ldb_val, ldd_val,
        a_offs_stride, a_idx_stride,
        b_offs_stride, b_idx_stride,
        d_offs_stride, d_idx_stride,
        mArray, nArray, kArray,
        ldaArray, ldbArray, lddArray,
        APtrArray, BPtrArray, DPtrArray,
        alphaPtrArray, betaPtrArray,
        alpha_scalar, beta_scalar,
        base_scale_a, base_scale_b,
        scale_a_stride_bytes, scale_b_stride_bytes,
        scale_a_inner, scale_a_outer,
        scale_b_inner, scale_b_outer,
        scaleAPtrArray, scaleBPtrArray,
        scale_a_mode, scale_b_mode,
        stream);

  // For per-group scales, point to the device-side pointer arrays
  // instead of the raw data pointer
  if (mata_needs_ptr) {
    scale_mata_ptr = scaleAPtrArray;
  }
  if (matb_needs_ptr) {
    scale_matb_ptr = scaleBPtrArray;
  }
}
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020

} // namespace at::native
