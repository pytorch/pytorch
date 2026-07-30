#include <ATen/native/cuda/CublasGroupedArgs.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13030

namespace {

void check_cublaslt_grouped_alignment(
    bool a_is_2d,
    bool b_is_2d,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t alignment,
    char transa,
    char transb) {
  if (a_is_2d && b_is_2d) {
    TORCH_CHECK(
        k % alignment == 0 || (transa == 't' && transb == 'n'),
        "cublasLt grouped GEMM with jagged K not aligned to 16 bytes requires transa=t and transb=n, got transa=",
        transa,
        " transb=",
        transb,
        " K=",
        k);
  } else if (a_is_2d && !b_is_2d) {
    TORCH_CHECK(
        m % alignment == 0 || !(transa == 't' && transb == 't'),
        "cublasLt grouped GEMM with jagged M not aligned to 16 bytes does not support transa=t and transb=t, got transa=",
        transa,
        " transb=",
        transb,
        " M=",
        m);
  } else if (!a_is_2d && b_is_2d) {
    TORCH_CHECK(
        n % alignment == 0 || !(transa == 'n' && transb == 'n'),
        "cublasLt grouped GEMM with jagged N not aligned to 16 bytes does not support transa=n and transb=n, got transa=",
        transa,
        " transb=",
        transb,
        " N=",
        n);
  } else {
    TORCH_CHECK(
        k % alignment == 0 || (transa == 't' && transb == 'n'),
        "cublasLt grouped GEMM with K not aligned to 16 bytes requires transa=t and transb=n, got transa=",
        transa,
        " transb=",
        transb,
        " K=",
        k);
  }
}

template <typename IndexType>
__global__ void populate_cublas_grouped_args_kernel(
    const int32_t* __restrict__ offs,
    int64_t base_A, int64_t base_B, int64_t base_D,
    IndexType cublas_m, IndexType cublas_n, IndexType cublas_k,
    bool m_is_delta, bool n_is_delta, bool k_is_delta,
    IndexType lda_val, IndexType ldb_val, IndexType ldd_val,
    int64_t a_offs_stride, int64_t a_idx_stride,
    int64_t b_offs_stride, int64_t b_idx_stride,
    int64_t d_offs_stride, int64_t d_idx_stride,
    IndexType* __restrict__ m_out, IndexType* __restrict__ n_out, IndexType* __restrict__ k_out,
    IndexType* __restrict__ lda_out, IndexType* __restrict__ ldb_out, IndexType* __restrict__ ldd_out,
    int64_t* __restrict__ APtr_out, int64_t* __restrict__ BPtr_out, int64_t* __restrict__ DPtr_out,
    int64_t* __restrict__ alphaPtr_out, int64_t* __restrict__ betaPtr_out,
    float* __restrict__ alpha_ptr, float* __restrict__ beta_ptr) {
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
    CUDA_KERNEL_ASSERT(delta >= 0 && "expected gemm dimension to be greater or equal 0\n");

    if (m_is_delta) {
      CUDA_KERNEL_ASSERT(end <= cublas_m && "expected offset to be less than tensor size\n");
    } else if (n_is_delta) {
      CUDA_KERNEL_ASSERT(end <= cublas_n && "expected offset to be less than tensor size\n");
    } else if (k_is_delta) {
      CUDA_KERNEL_ASSERT(end <= cublas_k && "expected offset to be less than tensor size\n");
    }

    if (i < blockDim.x - 1) {
      if (a_offs_stride != 0) {
        CUDA_KERNEL_ASSERT(
            (static_cast<int64_t>(delta) * a_offs_stride) % 16 == 0 &&
            "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }
      if (b_offs_stride != 0) {
        CUDA_KERNEL_ASSERT(
            (static_cast<int64_t>(delta) * b_offs_stride) % 16 == 0 &&
            "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }
      if ((m_is_delta || n_is_delta) && d_offs_stride != 0) {
        CUDA_KERNEL_ASSERT(
            (static_cast<int64_t>(delta) * d_offs_stride) % 16 == 0 &&
            "expected output tensor dynamic dimension byte size to be non-negative multiple of 16\n");
      }
    }

    group_start = static_cast<int64_t>(start_val);
  }

  m_out[i] = m_is_delta ? static_cast<IndexType>(delta) : cublas_m;
  n_out[i] = n_is_delta ? static_cast<IndexType>(delta) : cublas_n;
  k_out[i] = k_is_delta ? static_cast<IndexType>(delta) : cublas_k;

  lda_out[i] = lda_val;
  ldb_out[i] = ldb_val;
  ldd_out[i] = ldd_val;

  APtr_out[i] = base_A + group_start * a_offs_stride + i * a_idx_stride;
  BPtr_out[i] = base_B + group_start * b_offs_stride + i * b_idx_stride;
  DPtr_out[i] = base_D + group_start * d_offs_stride + i * d_idx_stride;

  // Store device pointer as int64_t — cublasLt grouped GEMM expects
  // per-group alpha/beta as arrays of device pointers.
  alphaPtr_out[i] = reinterpret_cast<int64_t>(alpha_ptr);
  betaPtr_out[i] = reinterpret_cast<int64_t>(beta_ptr);
}

// Carves up args.buf into typed sub-arrays, stores the pointers on `args`,
// and launches the kernel that fills them in. Layout of buf:
//   [m, n, k, lda, ldb, ldd]   -- 6 x batchCount of IndexType
//   [A*, B*, D*, alpha*, beta*]-- 5 x batchCount of int64_t (device ptrs)
//   [alpha_scalar, beta_scalar]-- 2 x float
template <typename IndexType>
void launch_populate_cublas_grouped_args(
    cublasGroupedArgs& args,
    int batchCount,
    const int32_t* offs,
    int64_t base_A, int64_t base_B, int64_t base_D,
    int64_t cublas_m, int64_t cublas_n, int64_t cublas_k,
    bool m_is_delta, bool n_is_delta, bool k_is_delta,
    int64_t lda_val, int64_t ldb_val, int64_t ldd_val,
    int64_t a_offs_stride, int64_t a_idx_stride,
    int64_t b_offs_stride, int64_t b_idx_stride,
    int64_t d_offs_stride, int64_t d_idx_stride,
    cudaStream_t stream) {
  IndexType* m_arr   = reinterpret_cast<IndexType*>(args.buf.data_ptr());
  IndexType* n_arr   = m_arr + batchCount;
  IndexType* k_arr   = n_arr + batchCount;
  IndexType* lda_arr = k_arr + batchCount;
  IndexType* ldb_arr = lda_arr + batchCount;
  IndexType* ldd_arr = ldb_arr + batchCount;

  args.APtrArray     = reinterpret_cast<int64_t*>(ldd_arr + batchCount);
  args.BPtrArray     = args.APtrArray + batchCount;
  args.DPtrArray     = args.BPtrArray + batchCount;
  args.alphaPtrArray = args.DPtrArray + batchCount;
  args.betaPtrArray  = args.alphaPtrArray + batchCount;

  args.alphaScalar = reinterpret_cast<float*>(args.betaPtrArray + batchCount);
  args.betaScalar  = args.alphaScalar + 1;

  args.mArray   = m_arr;
  args.nArray   = n_arr;
  args.kArray   = k_arr;
  args.ldaArray = lda_arr;
  args.ldbArray = ldb_arr;
  args.lddArray = ldd_arr;

  populate_cublas_grouped_args_kernel<IndexType><<<1, batchCount, 0, stream>>>(
      offs, base_A, base_B, base_D,
      static_cast<IndexType>(cublas_m), static_cast<IndexType>(cublas_n), static_cast<IndexType>(cublas_k),
      m_is_delta, n_is_delta, k_is_delta,
      static_cast<IndexType>(lda_val), static_cast<IndexType>(ldb_val), static_cast<IndexType>(ldd_val),
      a_offs_stride, a_idx_stride,
      b_offs_stride, b_idx_stride,
      d_offs_stride, d_idx_stride,
      m_arr, n_arr, k_arr,
      lda_arr, ldb_arr, ldd_arr,
      args.APtrArray, args.BPtrArray, args.DPtrArray,
      args.alphaPtrArray, args.betaPtrArray,
      args.alphaScalar, args.betaScalar);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

cublasGroupedArgs::cublasGroupedArgs(
    const Tensor& mat1,
    const Tensor& mat2,
    const std::optional<Tensor>& offs,
    Tensor& c,
    int batchCount_,
    bool needs_int64) {
  const bool a_is_2d = mat1.dim() == 2;
  const bool b_is_2d = mat2.dim() == 2;
  if (a_is_2d || b_is_2d) {
    TORCH_CHECK(offs.has_value(), "Offsets tensor must be provided when at least one input is 2D");
  }

  transa = mat1.stride(-1) == 1 ? 'n' : 't';
  transb = mat2.stride(-1) == 1 ? 'n' : 't';

  const int64_t element_size = mat1.element_size();
  const int64_t out_element_size = c.element_size();

  batchCount = batchCount_;
  use_int64 = needs_int64;

  const int64_t cublas_m = mat1.size(-2);
  const int64_t cublas_n = mat2.size(-1);
  const int64_t cublas_k = std::min(mat1.size(-1), mat2.size(-2));
  const int64_t lda_val = transa == 'n' ? mat1.stride(-2) : mat1.stride(-1);
  const int64_t ldb_val = transb == 'n' ? mat2.stride(-2) : mat2.stride(-1);
  const int64_t ldd_val = c.stride(-2);

  // Determine per-case which dimensions are variable (delta-based)
  // and how pointer strides work
  bool m_is_delta = false, n_is_delta = false, k_is_delta = false;
  int64_t a_offs_stride = 0, a_idx_stride = 0;
  int64_t b_offs_stride = 0, b_idx_stride = 0;
  int64_t d_offs_stride = 0, d_idx_stride = 0;

  if (a_is_2d && b_is_2d) {
    // 2D x 2D: jagged K
    k_is_delta = true;
    a_offs_stride = mat1.stride(-1) * element_size;
    b_offs_stride = mat2.stride(-2) * element_size;
    d_idx_stride = c.stride(0) * out_element_size;
    m = cublas_m;
    n = cublas_n;
    k = cublas_k / batchCount;
  } else if (a_is_2d && !b_is_2d) {
    // 2D x 3D: jagged M
    m_is_delta = true;
    a_offs_stride = mat1.stride(-2) * element_size;
    b_idx_stride = mat2.stride(0) * element_size;
    d_offs_stride = c.stride(-2) * out_element_size;
    m = cublas_m / batchCount;
    n = cublas_n;
    k = cublas_k;
  } else if (!a_is_2d && b_is_2d) {
    // 3D x 2D: jagged N
    n_is_delta = true;
    a_idx_stride = mat1.stride(0) * element_size;
    b_offs_stride = mat2.stride(-1) * element_size;
    d_offs_stride = c.stride(-1) * out_element_size;
    m = cublas_m;
    n = cublas_n / batchCount;
    k = cublas_k;
  } else {
    // 3D x 3D: all dimensions fixed
    a_idx_stride = mat1.stride(0) * element_size;
    b_idx_stride = mat2.stride(0) * element_size;
    d_idx_stride = c.stride(0) * out_element_size;
    m = cublas_m;
    n = cublas_n;
    k = cublas_k;
  }

  const int64_t alignment = 16 / element_size;
  check_cublaslt_grouped_alignment(
      a_is_2d,
      b_is_2d,
      cublas_m,
      cublas_n,
      cublas_k,
      alignment,
      transa,
      transb);

  // Determine element size for dimension arrays
  const size_t dim_elem_size = use_int64 ? sizeof(int64_t) : sizeof(int32_t);

  // Single device allocation for all arrays:
  //   6 x dim_elem_size[batchCount]  (m, n, k, lda, ldb, ldd)
  //   5 x int64[batchCount]          (A, B, D, alpha, beta ptrs)
  //   2 x float                      (alpha, beta scalars)
  const int64_t buf_bytes =
      static_cast<int64_t>(batchCount) * 6 * dim_elem_size +
      static_cast<int64_t>(batchCount) * 5 * sizeof(int64_t) +
      2 * sizeof(float);
  buf = at::empty({buf_bytes}, mat1.options().dtype(at::kByte));

  const int64_t base_A = reinterpret_cast<int64_t>(mat1.data_ptr());
  const int64_t base_B = reinterpret_cast<int64_t>(mat2.data_ptr());
  const int64_t base_D = reinterpret_cast<int64_t>(c.data_ptr());

  const int32_t* offs_ptr = offs.has_value()
      ? static_cast<const int32_t*>(offs.value().data_ptr())
      : nullptr;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto launch = use_int64
      ? &launch_populate_cublas_grouped_args<int64_t>
      : &launch_populate_cublas_grouped_args<int32_t>;
  launch(
      *this, batchCount, offs_ptr,
      base_A, base_B, base_D,
      cublas_m, cublas_n, cublas_k,
      m_is_delta, n_is_delta, k_is_delta,
      lda_val, ldb_val, ldd_val,
      a_offs_stride, a_idx_stride,
      b_offs_stride, b_idx_stride,
      d_offs_stride, d_idx_stride,
      stream);
}

#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13030

} // namespace at::native
