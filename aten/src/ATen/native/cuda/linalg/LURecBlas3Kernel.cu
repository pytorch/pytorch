#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/util/complex.h>
#include <ATen/native/cuda/MiscUtils.h>

#include <thrust/swap.h>

/*
  The following file contains implementation for a batched LU-factorization with partial pivoting.
  The approach is a recursive panel factorization with trailing matrix updates delegated to GEMMs/TRSMs.
  NOTE: meant as a temporary kernel before/when cuCUSOLVER/cuBLAS catches up (meant for very small matrices).

  Performance plots: https://github.com/nikitaved/custom_lu_batched_kernel_bench/tree/main/benchmarks/plots.
  Tested against MAGMA 2.10.0.

  Based off:

  @inproceedings{abdelfattah2019progressive,
    title={Progressive optimization of batched LU factorization on GPUs},
    author={Abdelfattah, Ahmad and Tomov, Stanimire and Dongarra, Jack},
    booktitle={2019 IEEE High Performance Extreme Computing Conference (HPEC)},
    pages={1--6},
    year={2019},
    organization={IEEE}
  }

*/


namespace at::native {

namespace {

#define LinOff(i, j, lda) i + static_cast<size_t>(j) * lda

// Small tile width for high occupancy (matches MAGMA's SWP_WIDTH=4)
constexpr int SWP_WIDTH = 4;

// Max possible panel width for the register-resident panel LU factozization.
constexpr int MAX_RECNB = 32;

// Nb values for the base case in the recursive call,
// when dispatching to the register-resident panel LU kernel
struct LURecnbRegisterResidentConfig {
  int nb_float;
  int nb_double;
  int nb_cfloat;
  int nb_cdouble;
};

struct LUNbConfig {
  int nb_small; // outer loop blocking factor when n < nb_crossover_n
  int nb_large; // outer loop blocking factor when n >= nb_crossover_n
};

// Global LU tuning
struct LUTuning {
  LURecnbRegisterResidentConfig recnb_reg; // recursive panel base-case width (rows <= 1024)
  int panel_threshold; // rows above this use block size (BS) 1024 tall-panel kernel
  int recnb_colserial; // recursive panel base-case width (flat column-by-column below this)
  int nb_crossover_n; // matrix size threshold: n >= this selects nb_large
  LUNbConfig nb_real; // blocking factors for float/double
  LUNbConfig nb_complex; // blocking factors for cfloat/cdouble
};

// Pre-tuned constants per compute capability
constexpr LUTuning tuning_sm80  = {{44, 44, 24, 16}, 768, 10, 512, {56, 256}, {64, 256}};  // A100
constexpr LUTuning tuning_sm89  = {{40, 32, 20, 24}, 768, 12, 256, {104, 384}, {104, 256}};  // L40S
constexpr LUTuning tuning_sm90  = {{52, 36, 52, 24}, 512, 10, 512, {40, 256}, {64, 256}};  // H100
constexpr LUTuning tuning_sm100 = {{48, 32, 32, 28}, 512, 10, 512, {72, 256}, {64, 256}};  // GB200

inline LUTuning get_tuning() {
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  const auto compcap = prop->major * 10 + prop->minor;
  switch (compcap) {
    case 80: return tuning_sm80;
    case 89: return tuning_sm89;
    case 90: return tuning_sm90;
    case 100: return tuning_sm100;
    default:
      // Fallback to sm_80
      return tuning_sm80;
  };
}

// Workspace -- pointer arrays needed by cuBLAS batched TRSM + pivinfo for parallel swaps.
// pivinfo: absolute permutation vector (one per batch, size m).
template <typename scalar_t>
struct LUWorkspace {
  LUWorkspace(const Tensor& input, int nb, bool compute_pivots) {
    int batch_count = cuda_int_cast(batchCount(input), "batchCount");
    int m = cuda_int_cast(input.size(-2), "input.size(-2)");
    int n = cuda_int_cast(input.size(-1), "input.size(-1)");

    // Pointer arrays for cuBLAS batched TRSM (64-bit addresses)
    buffer = at::empty({2, batch_count}, input.options().dtype(at::kLong));
    dL11_array = static_cast<scalar_t**>(buffer.select(0, 0).data_ptr());
    dA12_array = static_cast<scalar_t**>(buffer.select(0, 1).data_ptr());

    // Permutation vector workspace: m ints per batch
    pivinfo_buffer = compute_pivots ? at::empty({batch_count, m}, input.options().dtype(at::kInt)) : Tensor{};
    pivinfo = compute_pivots ? static_cast<int*>(pivinfo_buffer.data_ptr()) : nullptr;
    pivinfo_stride = compute_pivots ? m : 0;
  }

  Tensor buffer;

  // TRSM arrays
  scalar_t** dL11_array;
  scalar_t** dA12_array;

  // Permutation workspace
  Tensor pivinfo_buffer;
  int* pivinfo; // device pointer, batch_count * m ints
  int pivinfo_stride; // number of rows (stride between batches)
};

// Device-side pointer array computation for TRSM.
template <typename scalar_t>
__global__ void build_trsm_ptr_kernel(
  scalar_t* __restrict__ dA, int64_t matrix_stride, int lda, int batch_count,
  scalar_t** __restrict__ dL11_array,
  scalar_t** __restrict__ dA12_array,
  int diag_offset, int panel_width
) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_count) return;
  auto* base = dA + b * matrix_stride;
  dL11_array[b] = base + diag_offset + static_cast<size_t>(diag_offset) * lda;
  dA12_array[b] = base + diag_offset + static_cast<size_t>(diag_offset + panel_width) * lda;
}

// TRSM + GEMM trailing-matrix update.
// Solves L11 \ A12 (TRSM), then updates A22 -= L21 @ U12 (GEMM).
// All sub-blocks are relative to (diag_offset, diag_offset) on the diagonal:
//   L11: panel_width x panel_width, unit lower triangular
//   A12: panel_width x n_right (overwritten with U12)
//   L21: m_below x panel_width
//   A22: m_below x n_right
template <typename scalar_t>
void trailing_matrix_update(
  cublasHandle_t handle,
  scalar_t* dA,
  int64_t matrix_stride,
  LUWorkspace<scalar_t>& ws,
  int lda,
  int diag_offset,
  int panel_width,
  int n_right,
  int m_below,
  int batch_count
) {
  if (n_right <= 0) return;

  // Construct TRSM scalar_t** arrays {
  constexpr int threads = 64;
  int blocks = (batch_count + threads - 1) / threads;
  build_trsm_ptr_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    dA, matrix_stride, lda, batch_count,
    ws.dL11_array, ws.dA12_array,
    diag_offset, panel_width
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // }

  constexpr auto one = static_cast<scalar_t>(1);
  constexpr auto neg_one = static_cast<scalar_t>(-1);
  at::cuda::blas::trsmBatched(
    handle,
    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
    CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
    panel_width, n_right, &one,
    ws.dL11_array, lda,
    ws.dA12_array, lda,
    batch_count
  );

  if (m_below > 0) {
    size_t off_L21 = (diag_offset + panel_width) + static_cast<size_t>(diag_offset) * lda;
    size_t off_U12 = diag_offset + static_cast<size_t>(diag_offset + panel_width) * lda;
    size_t off_A22 = (diag_offset + panel_width) + static_cast<size_t>(diag_offset + panel_width) * lda;

    at::cuda::blas::bgemm(
      'n', 'n',
      m_below, n_right, panel_width,
      neg_one,
      dA + off_L21, lda, matrix_stride,
      dA + off_U12, lda, matrix_stride,
      one,
      dA + off_A22, lda, matrix_stride,
      batch_count
    );
  }
}

// Argmax Abs helpers {
#define AGGREGATE_ARGMAX(val, idx, other_val, other_idx) \
  if ((other_val > val) || (other_val == val && other_idx < idx)) { \
    val = other_val; \
    idx = other_idx; \
  }

template <typename real_t>
__device__ __forceinline__ void warp_argmax(real_t& val, int& idx) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    real_t other_val = __shfl_down_sync(0xffffffff, val, offset);
    int    other_idx = __shfl_down_sync(0xffffffff, idx, offset);
    AGGREGATE_ARGMAX(val, idx, other_val, other_idx);
  }
}

template <typename real_t, int BS>
__device__ __forceinline__ int block_argmax(
  real_t my_max, int my_idx,
  real_t* sdata, int* sidx, int tid
) {
  warp_argmax(my_max, my_idx);
  int warp_id = tid / 32;
  int lane = tid % 32;

  if (lane == 0) {
    sdata[warp_id] = my_max;
    sidx[warp_id] = my_idx;
  }
  __syncthreads();

  constexpr auto NWARPS = BS / 32;
  if (tid < 32) {
    auto v = (tid < NWARPS) ? sdata[tid] : static_cast<real_t>(-1);
    auto i = (tid < NWARPS) ? sidx[tid] : -1;
    warp_argmax(v, i);
    if (tid == 0) {
      sidx[0] = i;
    }
  }
  __syncthreads();

  return sidx[0];
}
// }

// Convert LAPACK-style sequential swap ipiv into an absolute permutation vector.
// After this kernel, pivinfo[i] (0-based) gives the source row for destination
// row (row_offset + i). Only rows [row_offset, row_offset + nrows) participate.
//
// Algorithm (same as MAGMA's setup_pivinfo_devfunc):
//   1. All threads initialize pivinfo as identity: pivinfo[i] = row_offset + i
//   2. Thread 0 replays the nb swaps sequentially on the identity.
//
// Launch: one block per batch, blockDim.x >= nrows (or loop if nrows > BS).
template <int BS>
__global__ void __launch_bounds__(BS)
setup_pivinfo_kernel(
  int* __restrict__ pivinfo,    // output: [batch_count, pivinfo_stride]
  int pivinfo_stride,           // stride between batches in pivinfo
  const int* __restrict__ ipiv, // input: LAPACK pivot indices (1-based)
  int ipiv_stride,              // stride between batches in ipiv
  int row_offset,               // first row index (= col_start)
  int nrows,                    // number of rows in submatrix (= m - col_start)
  int nb                        // number of pivots to replay
) {
  int batch = blockIdx.x;
  int tid = threadIdx.x;

  int* piv = pivinfo + batch * pivinfo_stride;
  const int* ip = ipiv + batch * ipiv_stride;

  // Initialize identity (1-based absolute row indices, like MAGMA)
  for (int dst = tid + row_offset; dst < row_offset + nrows; dst += BS) {
    piv[dst] = dst + 1;
  }
  __syncthreads();

  // Thread 0 replays the sequential swaps
  if (tid == 0) {
    for (int src = row_offset; src < row_offset + nb; ++src) {
      auto dst = ip[src] - 1;
      if (src != dst) {
        thrust::swap(piv[src], piv[dst]);
      }
    }
  }
}

void setup_pivinfo(
  int m,
  int col_start,
  int nb,
  const int* dipiv,
  int ipiv_stride,
  int* dpivinfo,
  int pivinfo_stride,
  int batch_count
) {
  int nrows = m - col_start;
  constexpr int BS = 256;
  setup_pivinfo_kernel<BS><<<batch_count, BS, 0, at::cuda::getCurrentCUDAStream()>>>(
    dpivinfo, pivinfo_stride,
    dipiv, ipiv_stride,
    col_start, nrows, nb
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Row-parallel swap: similar to MAGMA's dlaswp_rowparallel_devfunc.
// nb threads, each handles one row. Gathers source row into shared memory (strided),
// patches dA, then copies from shared memory into dA (coalesced).
// Direct swaps inflict strided reads/writes.
// pivinfo is 1-based. blockDim.x = nb (= height). Tiles across columns via grid.x.
template <typename scalar_t>
__global__ void
laswp_rowparallel_kernel(
  scalar_t* __restrict__ dA, int64_t matrix_stride,
  int lda,
  const int* __restrict__ pivinfo, // [batch_count, pivinfo_stride], 1-based
  int pivinfo_stride,
  int row_offset,   // = col_start
  int nb,           // number of rows = height = blockDim.x
  int ncols,        // total columns
  int col_offset,   // first column (absolute)
  int swp_width     // columns per tile
) {
  extern __shared__ char smem_raw[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem_raw);

  int batch = blockIdx.z;
  int tid = threadIdx.x;

  auto* A = dA + batch * matrix_stride;
  const int* piv = pivinfo + batch * pivinfo_stride + row_offset;

  // This tile's column range
  int tile_col_start = blockIdx.x * swp_width;
  int tile_width = ::min(swp_width, ncols - tile_col_start);

  if (tid < nb) {
    // src/dst rows
    int src = piv[tid] - 1;
    int dst = piv[src - row_offset] - 1;

    // Pass 1: gather source into shared memory, patch dA.
    // Strided read/write.
    for (int i = 0; i < tile_width; ++i) {
      int col = col_offset + tile_col_start + i;
      sdata[tid + i * nb] = A[LinOff(src, col, lda)];
      A[LinOff(src, col, lda)] = A[LinOff(dst, col, lda)];
    }
  }
  __syncthreads();

  if (tid < nb) {
    // Pass 2: write shared memory back -- coalesced write
    auto row = row_offset + tid;
    for (int i = 0; i < tile_width; ++i) {
      auto col = col_offset + tile_col_start + i;
      A[LinOff(row, col, lda)] = sdata[tid + i * nb];
    }
  }
}

// Parallel swap can be done over an opaque type,
// so double and cfloat share the same dispatch.
template <int N> struct alignas(N) OpaqueType { char data[N]; };

// Parallel pivot application using permutation vector.
// Gathers permuted rows into shared memory (strided access),
// then copies them back (coalesced write).
// Direct swaps inflict strided reads and writes.
template <typename scalar_t>
void batched_apply_pivots_parallel(
  scalar_t* dA,
  int64_t matrix_stride,
  int lda,
  int m,
  int col_start,
  int nb,
  const int* dipiv,
  int ipiv_stride,
  const int* dpivinfo,
  int pivinfo_stride,
  int col_lo,
  int col_hi,
  int batch_count
) {
  auto ncols = col_hi - col_lo;
  if (ncols <= 0 || nb <= 0) return;

  int swp_width = std::min(SWP_WIDTH, ncols);
  int col_tiles = (ncols + swp_width - 1) / swp_width;
  size_t shmem = nb * swp_width * sizeof(scalar_t);
  auto grid = dim3(col_tiles, 1, batch_count);

  laswp_rowparallel_kernel<<<grid, nb, shmem, at::cuda::getCurrentCUDAStream()>>>(
    dA, matrix_stride, lda,
    dpivinfo, pivinfo_stride,
    col_start, nb,
    ncols, col_lo, swp_width
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Register-resident fused panel factorization (similar to MAGMA's sgetf2_fused_device).
// Each thread owns one row of the panel in registers (rA[NB]).
// Pivot search via shared-memory parallel reduction, virtual row swap via rowid tracking,
// in-register scale and rank-1 update. One global read at start, one write at end.
// blockDim.x = nrows (number of rows in the submatrix), one block per batch.
// Constraint: nrows <= 1024 (max threads per block).
template <typename scalar_t>
__global__ void
batched_panel_register_resident_fused_kernel(
  scalar_t* __restrict__ dA, int64_t matrix_stride,
  int lda, int m,
  int col_start,
  int nb,
  int ipiv_stride,
  int* __restrict__ dipiv,
  int* __restrict__ dinfo
) {
  using real_t = c10::scalar_value_type<scalar_t>::type;

  const int tid = threadIdx.x;
  const int batch = blockIdx.x;
  const int nrows = m - col_start;
  auto* A = dA + batch * matrix_stride;
  int curr_row = tid; // tracks "virtual" row swaps
  int linfo = (col_start == 0) ? 0 : dinfo[batch];

  // Shared memory layout:
  // spivrow[NB] - pivot row values
  // sabsval[nrows] - abs values (for argmax reduction to find pivots)
  // sargmax[nrows] - argmax indices of abs values (for argmax reduction to find pivots)
  // sipiv[NB]   - pivot indices
  extern __shared__ char smem_raw[];
  scalar_t* spivrow = reinterpret_cast<scalar_t*>(smem_raw);
  real_t* sabsval = reinterpret_cast<real_t*>(spivrow + nb);
  int* sargmax = reinterpret_cast<int*>(sabsval + nrows);
  int* sipiv = reinterpret_cast<int*>(sargmax + nrows);

  // Each thread owns its full row stored in registers
  scalar_t rA[MAX_RECNB];
  #pragma unroll
  for (int i = 0; i < nb; ++i) {
    rA[i] = (tid < nrows)
      ? A[LinOff(col_start + tid, col_start + i, lda)]
      : static_cast<scalar_t>(0);
  }

  if (tid < nb) { sipiv[tid] = 0; };

  for (int i = 0, ir = i + tid, irows = nrows; i < nb; ++i, ++ir, --irows) {
    // 1. Write abs value to shared memory using current logical row position
    sabsval[curr_row] = std::abs(rA[i]);
    sargmax[tid] = tid;
    __syncthreads();

    // 2. Parallel reduction for argmax over rows [i, nrows)
    if (irows > 512) { if (tid < 512 && tid + 512 < irows) { AGGREGATE_ARGMAX(sabsval[ir], sargmax[ir], sabsval[ir + 512], sargmax[ir + 512]); } __syncthreads(); }
    if (irows > 256) { if (tid < 256 && tid + 256 < irows) { AGGREGATE_ARGMAX(sabsval[ir], sargmax[ir], sabsval[ir + 256], sargmax[ir + 256]); } __syncthreads(); }
    if (irows > 128) { if (tid < 128 && tid + 128 < irows) { AGGREGATE_ARGMAX(sabsval[ir], sargmax[ir], sabsval[ir + 128], sargmax[ir + 128]); } __syncthreads(); }
    if (irows >  64) { if (tid <  64 && tid +  64 < irows) { AGGREGATE_ARGMAX(sabsval[ir], sargmax[ir], sabsval[ir +  64], sargmax[ir +  64]); } __syncthreads(); }
    if (tid < 32) {
      auto val = (tid < irows) ? sabsval[ir] : static_cast<real_t>(-1);
      auto idx = (tid < irows) ? sargmax[ir] : tid;
      if (tid + 32 < irows) {
        auto other_val = sabsval[ir + 32];
        auto other_idx = sargmax[ir + 32];
        AGGREGATE_ARGMAX(val, idx, other_val, other_idx);
      }
      warp_argmax(val, idx);
      if (tid == 0) { sabsval[i] = val; sargmax[i] = idx; }
    }
    __syncthreads();

    auto abs_max = sabsval[i];
    auto argmax = sargmax[i];
    linfo = (abs_max == static_cast<real_t>(0) && linfo == 0) ? (col_start + i + 1) : linfo;

    if (tid == 0) {
      sipiv[i] = argmax;
    }
    __syncthreads();

    // 3. Pivot row broadcasts its values to shared memory
    if (curr_row == argmax) {
      #pragma unroll
      for (int j = 0; j < nb; ++j) { spivrow[j] = rA[j]; }
    }
    __syncthreads();

    // 4. Virtual row swap
    if (abs_max != static_cast<real_t>(0)) {
      if (curr_row == argmax) {
        curr_row = i;
      } else if (curr_row == i) {
        curr_row = argmax;
      }
    }

    // 5. Scale and rank-1 update (in registers)
    if (curr_row > i) {
      rA[i] /= spivrow[i];
      #pragma unroll
      for (int j = i + 1; j < nb; ++j) {
        rA[j] -= rA[i] * spivrow[j];
      }
    }
  }

  // Write info
  if (tid == 0) { dinfo[batch] = linfo; }

  // Write pivots (1-based, absolute)
  if (tid < nb) {
    dipiv[batch * ipiv_stride + col_start + tid] = sipiv[tid] + col_start + 1;
  }

  // Write back results using curr_row
  if (tid < nrows) {
    #pragma unroll
    for (int i = 0; i < nb; ++i) {
      A[LinOff(col_start + curr_row, col_start + i, lda)] = rA[i];
    }
  }
}

// Dispatch helper for register-resident fused panel kernel (NB 1-MAX_RECNB)
template <typename scalar_t>
bool try_launch_fused_panel_register_resident(
  bool compute_pivots,
  scalar_t* dA, int64_t matrix_stride, int lda, int m,
  int col_start, int nb,
  int* dipiv, int ipiv_stride,
  int* dinfo, int batch_count
) {
  int nrows = m - col_start;
  // Fused kernel needs one thread per row, max 1024.
  if (nrows > 1024 || nb > MAX_RECNB) return false;

  using real_t = c10::scalar_value_type<scalar_t>::type;
  size_t shmem = nb * sizeof(scalar_t) + nrows * sizeof(real_t) + nrows * sizeof(int) + nb * sizeof(int);

  dim3 grid(batch_count);
  dim3 threads(nrows);

  auto stream = at::cuda::getCurrentCUDAStream();

  batched_panel_register_resident_fused_kernel<scalar_t><<<grid, threads, shmem, stream>>>(
    dA, matrix_stride, lda, m, col_start, nb, ipiv_stride, dipiv, dinfo
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

// Batched panel LU factorization for nrows > 1024.
template <typename scalar_t, int BS>
__global__ void __launch_bounds__(BS)
batched_panel_colserial_fused_kernel(
  scalar_t* __restrict__ dA, int64_t matrix_stride,
  int lda, int m,
  int col_start, int nb,
  int ipiv_stride,
  int* __restrict__ dipiv,
  int* __restrict__ dinfo
) {
  using real_t = c10::scalar_value_type<scalar_t>::type;

  constexpr int NWARPS = BS / 32;
  __shared__ real_t sdata[NWARPS];
  __shared__ int sidx[NWARPS];
  __shared__ scalar_t sdiag;

  int batch = blockIdx.z;
  auto* A = dA + batch * matrix_stride;
  int tid = threadIdx.x;
  int panel_end = col_start + nb;

  for (int k = col_start; k < panel_end; ++k) {
    int rows_below = m - k - 1;
    int update_cols = panel_end - k - 1;

    // 1. Pivot find (warp-shuffle reduction)
    auto my_max = static_cast<real_t>(-1);
    auto my_idx = -1;
    for (int i = k + tid; i < m; i += BS) {
      auto v = std::abs(A[LinOff(i, k, lda)]);
      if (v > my_max) {
        my_max = v;
        my_idx = i;
      }
    }
    int pivot_row = block_argmax<real_t, BS>(my_max, my_idx, sdata, sidx, tid);
    if (tid == 0) {
      dipiv[batch * ipiv_stride + k] = pivot_row + 1; // 1-based!
    }

    // 2. Row swaps
    if (pivot_row != k) {
      for (int j = tid + col_start; j < nb + col_start; j += BS) {
        auto src = LinOff(k, j, lda);
        auto dst = LinOff(pivot_row, j, lda);
        thrust::swap(A[src], A[dst]);
      }
    }
    __syncthreads();

    // 3. Scale (divide by diagonal - skip if zero for singular matrices)
    if (tid == 0) {
      sdiag = A[LinOff(k, k, lda)];
      if (std::abs(sdiag) == 0 && dinfo[batch] == 0) {
        dinfo[batch] = k + 1; // 1-based!
      }
    }
    __syncthreads();

    if (std::abs(sdiag) != 0) {
      for (int i = k + 1 + tid; i < m; i += BS) {
        A[LinOff(i, k, lda)] /= sdiag;
      }
    }
    __syncthreads();

    // 4. Rank-1 update (linearized)
    if (rows_below > 0 && update_cols > 0) {
      auto numel = rows_below * update_cols;
      for (int idx = tid; idx < numel; idx += BS) {
        auto local_row = idx % rows_below;
        auto local_col = idx / rows_below;
        auto i = k + 1 + local_row;
        auto j = k + 1 + local_col;
        A[LinOff(i, j, lda)] -= A[LinOff(i, k, lda)] * A[LinOff(k, j, lda)];
      }
    }
  } // for cols in the panel
}

template <typename scalar_t>
void lu_batched_panel_recursive(
  cublasHandle_t handle,
  scalar_t* dA,
  int64_t matrix_stride,
  int lda,
  int m,
  int col_start,
  int nb,
  int* dipiv,
  int ipiv_stride,
  int* dinfo,
  int batch_count,
  LUWorkspace<scalar_t>& ws,
  const LUTuning& tuning,
  bool compute_pivots
) {
  int nrows = m - col_start;
  int recnb;
  if (nrows < 1024) {
    // Register-resident panel LU kernel
    if constexpr (std::is_same_v<float, scalar_t>) {
      recnb = tuning.recnb_reg.nb_float;
    } else if constexpr (std::is_same_v<double, scalar_t>) {
      recnb = tuning.recnb_reg.nb_double;
    } else if constexpr (std::is_same_v<std::complex<float>, scalar_t>) {
      recnb = tuning.recnb_reg.nb_cfloat;
    } else {
      recnb = tuning.recnb_reg.nb_cdouble;
    }
    // Cap for less register pressure
    recnb = std::min(recnb, MAX_RECNB);
  } else {
    // Colserial panel LU kernel
    recnb = tuning.recnb_colserial;
  }
  // Base case: use fused register-resident panel if possible, else fall back
  if (nb <= recnb) {
    if (try_launch_fused_panel_register_resident(
          compute_pivots,
          dA, matrix_stride, lda, m,
          col_start, nb, dipiv, ipiv_stride, dinfo, batch_count)) {
      return;
    }
    // Fallback: nrows > 1024 or nb is larger than what the register-resident kernel requires
    auto grid = dim3(1, 1, batch_count);
    if ((m - col_start) > tuning.panel_threshold) {
      batched_panel_colserial_fused_kernel<scalar_t, 1024><<<grid, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
        dA, matrix_stride, lda, m,
        col_start, nb,
        ipiv_stride, dipiv, dinfo
      );
    } else {
      batched_panel_colserial_fused_kernel<scalar_t, 256><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        dA, matrix_stride, lda, m,
        col_start, nb,
        ipiv_stride, dipiv, dinfo
      );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  auto n1 = nb / 2;
  auto n2 = nb - n1;

  // 1. Factor left half: columns [col_start, col_start + n1)
  lu_batched_panel_recursive(
    handle,
    dA, matrix_stride, lda, m,
    col_start, n1,
    dipiv, ipiv_stride, dinfo,
    batch_count, ws, tuning,
    compute_pivots
  );

  // 2. Apply left-half pivots to right half columns [col_start + n1, col_start + nb)
  if (compute_pivots) {
    using opaque_t = OpaqueType<sizeof(scalar_t)>;
    setup_pivinfo(m, col_start, n1, dipiv, ipiv_stride, ws.pivinfo, ws.pivinfo_stride, batch_count);
    batched_apply_pivots_parallel(
      reinterpret_cast<opaque_t*>(dA), matrix_stride, lda, m,
      col_start, n1,
      dipiv, ipiv_stride,
      ws.pivinfo, ws.pivinfo_stride,
      col_start + n1, col_start + nb, batch_count
    );
  }

  // 3. TRSM + GEMM: trailing update
  trailing_matrix_update(
    handle, dA, matrix_stride, ws, lda,
    col_start, n1, n2, m - col_start - n1, batch_count
  );

  // 4. Factor right half: columns [col_start + n1, col_start + nb)
  lu_batched_panel_recursive(
    handle,
    dA, matrix_stride, lda, m,
    col_start + n1, n2,
    dipiv, ipiv_stride, dinfo,
    batch_count, ws, tuning,
    compute_pivots
  );

  // 5. Apply right-half pivots back to left half columns [col_start, col_start + n1)
  if (compute_pivots) {
    using opaque_t = OpaqueType<sizeof(scalar_t)>;
    setup_pivinfo(m, col_start + n1, n2, dipiv, ipiv_stride, ws.pivinfo, ws.pivinfo_stride, batch_count);
    batched_apply_pivots_parallel(
      reinterpret_cast<opaque_t*>(dA), matrix_stride, lda, m,
      col_start + n1, n2,
      dipiv, ipiv_stride,
      ws.pivinfo, ws.pivinfo_stride,
      col_start, col_start + n1, batch_count
    );
  }
}

} // anonymous namespace

void lu_batched_blas3_kernel(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  const auto tuning = get_tuning();
  int batch_count = cuda_int_cast(batchCount(input), "batchCount");
  int m = cuda_int_cast(input.size(-2), "input.size(-2)");
  int n = cuda_int_cast(input.size(-1), "input.size(-1)");
  int64_t matrix_stride = matrixStride(input);
  int lda = std::max(cuda_int_cast(input.stride(-1), "input.stride(-1)"), std::max(1, m));

  NoTF32Guard disable_tf32;
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  infos.zero_();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "linalg_lu_batched_blas3_kernel", [&] {
    auto* dA = static_cast<scalar_t*>(input.data_ptr());
    auto* dipiv = static_cast<int*>(pivots.data_ptr());
    auto* dinfo = static_cast<int*>(infos.data_ptr());

    LUNbConfig nbc;
    if constexpr (c10::is_complex<scalar_t>::value) {
      nbc = tuning.nb_complex;
    } else {
      nbc = tuning.nb_real;
    }

    int nb = (n >= tuning.nb_crossover_n) ? nbc.nb_large : nbc.nb_small;
    auto ws = LUWorkspace<scalar_t>(input, nb, compute_pivots);
    auto min_mn = std::min(m, n);
    auto ipiv_stride = min_mn;

    // Right-looking blocked LU: step through columns in blocks of nb.
    // Each iteration factors one panel of width actual_nb, then updates the
    // trailing matrix to the right.
    // The panel itself is factored recursively (splitting its width in half
    // down to recnb, same algorithm as MAGMA's dgetrf_recpanel_batched).
    for (int j = 0; j < min_mn; j += nb) {
      auto actual_nb = std::min(nb, min_mn - j);

      // 1. Panel factorization
      lu_batched_panel_recursive(
        handle,
        dA, matrix_stride, lda, m,
        j, actual_nb,
        dipiv, ipiv_stride, dinfo,
        batch_count, ws, tuning,
        compute_pivots
      );

      // 2. Propagate pivots to columns outside the panel (row-parallel)
      //    Left side: cols [0, j)
      if (compute_pivots) {
        using opaque_t = OpaqueType<sizeof(scalar_t)>;
        setup_pivinfo(m, j, actual_nb, dipiv, ipiv_stride, ws.pivinfo, ws.pivinfo_stride, batch_count);
        batched_apply_pivots_parallel(
          reinterpret_cast<opaque_t*>(dA), matrix_stride, lda, m,
          j, actual_nb,
          dipiv, ipiv_stride,
          ws.pivinfo, ws.pivinfo_stride,
          0, j, batch_count
        );
        //    Right side: cols [j + actual_nb, n)
        batched_apply_pivots_parallel(
          reinterpret_cast<opaque_t*>(dA), matrix_stride, lda, m,
          j, actual_nb,
          dipiv, ipiv_stride,
          ws.pivinfo, ws.pivinfo_stride,
          j + actual_nb, n, batch_count
        );
      }

      // 3. Trailing matrix update
      trailing_matrix_update(
        handle, dA, matrix_stride, ws, lda,
        j, actual_nb, n - j - actual_nb, m - j - actual_nb, batch_count
      );
    }
  });
}

} // at::native
