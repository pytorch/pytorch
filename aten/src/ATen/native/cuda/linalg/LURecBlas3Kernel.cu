#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/util/complex.h>
#include <ATen/native/cuda/MiscUtils.h>

#include <thrust/swap.h>
#include <cooperative_groups.h>

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

constexpr auto LinOff(auto i, auto j, auto lda) {
  return i + static_cast<size_t>(j) * lda;
}
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
  LUWorkspace(const Tensor& input) {
    batch_count = cuda_int_cast(batchCount(input), "batchCount");
    int m = cuda_int_cast(input.size(-2), "input.size(-2)");

    // Pointer arrays for cuBLAS batched TRSM (64-bit addresses)
    buffer = at::empty({2, batch_count}, input.options().dtype(at::kLong));
    dL11_array = static_cast<scalar_t**>(buffer.select(0, 0).data_ptr());
    dA12_array = static_cast<scalar_t**>(buffer.select(0, 1).data_ptr());

    // Permutation vector workspace: m ints per batch
    pivinfo_buffer = at::empty({batch_count, m}, input.options().dtype(at::kInt));
    pivinfo = static_cast<int*>(pivinfo_buffer.data_ptr());
    pivinfo_stride = m;
  }

  int batch_count;
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
constexpr void AGGREGATE_ARGMAX(auto& val, auto& idx, const auto& other_val, const auto& other_idx) {
  if ((other_val > val) || (other_val == val && other_idx < idx)) {
    val = other_val;
    idx = other_idx;
  }
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
  int* sargmax = reinterpret_cast<int*>(sabsval + blockDim.x);
  int* sipiv = reinterpret_cast<int*>(sargmax + blockDim.x);

  // Each thread owns its full row stored in registers
  scalar_t rA[MAX_RECNB];
  #pragma unroll
  for (int i = 0; i < nb; ++i) {
    rA[i] = (tid < nrows)
      ? A[LinOff(col_start + tid, col_start + i, lda)]
      : static_cast<scalar_t>(0);
  }

  for (int i = 0, ir = i + tid, irows = blockDim.x; i < nb; ++i, ++ir, --irows) {
    // 1. Write abs value to shared memory using current logical row position
    sabsval[curr_row] = std::abs(rA[i]);
    sargmax[tid] = tid;
    __syncthreads();

    // 2. Parallel reduction for argmax over rows [i, blockDim.x)
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
    if (curr_row > i && abs_max != static_cast<real_t>(0)) {
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
  scalar_t* dA, int64_t matrix_stride, int lda, int m,
  int col_start, int nb,
  int* dipiv, int ipiv_stride,
  int* dinfo, int batch_count
) {
  int nrows = m - col_start;
  // Fused kernel needs one thread per row, max 1024.
  if (nrows > 1024 || nb > MAX_RECNB) return false;

  using real_t = c10::scalar_value_type<scalar_t>::type;

  dim3 grid(batch_count);
  auto stream = at::cuda::getCurrentCUDAStream();

  int padded_nrows = std::max(32, nrows);
  size_t shmem = nb * sizeof(scalar_t) + padded_nrows * sizeof(real_t) + padded_nrows * sizeof(int) + nb * sizeof(int);
  batched_panel_register_resident_fused_kernel<<<grid, padded_nrows, shmem, stream>>>(
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
  const LUTuning& tuning
) {
  int nrows = m - col_start;
  int recnb;
  if (nrows < 1024) {
    // Register-resident panel LU kernel
    if constexpr (std::is_same_v<float, scalar_t>) {
      recnb = tuning.recnb_reg.nb_float;
    } else if constexpr (std::is_same_v<double, scalar_t>) {
      recnb = tuning.recnb_reg.nb_double;
    } else if constexpr (std::is_same_v<c10::complex<float>, scalar_t>) {
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
    batch_count, ws, tuning
  );

  // 2. Apply left-half pivots to right half columns [col_start + n1, col_start + nb)
  using opaque_t = OpaqueType<sizeof(scalar_t)>;
  setup_pivinfo(m, col_start, n1, dipiv, ipiv_stride, ws.pivinfo, ws.pivinfo_stride, batch_count);
  batched_apply_pivots_parallel(
    reinterpret_cast<opaque_t*>(dA), matrix_stride, lda, m,
    col_start, n1,
    dipiv, ipiv_stride,
    ws.pivinfo, ws.pivinfo_stride,
    col_start + n1, col_start + nb, batch_count
  );

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
    batch_count, ws, tuning
  );

  // 5. Apply right-half pivots back to left half columns [col_start, col_start + n1)
  setup_pivinfo(m, col_start + n1, n2, dipiv, ipiv_stride, ws.pivinfo, ws.pivinfo_stride, batch_count);
  batched_apply_pivots_parallel(
    reinterpret_cast<opaque_t*>(dA), matrix_stride, lda, m,
    col_start + n1, n2,
    dipiv, ipiv_stride,
    ws.pivinfo, ws.pivinfo_stride,
    col_start, col_start + n1, batch_count
  );
}

} // anonymous namespace

void lu_batched_blas3_kernel(const Tensor& input, const Tensor& pivots, const Tensor& infos) {
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
    auto ws = LUWorkspace<scalar_t>(input);
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
        batch_count, ws, tuning
      );

      // 2. Propagate pivots to columns outside the panel (row-parallel)
      //    Left side: cols [0, j)
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

      // 3. Trailing matrix update
      trailing_matrix_update(
        handle, dA, matrix_stride, ws, lda,
        j, actual_nb, n - j - actual_nb, m - j - actual_nb, batch_count
      );
    }
  });
}


namespace ldl {

// Panel width. Wider panels cut the memory traffic of the trailing GEMMs,
// at the price of more work in the panel.
template <typename scalar_t>
int panel_width(int n) {
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major * 10 + prop->minor == 100) {
    // Tuned on GB200 for indefinite and positive definite inputs
    if constexpr (std::is_same_v<scalar_t, float>) {
      return n < 1024 ? 32 : (n <= 8192 ? 24 : (n <= 16384 ? 32 : 48));
    } else if constexpr (std::is_same_v<scalar_t, double>) {
      return n <= 768 ? 24 : (n <= 6144 ? 16 : (n < 16384 ? 32 : 96));
    } else if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
      return n <= 1536 ? 24 : (n <= 4096 ? 16 : (n <= 12288 ? 32 : 96));
    } else {
      return n <= 256 ? 32 : (n <= 6144 ? 16 : (n <= 12288 ? 48 : (n <= 20480 ? 64 : 96)));
    }
  }
  // Tuned on H100 for all dtypes
  return n <= 4096 ? 32 : (n <= 8192 ? 64 : (n <= 16384 ? 96 : 128));
}

// Width of the panel starting at step. cuBLAS runs the trailing GEMM at full speed
// only with 256-byte aligned operands, so panels end on such a boundary (as far as
// nb allows), while staying at least nb / 2 wide.
template <typename scalar_t>
int aligned_panel_width(int step, int nb, int n) {
  const int boundary = std::min<int>(256 / sizeof(scalar_t), nb);
  int panel_end = (step + nb) / boundary * boundary;
  if (panel_end - step < nb / 2) {
    panel_end += boundary;
  }
  return std::min(panel_end, n) - step;
}

// Panels with more rows than this are factored by a cooperative grid of
// blocks, one row per thread, rather than by a single block.
constexpr int COOP_MIN_ROWS = 2048;
constexpr int COOP_NTHREADS = 256;

// LDL factorization is square-root-free, hence,
// as in LAPACK, abs(a + ib) = abs(a) + abs(b).
template <typename scalar_t>
__device__ __forceinline__
auto abs(const scalar_t& v) {
  if constexpr (c10::is_complex<scalar_t>::value) {
    return std::abs(v.real()) + std::abs(v.imag());
  } else {
    return std::abs(v);
  }
}

template <typename scalar_t>
__device__ __forceinline__
auto real(const scalar_t& v) {
  if constexpr (c10::is_complex<scalar_t>::value) {
    return v.real();
  } else {
    return v;
  }
}

// A[j, i] given A[i, j] for a Hermitian (hermitian == true) or symmetric matrix
template <bool hermitian, typename scalar_t>
__device__ __forceinline__
scalar_t mirror(const scalar_t& v) {
  if constexpr (hermitian && c10::is_complex<scalar_t>::value) {
    return std::conj(v);
  } else {
    return v;
  }
}

} // namespace ::ldl


// The rows of the panel are spread cyclically over the threads of a single block
// or, if cooperative, over a grid of blocks that synchronize grid-wide.
template <typename scalar_t, int BS, bool hermitian, bool cooperative>
__global__ void __launch_bounds__(BS)
ldl_diagonal_panel_fused_kernel(
  scalar_t* __restrict__ dLD, int n, int lda,
  int nb, int curr_step, int* dcurr_step,
  int* dipiv, int* dinfo,
  // cooperative only: scratch for the grid-wide argmax, 2 * gridDim.x elements each
  typename c10::scalar_value_type<scalar_t>::type* __restrict__ dpart_max,
  int* __restrict__ dpart_idx
) {
  using real_t = c10::scalar_value_type<scalar_t>::type;
  const real_t ALPHA = (1 + std::sqrt(17)) / 8;
  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * BS + tid;
  const int nthreads = gridDim.x * BS;
  const auto panel_start = curr_step;
  const auto panel_end = panel_start + nb;

  scalar_t D[2][2];

  const auto sync = [] {
    if constexpr (cooperative) {
      cooperative_groups::this_grid().sync();
    } else {
      __syncthreads();
    }
  };

  // Argmax over all threads. The result gets a shared slot of its own, so that
  // the next call may start before every thread has read it. If cooperative,
  // the block results go through dpart_max/dpart_idx, and the grid-wide sync
  // in between also orders the global memory accesses made before the call.
  // Those have two slots per block, as a fast block may already write the next
  // partials while slower blocks still read these.
  constexpr int NWARPS = BS / 32;
  __shared__ real_t smax[NWARPS + 1];
  __shared__ int sidx[NWARPS + 1];
  int parity = 0;
  const auto reduce_argmax = [&](const std::tuple<real_t, int>& thread_max) {
    real_t my_max = std::get<0>(thread_max);
    int my_idx = std::get<1>(thread_max);
    const int warp = tid / 32;
    const int lane = tid % 32;
    warp_argmax(my_max, my_idx);
    if (lane == 0) {
      smax[warp] = my_max;
      sidx[warp] = my_idx;
    }
    __syncthreads();
    if (warp == 0) {
      my_max = lane < NWARPS ? smax[lane] : static_cast<real_t>(-1);
      my_idx = lane < NWARPS ? sidx[lane] : -1;
      warp_argmax(my_max, my_idx);
    }
    if constexpr (cooperative) {
      auto* part_max = dpart_max + parity * gridDim.x;
      auto* part_idx = dpart_idx + parity * gridDim.x;
      parity ^= 1;
      if (tid == 0) {
        part_max[blockIdx.x] = my_max;
        part_idx[blockIdx.x] = my_idx;
      }
      cooperative_groups::this_grid().sync();
      if (warp == 0) {
        my_max = static_cast<real_t>(-1);
        my_idx = -1;
        for (int b = lane; b < gridDim.x; b += 32) {
          AGGREGATE_ARGMAX(my_max, my_idx, part_max[b], part_idx[b]);
        }
        warp_argmax(my_max, my_idx);
      }
    }
    if (tid == 0) {
      smax[NWARPS] = my_max;
      sidx[NWARPS] = my_idx;
    }
    __syncthreads();
    return std::make_tuple(smax[NWARPS], sidx[NWARPS]);
  };

  // The panel is processed left to right. Rows/cols >= curr_step are stale,
  // i.e. they lack the updates from the panel's factored columns to the left,
  // and are made current only when needed. With
  //   Lprev = dLD[:, panel_start:curr_step],
  //   Uprev = dLD[panel_start:curr_step, :],
  // this subtracts (or adds back, if undo) Lprev @ Uprev from dLD[first:, k],
  // leaving out index skip. Subtracting one column at a time in place rounds
  // like a right-looking update. Lprev @ Uprev = Lprev @ D @ op(Lprev) is
  // symmetric/Hermitian, so dLD[k, first:] is the mirror of the column
  // and is copied rather than recomputed with strided reads.
  //
  // Interchanges do not touch Lprev/Uprev, so the stale data moved to a
  // position p is later updated with row/col p of Lprev/Uprev.
  // To keep that consistent, only current data is ever moved to p,
  // and adding the update for p back to it makes it stale again.
  //
  // Also returns the thread's share of the argmax of the off-diagonal
  // |dLD[first:, k]| it has written, for reduce_argmax.
  const auto update_rowcol = [&](const int k, const int first, const int skip, const bool undo = false) {
    auto my_max = static_cast<real_t>(-1);
    auto my_idx = -1;
    // No early exit when curr_step == panel_start: the row is still copied
    // from the column. Every row of U has to come from a column, otherwise
    // the rounding mismatch between the triangles left by the trailing GEMMs
    // leaks into L and D, and grows from panel to panel.
    for (int i = first + gtid; i < n; i += nthreads) {
      if (i == skip) continue;
      auto a = dLD[LinOff(i, k, lda)];
      for (int j = panel_start; j < curr_step; ++j) {
        if (undo) {
          a += dLD[LinOff(i, j, lda)] * dLD[LinOff(j, k, lda)];
        } else {
          a -= dLD[LinOff(i, j, lda)] * dLD[LinOff(j, k, lda)];
        }
      }
      if (i == k) {
        // A Hermitian diagonal is real, but complex rounding leaves an imaginary
        // residue. The mirrored rows assume a real pivot, so drop it (as zhetf2 does).
        if constexpr (hermitian) {
          a = ldl::real(a);
        }
      } else {
        dLD[LinOff(k, i, lda)] = ldl::mirror<hermitian>(a);
        AGGREGATE_ARGMAX(my_max, my_idx, ldl::abs(a), i);
      }
      dLD[LinOff(i, k, lda)] = a;
    }
    return std::make_tuple(my_max, my_idx);
  };

  // The processed block will factor nb or nb+1 rows/cols
  while (curr_step < panel_end) {
    int piv;
    int pivot_rank = 1;

    // Bring the current row/col up to date, finding its off-diagonal max
    // on the way. Argmax index is global! The sync inside reduce_argmax
    // is also the barrier after the update.
    const auto [lambda, ilambda] = reduce_argmax(update_rowcol(curr_step, /*first=*/curr_step, /*skip=*/-1));

    if (curr_step == panel_end - 1) {
      break;
    }

    // Bunch-Kaufman pivoting.
    // We follow p192 of
    // Golub, G. H., & Van Loan, C. F. (2013).
    // Matrix computations (4th ed.). Johns Hopkins University Press. {
    const auto diag_abs = ldl::abs(dLD[LinOff(curr_step, curr_step, lda)]);
    bool ilambda_updated = false;

    // ilambda is -1 when the scan found no candidate at all, which happens once
    // NaNs reach the column: every comparison against a NaN is false, so the
    // argmax never leaves its sentinel. Without this guard the sentinel is used
    // as a column index below and reads off the front of the buffer.
    if (ilambda < 0 || diag_abs >= ALPHA * lambda) {
      // No permutation, 1x1 pivot
      piv = curr_step;
    } else {
      // Bring the candidate row/col up to date. Its entries in row/col
      // curr_step are already current, and |dLD[curr_step, ilambda]| == lambda,
      // being the mirror of dLD[ilambda, curr_step].
      const auto [sigma_below, _] = reduce_argmax(update_rowcol(ilambda, /*first=*/curr_step + 1, /*skip=*/-1));
      const auto sigma = sigma_below > lambda ? sigma_below : lambda;
      ilambda_updated = true;
      // Checking whether ilambda diagonal pivot is "stable"
      if (sigma * diag_abs >= ALPHA * lambda * lambda) {
        // No permutation, 1x1 pivot
        piv = curr_step;
      } else if (ldl::abs(dLD[LinOff(ilambda, ilambda, lda)]) >= ALPHA * sigma) {
        // New 1x1 pivot
        piv = ilambda;
      } else {
        // New 2x2 pivot
        piv = ilambda;
        pivot_rank = 2;
        // The swap below moves row/col curr_step + 1 to ilambda, so it has
        // to be current too. The entries shared with ilambda already are.
        if (ilambda != curr_step + 1) {
          update_rowcol(curr_step + 1, /*first=*/curr_step + 1, /*skip=*/ilambda);
          sync();
        }
      }
    }
    // }

    // Update piv vector. info is set further below, once D is known: for a 2x2
    // block a zero diagonal is the normal case (it is why the block was chosen),
    // so singularity there is det(D) == 0, not a zero entry.
    if (gtid == 0) {
      if (pivot_rank == 1) {
        dipiv[curr_step] = piv + 1;
      } else {
        dipiv[curr_step + 0] = -(piv + 1);
        dipiv[curr_step + 1] = -(piv + 1);
      }
    }

    // Column/Row swaps {
    // 1x1 pivot -> swap with the current diagonal,
    // 2x2 pivot -> swap with the next to the current diagonal
    const int swp = curr_step + pivot_rank - 1;
    // The swaps and the undo below overwrite entries the pivot test above has read
    if (swp != piv || ilambda_updated) {
      sync();
    }
    if (swp != piv) {
      // Swap columns -- contiguous access
      for (int i = curr_step + gtid; i < n; i += nthreads) {
        thrust::swap(dLD[LinOff(i, swp, lda)], dLD[LinOff(i, piv, lda)]);
      }
      sync();
      // Swap rows -- noncontiguous access -- paying penatly here
      for (int i = curr_step + gtid; i < n; i += nthreads) {
        thrust::swap(dLD[LinOff(swp, i, lda)], dLD[LinOff(piv, i, lda)]);
      }
      sync();
    }
    // }

    // Unless it is part of the pivot, row/col ilambda now holds current data,
    // yet it gets updated later either here (in-panel) or by the GEMM.
    // Make it stale again.
    if (ilambda_updated && ilambda >= curr_step + pivot_rank) {
      update_rowcol(ilambda, /*first=*/curr_step + pivot_rank, /*skip=*/-1, /*undo=*/true);
      sync();
    }

    // Update L21 {
    // L21 = dLD[curr_step + pivot_rank:, curr_step:curr_step + pivot_rank]
    // L21 = L21 @ inv(D)
    // NOTE: keeping D11 and det as scalar_t as to not cause desyncs
    // between U12 and D
    bool D_is_singular = false;
    if (pivot_rank == 1) {
      auto D11 = dLD[LinOff(curr_step, curr_step, lda)];
      if (ldl::abs(D11) == static_cast<real_t>(0)) {
        D_is_singular = true;
      } else {
        for (int i = curr_step + pivot_rank + gtid; i < n; i += nthreads) {
          dLD[LinOff(i, curr_step, lda)] /= D11;
        }
      }
    } else {
      // NOTE: D stores inv(D) * det(D)
      D[1][1] = dLD[LinOff(curr_step, curr_step, lda)];
      D[0][1] = -dLD[LinOff(curr_step, curr_step + 1, lda)];
      D[1][0] = -dLD[LinOff(curr_step + 1, curr_step, lda)];
      D[0][0] = dLD[LinOff(curr_step + 1, curr_step + 1, lda)];

      // scale by det(D)
      auto det = D[0][0] * D[1][1] - D[0][1] * D[1][0];
      if (ldl::abs(det) == static_cast<real_t>(0)) {
        D_is_singular = true;
      } else {
        D[1][1] /= det;
        D[0][0] /= det;
        D[0][1] /= det;
        D[1][0] /= det;

        for (int i = curr_step + pivot_rank + gtid; i < n; i += nthreads) {
          auto l0 = dLD[LinOff(i, curr_step + 0, lda)];
          auto l1 = dLD[LinOff(i, curr_step + 1, lda)];
          dLD[LinOff(i, curr_step + 0, lda)] = l0 * D[0][0] + l1 * D[1][0];
          dLD[LinOff(i, curr_step + 1, lda)] = l0 * D[0][1] + l1 * D[1][1];
        }
      }
    }
    // Update info if singular and if detected for the first time
    if (D_is_singular && gtid == 0 && *dinfo == 0) {
      *dinfo = curr_step + 1;
    }
    // }

    // Finish iteration. No barrier needed: the only L21 entries the next update
    // reads right away are its own rows, which this thread has just scaled --
    // both loops start at curr_step + pivot_rank.
    curr_step += pivot_rank;
  }

  if (gtid == 0) {
    // Panel is processed -- update curr_step in the global memory
    *dcurr_step = curr_step;

    // Implies the whole computation is done, but
    // dipiv[-1] still needs to be updated
    if (curr_step == n - 1) {
      dipiv[n - 1] = n;
    }
  }
}

// Upper bound on the grid of the cooperative panel kernel, as all of its blocks
// have to be resident at once. 0 if the device cannot launch cooperative kernels.
template <typename scalar_t, bool hermitian>
int max_cooperative_panel_blocks() {
  const auto* props = at::cuda::getCurrentDeviceProperties();
  if (!props->cooperativeLaunch) {
    return 0;
  }
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm,
    ldl_diagonal_panel_fused_kernel<scalar_t, ldl::COOP_NTHREADS, hermitian, /*cooperative=*/true>,
    ldl::COOP_NTHREADS, 0
  ));
  return blocks_per_sm * props->multiProcessorCount;
}

template <typename scalar_t, bool hermitian>
void ldl_diagonal_panel(
  scalar_t* dLD, int n, int lda,
  int nb, int curr_step, int* dcurr_step,
  int* dipiv, int* dinfo,
  typename c10::scalar_value_type<scalar_t>::type* dpart_max, int* dpart_idx,
  int max_coop_blocks
) {
  constexpr int PANEL_THRESHOLD = 512;
  constexpr int LARGE_PANEL_NTHREADS = 1024;
  constexpr int SMALL_PANEL_NTHREADS = 256;
  // TODO: can be easily extended to the batched case
  const auto stream = at::cuda::getCurrentCUDAStream();

  auto problem_dim = n - curr_step;
  if (problem_dim > ldl::COOP_MIN_ROWS && max_coop_blocks > 1) {
    const auto nblocks = std::min((problem_dim + ldl::COOP_NTHREADS - 1) / ldl::COOP_NTHREADS, max_coop_blocks);
    void* args[] = {&dLD, &n, &lda, &nb, &curr_step, &dcurr_step, &dipiv, &dinfo, &dpart_max, &dpart_idx};
    const auto err = cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(ldl_diagonal_panel_fused_kernel<scalar_t, ldl::COOP_NTHREADS, hermitian, /*cooperative=*/true>),
      nblocks, ldl::COOP_NTHREADS, args, 0, stream
    );
    // The grid may not fit when the device is shared (e.g. under MPS). Both kernels
    // produce identical results, so fall back to the single-block one.
    if (err != cudaErrorCooperativeLaunchTooLarge) {
      C10_CUDA_CHECK(err);
      return;
    }
    (void)cudaGetLastError();
  }

  if (problem_dim > PANEL_THRESHOLD) {
    ldl_diagonal_panel_fused_kernel<scalar_t, LARGE_PANEL_NTHREADS, hermitian, /*cooperative=*/false><<<1, LARGE_PANEL_NTHREADS, 0, stream>>>(
      dLD, n, lda,
      nb, curr_step, dcurr_step,
      dipiv, dinfo, nullptr, nullptr
    );
  } else {
    ldl_diagonal_panel_fused_kernel<scalar_t, SMALL_PANEL_NTHREADS, hermitian, /*cooperative=*/false><<<1, SMALL_PANEL_NTHREADS, 0, stream>>>(
      dLD, n, lda,
      nb, curr_step, dcurr_step,
      dipiv, dinfo, nullptr, nullptr
    );
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, bool hermitian>
void ldl_factor_panels(const Tensor& LD, const Tensor& pivots, const Tensor& info, int n, int lda) {
  using real_t = c10::scalar_value_type<scalar_t>::type;
  auto* dLD = static_cast<scalar_t*>(LD.data_ptr());
  auto* dipiv = static_cast<int*>(pivots.data_ptr());
  auto* dinfo = static_cast<int*>(info.data_ptr());

  auto panel_step_holder = at::empty({1}, LD.options().dtype(at::kInt));
  auto* dstep = static_cast<int*>(panel_step_holder.data_ptr());

  // Scratch for the grid-wide argmax of the cooperative panel kernel
  const auto max_coop_blocks = max_cooperative_panel_blocks<scalar_t, hermitian>();
  const auto coop_blocks = std::min(max_coop_blocks, (n + ldl::COOP_NTHREADS - 1) / ldl::COOP_NTHREADS);
  auto part_max = at::empty({2 * coop_blocks}, LD.options().dtype(toRealValueType(LD.scalar_type())));
  auto part_idx = at::empty({2 * coop_blocks}, LD.options().dtype(at::kInt));
  auto* dpart_max = static_cast<real_t*>(part_max.data_ptr());
  auto* dpart_idx = static_cast<int*>(part_idx.data_ptr());

  const auto nb = ldl::panel_width<scalar_t>(n);
  int step = 0;

  // Right-Down-Diagonal-looking blocked LDLT/LDLH:
  // step through columns/rows in blocks of NB or NB-1 (pivots are 1x1 or 2x2)
  // and factor diagonal panels, then update the trailing matrix with a GEMM
  while (step < n - 1) {
    // 1. Panel factorization {
    const auto curr_nb = ldl::aligned_panel_width<scalar_t>(step, nb, n);
    ldl_diagonal_panel<scalar_t, hermitian>(
      dLD, n, lda,
      curr_nb, step, dstep,
      dipiv, dinfo,
      dpart_max, dpart_idx, max_coop_blocks
    );
    // }

    // 2. Trailing matrix update of B[step + curr_nb: step + curr_nb:] {
    // D2H to update the step on the host
    auto curr_step = panel_step_holder.item().toInt();
    if (step + curr_nb < n) {
      // TODO: LD22 is symmetric/Hermitian, so updating its lower triangle suffices.
      // cuBLAS syrkx/herkx with W = op(U12) as an m x k matrix (a transpose, e.g. via
      // cublas<t>geam) beats this gemm by 1.2-1.9x for n >= 16384 on H100 (1.6-1.9x
      // at n = 24576). That requires the panel kernel to stop reading the upper
      // triangle of LD22 (candidate column, swaps).
      at::cuda::blas::gemm(
        'n', 'n',
        n - step - curr_nb, n - step - curr_nb, curr_step - step,
        /*alpha=*/static_cast<scalar_t>(-1),
        /*L21=*/dLD + LinOff(step + curr_nb, step, lda), lda,
        /*U12=*/dLD + LinOff(step, step + curr_nb, lda), lda,
        /*beta=*/static_cast<scalar_t>(1),
        /*LD22=*/dLD + LinOff(step + curr_nb, step + curr_nb, lda), lda
      );
    }
    // }

    // Finish iteration
    step = curr_step;
  }
}

void ldl_factor_blas3_kernel(const Tensor& LD, const Tensor& pivots, const Tensor& info, bool hermitian) {
  // LD is lower triangular.
  // We materialize the upper triangular part for GEMM-friendly residual updates,
  // and that also spares us from swaps that restore the initial triangular structure.
  LD.add_(hermitian ? LD.tril(-1).mH() : LD.tril(-1).mT());
  int n = cuda_int_cast(LD.size(-1), "LD.size(-1)");
  int lda = std::max(cuda_int_cast(LD.stride(-1), "LD.stride(-1)"), std::max(1, n));
  info.zero_();

  // Disabling TF32 in GEMMs
  NoTF32Guard disable_tf32;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LD.scalar_type(), "ldl_factor_blas3_kernel", [&] {
    // Real types: symmetric and Hermitian coincide, so one instantiation suffices
    constexpr bool is_complex = c10::is_complex<scalar_t>::value;
    if (hermitian && is_complex) {
      ldl_factor_panels<scalar_t, /*hermitian=*/is_complex>(LD, pivots, info, n, lda);
    } else {
      ldl_factor_panels<scalar_t, /*hermitian=*/false>(LD, pivots, info, n, lda);
    }
  });
}

} // at::native
