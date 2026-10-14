// Standalone copy of ldl_diagonal_panel_fused_kernel (LURecBlas3Kernel.cu), so
// that the grid-wide argmax can be driven with grid sizes that the library
// never picks (e.g. the cap in max_cooperative_panel_blocks(), which binds only
// for n > ~169k). Loaded by test_ldl_custom.py via torch.utils.cpp_extension.
// The code between the BEGIN/END markers must stay identical to the original;
// test_standalone_kernel_matches_original enforces it.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/complex.h>
#include <thrust/swap.h>
#include <cooperative_groups.h>
#include <tuple>

namespace {

constexpr auto LinOff(auto i, auto j, auto lda) {
  return i + static_cast<size_t>(j) * lda;
}

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

namespace ldl {

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

template <bool hermitian, typename scalar_t>
__device__ __forceinline__
scalar_t mirror(const scalar_t& v) {
  if constexpr (hermitian && c10::is_complex<scalar_t>::value) {
    return std::conj(v);
  } else {
    return v;
  }
}

} // namespace ldl

// BEGIN COPY
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
      // The last column is a 1x1 pivot of its own: nothing is left to eliminate
      // but its singularity, which is checked here. Advancing curr_step ends the
      // factorization.
      if (curr_step == n - 1) {
        if (gtid == 0) {
          dipiv[n - 1] = n;
          if (ldl::abs(dLD[LinOff(n - 1, n - 1, lda)]) == static_cast<real_t>(0) && *dinfo == 0) {
            *dinfo = n;
          }
        }
        curr_step = n;
      }
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
      // Checking whether ilambda diagonal pivot is "stable".
      // As in LAPACK, lambda / sigma <= 1 avoids the overflowing lambda * lambda.
      if (diag_abs >= ALPHA * lambda * (lambda / sigma)) {
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

    // Update piv vector. info is set further below, for a 1x1 pivot only:
    // a 2x2 block is never singular (see there).
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
      // Swap rows -- noncontiguous access -- paying penalty here
      for (int i = curr_step + gtid; i < n; i += nthreads) {
        thrust::swap(dLD[LinOff(swp, i, lda)], dLD[LinOff(piv, i, lda)]);
      }
      sync();
    }
    // }

    // Unless it is part of the pivot, row/col ilambda now holds current data,
    // yet it gets updated later either here (in-panel) or by the GEMM.
    // Make it stale again.
    // NOTE: this costs accuracy. Adding the update back and subtracting it again
    // later each round by about eps * |update|, which is large next to a column that
    // cancels down to much less than its update. LAPACK's ?lasyf/?lahef never undo:
    // they update a copy of the candidate column in their workspace W. So, with the
    // same pivots as LAPACK, the solve residual averages about 1.2x LAPACK's on
    // random indefinite matrices, and a pivot that results from heavy cancellation,
    // e.g. in a graded matrix, can lose several digits (normwise backward stability
    // holds). Avoiding the undo takes a saved copy of the stale candidate and, after
    // an interchange, swapping the rows of Lprev along with the data (and back after
    // the trailing GEMM), as ?lasyf does.
    if (ilambda_updated && ilambda >= curr_step + pivot_rank) {
      update_rowcol(ilambda, /*first=*/curr_step + pivot_rank, /*skip=*/-1, /*undo=*/true);
      sync();
    }

    // Update L21 {
    // L21 = dLD[curr_step + pivot_rank:, curr_step:curr_step + pivot_rank]
    // L21 = L21 @ inv(D)
    // NOTE: D is kept in scalar_t, not made real as in zhetf2,
    // so as not to desync it from U12
    // NOTE: as in LAPACK's ?sytrf/?hetrf, subnormal pivots below 1 / max are not
    // supported: t / d01 and t / d10 below overflow, and so does a complex /= D11,
    // as c10::complex multiplies by the reciprocal of the denominator. Supporting
    // them means dividing by d01, d10 inside the loop as ?lahef_rook does, with a
    // complex division that divides by the denominator instead.
    // Nor is a complex d01 close to overflow, e.g. with both parts above max / 2,
    // where the denominator of c10::complex division overflows: r0, r1, s0, s1, L
    // come out 0 rather than ~1 / |d01|. This is normwise stable, but loses the
    // components of a solve on that block. ?lahef does the same, while ?hetf2
    // divides by the real |d01| instead.
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
      // D = [[d00, d01], [d10, d11]]. Its determinant overflows past sqrt(max) and
      // underflows below sqrt(min), so, as in LAPACK, it is never formed. Instead,
      // det(D) = d01 * d10 * (r0 * r1 - 1) with r0 = d00 / d01, r1 = d11 / d10,
      // and its scale is applied in the loop. The pivot test bounds |r0 * r1|
      // well below 1, so a 2x2 block is never singular.
      const auto d01 = dLD[LinOff(curr_step, curr_step + 1, lda)];
      const auto d10 = dLD[LinOff(curr_step + 1, curr_step, lda)];
      const auto r0 = dLD[LinOff(curr_step, curr_step, lda)] / d01;
      const auto r1 = dLD[LinOff(curr_step + 1, curr_step + 1, lda)] / d10;
      const auto t = static_cast<scalar_t>(1) / (r0 * r1 - static_cast<scalar_t>(1));
      const auto s0 = t / d01;
      const auto s1 = t / d10;
      for (int i = curr_step + pivot_rank + gtid; i < n; i += nthreads) {
        const auto l0 = dLD[LinOff(i, curr_step + 0, lda)];
        const auto l1 = dLD[LinOff(i, curr_step + 1, lda)];
        dLD[LinOff(i, curr_step + 0, lda)] = s0 * (r1 * l0 - l1);
        dLD[LinOff(i, curr_step + 1, lda)] = s1 * (r0 * l1 - l0);
      }
    }
    // Update info if singular and if detected for the first time.
    // Not guaranteed to fire for a singular input, even one with an exactly zero
    // row/col: interchanges move data but not Lprev/Uprev, so a moved column is
    // updated with the factors of its new position, and the exact zero survives
    // only if the undo above and the trailing GEMM cancel bit for bit, which
    // they do not. LAPACK keeps the zero exact, as the factors travel with the data.
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
  }
}
// END COPY

template <typename scalar_t, bool hermitian>
int64_t run_panel(
    const at::Tensor& LD, const at::Tensor& ipiv, const at::Tensor& info,
    int64_t nb, int64_t curr_step, bool cooperative, int64_t nblocks, int64_t nthreads) {
  using real_t = c10::scalar_value_type<scalar_t>::type;
  const c10::cuda::CUDAGuard guard(LD.device());
  auto* dLD = static_cast<scalar_t*>(LD.data_ptr());
  int n = LD.size(-1);
  int lda = LD.stride(-1);
  int inb = nb;
  int step = curr_step;
  auto step_holder = at::zeros({1}, LD.options().dtype(at::kInt));
  auto* dstep = static_cast<int*>(step_holder.data_ptr());
  auto* dipiv = static_cast<int*>(ipiv.data_ptr());
  auto* dinfo = static_cast<int*>(info.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();

  if (cooperative) {
    TORCH_CHECK(nthreads == 256, "the cooperative kernel runs with 256 threads");
    auto part_max = at::empty({2 * nblocks}, LD.options().dtype(c10::toRealValueType(LD.scalar_type())));
    auto part_idx = at::empty({2 * nblocks}, LD.options().dtype(at::kInt));
    auto* dpart_max = static_cast<real_t*>(part_max.data_ptr());
    auto* dpart_idx = static_cast<int*>(part_idx.data_ptr());
    void* args[] = {&dLD, &n, &lda, &inb, &step, &dstep, &dipiv, &dinfo, &dpart_max, &dpart_idx};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(ldl_diagonal_panel_fused_kernel<scalar_t, 256, hermitian, true>),
        nblocks, 256, args, 0, stream));
  } else if (nthreads == 256) {
    ldl_diagonal_panel_fused_kernel<scalar_t, 256, hermitian, false><<<1, 256, 0, stream>>>(
        dLD, n, lda, inb, step, dstep, dipiv, dinfo, nullptr, nullptr);
  } else {
    TORCH_CHECK(nthreads == 1024, "nthreads must be 256 or 1024");
    ldl_diagonal_panel_fused_kernel<scalar_t, 1024, hermitian, false><<<1, 1024, 0, stream>>>(
        dLD, n, lda, inb, step, dstep, dipiv, dinfo, nullptr, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return step_holder.item<int>();
}

// Factors the panel starting at curr_step of the column-major LD (both triangles
// populated). Returns the step the panel ended at. hermitian only matters for
// complex types, as it does in the library.
int64_t panel(
    const at::Tensor& LD, const at::Tensor& ipiv, const at::Tensor& info,
    int64_t nb, int64_t curr_step, bool cooperative, int64_t nblocks, int64_t nthreads,
    bool hermitian) {
  TORCH_CHECK(LD.is_cuda() && LD.stride(-2) == 1, "LD must be a column-major CUDA matrix");
#define RUN(scalar_t, herm) run_panel<scalar_t, herm>(LD, ipiv, info, nb, curr_step, cooperative, nblocks, nthreads)
  switch (LD.scalar_type()) {
    case at::kFloat: return RUN(float, false);
    case at::kDouble: return RUN(double, false);
    case at::kComplexFloat: return hermitian ? RUN(c10::complex<float>, true) : RUN(c10::complex<float>, false);
    case at::kComplexDouble: return hermitian ? RUN(c10::complex<double>, true) : RUN(c10::complex<double>, false);
    default: TORCH_CHECK(false, "unsupported dtype ", LD.scalar_type());
  }
#undef RUN
}

} // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("panel", &panel);
}
