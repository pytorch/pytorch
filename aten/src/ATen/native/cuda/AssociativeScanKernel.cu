#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/AssociativeScanKernel.h>
#include <ATen/native/AssociativeScanUtils.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Load.h>
#include <c10/util/complex.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace at::native {

namespace {

constexpr int kBlockDimX = 256;
constexpr int kRowsPerBlock = 4;

// Host/device handles for the L parallel input/output arrays.
template <typename scalar_t, int L>
struct ScanPointers {
  const scalar_t* self[L];
  scalar_t* result[L];
};

// Warp shuffle up. CUDA requires an explicit 32-lane sync mask; HIP (ROCm)
// uses a 64-bit mask on gfx9 (or rejects the 32-bit one), so use the maskless
// intrinsic there.
template <typename T>
__device__ inline T shfl_up_val(T val, uint32_t delta) {
#if defined(USE_ROCM)
  return __shfl_up(val, delta);
#else
  return __shfl_up_sync(0xffffffffu, val, delta);
#endif
}

template <typename scalar_t, int L>
__device__ inline ScanVec<scalar_t, L> shfl_up(
    ScanVec<scalar_t, L> val,
    uint32_t delta) {
  ScanVec<scalar_t, L> res;
#pragma unroll
  for (int i = 0; i < L; ++i) {
    if constexpr (std::is_same_v<scalar_t, c10::Half>) {
      res.v[i] = c10::Half(
          static_cast<uint16_t>(shfl_up_val(val.v[i].x, delta)),
          c10::Half::from_bits());
    } else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
      res.v[i] = c10::BFloat16(
          static_cast<uint16_t>(shfl_up_val(val.v[i].x, delta)),
          c10::BFloat16::from_bits());
    } else if constexpr (c10::is_complex<scalar_t>::value) {
      using comp_t = typename scalar_t::value_type;
      res.v[i] = scalar_t(
          shfl_up_val(static_cast<comp_t>(val.v[i].real()), delta),
          shfl_up_val(static_cast<comp_t>(val.v[i].imag()), delta));
    } else {
      res.v[i] = shfl_up_val(val.v[i], delta);
    }
  }
  return res;
}

// Inclusive scan of a warp's segment using __shfl_up_sync (Hillis-Steele over
// lanes). Unselected lanes keep their own value which acts as the identity
// segment boundary.
template <typename scalar_t, int L, typename Combine>
__device__ inline ScanVec<scalar_t, L> warp_inclusive_scan(
    ScanVec<scalar_t, L> val,
    uint32_t lane) {
#pragma unroll
  for (uint32_t d = 1; d < C10_WARP_SIZE; d <<= 1) {
    ScanVec<scalar_t, L> t = shfl_up<scalar_t, L>(val, d);
    if (lane >= d) {
      val = Combine::combine(t, val);
    }
  }
  return val;
}

// Hierarchical block scan:
//   1. each warp independently scans its 32-element segment (warp shuffle),
//   2. warp totals are scanned by warp 0 (shared memory),
//   3. every warp adds the exclusive prefix of the preceding warps.
// The block iterates over chunks of the scan dimension while carrying a
// running block total (decoupled look-back), so arbitrarily long scans are
// supported with O(blockDim) shared memory.
template <typename scalar_t, int L, typename Combine>
__global__ void associative_scan_cuda_kernel(
    ScanPointers<scalar_t, L> ptrs,
    int64_t N,
    int64_t M) {
  alignas(sizeof(double)) extern __shared__ char smem[];
  // One entry per warp per row: the inclusive scan of the warp totals.
  ScanVec<scalar_t, L>* warp_totals =
      reinterpret_cast<ScanVec<scalar_t, L>*>(smem);
  constexpr int kNumWarps = kBlockDimX / C10_WARP_SIZE;
  const uint32_t lane = threadIdx.x % C10_WARP_SIZE;
  const uint32_t warp_id = threadIdx.x / C10_WARP_SIZE;

  for (int64_t row = blockIdx.x * kRowsPerBlock + threadIdx.y;
       row < M;
       row += gridDim.x * kRowsPerBlock) {
    ScanVec<scalar_t, L>* wt = warp_totals + threadIdx.y * kNumWarps;
    ScanVec<scalar_t, L> block_total = Combine::identity();

    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
      ScanVec<scalar_t, L> val;
      if (i < N) {
        for (int l = 0; l < L; ++l) {
          val.v[l] = c10::load(&ptrs.self[l][row * N + i]);
        }
      } else {
        val = Combine::identity();
      }

      val = warp_inclusive_scan<scalar_t, L, Combine>(val, lane);

      if (lane == C10_WARP_SIZE - 1) {
        wt[warp_id] = val;
      }
      __syncthreads();

      if (warp_id == 0) {
        ScanVec<scalar_t, L> w =
            (lane < kNumWarps) ? wt[lane] : Combine::identity();
        w = warp_inclusive_scan<scalar_t, L, Combine>(w, lane);
        if (lane < kNumWarps) {
          wt[lane] = w;
        }
      }
      __syncthreads();

      // Exclusive prefix of the preceding warps within this chunk, then the
      // running total of all preceding chunks.
      ScanVec<scalar_t, L> warp_prefix =
          (warp_id > 0) ? wt[warp_id - 1] : Combine::identity();
      val = Combine::combine(warp_prefix, val);
      val = Combine::combine(block_total, val);

      if (i < N) {
        for (int l = 0; l < L; ++l) {
          ptrs.result[l][row * N + i] = val.v[l];
        }
      }

      block_total = Combine::combine(block_total, wt[kNumWarps - 1]);
      // Ensure all warps finished reading `wt` before it is overwritten.
      __syncthreads();
    }
  }
}

template <typename scalar_t, int L, typename Combine>
void launch_associative_scan(
    const ScanPointers<scalar_t, L>& ptrs,
    const TensorBase& self) {
  const int64_t N = self.size(-1);
  const int64_t M = self.numel() / N;
  if (M == 0 || N == 0) {
    return;
  }
  dim3 threads(kBlockDimX, kRowsPerBlock);
  int64_t blocks = (M + kRowsPerBlock - 1) / kRowsPerBlock;
  blocks = std::min<int64_t>(
      blocks, at::cuda::getCurrentDeviceProperties()->maxGridSize[0]);
  const int num_warps = kBlockDimX / C10_WARP_SIZE;
  size_t smem_bytes = kRowsPerBlock * num_warps * sizeof(ScanVec<scalar_t, L>);
  associative_scan_cuda_kernel<scalar_t, L, Combine>
      <<<static_cast<uint32_t>(blocks), threads, smem_bytes,
         at::cuda::getCurrentCUDAStream()>>>(ptrs, N, M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_combine(
    const TensorBase& result,
    const TensorBase& self,
    const std::string& combine_mode) {
  ScanPointers<scalar_t, 1> ptrs;
  ptrs.self[0] = self.const_data_ptr<scalar_t>();
  ptrs.result[0] = result.mutable_data_ptr<scalar_t>();

  if (combine_mode == "add") {
    launch_associative_scan<scalar_t, 1, CombineAdd<scalar_t, 1>>(ptrs, self);
  } else if (combine_mode == "mul") {
    launch_associative_scan<scalar_t, 1, CombineMul<scalar_t, 1>>(ptrs, self);
  } else if (combine_mode == "max") {
    launch_associative_scan<scalar_t, 1, CombineMax<scalar_t>>(ptrs, self);
  } else if (combine_mode == "min") {
    launch_associative_scan<scalar_t, 1, CombineMin<scalar_t>>(ptrs, self);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported combine_mode: ", combine_mode);
  }
}

} // namespace

void associative_scan_cuda_kernel(
    const TensorBase& result,
    const TensorBase& self,
    const std::string& combine_mode) {
  if (self.numel() == 0) {
    return;
  }
  const cuda::OptionalCUDAGuard guard(self.device());
  if (combine_mode == "add" || combine_mode == "mul") {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kHalf,
        kBFloat16,
        self.scalar_type(),
        "associative_scan_cuda",
        [&] { dispatch_combine<scalar_t>(result, self, combine_mode); });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        kHalf,
        kBFloat16,
        self.scalar_type(),
        "associative_scan_cuda",
        [&] { dispatch_combine<scalar_t>(result, self, combine_mode); });
  }
}

void associative_scan_tensor_list_cuda_kernel(
    const std::vector<TensorBase>& result,
    const std::vector<TensorBase>& self,
    const std::string& combine_mode) {
  if (self.empty() || self[0].numel() == 0) {
    return;
  }
  const cuda::OptionalCUDAGuard guard(self[0].device());
  const int64_t N = self[0].size(-1);
  const int64_t M = self[0].numel() / N;
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      self[0].scalar_type(),
      "associative_scan_tensor_list_cuda",
      [&] {
        ScanPointers<scalar_t, 2> ptrs;
        ptrs.self[0] = self[0].const_data_ptr<scalar_t>();
        ptrs.self[1] = self[1].const_data_ptr<scalar_t>();
        ptrs.result[0] = result[0].mutable_data_ptr<scalar_t>();
        ptrs.result[1] = result[1].mutable_data_ptr<scalar_t>();
        launch_associative_scan<scalar_t, 2, CombineLinearRecurrence<scalar_t>>(
            ptrs, self[0]);
      });
}

REGISTER_CUDA_DISPATCH(associative_scan_stub, &associative_scan_cuda_kernel)
REGISTER_CUDA_DISPATCH(
    associative_scan_tensor_list_stub,
    &associative_scan_tensor_list_cuda_kernel)

} // namespace at::native
