#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
#include <cuda_fp8.h>
#include <limits>
#endif

namespace at::native {

namespace {

// Initial pool size for CUDA events per device.
constexpr size_t kInitialEventPoolSize = 8;

at::cuda::CUDAEventPool::Event getEventFromPool(const at::DeviceIndex device_idx) {
  // Pre-populate the pool with events to avoid stalls in creating events
  static auto* event_pool = new at::cuda::CUDAEventPool(kInitialEventPoolSize);
  return event_pool->get(device_idx);
}

} // namespace

void neg_kernel_cuda(TensorIteratorBase &iter);
void conj_kernel_cuda(TensorIteratorBase &iter);

template <typename SrcT, typename DstT>
void converting_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(SrcT value) {
        return static_cast<DstT>(value);
    });
}

template <typename SrcT>
struct ConvertToFloat8E4M3fnOp {
  __device__ __forceinline__ Float8_e4m3fn operator()(SrcT value) const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    __nv_fp8_storage_t x;
    if constexpr (std::is_same_v<SrcT, float>) {
      x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    } else if constexpr (std::is_same_v<SrcT, Half>) {
      x = __nv_cvt_halfraw_to_fp8(static_cast<__half>(value), __NV_SATFINITE, __NV_E4M3);
    } else if constexpr (std::is_same_v<SrcT, BFloat16>) {
      x = __nv_cvt_bfloat16raw_to_fp8(static_cast<__nv_bfloat16>(value), __NV_SATFINITE, __NV_E4M3);
    } else {
      x = __nv_cvt_float_to_fp8(static_cast<float>(value), __NV_SATFINITE, __NV_E4M3);
    }
    return Float8_e4m3fn(x, Float8_e4m3fn::from_bits());
#else
    return Float8_e4m3fn(value);
#endif
  }
};

// e5m2 intrinsics are correct but slower; only used for float on Blackwell
// to work around the ptxas subnormal codegen bug.
struct ConvertFloatToFloat8E5M2Op {
  __device__ __forceinline__ Float8_e5m2 operator()(float value) const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13020 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    auto x = __nv_cvt_float_to_fp8(value, __NV_NOSAT, __NV_E5M2);
    return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
    return Float8_e5m2(value);
#endif
  }
};

void float8_copy_kernel_cuda(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  ScalarType other_dtype = iter.dtype(1);
  if (dtype == kFloat8_e4m3fn) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, ConvertToFloat8E4M3fnOp<float>{});
         break;
      case kHalf:
         gpu_kernel_nocast(iter, ConvertToFloat8E4M3fnOp<Half>{});
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, ConvertToFloat8E4M3fnOp<BFloat16>{});
         break;
      default:
        gpu_kernel(iter, [] GPU_LAMBDA(Float8_e4m3fn x) { return x; });
        break;
    }
  } else if (dtype == kFloat8_e5m2) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, ConvertFloatToFloat8E5M2Op{});
         break;
      case kHalf:
         converting_copy_kernel_cuda<Half, Float8_e5m2>(iter);
         break;
      case kBFloat16:
         converting_copy_kernel_cuda<BFloat16, Float8_e5m2>(iter);
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e5m2 x) { return x; });
         break;
    }
  } else if (dtype == kFloat8_e4m3fnuz) {
    switch (other_dtype) {
      case kFloat:
         converting_copy_kernel_cuda<float, Float8_e4m3fnuz>(iter);
         break;
      case kHalf:
         converting_copy_kernel_cuda<Half, Float8_e4m3fnuz>(iter);
         break;
      case kBFloat16:
         converting_copy_kernel_cuda<BFloat16, Float8_e4m3fnuz>(iter);
         break;
      default:
        gpu_kernel(iter, [] GPU_LAMBDA(Float8_e4m3fnuz x) { return x; });
        break;
    }
  } else if (dtype == kFloat8_e5m2fnuz) {
    switch (other_dtype) {
      case kFloat:
         converting_copy_kernel_cuda<float, Float8_e5m2fnuz>(iter);
         break;
      case kHalf:
         converting_copy_kernel_cuda<Half, Float8_e5m2fnuz>(iter);
         break;
      case kBFloat16:
         converting_copy_kernel_cuda<BFloat16, Float8_e5m2fnuz>(iter);
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e5m2fnuz x) { return x; });
         break;
    }
  } else if (dtype == kFloat8_e8m0fnu) {
    // TODO(#146647): clean this up, too much copy-pasta
    switch (other_dtype) {
      case kFloat:
         converting_copy_kernel_cuda<float, Float8_e8m0fnu>(iter);
         break;
      case kHalf:
         converting_copy_kernel_cuda<Half, Float8_e8m0fnu>(iter);
         break;
      case kBFloat16:
         converting_copy_kernel_cuda<BFloat16, Float8_e8m0fnu>(iter);
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e8m0fnu x) { return x; });
         break;
    }
  } else {
    TORCH_CHECK(false, "This supposed to be called only for Float8 types");
  }
}

// TODO: We probably can use the opaque type trick to avoid creating duplicate
// kernels for equivalent bit lengths
void direct_copy_kernel_cuda(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_", [&] {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (isFloat8Type(dtype)) {
     float8_copy_kernel_cuda(iter);
  } else if (iter.dtype(1) == kFloat && (dtype == kBFloat16 || dtype == kHalf)) {
     if (dtype == kBFloat16) {
       converting_copy_kernel_cuda<float, BFloat16>(iter);
     } else {
       converting_copy_kernel_cuda<float, Half>(iter);
     }
  }
  else if ((iter.dtype(1) == kBFloat16 || iter.dtype(1) == kHalf) && dtype == kFloat) {
    if (iter.dtype(1) == kBFloat16) {
      converting_copy_kernel_cuda<BFloat16, float>(iter);
    } else {
      converting_copy_kernel_cuda<Half, float>(iter);
    }
  }
  else if (iter.dtype(1) == kInt && dtype == kLong) {
    converting_copy_kernel_cuda<int32_t, int64_t>(iter);
  }
  else if (iter.dtype(1) == kBool && dtype == kLong) {
    converting_copy_kernel_cuda<bool, int64_t>(iter);
  }
  else if (iter.dtype(1) == kBool && dtype == kDouble) {
    converting_copy_kernel_cuda<bool, double>(iter);
  }
  else if (isBitsType(dtype)) {
    TORCH_CHECK(dtype == iter.dtype(1), "copy_() does not support casting "
      "bits types to different bits types. Source dtype is ", iter.dtype(1), "target dtype is ", dtype);
    AT_DISPATCH_BIT_TYPES(dtype, "copy_", [&] {
      gpu_kernel_nocast(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (dtype == ScalarType::Float4_e2m1fn_x2) {
    TORCH_CHECK(dtype == iter.dtype(1), "copy_() does not support casting "
      "Float4_e2m1fn_x2 to different types. Source dtype is ", iter.dtype(1), "target dtype is ", dtype);
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(Float4_e2m1fn_x2 x) { return x; });
  } else {
    AT_DISPATCH_V2(
        dtype, "copy_", AT_WRAP([&] {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kHalf, kBool, kBFloat16, kComplexHalf, kBComplex32, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

void neg_conj_kernel_cuda(TensorIteratorBase &iter) {
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_cuda", [&] {
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return -std::conj(x); });
  });
}

using namespace at::cuda;

namespace {

constexpr int kTransposeTile = 32;
constexpr int kTransposeRows = 8;
constexpr int kTransposeFp32Tile = 64;
// Vectorized tiles move 16 bytes per thread per access: four 32-bit words,
// each holding one fp32 or several packed 1- or 2-byte elements.
constexpr int kTransposeWordSize = sizeof(uint32_t);
constexpr int kTransposeAccessSize = 4;
constexpr int kTransposeVecBytes = kTransposeWordSize * kTransposeAccessSize;
// Batched slices with fewer useful elements per block than this leave most
// of a tile idle and lose to the generic kernel (measured on H100 for 1-, 2-,
// 4- and 8-byte types). Narrower types need more elements to amortize a
// vectorized tile; their 32x32 scalar tile only wins when nearly full. A
// slice row or column under 16 elements never wins.
constexpr int kTransposeMinSliceDim = 16;
inline int64_t transpose_min_elements_per_block(int64_t element_size, bool vectorized) {
  if (element_size <= 2) return vectorized ? (element_size == 1 ? 1024 : 768) : 896;
  return 512;
}

// Shared-memory banks are 4 bytes wide, so what must be coprime with 32 is
// the tile row stride measured in 32-bit words, not in elements. Padding by
// one element only achieves that for 4-byte types; 1- and 2-byte types need
// a wider pad. For 8-byte types a warp's access splits into two 16-lane
// phases that already cover all 32 banks.
template <typename T>
struct TransposeTilePad {
  static constexpr int value = sizeof(T) == 1 ? 4    // 36 B = 9 words
                             : sizeof(T) == 2 ? 2    // 68 B = 17 words
                                              : 1;   // 4 B: 33 words
};

// Slice index -> {dst, src} offsets over the dims that are not transposed.
// Like OffsetCalculator, but with 32-bit divisors (num_slices < 2^31, see the
// launch check) and 64-bit strides, so the per-block divide stays cheap while
// offsets cannot overflow.
struct TransposeSliceOffsets {
  int dims;
  at::cuda::detail::IntDivider<uint32_t> sizes[MAX_DIMS];
  int64_t strides[MAX_DIMS][2];

  // A plain runtime loop: unrolling over MAX_DIMS with a runtime break (as
  // OffsetCalculator does) costs every block the full predicated chain, which
  // is measurable on small scalar tiles and pure waste for 2D (dims == 0).
  C10_HOST_DEVICE std::array<int64_t, 2> get(uint32_t idx) const {
    std::array<int64_t, 2> off{0, 0};
    for (int d = 0; d < dims; ++d) {
      const auto dm = sizes[d].divmod(idx);
      idx = dm.div;
      off[0] += dm.mod * strides[d][0];
      off[1] += dm.mod * strides[d][1];
    }
    return off;
  }
};

// A set of 2D transposes, one per slice s in [0, num_slices):
// dst_s[x][y] = src_s[y][x], with the slice bases given by `outer`.
// num_slices == 1 is the plain 2D transpose. All sizes and strides are in
// elements of the kernel's T (words for the packed 1- and 2-byte instantiations).
struct TransposeCopyArgs {
  int64_t width;        // elements per src row == rows of dst
  int64_t height;       // rows of src == elements per dst row
  int64_t src_pitch;    // elements between consecutive src rows
  int64_t dst_pitch;    // elements between consecutive dst rows
  uint32_t num_slices;  // product of the outer dims
  TransposeSliceOffsets outer;
};

// kBatched == false is the plain 2D transpose with blockIdx.y as the tile row;
// any per-block slice math there, even a never-taken loop, measurably slows
// small scalar-tile transposes (latency-bound, so it adds to every block).
template <typename T, bool kBatched, int kVectorSize = 1, int kAccessSize = 1, int kTileSize = kTransposeTile>
__global__ void transpose_copy_tiled_kernel(const T* __restrict__ src, T* __restrict__ dst, TransposeCopyArgs a) {
  __shared__ T tile[kVectorSize][kTileSize][kTileSize + TransposeTilePad<T>::value];
  using Vec = memory::aligned_vector<T, kAccessSize>;
  constexpr int rows = kTransposeRows * kAccessSize;
  const int64_t width = a.width;
  const int64_t height = a.height;
  const int64_t src_pitch = a.src_pitch;
  const int64_t dst_pitch = a.dst_pitch;

  int64_t by = blockIdx.y;
  const T* __restrict__ src_b = src;
  T* __restrict__ dst_b = dst;
  if constexpr (kBatched) {
    // (slice, tile row) pairs are flattened over gridDim.y x gridDim.z, since
    // each is capped at 65535; blocks past the end only occur in the last z
    // slab. The launch check guarantees tiles_y * num_slices < 2^31.
    const uint32_t tiles_y = at::ceil_div<int64_t>(height, kTileSize);
    const uint32_t t = blockIdx.z * gridDim.y + blockIdx.y;
    if (t >= tiles_y * a.num_slices) return;
    by = t % tiles_y;
    const auto slice = a.outer.get(t / tiles_y);
    src_b += slice[1];
    dst_b += slice[0];
  }
  #pragma unroll
  for (int col = 0; col < kTileSize; col += kTransposeTile) {
    const int64_t x = static_cast<int64_t>(blockIdx.x) * kTileSize + threadIdx.x * kAccessSize + col;
    const int64_t y = by * kTileSize + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < kTileSize; j += rows) {
      if (x < width && (y + j) < height) {
        Vec inputs[kVectorSize];
        #pragma unroll
        for (int i = 0; i < kVectorSize; ++i) {
          inputs[i] = *reinterpret_cast<const Vec*>(
              src_b + ((y + j) * kVectorSize + i) * src_pitch + x);
        }
        // Transpose each packed 2x2 or 4x4 block in registers. Separate
        // padded planes keep both shared-memory accesses bank-conflict free.
        #pragma unroll
        for (int k = 0; k < kAccessSize; ++k) {
          T values[kVectorSize];
          #pragma unroll
          for (int i = 0; i < kVectorSize; ++i) {
            values[i] = inputs[i].val[k];
          }
          // Lanes are numbered from the least significant 16 bits / byte.
          // 2x2 (two 16-bit lanes per word): [a0 a1], [b0 b1] -> [a0 b0], [a1 b1].
          // 4x4 (four bytes per word): first interleave word pairs into 16-bit
          // lane pairs, then interleave those, so row i of the output holds
          // byte i of each input word: [a_i b_i c_i d_i].
          if constexpr (kVectorSize == 2) {
            const auto a = values[0];
            const auto b = values[1];
            values[0] = __byte_perm(a, b, 0x5410);
            values[1] = __byte_perm(a, b, 0x7632);
          } else if constexpr (kVectorSize == 4) {
            const auto a = __byte_perm(values[0], values[1], 0x5140);
            const auto b = __byte_perm(values[0], values[1], 0x7362);
            const auto c = __byte_perm(values[2], values[3], 0x5140);
            const auto d = __byte_perm(values[2], values[3], 0x7362);
            values[0] = __byte_perm(a, c, 0x5410);
            values[1] = __byte_perm(a, c, 0x7632);
            values[2] = __byte_perm(b, d, 0x5410);
            values[3] = __byte_perm(b, d, 0x7632);
          }
          #pragma unroll
          for (int i = 0; i < kVectorSize; ++i) {
            tile[i][threadIdx.y + j][col + threadIdx.x * kAccessSize + k] = values[i];
          }
        }
      }
    }
  }
  __syncthreads();

  #pragma unroll
  for (int col = 0; col < kTileSize; col += kTransposeTile) {
    const int64_t x = by * kTileSize + threadIdx.x * kAccessSize + col;
    const int64_t y = static_cast<int64_t>(blockIdx.x) * kTileSize + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < kTileSize; j += rows) {
      if (x < height && (y + j) < width) {
        #pragma unroll
        for (int i = 0; i < kVectorSize; ++i) {
          Vec output;
          #pragma unroll
          for (int k = 0; k < kAccessSize; ++k) {
            output.val[k] = tile[i][col + threadIdx.x * kAccessSize + k][threadIdx.y + j];
          }
          *reinterpret_cast<Vec*>(dst_b + ((y + j) * kVectorSize + i) * dst_pitch + x) = output;
        }
      }
    }
  }
}

// Vectorized tiles run on 32-bit words (1- and 2-byte types packed); scalar
// tiles run on the tensor's own element type.
template <bool kBatched>
void launch_tiled_transpose(int64_t element_size, bool vectorized, dim3 grid, dim3 block,
                            cudaStream_t stream, const void* src, void* dst, const TransposeCopyArgs& args) {
  if (vectorized) {
    if (element_size == 1) {
      transpose_copy_tiled_kernel<uint32_t, kBatched, 4, kTransposeAccessSize><<<grid, block, 0, stream>>>(static_cast<const uint32_t*>(src), static_cast<uint32_t*>(dst), args);
    } else if (element_size == 2) {
      transpose_copy_tiled_kernel<uint32_t, kBatched, 2, kTransposeAccessSize><<<grid, block, 0, stream>>>(static_cast<const uint32_t*>(src), static_cast<uint32_t*>(dst), args);
    } else {
      transpose_copy_tiled_kernel<uint32_t, kBatched, 1, kTransposeAccessSize, kTransposeFp32Tile><<<grid, block, 0, stream>>>(static_cast<const uint32_t*>(src), static_cast<uint32_t*>(dst), args);
    }
  } else {
    switch (element_size) {
      case 1: transpose_copy_tiled_kernel<uint8_t, kBatched><<<grid, block, 0, stream>>>(static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst), args); break;
      case 2: transpose_copy_tiled_kernel<uint16_t, kBatched><<<grid, block, 0, stream>>>(static_cast<const uint16_t*>(src), static_cast<uint16_t*>(dst), args); break;
      case 4: transpose_copy_tiled_kernel<uint32_t, kBatched><<<grid, block, 0, stream>>>(static_cast<const uint32_t*>(src), static_cast<uint32_t*>(dst), args); break;
      case 8: transpose_copy_tiled_kernel<uint64_t, kBatched><<<grid, block, 0, stream>>>(static_cast<const uint64_t*>(src), static_cast<uint64_t*>(dst), args); break;
      default: TORCH_INTERNAL_ASSERT(false, "unsupported element size ", element_size);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Recognizes any permuted copy whose contiguous dim differs between src and
// dst: dim 0 is dst-contiguous (TensorIterator orders dims by increasing dst
// stride), some other dim is src-contiguous, and every remaining dim is an
// outer dim with arbitrary strides on both sides. This covers the dense 2D
// transpose, [B, C, L] <- [B, L, C] permutes, and higher-rank permutes that
// move the innermost dim. Permutes that keep the innermost dim have nothing
// to tile and, like any other arrangement, take the generic path.
bool maybe_tiled_transpose_copy(TensorIterator& iter) {
  const int ndim = iter.ndim();
  // Two transposed dims; everything else must fit the outer offset table.
  if (ndim < 2 || ndim - 2 > MAX_DIMS) return false;
  // The generic path normalizes bool bytes on load (NOTE [Loading boolean
  // values]); this kernel copies raw bytes, so leave bool to the generic path.
  if (iter.dtype(0) == kBool) return false;
  const int64_t element_size = iter.element_size(0);
  if (element_size > 8) return false;

  auto shape = iter.shape();
  auto os = iter.strides(0);   // bytes
  auto is = iter.strides(1);   // bytes
  if (os[0] != element_size) return false;
  int tdim = -1;
  for (int d = 1; d < ndim; ++d) {
    if (is[d] == element_size) { tdim = d; break; }
  }
  if (tdim < 0) return false;

  // Kernel convention: src is h rows of w contiguous elements; dst is w rows
  // of h contiguous elements. Rows must not overlap.
  const int64_t h = shape[0];
  const int64_t w = shape[tdim];
  if (os[tdim] < element_size * h || is[0] < element_size * w) return false;

  // Every other dim is an outer dim (byte strides, converted below).
  int64_t outer_sizes[MAX_DIMS], outer_dst[MAX_DIMS], outer_src[MAX_DIMS];
  int num_outer = 0;
  int64_t num_slices = 1;
  for (int d = 1; d < ndim; ++d) {
    if (d == tdim) continue;
    outer_sizes[num_outer] = shape[d];
    outer_dst[num_outer] = os[d];
    outer_src[num_outer] = is[d];
    num_slices *= shape[d];
    ++num_outer;
  }

  const void* sp = iter.tensor(1).const_data_ptr();
  void* dp = iter.tensor(0).mutable_data_ptr();
  // Vectorized tiles need every row, pitch, slice base and pointer 16-byte aligned.
  const auto aligned = [](int64_t bytes) { return bytes % kTransposeVecBytes == 0; };
  const bool vectorized = element_size <= kTransposeWordSize &&
      aligned(w * element_size) && aligned(h * element_size) && aligned(os[tdim]) && aligned(is[0]) &&
      std::all_of(outer_dst, outer_dst + num_outer, aligned) &&
      std::all_of(outer_src, outer_src + num_outer, aligned) &&
      aligned(reinterpret_cast<uintptr_t>(sp)) && aligned(reinterpret_cast<uintptr_t>(dp));
  // Narrow vectorized copies benefit from tiling at smaller sizes.
  const int64_t min_bytes = vectorized && element_size < kTransposeWordSize
      ? (int64_t(256) << 10) : (int64_t(4) << 20);
  if (num_slices * h * w * element_size < min_bytes) return false;

  // One word holds four bytes or two halfwords, giving 128x128 or 64x64 tiles.
  const int64_t unit = vectorized ? kTransposeWordSize : element_size;
  const int tile_size = vectorized && element_size == kTransposeWordSize
      ? kTransposeFp32Tile : kTransposeTile * (vectorized ? kTransposeWordSize / element_size : 1);
  const int64_t tiles_x = at::ceil_div<int64_t>(w, tile_size);
  const int64_t tiles_y = at::ceil_div<int64_t>(h, tile_size);
  // Measured on batched copies; 2D transposes keep their previous routing
  // (a skinny 2D transpose at 50% tile utilization still beats the generic
  // kernel, so the per-slice rule does not transfer).
  if (num_outer > 0 &&
      (std::min(h, w) < kTransposeMinSliceDim ||
       h * w < transpose_min_elements_per_block(element_size, vectorized) * tiles_x * tiles_y)) {
    return false;
  }

  // Plain 2D transposes map tile rows to gridDim.y directly. Batched copies
  // (and 2D ones with more tile rows than gridDim.y allows) flatten (slice,
  // tile row) pairs over gridDim.y x gridDim.z and index the kernel in 31
  // bits; tile columns take gridDim.x. Anything larger is left to the
  // generic kernel.
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const bool batched = num_outer > 0 || tiles_y > props->maxGridSize[1];
  const int64_t total_y = tiles_y * num_slices;
  const int64_t grid_y = std::min<int64_t>(total_y, props->maxGridSize[1]);
  const int64_t grid_z = batched ? at::ceil_div(total_y, grid_y) : 1;
  // IntDivider<uint32_t> requires divisors <= INT32_MAX, which bounds every outer size.
  if (total_y > std::numeric_limits<int32_t>::max() || tiles_x > props->maxGridSize[0] ||
      grid_z > props->maxGridSize[2]) {
    return false;
  }
  const int access_size = vectorized ? kTransposeAccessSize : 1;
  dim3 block(kTransposeTile / access_size, kTransposeRows * access_size);
  dim3 grid((unsigned)tiles_x, (unsigned)grid_y, (unsigned)grid_z);

  TransposeCopyArgs args{w * element_size / unit, h * element_size / unit,
                         /*src_pitch=*/is[0] / unit, /*dst_pitch=*/os[tdim] / unit,
                         static_cast<uint32_t>(num_slices), {}};
  args.outer.dims = num_outer;
  for (int i = 0; i < num_outer; ++i) {
    args.outer.sizes[i] = at::cuda::detail::IntDivider<uint32_t>(outer_sizes[i]);
    args.outer.strides[i][0] = outer_dst[i] / unit;
    args.outer.strides[i][1] = outer_src[i] / unit;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (batched) {
    launch_tiled_transpose<true>(element_size, vectorized, grid, block, stream, sp, dp, args);
  } else {
    launch_tiled_transpose<false>(element_size, vectorized, grid, block, stream, sp, dp, args);
  }
  return true;
}

// Below 4 MiB the generic kernel is as fast or faster for fp32 and small permutes (measured on GB300).
constexpr int64_t kStridedCopyMinBytes = int64_t(4) << 20;

// The iterator's byte offsets, except dim 0 counts kUnitBytes chunks instead of elements.
template <int kUnitBytes>
void launch_strided_unit_copy(TensorIteratorBase& iter) {
  using Word = uint16_t;
  constexpr int kWordsPerUnit = kUnitBytes / sizeof(Word);
  using Vec = memory::aligned_vector<Word, kWordsPerUnit>;
  auto offsets = make_offset_calculator<2>(iter);
  offsets.sizes_[0] = at::cuda::detail::IntDivider<uint32_t>(iter.shape()[0] * iter.element_size(0) / kUnitBytes);
  offsets.strides_[0][0] = offsets.strides_[0][1] = kUnitBytes;
  char* dst = static_cast<char*>(iter.data_ptr(0));
  const char* src = static_cast<const char*>(iter.data_ptr(1));
  launch_legacy_kernel<128, 4>(iter.numel() * iter.element_size(0) / kUnitBytes, [=] GPU_LAMBDA(int idx) {
    const auto offset = offsets.get(idx);
    *reinterpret_cast<Vec*>(dst + offset[0]) =
        memory::load_vector<kWordsPerUnit>(reinterpret_cast<const Word*>(src + offset[1]), 0);
  });
}

// Copies whose innermost dim is contiguous in both tensors but that are not contiguous overall
// (pitched rows, permutes that keep the last dim). Each row is moved as 16/8/4/2-byte chunks,
// so the offset is computed once per chunk instead of once per element.
bool maybe_strided_unit_copy(TensorIteratorBase& iter) {
  const int ndim = iter.ndim();
  // Raw bytes would skip bool normalization (NOTE [Loading boolean values]).
  // Keep P2P copies on the generic kernel until measured.
  if (ndim > MAX_DIMS || iter.dtype(0) == kBool || iter.device(0) != iter.device(1)) return false;
  // Conservatively exclude shared storage, including disjoint views.
  if (iter.tensor_base(0).is_alias_of(iter.tensor_base(1))) return false;
  const int64_t element_size = iter.element_size(0);
  // Reject empty and 0-dim iterators before reading shape[0].
  if (iter.numel() * element_size < kStridedCopyMinBytes || !iter.has_contiguous_first_dim()) return false;

  // Widest chunk (<= 16 B) that divides the row bytes, both base addresses and every outer
  // stride, so no chunk is misaligned or crosses a row.
  auto shape = iter.shape();
  uint64_t alignment = (shape[0] * element_size) |
      reinterpret_cast<uintptr_t>(iter.data_ptr(0)) | reinterpret_cast<uintptr_t>(iter.data_ptr(1));
  for (int d = 1; d < ndim; ++d) {
    if (shape[d] > 1) alignment |= iter.strides(0)[d] | iter.strides(1)[d];
  }
  int64_t unit_bytes = 16;
  while (alignment % unit_bytes != 0) unit_bytes /= 2;
  // A one-element chunk is just the generic kernel.
  if (unit_bytes <= element_size) return false;

  if (!iter.can_use_32bit_indexing()) {
    // Splitting can change alignment; recheck each piece.
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      if (!maybe_strided_unit_copy(sub_iter)) direct_copy_kernel_cuda(sub_iter);
    }
    return true;
  }

  switch (unit_bytes) {
    case 16: launch_strided_unit_copy<16>(iter); break;
    case 8: launch_strided_unit_copy<8>(iter); break;
    case 4: launch_strided_unit_copy<4>(iter); break;
    case 2: launch_strided_unit_copy<2>(iter); break;
    default: TORCH_INTERNAL_ASSERT(false, "unsupported unit size ", unit_bytes);
  }
  return true;
}

} // namespace

// device-to-device copy, does type conversion
void copy_device_to_device(TensorIterator& iter,
                           bool non_blocking,
                           bool p2p_enabled) {
  int64_t numel = iter.numel();

  // We can memcpy the memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool same_conj = iter.tensor(0).is_conj() == iter.tensor(1).is_conj();
  bool same_neg = iter.tensor(0).is_neg() == iter.tensor(1).is_neg();
  bool memcpy_eligible = same_type && same_conj && same_neg && iter.is_contiguous();

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  CUDAGuard device_guard(src_device);

  // We always perform the copy on the source device, using the current stream
  // on the source device, and we fully synchronize on both src and dst's
  // current streams for completion of the copy. We have to explicitly do this
  // for non-contig copies. This mimics the behavior of cross-device
  // cudaMemcpyAsync on the default stream.
  CUDAStream copy_stream = getCurrentCUDAStream(src_device.index());
  if (src_device != dst_device) {
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)

    // Use event pool for better performance instead of creating new events
    auto dst_ready = getEventFromPool(dst_device.index());
    device_guard.set_device(dst_device);
    dst_ready->record(getCurrentCUDAStream(dst_device.index()));

    device_guard.set_device(src_device);
    dst_ready->block(copy_stream);
  }

  if (memcpy_eligible) {
    void *dst = iter.data_ptr(0);
    void *src = iter.data_ptr(1);
    size_t size = numel * iter.element_size(0);
    if (src != dst || src_device != dst_device) {
      // Due to bizarre cuda driver intricacies, copies of
      // cudaMallocAsynced memory between devices that aren't
      // peer-to-peer-capable need "cudaMemcpyPeerAsync".
      // So we let the allocator implement the correct call
      // (either cudaMemcpyAsync or cudaMemcpyPeerAsync)
      AT_CUDA_CHECK(CUDACachingAllocator::memcpyAsync(
        dst, dst_device.index(),
        src, src_device.index(),
        size, copy_stream, p2p_enabled));
    }
  } else {
    if (same_type && same_neg && same_conj &&
        (maybe_tiled_transpose_copy(iter) || maybe_strided_unit_copy(iter))) {
      // handled by the tiled transpose or strided unit kernel
    } else if (same_neg) {
      if (!same_conj) {
        conj_kernel_cuda(iter);
      } else {
        direct_copy_kernel_cuda(iter);
      }
    } else {
      if (!same_conj) {
        neg_conj_kernel_cuda(iter);
      } else {
        neg_kernel_cuda(iter);
      }
    }
  }

  if (src_device != dst_device) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.

    // Still on src_device, record stream event
    auto src_ready = getEventFromPool(src_device.index());
    src_ready->record(copy_stream);

    device_guard.set_device(dst_device);
    src_ready->block(getCurrentCUDAStream(dst_device.index()));
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

static bool copy_requires_temporaries(TensorIterator& iter, bool p2p_enabled) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  if (dst_device == src_device) {
    // We never require temporaries for copies on the same GPU.
    TORCH_INTERNAL_ASSERT(dst_device.is_cuda() && src_device.is_cuda());
    return false;
  }

  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // Contiguous same-dtype copies can always use cudaMemcpyAsync
    return false;
  } else if (dst_device.is_cuda() && src_device.is_cuda()) {
    // Copies between GPUs can use the copy kernel if P2P is supported
    return !p2p_enabled;
  } else {
    // The remaining cases require temporaries. For example, this includes
    // non-contiguous copies between CPU and GPU.
    return true;
  }
}

static bool maybe_enable_p2p_access(Device dst_device, Device src_device) {
  if (dst_device.is_cpu() || src_device.is_cpu()) {
    return false;
  }
  return at::cuda::get_p2p_access(src_device.index(), dst_device.index());
}

static void copy_kernel_cuda(TensorIterator& iter, bool non_blocking) {
  TORCH_CHECK(iter.ntensors() == 2);

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  // Enable p2p access between devices. (No-op if it involves the CPU)
  bool p2p_enabled = maybe_enable_p2p_access(dst_device, src_device);

  if (copy_requires_temporaries(iter, p2p_enabled)) {
    // NB: this involves recursive calls to copy. Be careful that those copies
    // don't require temporaries or you will cause an infinite recursion!
    auto& dst = iter.tensor(0);
    Tensor dst_contig;
    Tensor src_contig;

    // If non_blocking is true - type conversions are performed on the GPU
    // For blocking transfers conversions are performed on CPU to avoid allocating
    // extra GPU memory
    // for GPU-GPU transfers conversions are performed on the source device
    auto conversion_device = non_blocking ? kCUDA : kCPU;
    if (iter.device_type(1) == conversion_device) {
      dst_contig = dst.is_contiguous() ? dst : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();
    } else {
      bool same_type = iter.dtype(0) == iter.dtype(1);
      dst_contig = (dst.is_contiguous() && same_type) ? dst : at::empty_like(dst, iter.dtype(1), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).expand_as(dst).contiguous();
    }

    // propagate the correct conjugate bit
    dst_contig._set_conj(dst.is_conj());
    src_contig._set_conj(iter.tensor(1).is_conj());

    dst_contig._set_neg(dst.is_neg());
    src_contig._set_neg(iter.tensor(1).is_neg());

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return;
  }

  // Copy on GPU (or between GPUs)
  if (dst_device.is_cuda() && src_device.is_cuda()) {
    copy_device_to_device(iter, non_blocking, p2p_enabled);
    return;
  }

  // Copy between CPU and GPU
  cuda::OptionalCUDAGuard device_guard;
  cudaMemcpyKind kind;
  const Tensor* host_tensor = nullptr;
  if (dst_device.is_cuda() && src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    kind = cudaMemcpyHostToDevice;
    host_tensor = &iter.tensor(1);
  } else if (dst_device.is_cpu() && src_device.is_cuda()) {
    device_guard.set_device(src_device);
    kind = cudaMemcpyDeviceToHost;
    host_tensor = &iter.tensor(0);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GPU copy_()");
  }

  // Check for unpinned CPU memory during CUDA graph capture
  if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
    TORCH_CHECK(
        host_tensor->is_pinned(),
        "Cannot copy between CPU and CUDA tensors during CUDA graph capture ",
        "unless the CPU tensor is pinned. Please use tensor.pin_memory() or ",
        "allocate the tensor with pin_memory=True.");
  }

  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  int64_t nbytes = iter.numel() * iter.element_size(0);
  CUDAStream stream = getCurrentCUDAStream();

  if (non_blocking) {
    AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));
    // we use both the storage context and the tensor data pointer as the key
    // for the caching host allocator. This allows us to better attribute the
    // events to the original tensor allocation correctly. The cases we seek to
    // handle are:

    // 1: a user can pass a pinned memory tensor with an alternative
    // context, for example if allocating memory directly from the pinned memory
    // allocator and constructing a tensor with torch::from_blob.

    // 2: a user can pass a tensor with a different base pointer to the original
    // allocation (via slicing).
    const auto& dst_tensor = iter.tensor(0);
    const auto& src_tensor = iter.tensor(1);
    const auto& host_tensor = (dst_device == kCPU ? dst_tensor : src_tensor);
    auto* ptr = (dst_device == kCPU ? dst : src);
    auto* ctx = host_tensor.storage().data_ptr().get_context();
    // TODO: warn on the return value.
    at::getHostAllocator(at::kCUDA)->record_event(ptr, ctx, stream.unwrap());
  } else {
    at::cuda::memcpy_and_sync(dst, src, nbytes, kind, stream);
  }

  if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
     iter.tensor(0).conj_physical_();
  }
  if (iter.tensor(0).is_neg() != iter.tensor(1).is_neg()) {
     iter.tensor(0).neg_();
  }
}

REGISTER_DISPATCH(copy_stub, &copy_kernel_cuda)

} // namespace at::native
