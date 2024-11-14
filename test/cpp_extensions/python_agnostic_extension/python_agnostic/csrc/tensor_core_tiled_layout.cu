#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800  // at least Ampere

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "");
  const uint64_t blocks = a / b + (a % b != 0);
  return blocks;
}
constexpr int32_t kWarpSize = 32;

//Simple data structure to represent 4 pairs of bfloat16s, used for vectorized dequantization
//https://github.com/pytorch/pytorch/blob/b6689e0fb83a1578959ab0d9c6d2d9e11f7df21a/aten/src/ATen/native/cuda/int4mm.cu#L178-L180
struct __align__(16) bf16x2x4 {
  __nv_bfloat162 vals[4];
};

//Copied from https://github.com/pytorch/pytorch/blob/b6689e0fb83a1578959ab0d9c6d2d9e11f7df21a/aten/src/ATen/native/cuda/int4mm.cu#L195C1-L241C1
inline __device__ bf16x2x4 convert_i4x8_to_bf16x2x4(uint32_t source) {
  bf16x2x4 result;
  constexpr int kElements = 8;

  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const source_i4s = source;

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

  // We don't have enough mantissa to remove as much shift overhead as FP16, so
  // we must loop. No shift needed for first item.
  uint32_t i4s = source_i4s;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
#pragma unroll
  for (int ii = 1; ii < kElements / 2; ++ii) {
    i4s >>= 4; // or is it 8?
    // (i4s & 0x000f000f) | 0x43004300
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[ii])
        : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  }

  // This is the BF16 {-136, -136} represented as an integer.
  static constexpr uint32_t BF16_BIAS = 0xC308C308;
  static constexpr uint32_t BF16_ONE = 0x3F803F80;

// Finally, we construct the output numbers.
#pragma unroll
  for (int ii = 0; ii < kElements / 2; ++ii) {
    // Since this section is for Ampere+, we use bf16 fma to do the bias
    // subtraction
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(h[ii])
        : "r"(h[ii]), "r"(BF16_ONE), "r"(BF16_BIAS));
  }

  return result;
}
// in size [ceil(n / 8)][ceil(k / (InnerKTiles * 16))][32][InnerKTiles / 2]
// scales_and_zeros size [numQGroups][n][2]
// out size [n][k]
template <typename Out_t, int InnerKTiles, int groupSize, bool kDequant = true>
__global__ void _dequantize_int4_kernel(
    const at::PackedTensorAccessor32<int32_t, 4, at::RestrictPtrTraits> in,
    at::PackedTensorAccessor32<Out_t, 2, at::RestrictPtrTraits> out,
    std::optional<const at::PackedTensorAccessor32<c10::BFloat16, 3, at::RestrictPtrTraits>> scales_and_zeros = std::nullopt)
{

  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto kOuterTile = blockIdx.x;
  auto nTile = blockIdx.y;
  auto t = threadIdx.x;

  // n dimension that this lane loads from
  auto n0 = nTile * kNTileSize + (t / 4);

  // 8 k-tile values, 4 per m16n8k16 mma.sync operand B
  // int32_t ks[8];
  //Only need 4 offsets since TC layout for single tile is 2x2 (2 pairs of 2 contiguous values)
  int32_t ks[4];

  // Store address base offset
  auto pOut = &out[n0][0];

// Unpack 2 k-tiles at a time since min pack size is InnerKTiles = 2
#pragma unroll
  for (int innerKTile = 0; innerKTile < InnerKTiles; innerKTile += 2) {
    //Tensor-core layout for m16n8k16 is such that each tile has 2 pairs of 2 contiguous values
    //Hence, we only need 4 offsets
    // Offsets of innerTile0
    auto kBase0 = (kOuterTile * InnerKTiles + innerKTile) * kKTileSize;
    ks[0] = kBase0 + (t % 4) * 2;
    ks[1] = ks[0] + 8;

    // Offsets of innerTile1
    auto kBase1 = kBase0 + kKTileSize;
    ks[2] = kBase1 + (t % 4) * 2;
    ks[3] = ks[2] + 8;

    // inner k-tiles unpack two at a time
    int32_t pack = in[nTile][kOuterTile][t][innerKTile / 2];

    if constexpr(kDequant) {
      // static_assert(scales_and_zeros.has_value(), "scales_and_zeros must be set when dequantizing");
      static_assert(std::is_same<Out_t, c10::BFloat16>::value, "Out must be BFloat16 when dequantizing");
      // __nv_bfloat16 v[8];

      // // Extract u4, convert to s4 by subtracting by 2 ** nbits / 2, then convert to bfloat16
      bf16x2x4 v_bf16x2x4 = convert_i4x8_to_bf16x2x4(pack);

      // All b values within a 16x16 tile should fall within the same q group
      // Hence we load 1 scale and zero per loop
      int qgroup = ks[0] /  groupSize;
      const __nv_bfloat16 *pSZ = reinterpret_cast<const __nv_bfloat16*>(&scales_and_zeros.value()[qgroup][n0][0]);

      // Vectorize scales and zeros
      __nv_bfloat162 scale2 = __bfloat162bfloat162(pSZ[0]);
      __nv_bfloat162 zero2 = __bfloat162bfloat162(pSZ[1]);

  #pragma unroll
      for (int i = 0; i < 4; i++) {
        reinterpret_cast<__nv_bfloat162*>(&pOut[ks[i]])[0] = __hfma2(v_bf16x2x4.vals[i], scale2, zero2);
      }
    }
    else {
      static_assert(std::is_same<Out_t, int32_t>::value, "Out must be int32_t when unpacking to int");
      int32_t v[8];

      v[0] = pack & 0x0000000f;
      v[2] = (pack >> 4) & 0x0000000f;
      v[4] = (pack >> 8) & 0x0000000f;
      v[6] = (pack >> 12) & 0x0000000f;
      v[1] = (pack >> 16) & 0x0000000f;
      v[3] = (pack >> 20) & 0x0000000f;
      v[5] = (pack >> 24) & 0x0000000f;
      v[7] = (pack >> 28) & 0x0000000f;
      int2* v_i32x2 = reinterpret_cast<int2 *>(v);

    #pragma unroll
      for (int i = 0; i < 4; ++i) {
        reinterpret_cast<int2 *>(&pOut[ks[i]])[0] = v_i32x2[i];
      }
    }
  }
}

// output is [n][k] (int32 dtype)
// input is [n / 8][k / (InnerKTiles * 16)][32][innerKTiles / 2]
// scales_and_zeros is [numQGroups][n][2]
// qGroupSize is 32, 64, 128 or 256
at::Tensor _dequantize_tensor_core_tiled_layout(
    const at::Tensor& packed_w,
    const at::Tensor& scales_and_zeros,
    int64_t group_size,
    int64_t innerKTiles)
{

  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  c10::cuda::CUDAGuard g(packed_w.device());

  // packed_w preconditions
  TORCH_CHECK(packed_w.dim() == 4);
  TORCH_CHECK(packed_w.dtype() == at::kInt);
  TORCH_CHECK(packed_w.is_contiguous());
  TORCH_CHECK(packed_w.size(2) == 32);
  TORCH_CHECK(packed_w.size(3) == innerKTiles / 2);
  TORCH_CHECK(innerKTiles == 2 || innerKTiles == 4 || innerKTiles == 8);

  auto numQGroups = scales_and_zeros.size(0);
  int N = packed_w.size(0) * kNTileSize;
  int K = packed_w.size(1) * innerKTiles * kKTileSize;

  // scales_and_zeros preconditions
  TORCH_CHECK(
      group_size == 32 || group_size == 64 || group_size == 128 ||
      group_size == 256);
  TORCH_CHECK(numQGroups == K / group_size);
  TORCH_CHECK(scales_and_zeros.dim() == 3);
  TORCH_CHECK(scales_and_zeros.size(1) == N);
  TORCH_CHECK(scales_and_zeros.size(2) == 2);

  auto nTiles = divUp(N, kNTileSize);
  auto kSuperTiles = divUp(K, innerKTiles * kKTileSize);
  auto out = at::empty(
      {N, K},
      at::TensorOptions().dtype(at::kBFloat16).device(packed_w.device()));

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(kSuperTiles, nTiles);

#define RUN_DEQUANT(QGROUPSIZE) \
  do { \
    switch(innerKTiles) { \
      case 2: \
        _dequantize_int4_kernel<c10::BFloat16, 2, QGROUPSIZE, true><<<grid, kWarpSize, 0, stream>>>( \
        packed_w.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>(), \
        out.packed_accessor32<c10::BFloat16, 2, at::RestrictPtrTraits>(), \
        scales_and_zeros.packed_accessor32<c10::BFloat16, 3, at::RestrictPtrTraits>()); \
        break; \
      case 4: \
        _dequantize_int4_kernel<c10::BFloat16, 4, QGROUPSIZE, true><<<grid, kWarpSize, 0, stream>>>( \
        packed_w.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>(), \
        out.packed_accessor32<c10::BFloat16, 2, at::RestrictPtrTraits>(), \
        scales_and_zeros.packed_accessor32<c10::BFloat16, 3, at::RestrictPtrTraits>()); \
        break; \
      case 8: \
        _dequantize_int4_kernel<c10::BFloat16, 8, QGROUPSIZE, true><<<grid, kWarpSize, 0, stream>>>( \
        packed_w.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>(), \
        out.packed_accessor32<c10::BFloat16, 2, at::RestrictPtrTraits>(), \
        scales_and_zeros.packed_accessor32<c10::BFloat16, 3, at::RestrictPtrTraits>()); \
        break; \
      default: \
        break; \
    } \
  } while(false)

#define DISPATCH_Q_GROUP() \
  do { \
    switch (group_size) { \
      case 32: \
        RUN_DEQUANT(32); \
        break; \
      case 64: \
        RUN_DEQUANT(64); \
        break; \
      case 128: \
        RUN_DEQUANT(128); \
        break; \
      case 256: \
        RUN_DEQUANT(256); \
        break; \
      default: \
        break; \
    } \
  } while(false)

  DISPATCH_Q_GROUP();
  #undef DISPATCH_Q_GROUP
  #undef RUN_DEQUANT

  return out;
}

// output is [n][k] (int32 dtype)
// input is [n / 8][k / (InnerKTiles * 16)][32][innerKTiles / 2]
at::Tensor _unpack_tensor_core_tiled_layout(
    const at::Tensor& packed_w,
    int64_t innerKTiles)
{

  c10::cuda::CUDAGuard g(packed_w.device());

  TORCH_CHECK(packed_w.dim() == 4);
  TORCH_CHECK(packed_w.dtype() == at::kInt);
  TORCH_CHECK(packed_w.is_contiguous());

  TORCH_CHECK(packed_w.size(2) == 32);
  TORCH_CHECK(packed_w.size(3) == innerKTiles / 2);
  TORCH_CHECK(innerKTiles == 2 || innerKTiles == 4 || innerKTiles == 8);

  int N = packed_w.size(0) * 8;
  int K = packed_w.size(1) * innerKTiles * 16;

  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto nTiles = divUp(N, kNTileSize);

  auto kSuperTiles = divUp(K, innerKTiles * kKTileSize);

  auto out = at::empty(
      {N, K},
      at::TensorOptions().dtype(at::kInt).device(packed_w.device()));

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(kSuperTiles, nTiles);

  if (innerKTiles == 2) {
    _dequantize_int4_kernel<int32_t, 2, 0, false><<<grid, kWarpSize, 0, stream>>>(
        packed_w.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>());
  }
   else if (innerKTiles == 4) {
    _dequantize_int4_kernel<int32_t, 4, 0, false><<<grid, kWarpSize, 0, stream>>>(
        packed_w.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>());
  } else if (innerKTiles == 8) {
    _dequantize_int4_kernel<int32_t, 8, 0, false><<<grid, kWarpSize, 0, stream>>>(
        packed_w.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>());
  }

  return out;
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::unpack_tensor_core_tiled_layout", &_unpack_tensor_core_tiled_layout);
  m.impl("torchao::dequantize_tensor_core_tiled_layout", &_dequantize_tensor_core_tiled_layout);

}

#endif
