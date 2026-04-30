// MPP (MetalPerformancePrimitives) matmul2d kernel for F.linear.
// Computes C[M,N] = A[M,K] @ B[N,K]^T with optional fused bias.
// NEEDS_PADDING=false: M and N must be tile-aligned; emits no per-tile
// bounds checks. NEEDS_PADDING=true: handles arbitrary M and N — interior
// tiles still take the fast static-extent path, edge tiles fall through to
// a dynamic-extent slice that lets matmul2d zero-extend OOB reads and clip
// OOB writes. The host picks the variant based on whether M and N are
// tile-aligned.
// Requires Metal 4.0 (macOS 26+).
#if __METAL_VERSION__ >= 400
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename T, bool HAS_BIAS, bool NEEDS_PADDING, int TILE_M, int TILE_N>
kernel void mpp_linear(
    device T* A [[buffer(0)]],
    device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    device T* bias [[buffer(3)]],
    constant uint3& sizes [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint M = sizes.x;
  const uint K = sizes.y;
  const uint N = sizes.z;
  const uint m_off = tgid.y * TILE_M;
  const uint n_off = tgid.x * TILE_N;

  // Convention: rank0 is the fast / contiguous dim (K for A and B, N for C),
  // rank1 is the slow / outer dim. A is [M,K] row-major, B is [N,K] row-major;
  // matmul2d transposes B at run time.
  using ext2d = dextents<int32_t, 2>;
  tensor<device T, ext2d, tensor_inline> mA_full(A, ext2d((int)K, (int)M));
  tensor<device T, ext2d, tensor_inline> mB_full(B, ext2d((int)K, (int)N));
  tensor<device T, ext2d, tensor_inline> mC_full(C, ext2d((int)N, (int)M));

  constexpr auto desc = matmul2d_descriptor(
      TILE_M, TILE_N, static_cast<int>(dynamic_extent), false, true);
  matmul2d<desc, execution_simdgroups<4>> op;

  // Static-extent slice: matmul2d treats the tile as fully in-bounds, no
  // checks. Plain slice: dynamic extents, matmul2d emits per-access bounds
  // checks. When NEEDS_PADDING is false the condition folds to true at compile
  // time.
  if (!NEEDS_PADDING || (m_off + TILE_M <= M && n_off + TILE_N <= N)) {
    auto mA = mA_full.template slice<dynamic_extent, TILE_M>(0, (int)m_off);
    auto mB = mB_full.template slice<dynamic_extent, TILE_N>(0, (int)n_off);
    auto mC = mC_full.template slice<TILE_N, TILE_M>((int)n_off, (int)m_off);
    op.run(mA, mB, mC);
  } else {
    auto mA = mA_full.slice(0, (int)m_off);
    auto mB = mB_full.slice(0, (int)n_off);
    auto mC = mC_full.slice((int)n_off, (int)m_off);
    op.run(mA, mB, mC);
  }

  if (HAS_BIAS) {
    constexpr uint TG_SIZE = 4 * 32;
    device T* bias_tile = bias + n_off;
    if (NEEDS_PADDING) {
      const uint m_lim = M - m_off < (uint)TILE_M ? M - m_off : (uint)TILE_M;
      const uint n_lim = N - n_off < (uint)TILE_N ? N - n_off : (uint)TILE_N;
      for (uint i = tid; i < TILE_M * TILE_N; i += TG_SIZE) {
        uint m = i / TILE_N;
        uint n = i % TILE_N;
        if (m < m_lim && n < n_lim) {
          C[(m_off + m) * N + (n_off + n)] += bias_tile[n];
        }
      }
    } else {
      device T* C_tile = C + m_off * N + n_off;
      for (uint i = tid; i < TILE_M * TILE_N; i += TG_SIZE) {
        uint m = i / TILE_N;
        uint n = i % TILE_N;
        C_tile[m * N + n] += bias_tile[n];
      }
    }
  }
}

#define INSTANTIATE_VARIANT(T, suffix, TM, TN, name, HAS_BIAS, PAD)      \
  template [[host_name("mpp_linear_" name "_" #TM "x" #TN "_" #suffix)]] \
  kernel void mpp_linear<T, HAS_BIAS, PAD, TM, TN>(                      \
      device T*,                                                         \
      device T*,                                                         \
      device T*,                                                         \
      device T*,                                                         \
      constant uint3&,                                                   \
      uint2,                                                             \
      uint);

#define INSTANTIATE_TILE(T, suffix, TM, TN)                           \
  INSTANTIATE_VARIANT(T, suffix, TM, TN, "aligned", false, false)     \
  INSTANTIATE_VARIANT(T, suffix, TM, TN, "aligned_bias", true, false) \
  INSTANTIATE_VARIANT(T, suffix, TM, TN, "dyn", false, true)          \
  INSTANTIATE_VARIANT(T, suffix, TM, TN, "dyn_bias", true, true)

#define INSTANTIATE(T, suffix)        \
  INSTANTIATE_TILE(T, suffix, 32, 32) \
  INSTANTIATE_TILE(T, suffix, 64, 64) \
  INSTANTIATE_TILE(T, suffix, 128, 64)

INSTANTIATE(float, float)
INSTANTIATE(half, half)
INSTANTIATE(bfloat, bfloat)
#undef INSTANTIATE
#undef INSTANTIATE_TILE
#undef INSTANTIATE_VARIANT
#endif // __METAL_VERSION__ >= 400
