"""AMX/AVX-512 interleaved GEMM helpers for the CPU FlexAttention template.

AMX tile registers only accept literal-immediate indices, so the two supported
row counts (16 and 32) are separate specializations with hardcoded tile indices
that follow ``AMXState``'s slot order (C tiles, then A, then B).
"""

# The C++ is a Jinja template only to substitute the kernel-name prefix so the
# symbols do not collide across multiple compiled kernels in one process.
FLEX_ATTENTION_AMX_HELPERS = r"""
// AMX bf16 accumulator block. C[NROWS,32] (+)= A[NROWS,K] @ Bp[K,32] (VNNI2).
//
// A hardware tile is at most 16 rows x 64 bytes. bf16 is 2 bytes, so one tile
// holds 16 rows x 32 bf16 elements. This block covers a 32x32 output by using
// a 2x2 grid of C tiles fed by 2 A tiles (32 rows) and 2 B tiles (32 columns).
template <bool accum, typename CB>
inline void {{kernel_name}}_amx_block32(
    AMXState& amx_state,
    const {{amx_t}}* A, const {{amx_t}}* B, float* C,
    int64_t K, int64_t lda, int64_t ldb, int64_t ldc, CB cb) {
  auto load_cfg = [](const amx_tilecfg& c) { _tile_loadconfig(&c); };
  // rows=16, colsb=64 bytes (=32 bf16 K per step); 2x2 grid -> 4 C tiles, 2 A, 2 B.
  amx_state.configure(16, 64, 2, 2, load_cfg);
  if constexpr (accum) {
    // Preload C so the tdp results add onto the running P@V (across kv blocks).
    // +16 selects the right-half columns; +16*ldc selects the bottom 16 rows.
    _tile_loadd(0, C, ldc * sizeof(float));
    _tile_loadd(1, C + 16, ldc * sizeof(float));
    _tile_loadd(2, C + 16 * ldc, ldc * sizeof(float));
    _tile_loadd(3, C + 16 * ldc + 16, ldc * sizeof(float));
  } else {
    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  }
  // Step by 32: each A tile consumes 32 K elements (64 bytes) per iteration.
  for (int64_t k = 0; k < K; k += 32) {
    const {{amx_t}}* Ak = A + k;
    const {{amx_t}}* Bk = B + k * ldb;
    int64_t kn = k + 32;
    if (kn < K) {
      // Prefetch the next K-step's B into L1: +32/+64 elems are +64B/+128B,
      // i.e. the first three consecutive 64-byte cache lines.
      const {{amx_t}}* Bp = B + kn * ldb;
      _mm_prefetch(reinterpret_cast<const char*>(Bp), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(Bp + 32), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(Bp + 64), _MM_HINT_T0);
    }
    // A is row-major bf16: row stride is lda elements.
    _tile_loadd(4, Ak, lda * sizeof({{amx_t}}));
    // B is VNNI2-packed [K/2, N, 2], so one packed row spans 2 logical K rows ->
    // stride ldb*2. The +32 offset is the next 16 N columns (16 N * 2 VNNI).
    _tile_loadd(6, Bk, ldb * 2 * sizeof({{amx_t}}));
    _tile_loadd(7, Bk + 32, ldb * 2 * sizeof({{amx_t}}));
    _tile_dpbf16ps(0, 4, 6);  // C[top-left]     = A[top]    @ B[left]
    _tile_loadd(5, Ak + 16 * lda, lda * sizeof({{amx_t}}));  // A rows 16-31
    _tile_dpbf16ps(1, 4, 7);  // C[top-right]    = A[top]    @ B[right]
    _tile_dpbf16ps(2, 5, 6);  // C[bottom-left]  = A[bottom] @ B[left]
    _tile_dpbf16ps(3, 5, 7);  // C[bottom-right] = A[bottom] @ B[right]
    cb();
  }
  _tile_stored(0, C, ldc * sizeof(float));
  _tile_stored(1, C + 16, ldc * sizeof(float));
  _tile_stored(2, C + 16 * ldc, ldc * sizeof(float));
  _tile_stored(3, C + 16 * ldc + 16, ldc * sizeof(float));
}

// 16-row variant: C[NROWS<=16,32] (+)= A[NROWS,K] @ Bp[K,32]. Covers the final
// 1..32 rows of a panel (a <16 remainder was zero-padded up to 16 by scale_q).
// Uses a single A tile (16 rows) x 2 B tiles (32 columns).
template <bool accum, typename CB>
inline void {{kernel_name}}_amx_block16(
    AMXState& amx_state,
    const {{amx_t}}* A, const {{amx_t}}* B, float* C,
    int64_t K, int64_t lda, int64_t ldb, int64_t ldc, CB cb) {
  auto load_cfg = [](const amx_tilecfg& c) { _tile_loadconfig(&c); };
  // 1x2 grid -> 2 C tiles, 1 A, 2 B.
  amx_state.configure(16, 64, 1, 2, load_cfg);
  if constexpr (accum) {
    _tile_loadd(0, C, ldc * sizeof(float));
    _tile_loadd(1, C + 16, ldc * sizeof(float));  // +16 -> right-half columns
  } else {
    _tile_zero(0); _tile_zero(1);
  }
  for (int64_t k = 0; k < K; k += 32) {
    const {{amx_t}}* Bk = B + k * ldb;
    int64_t kn = k + 32;
    if (kn < K) {
      const {{amx_t}}* Bp = B + kn * ldb;
      _mm_prefetch(reinterpret_cast<const char*>(Bp), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(Bp + 32), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(Bp + 64), _MM_HINT_T0);
    }
    _tile_loadd(2, A + k, lda * sizeof({{amx_t}}));
    // VNNI2 packing: packed-row stride ldb*2; +32 = next 16 N cols (16 N * 2 VNNI).
    _tile_loadd(3, Bk, ldb * 2 * sizeof({{amx_t}}));
    _tile_dpbf16ps(0, 2, 3);  // C[left]  = A @ B[left]
    _tile_loadd(4, Bk + 32, ldb * 2 * sizeof({{amx_t}}));
    _tile_dpbf16ps(1, 2, 4);  // C[right] = A @ B[right]
    cb();
  }
  _tile_stored(0, C, ldc * sizeof(float));
  _tile_stored(1, C + 16, ldc * sizeof(float));
}

// Main GEMM function - C[M,N] (+)= A[M,K] rowmajor(lda) @ Bp[K,N] VNNI2(ldb).
// Walks the output in 32-row x 32-col blocks (one AMX tile grid per block).
template <bool accum, typename CB>
inline void {{kernel_name}}_amx_gemm_cb(
    AMXState& amx_state,
    const {{amx_t}}* A, const {{amx_t}}* B, float* C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc, CB cb) {
  for (int64_t m = 0; m < M; m += 32) {
    // 32-row (2 M-tiles) panel when >=32 rows remain, else a 16-row panel that
    // covers the final 1..32 rows (rounding the last <16 remainder up to 16).
    int64_t nrows = (M - m) > 16 ? 32 : 16;
    for (int64_t n = 0; n < N; n += 32) {
      const {{amx_t}}* Ablk = A + m * lda;
      // VNNI2 layout [K/2, N, 2] interleaves 2 in the last dim, so column n
      // starts at element n*2 within a packed row.
      const {{amx_t}}* Bblk = B + n * 2;
      float* Cblk = C + m * ldc + n;
      if (nrows == 32)
        {{kernel_name}}_amx_block32<accum>(amx_state, Ablk, Bblk, Cblk, K, lda, ldb, ldc, cb);
      else
        {{kernel_name}}_amx_block16<accum>(amx_state, Ablk, Bblk, Cblk, K, lda, ldb, ldc, cb);
    }
  }
}

template <bool accum>
inline void {{kernel_name}}_amx_gemm(
    AMXState& amx_state,
    const {{amx_t}}* A, const {{amx_t}}* B, float* C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {
  {{kernel_name}}_amx_gemm_cb<accum>(amx_state, A, B, C, M, N, K, lda, ldb, ldc, []() {});
}
"""


def codegen_flex_attention_amx_helpers(kernel_name: str) -> str:
    """Render the AMX/AVX interleaving helpers for the given kernel-name prefix.

    ``amx_t`` is ``uint16_t`` -- the packed K/V and pre-scaled Q buffers are
    reinterpreted to 16-bit at the call sites (both BFloat16 and Half are 2-byte),
    matching what ``pack_vnni2`` and ``_tile_dpbf16ps`` consume.
    """
    from .common import KernelTemplate

    return KernelTemplate._template_from_string(FLEX_ATTENTION_AMX_HELPERS).render(
        dict(kernel_name=kernel_name, amx_t="uint16_t")
    )
