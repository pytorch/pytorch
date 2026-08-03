# mypy: allow-untyped-defs
"""AMX/AVX-512 interleaved GEMM helpers for the CPU FlexAttention template.

AMX tile registers only accept literal-immediate indices, so the two supported
row counts (16 and 32) are separate specializations with hardcoded tile indices
that follow ``AMXState``'s slot order (C tiles, then A, then B).
"""

# The C++ is a Jinja template only to substitute the kernel-name prefix so the
# symbols do not collide across multiple compiled kernels in one process.
FLEX_ATTENTION_AMX_HELPERS = r"""
// Zero-padded the rows up to a multiple of 16 so the AMX Q@K^T can read whole 16-row A tiles.
// Also fold the scale into the Q.
//   rows  -> real query rows; prows -> rows rounded up to a multiple of 16 (the AMX A-tile height)
//   cols  -> real head size;  pcols -> head size padded (eheadSize), the packed row stride of out_ptr
//   ldi   -> row stride of the source q_ptr
template <typename scalar_t>
inline void {{kernel_name}}_amx_scale_q(
    const scalar_t* q_ptr,
    scalar_t* out_ptr,
    float scale,
    int64_t rows,
    int64_t prows,
    int64_t cols,
    int64_t pcols,
    int64_t ldi) {
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t vec_size = Vec::size();
  auto fscale = at::vec::Vectorized<float>(scale);
  for (int64_t r = 0; r < rows; ++r) {
    const scalar_t* src = q_ptr + r * ldi;
    scalar_t* dst = out_ptr + r * pcols;
    int64_t c = 0;
    // bf16/half: widen to fp32, scale, narrow back so the multiply keeps fp32 precision.
    if constexpr (c10::is_reduced_floating_point_v<scalar_t>) {
      // vec_size * (cols / vec_size) is the largest whole-vector prefix of the row.
      for (; c < vec_size * (cols / vec_size); c += vec_size) {
        auto [v0, v1] = at::vec::convert_to_float<scalar_t>(Vec::loadu(src + c));
        at::vec::convert_from_float<scalar_t>(v0 * fscale, v1 * fscale).store(dst + c);
      }
      for (; c < cols; ++c) dst[c] = static_cast<scalar_t>(static_cast<float>(src[c]) * scale);
    } else {
      auto vscale = Vec(static_cast<scalar_t>(scale));
      for (; c < vec_size * (cols / vec_size); c += vec_size) {
        (Vec::loadu(src + c) * vscale).store(dst + c);
      }
      for (; c < cols; ++c) dst[c] = src[c] * static_cast<scalar_t>(scale);
    }
    // Zero the cols..pcols column padding so K^T reads garbage-free head-size lanes.
    for (int64_t p = cols; p < pcols; ++p) dst[p] = static_cast<scalar_t>(0);
  }
  // Zero the rows..prows row padding so the final A tile is a full 16 rows.
  for (int64_t r = rows; r < prows; ++r) {
    scalar_t* dst = out_ptr + r * pcols;
    for (int64_t p = 0; p < pcols; ++p) dst[p] = static_cast<scalar_t>(0);
  }
}

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

// One row of online softmax for the AMX bf16 path (scale already folded into Q).
// This is the AVX-512 work interleaved with the next block's AMX Q@K^T GEMM.
//   qk_row     -> this kv block's raw scores (overwritten in place)
//   p_row      -> output probabilities, fed to the P@V GEMM as VNNI2 A operand
//   row_max/row_sum -> running softmax statistics carried across kv blocks
//   dst_row    -> running P@V accumulator that must be rescaled when row_max grows
template <typename scalar_t>
inline void {{kernel_name}}_amx_online_softmax_row(
    float* qk_row,
    scalar_t* p_row,
    int64_t cur_kvSplitSize,
    float& row_max,
    float& row_sum,
    float* dst_row,
    int64_t headSize_v,
    bool first_block) {
  using Vec = at::vec::Vectorized<float>;
  float block_max = -std::numeric_limits<float>::infinity();
  // scale=1: scale is already folded into Q, so this just finds this block's max.
  {{kernel_name}}_mul_reduce_max_fusion_kernel(
      qk_row, static_cast<float>(1), cur_kvSplitSize, qk_row, block_max);
  float new_max = row_max > block_max ? row_max : block_max;
  if (new_max == -std::numeric_limits<float>::infinity()) {
    // Whole row masked out: emit zero probabilities, leave stats untouched.
    {{kernel_name}}_fill_stub(p_row, static_cast<scalar_t>(0), cur_kvSplitSize);
  } else {
    // exp_reduce_sum seeds val with new_max, so it computes exp(qk - new_max)
    // and returns sum() in block_sum.
    float block_sum = new_max;
    {{kernel_name}}_exp_reduce_sum_fusion_kernel(
        qk_row, cur_kvSplitSize, p_row, block_sum);
    // Rescale the previous running sum/accumulator from old row_max to new_max.
    float exp_tmp = std::exp(row_max - new_max);
    row_sum = block_sum + exp_tmp * row_sum;
    if (!first_block) {
      at::vec::map<float>(
          [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
          dst_row, dst_row, headSize_v);
    }
  }
  row_max = new_max;
  // P is the VNNI2 A operand of P@V; VNNI2 packs K in pairs, so an odd
  // cur_kvSplitSize needs one zero-padded column to complete the last pair.
  if (cur_kvSplitSize % 2 != 0) {
    p_row[cur_kvSplitSize] = static_cast<scalar_t>(0);
  }
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
