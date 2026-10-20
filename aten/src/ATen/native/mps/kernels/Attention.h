// Parameters and constants shared by Attention.mm and the attention kernels.
#pragma once
#include <c10/metal/common.h>

// Two-pass decode splits the keys of a (batch, head) row across kDecodeBlocks
// threadgroups of kDecodeBlockSimdgroups simdgroups (one key per simdgroup per
// step); one-pass decode uses a single threadgroup of kDecodeSimdgroups.
C10_METAL_CONSTEXPR int kDecodeBlocks = 32;
C10_METAL_CONSTEXPR int kDecodeSimdgroups = 32;
C10_METAL_CONSTEXPR int kDecodeBlockSimdgroups = 8;
// Threads sharing the partial blocks of one merged output element.
C10_METAL_CONSTEXPR int kMergeLanes = 8;
// Simdgroups per prefill threadgroup, stacked along the Q block: one 8-row
// simdgroup-matrix fragment or one 16-row MPP fragment each.
C10_METAL_CONSTEXPR int kPrefillSimdgroups = 4;
C10_METAL_CONSTEXPR int kPrefillBQ = kPrefillSimdgroups * 8;
C10_METAL_CONSTEXPR int kMppBQ = kPrefillSimdgroups * 16;
C10_METAL_CONSTEXPR int kMppBK = 32;
// Keys per K block of a simdgroup-matrix tile: the wide tiles take fewer so
// fp32 fits the 32 KB threadgroup memory (padded against bank conflicts at
// 128 and 160, unpadded at 192).
constexpr int prefill_bk(int bd) {
  return bd < 128 ? 32 : bd == 128 ? 16 : 8;
}

struct DecodeAttnParams {
  uint32_t N; // key sequence length
  uint32_t D; // QK head dim
  uint32_t Dv; // V head dim
  uint32_t gqa_factor; // num_q_heads / num_kv_heads
  uint32_t num_heads;
  float scale;
  // Element strides; the head-dim stride is 1 and the single query row has no
  // sequence stride. Mask strides are 0 along broadcast dims.
  ::c10::metal::array<uint32_t, 2> Q_strides; // (B, H)
  ::c10::metal::array<uint32_t, 3> K_strides; // (B, H, L)
  ::c10::metal::array<uint32_t, 3> V_strides; // (B, H, L)
  ::c10::metal::array<uint32_t, 3> M_strides; // (B, H, L)
};

struct PrefillAttnParams {
  int B; // batch size
  int H; // number of query heads
  int D; // QK head dim; may exceed the tile width and is then chunked
  int qL; // query sequence length
  int kL; // key sequence length
  int gqa_factor; // num_q_heads / num_kv_heads
  float scale;
  int Dv; // V/O columns per slab, <= the tile width
  int n_splits; // threadgroups sharing a Q block along kL; > 1 writes partials
  int n_slabs; // Dv-wide V/O column slabs, the outermost factor of the grid's z
  int n_score_splits; // splits of the staged scores pass (P @ V merges them)
  // Strides (B, H, L) - element stride in last dim is assumed to be 1.
  // 64-bit: a batch/head stride is a product of the other dimensions and
  // overflows 32 bits on large inputs, and the mask qL stride is multiplied
  // by an absolute query position, reaching qL * kL.
  ::c10::metal::array<int64_t, 3> Q_strides;
  ::c10::metal::array<int64_t, 3> K_strides;
  ::c10::metal::array<int64_t, 3> V_strides;
  ::c10::metal::array<int64_t, 3> O_strides;
};

struct PrefillAttnMaskParams {
  ::c10::metal::array<int64_t, 4> M_strides; // (B, H, qL, kL)
};

#ifdef __METAL__
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>
#endif
