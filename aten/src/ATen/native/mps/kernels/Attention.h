// Parameters and constants shared by Attention.mm and the attention kernels.
#pragma once
#include <c10/metal/common.h>

// Decode takes up to 8 (kDecodeMaxQueries) query rows. Each row runs on one
// threadgroup of 1024 threads (kDecodeSimdgroups). From 1024 (kTwoPassMinKeys)
// keys and upwards, pass 1 instead runs each row on 32 (kDecodeBlocks)
// threadgroups of 256 threads (kDecode2PassSimdgroups), however many keys there
// are, to fill the GPU at small batch. Pass 2 merges the 32 partial results on
// one threadgroup of 1024 threads.
// DecodeAttention.h instantiates the kernels for each of 32, 64, 96, 128, 256
// and 512 (kDecodeTiles).
C10_METAL_CONSTEXPR int kDecodeMaxQueries = 8;
C10_METAL_CONSTEXPR int kTwoPassMinKeys = 1024;
C10_METAL_CONSTEXPR int kDecodeSimdgroups = 32;
C10_METAL_CONSTEXPR int kDecodeBlocks = 32;
C10_METAL_CONSTEXPR int kDecode2PassSimdgroups = 8;
C10_METAL_CONSTEXPR ::c10::metal::array<int, 6> kDecodeTiles =
    {32, 64, 96, 128, 256, 512};
static_assert(
    kDecodeSimdgroups * ::c10::metal::simdgroup_size <=
            ::c10::metal::max_threadgroup_size &&
        kDecodeBlocks * ::c10::metal::simdgroup_size <=
            ::c10::metal::max_threadgroup_size,
    "exceeds the Metal threadgroup size limit");

// Prefill tiles (padded head dims); PrefillAttention.h and MppAttention.h
// instantiate the kernels for each. Head dims above the widest tile run on
// 128 (kChunkTile): the QK dim in 128-wide chunks, V/O in 128-wide column
// slabs. The 256 MPP tile (kMppWideTile) is register-bound on Apple10, where
// the chunked 128 tile runs 1.7x faster, so it only serves an exact 256 head
// dim on earlier GPUs.
C10_METAL_CONSTEXPR ::c10::metal::array<int, 9> kPrefillTiles =
    {32, 64, 72, 80, 96, 128, 160, 192, 256};
C10_METAL_CONSTEXPR ::c10::metal::array<int, 3> kMppTiles = {64, 96, 128};
C10_METAL_CONSTEXPR int kMppWideTile = 256;
C10_METAL_CONSTEXPR int kChunkTile = 128;

// A prefill threadgroup takes a block of query rows and loops over the keys in
// blocks. It runs 4 (kPrefillSimdgroups) simdgroups, each owning 8 query rows
// on the simdgroup-matrix path or 16 on the MPP path.
// MPP path: 64-row Q blocks (kMppBQ), 32 keys at a time (kMppBK).
C10_METAL_CONSTEXPR int kPrefillSimdgroups = 4;
C10_METAL_CONSTEXPR int kMppBQ = kPrefillSimdgroups * 16;
C10_METAL_CONSTEXPR int kMppBK = 32;
// Simdgroup-matrix path: 32-row Q blocks (prefill_bq), 32 keys at a time
// (prefill_bk). Wide tiles take less so fp32 fits the 32 KB threadgroup memory:
// 16 keys on the 128 tile, 8 keys from 160 up, and 16 rows (2 simdgroups) on
// the 256 tile.
constexpr int prefill_bk(int bd) {
  return bd < 128 ? 32 : bd == 128 ? 16 : 8;
}
constexpr int prefill_simdgroups(int bd) {
  return bd == 256 ? 2 : kPrefillSimdgroups;
}
constexpr int prefill_bq(int bd) {
  return prefill_simdgroups(bd) * 8;
}

struct PrefillAttnParams {
  int B; // batch size
  int H; // number of query heads
  int D; // QK head dim; may exceed the tile width and is then chunked
  int qL; // query sequence length
  int kL; // key sequence length
  int gqa_factor; // num_q_heads / num_kv_heads
  float scale;
  int Dv; // V/O columns per slab, <= the tile width
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
// An additive mask enters the exp2 softmax as mask * log2(e). The product
// overflows finite sentinels such as finfo(float32).min to -inf, which would
// zero a fully masked row where the CPU math path (fp32) averages it:
// saturate instead, keeping a true -inf and NaN as they are.
METAL_FUNC float mask_log2e(float m) {
  const float m2 = m * M_LOG2E_F;
  return m2 == -INFINITY && m != -INFINITY ? -FLT_MAX : m2;
}
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>
#endif
